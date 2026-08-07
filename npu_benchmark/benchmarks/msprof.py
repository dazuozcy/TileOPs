"""msprof op-based kernel profiling for Ascend NPU.

Uses ``msprof op --kernel-name=xxx --output=xxx --launch-count=N
--warm-up=M python xxx.py`` to measure kernel latency at hardware level.
Parses ``OpBasicInfo_*.csv`` for the ``Task Duration(us)`` field.

Activate via ``NPU_BENCH_TIMING=msprof`` env var.

Env vars:
  NPU_BENCH_MSPROF_KERNEL_NAME   Kernel name for --kernel-name (auto-detected)
  NPU_BENCH_MSPROF_LAUNCH_COUNT  Timed launches collected  (default 10)
  NPU_BENCH_MSPROF_WARM_UP       Warm-up launches skipped  (default 5)
  NPU_BENCH_MSPROF_OUTPUT        Fixed output directory    (temp dir if unset)
  NPU_BENCH_MSPROF_KEEP_OUTPUT   Set to "0" to auto-delete  (default keep)
"""

from __future__ import annotations

import csv
import glob
import inspect
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Optional

_logger = logging.getLogger("npu_bench")

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_NPU_BENCHMARK_DIR = os.path.dirname(_BENCH_DIR)

_SKIP_PARAMS = {"self", "kernel_map"}

_DURATION_COL_CANDIDATES = [
    "Task Duration(us)",
    "Task Duration",
    "Task Duration (us)",
    "task duration(us)",
    "Task Duration(\xb5s)",
]


def _get_importable_class(obj: Any, package_prefix: str) -> type:
    """Walk MRO to find the first class defined in *package_prefix*."""
    for cls in type(obj).__mro__:
        mod = getattr(cls, "__module__", "")
        if mod.startswith(package_prefix) and mod != "__main__":
            return cls
    return type(obj)


def _get_reconstruct_kwargs(obj: Any) -> dict[str, Any]:
    """Introspect constructor args stored as same-named instance attributes."""
    sig = inspect.signature(type(obj).__init__)
    kwargs: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name in _SKIP_PARAMS:
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL,
                          inspect.Parameter.VAR_KEYWORD):
            continue
        if hasattr(obj, name):
            kwargs[name] = getattr(obj, name)
    return kwargs


def _generate_launch_script(
    op_module: str,
    op_class_name: str,
    op_kwargs: dict[str, Any],
    workload_module: str,
    workload_class_name: str,
    workload_args: dict[str, Any],
    warm_up: int,
    launch_count: int,
) -> str:
    """Generate a standalone Python script that launches the kernel."""
    return f'''import os
import sys

os.environ.setdefault("TILELANG_ASCEND_MODE", "Dev")
sys.path.insert(0, {_NPU_BENCHMARK_DIR!r})

import torch
import torch_npu  # noqa: F401  registers .npu() device

from {op_module} import {op_class_name}
from {workload_module} import {workload_class_name}

workload = {workload_class_name}(**{workload_args!r})
inputs = workload.gen_inputs()
op = {op_class_name}(**{op_kwargs!r})

warm_up = {warm_up}
launch_count = {launch_count}

with torch.no_grad():
    for _ in range(warm_up + launch_count):
        op(*inputs)
torch.npu.synchronize()
print("msprof launch script done")
'''


def _find_duration_column(header: list[str]) -> Optional[str]:
    """Find the Task Duration column in a CSV header."""
    for candidate in _DURATION_COL_CANDIDATES:
        if candidate in header:
            return candidate
    for col in header:
        low = col.strip().lower()
        if "task duration" in low or "taskduration" in low:
            return col
    return None


def _parse_op_basic_info(output_dir: str) -> list[float]:
    """Parse ``OpBasicInfo_*.csv`` for ``Task Duration(us)`` values."""
    pattern = os.path.join(output_dir, "**", "OpBasicInfo*.csv")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No OpBasicInfo_*.csv found under {output_dir}")

    durations: list[float] = []
    for filepath in files:
        with open(filepath, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue
            col = _find_duration_column(list(reader.fieldnames))
            if col is None:
                _logger.warning(
                    "Task Duration column not found in %s. Headers: %s",
                    filepath, reader.fieldnames)
                continue
            for row in reader:
                val = (row.get(col) or "").strip()
                if val:
                    try:
                        durations.append(float(val))
                    except ValueError:
                        pass
    return durations


def bench_kernel_msprof(
    op: Any,
    workload: Any,
    kernel_name: str = "main",
    launch_count: Optional[int] = None,
    warm_up: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> float:
    """Benchmark a kernel using ``msprof op``.

    Generates a launch script, runs ``msprof op``, parses
    ``OpBasicInfo_*.csv`` for ``Task Duration(us)``, and returns the
    median latency in **milliseconds**.

    The Op is reconstructed in a fresh subprocess with ``tune=False``
    (default config).  Autotuned configs from the parent process are
    not transferred — the default configs are already tuned per
    ``Kernel.default_config``.
    """
    launch_count = launch_count or int(
        os.getenv("NPU_BENCH_MSPROF_LAUNCH_COUNT", "10"))
    warm_up = warm_up or int(
        os.getenv("NPU_BENCH_MSPROF_WARM_UP", "5"))

    keep_output = os.getenv("NPU_BENCH_MSPROF_KEEP_OUTPUT", "1") != "0"
    if output_dir is None:
        output_dir = os.getenv("NPU_BENCH_MSPROF_OUTPUT")
    _is_temp_dir = False
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="msprof_")
        _is_temp_dir = True

    os.makedirs(output_dir, exist_ok=True)

    op_class = _get_importable_class(op, "ops")
    workload_class = _get_importable_class(workload, "workloads")

    op_kwargs = _get_reconstruct_kwargs(op)
    op_kwargs["tune"] = False
    workload_args = _get_reconstruct_kwargs(workload)

    script = _generate_launch_script(
        op_class.__module__, op_class.__name__, op_kwargs,
        workload_class.__module__, workload_class.__name__, workload_args,
        0, 1,
    )
    script_path = os.path.join(output_dir, "launch_kernel.py")
    with open(script_path, "w") as f:
        f.write(script)

    cmd = [
        "msprof", "op",
        f"--kernel-name={kernel_name}",
        f"--output={output_dir}",
        f"--launch-count={launch_count}",
        f"--warm-up={warm_up}",
        # These two lines below avoid excessive profiling time caused by parsing
        # too much data; here we only care about latency, not the detailed specifics.
        f"--aic-metrics=TimelineDetail",
        f"--dump=off",
        sys.executable, script_path,
    ]
    _logger.info("Running msprof: %s", " ".join(cmd))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=600)
    except FileNotFoundError:
        raise RuntimeError(
            "msprof command not found. Please install CANN msprof.")
    except subprocess.TimeoutExpired:
        raise RuntimeError("msprof op timed out after 600 s")

    if result.returncode != 0:
        _logger.error("msprof stdout:\n%s", result.stdout)
        _logger.error("msprof stderr:\n%s", result.stderr)
        raise RuntimeError(
            f"msprof op failed with code {result.returncode}")

    if result.stdout:
        tail = result.stdout[-800:]
        _logger.info("msprof stdout (tail):\n%s", tail)

    try:
        durations_us = _parse_op_basic_info(output_dir)
    finally:
        if _is_temp_dir and not keep_output:
            shutil.rmtree(output_dir, ignore_errors=True)

    if not durations_us:
        raise RuntimeError(
            f"No Task Duration values found in {output_dir}. "
            f"Check msprof output for errors.")

    _logger.info("Collected %d duration samples (us): [%s, ...]",
                 len(durations_us),
                 ", ".join(f"{d:.2f}" for d in durations_us[:10]))

    durations_us.sort()
    median_us = durations_us[len(durations_us) // 2]
    return median_us / 1000.0

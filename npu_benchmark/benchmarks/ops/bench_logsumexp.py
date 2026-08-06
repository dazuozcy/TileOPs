"""Benchmark for the LogSumExp forward op on Ascend NPU.

Workload shapes and dtypes come from the manifest; roofline FLOP and byte
counts come from the op's ``eval_roofline()`` via
:class:`ManifestBenchmark`.

NOTE: The NPU ``LogSumExpKernel`` is currently a placeholder stub
(``forward()`` raises ``NotImplementedError``).  This benchmark is marked
``skip`` until the kernel is implemented — remove the marker when the
real NPU reduction kernel lands.
"""

from __future__ import annotations

import pytest
import torch

from benchmarks.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workload_field_params,
)
from manifest import load_workloads
from ops import LogSumExpFwdOp
from workloads.logsumexp import LogSumExpWorkload

_TUNE = True
_LOGSUMEXP_OP = "LogSumExpFwdOp"
_LOGSUMEXP_PARAMS = workload_field_params(
    load_workloads(_LOGSUMEXP_OP),
    ("x_shape", "dtype", "dim", "keepdim"),
)


@pytest.mark.skip(
    reason="LogSumExpKernel NPU implementation is a placeholder",
)
@pytest.mark.parametrize(
    "x_shape, dtype, dim, keepdim",
    _LOGSUMEXP_PARAMS,
)
def test_logsumexp_bench(x_shape, dtype: torch.dtype,
                         dim: int, keepdim: bool) -> None:
    test = LogSumExpWorkload(x_shape, dtype, dim=dim, keepdim=keepdim)
    inputs = test.gen_inputs()

    op = LogSumExpFwdOp(dim=dim, keepdim=keepdim, tune=_TUNE)
    bm = ManifestBenchmark(_LOGSUMEXP_OP, op, test)

    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="kernel")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])

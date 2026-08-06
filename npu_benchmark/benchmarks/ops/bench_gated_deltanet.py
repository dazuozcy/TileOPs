"""Benchmark for the Gated DeltaNet forward op on Ascend NPU.

The NPU ``GatedDeltaNetFwdKernel`` is currently a stub (empty body).  The
kernel compiles and runs (producing undefined output) so the benchmark
timing / roofline flow executes normally.
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
from ops import GatedDeltaNetFwdOp
from workloads.gated_deltanet import GatedDeltaNetFwdWorkload

_TUNE = True
_OP_NAME = "GatedDeltaNetFwdOp"
_PARAMS = workload_field_params(
    load_workloads(_OP_NAME),
    ("q_shape", "k_shape", "v_shape", "g_shape", "beta_shape",
     "chunk_size", "dtype"),
)


@pytest.mark.parametrize(
    "q_shape, k_shape, v_shape, g_shape, beta_shape, chunk_size, dtype",
    _PARAMS,
)
def test_gated_deltanet_bench(q_shape, k_shape, v_shape, g_shape,
                              beta_shape, chunk_size,
                              dtype: torch.dtype) -> None:
    test = GatedDeltaNetFwdWorkload(q_shape, k_shape, v_shape, g_shape,
                                    beta_shape, chunk_size, dtype)
    inputs = test.gen_inputs()

    op = GatedDeltaNetFwdOp(chunk_size=chunk_size, tune=_TUNE)
    bm = ManifestBenchmark(_OP_NAME, op, test)

    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="kernel")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])

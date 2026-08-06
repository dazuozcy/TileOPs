"""Benchmark for the Argmax forward op on Ascend NPU.

The NPU ``ArgmaxKernel`` is currently a stub (empty body).  The kernel
compiles and runs (producing undefined output) so the benchmark timing
/ roofline flow executes normally.
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
from ops import ArgmaxFwdOp
from workloads.argmax import ArgmaxWorkload

_TUNE = True
_ARGMAX_OP = "ArgmaxFwdOp"
_PARAMS = workload_field_params(
    load_workloads(_ARGMAX_OP),
    ("x_shape", "dtype", "dim", "keepdim"),
)


@pytest.mark.parametrize(
    "x_shape, dtype, dim, keepdim",
    _PARAMS,
)
def test_argmax_bench(x_shape, dtype: torch.dtype,
                      dim: int, keepdim: bool) -> None:
    test = ArgmaxWorkload(x_shape, dtype, dim=dim, keepdim=keepdim)
    inputs = test.gen_inputs()

    op = ArgmaxFwdOp(dim=dim, keepdim=keepdim, tune=_TUNE)
    bm = ManifestBenchmark(_ARGMAX_OP, op, test)

    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="kernel")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])

"""Benchmark for the MaxPool3d forward op on Ascend NPU.

The NPU ``MaxPool3dKernel`` is currently a stub (empty body).  The kernel
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
from ops import MaxPool3dFwdOp
from workloads.max_pool3d import MaxPool3dWorkload

_TUNE = True
_MAX_POOL3D_OP = "MaxPool3dFwdOp"
_PARAMS = workload_field_params(
    load_workloads(_MAX_POOL3D_OP),
    ("input_shape", "kernel_size", "stride", "padding",
     "dilation", "ceil_mode", "dtype"),
)


@pytest.mark.parametrize(
    "input_shape, kernel_size, stride, padding, dilation, ceil_mode, dtype",
    _PARAMS,
)
def test_max_pool3d_bench(input_shape, kernel_size, stride, padding,
                          dilation, ceil_mode,
                          dtype: torch.dtype) -> None:
    test = MaxPool3dWorkload(input_shape, kernel_size, dtype,
                             stride=stride, padding=padding,
                             dilation=dilation, ceil_mode=ceil_mode)
    inputs = test.gen_inputs()

    op = MaxPool3dFwdOp(kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation,
                        ceil_mode=ceil_mode, tune=_TUNE)
    bm = ManifestBenchmark(_MAX_POOL3D_OP, op, test)

    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="kernel")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])

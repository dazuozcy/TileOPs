"""Benchmark for the AvgPool2d forward op on Ascend NPU.

The NPU ``AvgPool2dKernel`` is currently a stub (empty body).  The kernel
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
from ops import AvgPool2dFwdOp
from workloads.avg_pool2d import AvgPool2dWorkload

_TUNE = True
_AVG_POOL2D_OP = "AvgPool2dFwdOp"
_PARAMS = workload_field_params(
    load_workloads(_AVG_POOL2D_OP),
    ("input_shape", "kernel_size", "stride", "padding",
     "ceil_mode", "count_include_pad", "dtype"),
)


@pytest.mark.parametrize(
    "input_shape, kernel_size, stride, padding, ceil_mode, count_include_pad, dtype",
    _PARAMS,
)
def test_avg_pool2d_bench(input_shape, kernel_size, stride, padding,
                          ceil_mode, count_include_pad,
                          dtype: torch.dtype) -> None:
    test = AvgPool2dWorkload(input_shape, kernel_size, dtype,
                             stride=stride, padding=padding,
                             ceil_mode=ceil_mode,
                             count_include_pad=count_include_pad)
    inputs = test.gen_inputs()

    op = AvgPool2dFwdOp(kernel_size=kernel_size, stride=stride,
                        padding=padding, ceil_mode=ceil_mode,
                        count_include_pad=count_include_pad, tune=_TUNE)
    bm = ManifestBenchmark(_AVG_POOL2D_OP, op, test)

    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="kernel")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])

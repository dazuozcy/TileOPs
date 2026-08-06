"""Benchmark for the Conv2d forward op on Ascend NPU.

The NPU ``Conv2dKernel`` is currently a stub (empty body).  The kernel
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
from ops import Conv2dFwdOp
from workloads.conv2d import Conv2dWorkload

_TUNE = True
_CONV2D_OP = "Conv2dFwdOp"
_PARAMS = workload_field_params(
    load_workloads(_CONV2D_OP),
    ("input_shape", "C_out", "kH", "kW",
     "stride", "padding", "dilation", "groups", "dtype"),
)


@pytest.mark.parametrize(
    "input_shape, C_out, kH, kW, stride, padding, dilation, groups, dtype",
    _PARAMS,
)
def test_conv2d_bench(input_shape, C_out, kH, kW,
                      stride, padding, dilation, groups,
                      dtype: torch.dtype) -> None:
    test = Conv2dWorkload(input_shape, C_out, kH, kW, dtype,
                          stride=stride, padding=padding,
                          dilation=dilation, groups=groups)
    inputs = test.gen_inputs()

    op = Conv2dFwdOp(stride=stride, padding=padding,
                     dilation=dilation, groups=groups, tune=_TUNE)
    bm = ManifestBenchmark(_CONV2D_OP, op, test)

    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="kernel")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])

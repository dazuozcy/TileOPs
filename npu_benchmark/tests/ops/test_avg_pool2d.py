"""Correctness test for the AvgPool2d forward op.

Compares kernel output against ``torch.nn.functional.avg_pool2d`` with
element-wise allclose.  Tolerances are dtype-aware (see
``tests.tolerances``).

NOTE: The NPU ``AvgPool2dKernel`` is currently a stub (empty body).
The kernel compiles and runs but produces undefined output values, so
correctness tests are expected to fail until the real pooling body
is implemented.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from manifest import load_workloads
from ops import AvgPool2dFwdOp
from benchmarks.benchmark_base import workload_field_params
from workloads.avg_pool2d import AvgPool2dWorkload
from tests.tolerances import dtype_tolerances

_PARAMS = workload_field_params(
    load_workloads("AvgPool2dFwdOp"),
    ("input_shape", "kernel_size", "stride", "padding",
     "ceil_mode", "count_include_pad", "dtype"),
)


@pytest.mark.parametrize(
    "input_shape, kernel_size, stride, padding, ceil_mode, count_include_pad, dtype",
    _PARAMS,
)
def test_avg_pool2d_fwd_op(input_shape, kernel_size, stride, padding,
                           ceil_mode, count_include_pad,
                           dtype: torch.dtype) -> None:
    wl = AvgPool2dWorkload(input_shape, kernel_size, dtype,
                           stride=stride, padding=padding,
                           ceil_mode=ceil_mode,
                           count_include_pad=count_include_pad)
    inputs = wl.gen_inputs()
    op = AvgPool2dFwdOp(kernel_size=kernel_size, stride=stride,
                        padding=padding, ceil_mode=ceil_mode,
                        count_include_pad=count_include_pad, tune=False)
    output = op(*inputs)

    input = inputs[0]
    ref = F.avg_pool2d(input, kernel_size, stride=stride, padding=padding,
                       ceil_mode=ceil_mode,
                       count_include_pad=count_include_pad)
    rtol, atol = dtype_tolerances(dtype)
    torch.testing.assert_close(output, ref, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])

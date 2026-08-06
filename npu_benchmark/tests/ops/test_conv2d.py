"""Correctness test for the Conv2d forward op.

Compares kernel output against ``torch.nn.functional.conv2d`` with
element-wise allclose.  Tolerances are dtype-aware (see
``tests.tolerances``).

NOTE: The NPU ``Conv2dKernel`` is currently a stub (empty body).
The kernel compiles and runs but produces undefined output values, so
correctness tests are expected to fail until the real convolution body
is implemented.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from manifest import load_workloads
from ops import Conv2dFwdOp
from benchmarks.benchmark_base import workload_field_params
from workloads.conv2d import Conv2dWorkload
from tests.tolerances import dtype_tolerances

_PARAMS = workload_field_params(
    load_workloads("Conv2dFwdOp"),
    ("input_shape", "C_out", "kH", "kW",
     "stride", "padding", "dilation", "groups", "dtype"),
)


@pytest.mark.parametrize(
    "input_shape, C_out, kH, kW, stride, padding, dilation, groups, dtype",
    _PARAMS,
)
def test_conv2d_fwd_op(input_shape, C_out, kH, kW,
                       stride, padding, dilation, groups,
                       dtype: torch.dtype) -> None:
    wl = Conv2dWorkload(input_shape, C_out, kH, kW, dtype,
                        stride=stride, padding=padding,
                        dilation=dilation, groups=groups)
    inputs = wl.gen_inputs()
    op = Conv2dFwdOp(stride=stride, padding=padding,
                     dilation=dilation, groups=groups, tune=False)
    output = op(*inputs)

    input, weight = inputs
    ref = F.conv2d(input, weight, stride=stride, padding=padding,
                   dilation=dilation, groups=groups)
    rtol, atol = dtype_tolerances(dtype)
    torch.testing.assert_close(output, ref, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])

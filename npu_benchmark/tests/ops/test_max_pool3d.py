"""Correctness test for the MaxPool3d forward op.

Compares kernel output against ``torch.nn.functional.max_pool3d`` with
element-wise allclose.  Tolerances are dtype-aware (see
``tests.tolerances``).

NOTE: The NPU ``MaxPool3dKernel`` is currently a stub (empty body).
The kernel compiles and runs but produces undefined output values, so
correctness tests are expected to fail until the real pooling body
is implemented.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from manifest import load_workloads
from ops import MaxPool3dFwdOp
from benchmarks.benchmark_base import workload_field_params
from workloads.max_pool3d import MaxPool3dWorkload
from tests.tolerances import dtype_tolerances

_PARAMS = workload_field_params(
    load_workloads("MaxPool3dFwdOp"),
    ("input_shape", "kernel_size", "stride", "padding",
     "dilation", "ceil_mode", "dtype"),
)


@pytest.mark.parametrize(
    "input_shape, kernel_size, stride, padding, dilation, ceil_mode, dtype",
    _PARAMS,
)
def test_max_pool3d_fwd_op(input_shape, kernel_size, stride, padding,
                           dilation, ceil_mode,
                           dtype: torch.dtype) -> None:
    wl = MaxPool3dWorkload(input_shape, kernel_size, dtype,
                           stride=stride, padding=padding,
                           dilation=dilation, ceil_mode=ceil_mode)
    inputs = wl.gen_inputs()
    op = MaxPool3dFwdOp(kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation,
                        ceil_mode=ceil_mode, tune=False)
    output = op(*inputs)

    input = inputs[0]
    ref = F.max_pool3d(input, kernel_size, stride=stride, padding=padding,
                       dilation=dilation, ceil_mode=ceil_mode)
    rtol, atol = dtype_tolerances(dtype)
    torch.testing.assert_close(output, ref, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])

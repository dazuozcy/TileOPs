"""Correctness test for the LogSumExp forward op.

Compares kernel output against ``torch.logsumexp`` with element-wise
allclose.  Tolerances are dtype-aware (see ``tests.tolerances``).

NOTE: The NPU ``LogSumExpKernel`` is currently a placeholder stub
(``forward()`` raises ``NotImplementedError``).  This test is marked
``skip`` until the kernel is implemented — remove the marker when the
real NPU reduction kernel lands.
"""

from __future__ import annotations

import pytest
import torch

from manifest import load_workloads
from ops import LogSumExpFwdOp
from benchmarks.benchmark_base import workload_field_params
from workloads.logsumexp import LogSumExpWorkload
from tests.tolerances import dtype_tolerances

_PARAMS = workload_field_params(
    load_workloads("LogSumExpFwdOp"),
    ("x_shape", "dtype", "dim", "keepdim"),
)


@pytest.mark.skip(
    reason="LogSumExpKernel NPU implementation is a placeholder",
)
@pytest.mark.parametrize(
    "x_shape, dtype, dim, keepdim",
    _PARAMS,
)
def test_logsumexp_fwd_op(x_shape, dtype: torch.dtype,
                          dim: int, keepdim: bool) -> None:
    wl = LogSumExpWorkload(x_shape, dtype, dim=dim, keepdim=keepdim)
    inputs = wl.gen_inputs()
    op = LogSumExpFwdOp(dim=dim, keepdim=keepdim, tune=False)
    output = op(*inputs)

    x = inputs[0]
    ref = torch.logsumexp(x, dim=dim, keepdim=keepdim)
    rtol, atol = dtype_tolerances(dtype)
    torch.testing.assert_close(output, ref, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])

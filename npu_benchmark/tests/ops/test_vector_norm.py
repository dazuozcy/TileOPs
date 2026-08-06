"""Correctness tests for vector-norm ops (L1, L2, inf).

Compares kernel output against ``torch.linalg.vector_norm`` with
element-wise allclose.  Tolerances are dtype-aware (see
``tests.tolerances``).

NOTE: The NPU ``VectorNormKernel`` is currently a stub (empty body).
The kernel compiles and runs but produces undefined output values, so
correctness tests are expected to fail until the real reduction body
is implemented.
"""

from __future__ import annotations

import pytest
import torch

from manifest import load_workloads
from ops import L1NormFwdOp, L2NormFwdOp, InfNormFwdOp
from benchmarks.benchmark_base import workload_field_params
from workloads.vector_norm import VectorNormWorkload
from tests.tolerances import dtype_tolerances

_PARAMS = workload_field_params(
    load_workloads("L1NormFwdOp"),
    ("x_shape", "dtype", "dim", "keepdim"),
)


@pytest.mark.parametrize("x_shape, dtype, dim, keepdim", _PARAMS)
def test_l1_norm_fwd_op(x_shape, dtype: torch.dtype,
                        dim: int, keepdim: bool) -> None:
    wl = VectorNormWorkload(x_shape, dtype, dim=dim, keepdim=keepdim)
    inputs = wl.gen_inputs()
    op = L1NormFwdOp(dim=dim, keepdim=keepdim, tune=False)
    output = op(*inputs)

    x = inputs[0]
    ref = torch.linalg.vector_norm(x, ord=1, dim=dim, keepdim=keepdim)
    rtol, atol = dtype_tolerances(dtype)
    torch.testing.assert_close(output, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("x_shape, dtype, dim, keepdim", _PARAMS)
def test_l2_norm_fwd_op(x_shape, dtype: torch.dtype,
                        dim: int, keepdim: bool) -> None:
    wl = VectorNormWorkload(x_shape, dtype, dim=dim, keepdim=keepdim)
    inputs = wl.gen_inputs()
    op = L2NormFwdOp(dim=dim, keepdim=keepdim, tune=False)
    output = op(*inputs)

    x = inputs[0]
    ref = torch.linalg.vector_norm(x, ord=2, dim=dim, keepdim=keepdim)
    rtol, atol = dtype_tolerances(dtype)
    torch.testing.assert_close(output, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("x_shape, dtype, dim, keepdim", _PARAMS)
def test_inf_norm_fwd_op(x_shape, dtype: torch.dtype,
                         dim: int, keepdim: bool) -> None:
    wl = VectorNormWorkload(x_shape, dtype, dim=dim, keepdim=keepdim)
    inputs = wl.gen_inputs()
    op = InfNormFwdOp(dim=dim, keepdim=keepdim, tune=False)
    output = op(*inputs)

    x = inputs[0]
    ref = torch.linalg.vector_norm(x, ord=float("inf"), dim=dim, keepdim=keepdim)
    rtol, atol = dtype_tolerances(dtype)
    torch.testing.assert_close(output, ref, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])

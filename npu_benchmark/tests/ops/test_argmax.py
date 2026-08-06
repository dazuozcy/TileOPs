"""Correctness test for the Argmax forward op.

Compares kernel output against ``torch.argmax`` with element-wise
allclose.  Tolerances are dtype-aware (see ``tests.tolerances``).

NOTE: The NPU ``ArgmaxKernel`` is currently a stub (empty body).
The kernel compiles and runs but produces undefined output values, so
correctness tests are expected to fail until the real reduction body
is implemented.
"""

from __future__ import annotations

import pytest
import torch

from manifest import load_workloads
from ops import ArgmaxFwdOp
from benchmarks.benchmark_base import workload_field_params
from workloads.argmax import ArgmaxWorkload
from tests.tolerances import dtype_tolerances

_PARAMS = workload_field_params(
    load_workloads("ArgmaxFwdOp"),
    ("x_shape", "dtype", "dim", "keepdim"),
)


@pytest.mark.parametrize(
    "x_shape, dtype, dim, keepdim",
    _PARAMS,
)
def test_argmax_fwd_op(x_shape, dtype: torch.dtype,
                       dim: int, keepdim: bool) -> None:
    wl = ArgmaxWorkload(x_shape, dtype, dim=dim, keepdim=keepdim)
    inputs = wl.gen_inputs()
    op = ArgmaxFwdOp(dim=dim, keepdim=keepdim, tune=False)
    output = op(*inputs)

    x = inputs[0]
    ref = torch.argmax(x, dim=dim, keepdim=keepdim)
    rtol, atol = dtype_tolerances(dtype)
    torch.testing.assert_close(output, ref, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])

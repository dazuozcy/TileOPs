"""Correctness test for the Mish activation op.

Compares kernel output against torch reference with element-wise allclose.
Tolerances are dtype-aware (see ``tests.tolerances``).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from manifest import load_workloads
from ops import MishFwdOp
from benchmarks.benchmark_base import workload_field_params
from workloads.mish import MishWorkload
from tests.tolerances import dtype_tolerances

_PARAMS = workload_field_params(
    load_workloads("MishFwdOp"),
    ("input_shape", "dtype"),
)


@pytest.mark.parametrize(
    "input_shape, dtype",
    _PARAMS,
)
def test_mish_fwd_op(input_shape, dtype: torch.dtype) -> None:
    wl = MishWorkload(input_shape, dtype)
    inputs = wl.gen_inputs()
    op = MishFwdOp(tune=False)
    output = op(*inputs)

    input = inputs[0]
    ref = input * torch.tanh(F.softplus(input))
    rtol, atol = dtype_tolerances(dtype)
    torch.testing.assert_close(output, ref, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])

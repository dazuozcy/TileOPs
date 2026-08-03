"""Correctness test for the tensor-weight lerp op.

Compares kernel output against torch.lerp with element-wise allclose.
Tolerances are dtype-aware (see ``tests.tolerances``).
"""

from __future__ import annotations

import pytest
import torch

from manifest import load_workloads
from ops import LerpTensorOp
from benchmarks.benchmark_base import workload_field_params
from workloads.lerp_tensor import LerpTensorWorkload
from tests.tolerances import dtype_tolerances

_PARAMS = workload_field_params(
    load_workloads("LerpTensorOp"),
    ("input_shape", "dtype"),
)


@pytest.mark.parametrize(
    "input_shape, dtype",
    _PARAMS,
)
def test_lerp_tensor_op(input_shape, dtype: torch.dtype) -> None:
    wl = LerpTensorWorkload(input_shape, dtype)
    inputs = wl.gen_inputs()
    op = LerpTensorOp(tune=False)
    output = op(*inputs)

    input, end, weight = inputs
    ref = torch.lerp(input, end, weight)
    rtol, atol = dtype_tolerances(dtype)
    torch.testing.assert_close(output, ref, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])

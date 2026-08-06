"""Correctness test for the MoE grouped GEMM forward op.

Compares kernel output against a per-expert PyTorch matmul reference.
Tolerances are dtype-aware (see ``tests.tolerances``).

NOTE: The NPU ``MoeGroupedGemmNopadKernel`` is currently a stub (empty
body).  The kernel compiles and runs but produces undefined output values,
so correctness tests are expected to fail until the real GEMM body is
implemented.
"""

from __future__ import annotations

import pytest
import torch

from manifest import load_workloads
from ops import MoeGroupedGemmNopadFwdOp
from benchmarks.benchmark_base import workload_field_params
from workloads.moe_grouped_gemm_nopad import MoeGroupedGemmNopadFwdWorkload
from tests.tolerances import dtype_tolerances

_PARAMS = workload_field_params(
    load_workloads("MoeGroupedGemmNopadFwdOp"),
    ("numel", "num_experts", "n", "k", "dtype"),
)


def _ref_moe_grouped_gemm(a, b, true_sizes, true_offsets):
    """Per-expert NT GEMM reference."""
    numel, K = a.shape
    E, N, _ = b.shape

    c = torch.zeros(numel, N, dtype=a.dtype, device=a.device)
    af = a.float()
    bf = b.float()
    cf = torch.zeros(numel, N, dtype=torch.float32, device=a.device)

    for e in range(E):
        start = int(true_offsets[e])
        size = int(true_sizes[e])
        if size > 0:
            cf[start:start + size] = af[start:start + size] @ bf[e].T

    return cf.to(a.dtype)


@pytest.mark.parametrize(
    "numel, num_experts, n, k, dtype",
    _PARAMS,
)
def test_moe_grouped_gemm_nopad_fwd_op(numel, num_experts, n, k,
                                       dtype: torch.dtype) -> None:
    wl = MoeGroupedGemmNopadFwdWorkload(numel, num_experts, n, k, dtype)
    inputs = wl.gen_inputs()
    op = MoeGroupedGemmNopadFwdOp(numel, num_experts, n, k, dtype=dtype,
                                 tune=False)
    output = op(*inputs)

    ref_c = _ref_moe_grouped_gemm(*inputs)

    rtol, atol = dtype_tolerances(dtype)
    torch.testing.assert_close(output, ref_c, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])

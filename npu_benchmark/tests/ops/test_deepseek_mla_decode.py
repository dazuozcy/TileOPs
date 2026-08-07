"""Correctness test for the MLA decode op.

Compares kernel output against a naive PyTorch reference.  Tolerances are
dtype-aware (see ``tests.tolerances``).

NOTE: The NPU ``MLADecodeKernel`` is currently a stub (empty body).
The kernel compiles and runs but produces undefined output values, so
correctness tests are expected to fail until the real forward body is
implemented.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from manifest import load_workloads
from ops import MultiHeadLatentAttentionDecodeWithKVCacheFwdOp
from benchmarks.benchmark_base import workload_field_params
from workloads.deepseek_mla_decode import MlaDecodeWorkload
from tests.tolerances import dtype_tolerances

_PARAMS = workload_field_params(
    load_workloads("MultiHeadLatentAttentionDecodeWithKVCacheFwdOp"),
    ("q_shape", "q_pe_shape", "kv_shape", "k_pe_shape", "pe_dim", "dtype"),
)


def _ref_mla_decode(q, q_pe, k, k_pe):
    """Naive MLA decode reference (PyTorch, float32 intermediate).

    Inputs:
      q     [B, H, D]
      q_pe  [B, H, pe_dim]
      k     [B, N_kv, H_kv, D]
      k_pe  [B, N_kv, H_kv, pe_dim]
    Output:
      o     [B, H, D]
    """
    B, H, D = q.shape
    pe_dim = q_pe.shape[-1]
    _, S_kv, H_kv, _ = k.shape
    num_head_groups = H // H_kv
    scale = (D + pe_dim) ** 0.5

    Q = q.view(B, H_kv, num_head_groups, D).permute(0, 2, 1, 3)
    Q_pe = q_pe.view(B, H_kv, num_head_groups, pe_dim).permute(0, 2, 1, 3)
    KV = k.permute(0, 2, 1, 3)
    K_pe = k_pe.permute(0, 2, 1, 3)

    query = torch.cat([Q, Q_pe], dim=-1)
    key = torch.cat([KV, K_pe], dim=-1)

    scores = torch.einsum(
        'bghd,bhsd->bghs', query.float(), key.float()) / scale
    attention = F.softmax(scores, dim=-1).to(q.dtype)

    out = torch.einsum(
        'bghs,bhsd->bghd', attention.float(), KV.float()).to(q.dtype)
    out = out.permute(0, 2, 1, 3).reshape(B, H, D)
    return out


@pytest.mark.parametrize(
    "q_shape, q_pe_shape, kv_shape, k_pe_shape, pe_dim, dtype",
    _PARAMS,
)
def test_mla_decode_fwd_op(q_shape, q_pe_shape, kv_shape, k_pe_shape,
                           pe_dim, dtype: torch.dtype) -> None:
    wl = MlaDecodeWorkload(q_shape, q_pe_shape, kv_shape, k_pe_shape,
                           pe_dim, dtype)
    inputs = wl.gen_inputs()
    op = MultiHeadLatentAttentionDecodeWithKVCacheFwdOp(pe_dim=pe_dim,
                                                        tune=False)
    output = op(*inputs)

    q, q_pe, k, k_pe = inputs
    ref = _ref_mla_decode(q, q_pe, k, k_pe)

    rtol, atol = dtype_tolerances(dtype)
    torch.testing.assert_close(output, ref, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])

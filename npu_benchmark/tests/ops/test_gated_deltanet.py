"""Correctness test for the Gated DeltaNet forward op.

Compares kernel output against a naive PyTorch reference (per-token
recurrence).  Tolerances are dtype-aware (see ``tests.tolerances``).

NOTE: The NPU ``GatedDeltaNetFwdKernel`` is currently a stub (empty body).
The kernel compiles and runs but produces undefined output values, so
correctness tests are expected to fail until the real forward body is
implemented.
"""

from __future__ import annotations

import pytest
import torch

from manifest import load_workloads
from ops import GatedDeltaNetFwdOp
from benchmarks.benchmark_base import workload_field_params
from workloads.gated_deltanet import GatedDeltaNetFwdWorkload
from tests.tolerances import dtype_tolerances

_PARAMS = workload_field_params(
    load_workloads("GatedDeltaNetFwdOp"),
    ("q_shape", "k_shape", "v_shape", "g_shape", "beta_shape",
     "chunk_size", "dtype"),
)


def _ref_gated_deltanet_fwd(q, k, v, g, beta, chunk_size):
    """Naive O(S * DK * DV) Gated DeltaNet reference (vectorised over B, H)."""
    B, H, S, DK = q.shape
    DV = v.shape[-1]
    NC = S // chunk_size

    qf, kf, vf = q.float(), k.float(), v.float()
    gf, bf = g.float(), beta.float()

    o = torch.zeros(B, H, S, DV, dtype=torch.float32, device=q.device)
    states = torch.zeros(B, H, NC + 1, DK, DV, dtype=torch.float32, device=q.device)
    Aw = torch.zeros(B, H, S, chunk_size, dtype=torch.float32, device=q.device)
    Au = torch.zeros(B, H, S, chunk_size, dtype=torch.float32, device=q.device)

    state = torch.zeros(B, H, DK, DV, dtype=torch.float32, device=q.device)
    for c in range(NC):
        states[:, :, c] = state
        for l in range(chunk_size):
            t = c * chunk_size + l
            alpha = torch.sigmoid(gf[:, :, t])  # [B, H]
            kv = kf[:, :, t, :].unsqueeze(-1) * vf[:, :, t, :].unsqueeze(-2)
            state = alpha.unsqueeze(-1).unsqueeze(-1) * state + \
                bf[:, :, t].unsqueeze(-1).unsqueeze(-1) * kv
            o[:, :, t, :] = torch.einsum("bhd,bhde->bhe", qf[:, :, t, :], state)
    states[:, :, NC] = state

    return o.to(q.dtype), states, Aw, Au


@pytest.mark.parametrize(
    "q_shape, k_shape, v_shape, g_shape, beta_shape, chunk_size, dtype",
    _PARAMS,
)
def test_gated_deltanet_fwd_op(q_shape, k_shape, v_shape, g_shape,
                               beta_shape, chunk_size,
                               dtype: torch.dtype) -> None:
    wl = GatedDeltaNetFwdWorkload(q_shape, k_shape, v_shape, g_shape,
                                  beta_shape, chunk_size, dtype)
    inputs = wl.gen_inputs()
    op = GatedDeltaNetFwdOp(chunk_size=chunk_size, tune=False)
    output = op(*inputs)

    q, k, v, g, beta = inputs
    ref_o, ref_S, ref_Aw, ref_Au = _ref_gated_deltanet_fwd(
        q, k, v, g, beta, chunk_size)

    o = output[0]
    rtol, atol = dtype_tolerances(dtype)
    torch.testing.assert_close(o, ref_o, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])

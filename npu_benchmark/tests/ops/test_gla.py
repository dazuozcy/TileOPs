"""Correctness test for the GLA forward op.

Compares kernel output against a naive PyTorch reference (per-token
recurrence).  Tolerances are dtype-aware (see ``tests.tolerances``).

NOTE: The NPU ``GLAFwdKernel`` is currently a stub (empty body).
The kernel compiles and runs but produces undefined output values, so
correctness tests are expected to fail until the real forward body is
implemented.
"""

from __future__ import annotations

import pytest
import torch

from manifest import load_workloads
from ops import GLAFwdOp
from benchmarks.benchmark_base import workload_field_params
from workloads.gla import GLAFwdWorkload
from tests.tolerances import dtype_tolerances

_PARAMS = workload_field_params(
    load_workloads("GLAFwdOp"),
    ("q_shape", "k_shape", "v_shape", "g_shape",
     "chunk_size", "scale", "dtype"),
)


def _ref_gla_fwd(q, k, v, g, chunk_size, scale):
    """Naive O(S * DK * DV) GLA reference (vectorised over B, H)."""
    B, S, H, DK = q.shape
    DV = v.shape[-1]

    qf, kf, vf, gf = q.float(), k.float(), v.float(), g.float()

    if scale < 0:
        scale = DK ** -0.5

    o = torch.zeros(B, S, H, DV, dtype=torch.float32, device=q.device)
    final_state = torch.zeros(B, H, DK, DV, dtype=torch.float32, device=q.device)

    state = torch.zeros(B, H, DK, DV, dtype=torch.float32, device=q.device)
    for t in range(S):
        gate = torch.exp(gf[:, t, :, :])  # [B, H, DK]
        kv = kf[:, t, :, :].unsqueeze(-1) * vf[:, t, :, :].unsqueeze(-2)
        state = state * gate.unsqueeze(-1) + kv
        o[:, t, :, :] = torch.einsum(
            "bhd,bhde->bhe", qf[:, t, :, :] * scale, state)
    final_state = state

    return o.to(q.dtype), final_state.to(q.dtype)


@pytest.mark.parametrize(
    "q_shape, k_shape, v_shape, g_shape, chunk_size, scale, dtype",
    _PARAMS,
)
def test_gla_fwd_op(q_shape, k_shape, v_shape, g_shape,
                    chunk_size, scale, dtype: torch.dtype) -> None:
    wl = GLAFwdWorkload(q_shape, k_shape, v_shape, g_shape,
                        chunk_size, dtype, scale=scale)
    inputs = wl.gen_inputs()
    op = GLAFwdOp(chunk_size=chunk_size, scale=scale,
                  output_final_state=True, tune=False)
    output = op(*inputs)

    q, k, v, g = inputs
    ref_o, ref_final_state = _ref_gla_fwd(q, k, v, g, chunk_size, scale)

    o = output[0]
    rtol, atol = dtype_tolerances(dtype)
    torch.testing.assert_close(o, ref_o, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])

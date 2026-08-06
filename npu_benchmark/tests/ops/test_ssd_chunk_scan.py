"""Correctness test for the SSD chunk scan forward op.

Compares kernel output against a naive PyTorch reference (per-token
loop).  Tolerances are dtype-aware (see ``tests.tolerances``).

NOTE: The NPU ``SSDChunkScanFwdKernel`` is currently a stub (empty body).
The kernel compiles and runs but produces undefined output values, so
correctness tests are expected to fail until the real scan body is
implemented.
"""

from __future__ import annotations

import pytest
import torch

from manifest import load_workloads
from ops import SSDChunkScanFwdOp
from benchmarks.benchmark_base import workload_field_params
from workloads.ssd_chunk_scan import SSDChunkScanFwdWorkload
from tests.tolerances import dtype_tolerances

_PARAMS = workload_field_params(
    load_workloads("SSDChunkScanFwdOp"),
    ("x_shape", "cb_shape", "dA_cumsum_shape", "C_shape",
     "prev_states_shape", "dt_shape", "dtype"),
)


def _ref_ssd_chunk_scan(x, cb, dA_cumsum, C, prev_states, dt):
    """Naive SSD chunk scan reference."""
    B, S, H, P = x.shape
    NC = cb.shape[1]
    G = cb.shape[2]
    Q = cb.shape[3]
    N = C.shape[-1]

    xf = x.float()
    cbf = cb.float()
    dAf = dA_cumsum.float()
    Cf = C.float()
    psf = prev_states.float()
    dtf = dt.float()

    y = torch.zeros(B, S, H, P, dtype=torch.float32, device=x.device)

    for c in range(NC):
        for h in range(H):
            g = h // (H // G) if H > G else 0
            for l in range(Q):
                t = c * Q + l
                hist = torch.einsum(
                    "bn,bpn->bp",
                    Cf[:, t, g, :],
                    psf[:, c, h, :, :],
                )
                hist = hist * torch.exp(dAf[:, h, c, l]).unsqueeze(-1)

                intra = torch.zeros(B, P, dtype=torch.float32, device=x.device)
                for s in range(l + 1):
                    ts = c * Q + s
                    coef = (cbf[:, c, g, l, s]
                            * torch.exp(dAf[:, h, c, l] - dAf[:, h, c, s])
                            * dtf[:, h, c, s])
                    intra += coef.unsqueeze(-1) * xf[:, ts, h, :]

                y[:, t, h, :] = hist + intra

    return y


@pytest.mark.parametrize(
    "x_shape, cb_shape, dA_cumsum_shape, C_shape, "
    "prev_states_shape, dt_shape, dtype",
    _PARAMS,
)
def test_ssd_chunk_scan_fwd_op(x_shape, cb_shape, dA_cumsum_shape, C_shape,
                               prev_states_shape, dt_shape,
                               dtype: torch.dtype) -> None:
    wl = SSDChunkScanFwdWorkload(x_shape, cb_shape, dA_cumsum_shape, C_shape,
                                 prev_states_shape, dt_shape, dtype)
    inputs = wl.gen_inputs()
    op = SSDChunkScanFwdOp(tune=False)
    output = op(*inputs)

    ref_y = _ref_ssd_chunk_scan(*inputs)

    rtol, atol = dtype_tolerances(torch.float32)
    torch.testing.assert_close(output, ref_y, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])

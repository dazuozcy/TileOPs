from typing import Dict, Optional

import torch

from kernels.kernel_base import Kernel
from kernels.ssd_chunk_scan import SSDChunkScanFwdKernel
from perf.formulas import ssd_chunk_scan_fwd_roofline

from .op_base import Op

__all__ = ["SSDChunkScanFwdOp"]


class SSDChunkScanFwdOp(Op):
    """Mamba-2 SSD fused chunk scan forward operator.

    Fuses the history (prev_states) contribution and intra-chunk causal
    decay into a single pass.

    Input:  x            [B, S, H, P]
            cb           [B, NC, G, Q, Q]
            dA_cumsum    [B, H, NC, Q]          (float32)
            C            [B, S, G, N]
            prev_states  [B, NC, H, P, N]       (float32)
            dt           [B, H, NC, Q]
    Output: y            [B, S, H, P]           (float32)

    Backed by the TileLang ``SSDChunkScanFwdKernel`` (NPU -- currently a
    placeholder stub; see ``kernels/ssd_chunk_scan.py``).

    Args:
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """

    _SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)

    def __init__(self,
                 *,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        super().__init__()
        self.tune = tune

        self.batch = None
        self.num_chunks = None
        self.chunk_len = None
        self.n_heads = None
        self.d_head = None
        self.d_state = None
        self.n_groups = None
        self.dtype = None

        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"ssd_chunk_scan_fwd": SSDChunkScanFwdKernel}

    def _get_kernel(self, batch, num_chunks, chunk_len, n_heads, d_head,
                    d_state, n_groups, dtype, device_index) -> Kernel:
        key = (batch, num_chunks, chunk_len, n_heads, d_head, d_state,
               n_groups, dtype, device_index, self.tune)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["ssd_chunk_scan_fwd"](
                batch, num_chunks, chunk_len, n_heads, d_head, d_state,
                n_groups, dtype=dtype, tune=self.tune)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor, cb: torch.Tensor,
                dA_cumsum: torch.Tensor, C: torch.Tensor,
                prev_states: torch.Tensor, dt: torch.Tensor
                ) -> torch.Tensor:
        from utils import is_available

        if not is_available():
            raise ValueError("SSDChunkScanFwdOp expects an accelerator device")
        if x.ndim != 4:
            raise ValueError("x must have shape [B, S, H, P]")
        batch, seq_len, n_heads, d_head = x.shape
        if x.dtype not in self._SUPPORTED_DTYPES:
            names = ", ".join(str(dt) for dt in self._SUPPORTED_DTYPES)
            raise ValueError(
                f"SSDChunkScanFwdOp does not support dtype {x.dtype}. "
                f"Supported: [{names}]")

        if dA_cumsum.ndim != 4:
            raise ValueError("dA_cumsum must have shape [B, H, NC, Q]")
        if dA_cumsum.shape[0] != batch or dA_cumsum.shape[1] != n_heads:
            raise ValueError("dA_cumsum must match x batch and n_heads")
        num_chunks, chunk_len = dA_cumsum.shape[2], dA_cumsum.shape[3]
        if seq_len != num_chunks * chunk_len:
            raise ValueError("x seq_len must equal num_chunks * chunk_len")

        if cb.ndim != 5:
            raise ValueError("cb must have shape [B, NC, G, Q, Q]")
        if cb.shape[0] != batch or cb.shape[1] != num_chunks:
            raise ValueError("cb must match x batch and num_chunks")
        if cb.shape[3] != chunk_len or cb.shape[4] != chunk_len:
            raise ValueError("cb must match chunk_len")
        n_groups = cb.shape[2]

        if C.ndim != 4 or C.shape[0] != batch or C.shape[1] != seq_len:
            raise ValueError("C must have shape [B, S, G, N]")
        if C.shape[2] != n_groups:
            raise ValueError("C n_groups must match cb")
        d_state = C.shape[3]

        if n_heads % n_groups != 0:
            raise ValueError("n_heads must be divisible by n_groups")

        if prev_states.shape != (batch, num_chunks, n_heads, d_head, d_state):
            raise ValueError(
                "prev_states must have shape [B, NC, H, P, N]")
        if dt.shape != (batch, n_heads, num_chunks, chunk_len):
            raise ValueError("dt must have shape [B, H, NC, Q]")

        self.batch = batch
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.dtype = x.dtype

        self.kernel = self._get_kernel(
            batch, num_chunks, chunk_len, n_heads, d_head, d_state,
            n_groups, x.dtype, x.device.index)

        x = x.contiguous()
        cb = cb.contiguous()
        dA_cumsum = dA_cumsum.contiguous()
        C = C.contiguous()
        prev_states = prev_states.contiguous()
        dt = dt.contiguous()

        return self.kernel(x, cb, dA_cumsum, C, prev_states, dt)

    def eval_roofline(self) -> tuple[int, int]:
        return ssd_chunk_scan_fwd_roofline(self)

from typing import Dict, Optional, Tuple

import torch

from kernels.kernel_base import Kernel
from kernels.gated_deltanet import GatedDeltaNetFwdKernel
from perf.formulas import gated_deltanet_fwd_roofline

from .op_base import Op

__all__ = ["GatedDeltaNetFwdOp"]


class GatedDeltaNetFwdOp(Op):
    """Gated DeltaNet forward operator.

    Chunked Gated DeltaNet forward pass producing ``(o, S, Aw, Au)``.

    Layout: BHSD (batch, head, seq_len, dim).

    Input:  q    [B, H, S, DK]
            k    [B, H, S, DK]
            v    [B, H, S, DV]
            g    [B, H, S]
            beta [B, H, S]
    Output: o        [B, H, S, DV]          (same dtype as input)
            S        [B, H, NC+1, DK, DV]   (float32)
            Aw       [B, H, S, chunk_size]  (float32)
            Au       [B, H, S, chunk_size]  (float32)

    Backed by the TileLang ``GatedDeltaNetFwdKernel`` (NPU -- currently a
    placeholder stub; see ``kernels/gated_deltanet.py``).

    Args:
        chunk_size: Chunk size for chunked linear attention (default 64).
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """

    _SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(self,
                 chunk_size: int = 64,
                 *,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.tune = tune

        self.batch = None
        self.heads = None
        self.seq_len = None
        self.dim_k = None
        self.dim_v = None
        self.dtype = None

        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"gated_deltanet_fwd_kernel": GatedDeltaNetFwdKernel}

    def _get_kernel(self, batch, heads, seq_len, dim_k, dim_v,
                    dtype, device_index) -> Kernel:
        key = (batch, heads, seq_len, self.chunk_size,
               dim_k, dim_v, dtype, device_index, self.tune)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["gated_deltanet_fwd_kernel"](
                batch, heads, seq_len, self.chunk_size, dim_k, dim_v,
                dtype=dtype, tune=self.tune)
        return self._kernel_cache[key]

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                g: torch.Tensor, beta: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from utils import is_available

        if not is_available():
            raise ValueError("GatedDeltaNetFwdOp expects an accelerator device")
        if q.ndim != 4:
            raise ValueError(
                f"q must have shape [B, H, S, DK], got {q.ndim}D")
        batch, heads, seq_len, dim_k = q.shape
        if k.shape != (batch, heads, seq_len, dim_k):
            raise ValueError("k must match q shape")
        if v.ndim != 4 or v.shape[:3] != (batch, heads, seq_len):
            raise ValueError("v must have shape [B, H, S, DV]")
        dim_v = v.shape[-1]
        if g.shape != (batch, heads, seq_len):
            raise ValueError("g must have shape [B, H, S]")
        if beta.shape != (batch, heads, seq_len):
            raise ValueError("beta must have shape [B, H, S]")
        if q.dtype not in self._SUPPORTED_DTYPES:
            names = ", ".join(str(dt) for dt in self._SUPPORTED_DTYPES)
            raise ValueError(
                f"GatedDeltaNetFwdOp does not support dtype {q.dtype}. "
                f"Supported: [{names}]")
        for name, tensor in (("k", k), ("v", v), ("g", g), ("beta", beta)):
            if tensor.dtype != q.dtype:
                raise ValueError(
                    f"{name}.dtype must be {q.dtype}, got {tensor.dtype}")
        if seq_len % self.chunk_size != 0:
            raise ValueError(
                f"seq_len ({seq_len}) must be divisible by chunk_size "
                f"({self.chunk_size})")

        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = q.dtype

        self.kernel = self._get_kernel(
            batch, heads, seq_len, dim_k, dim_v, q.dtype, q.device.index)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        g = g.contiguous()
        beta = beta.contiguous()
        return self.kernel(q, k, v, g, beta)

    def eval_roofline(self) -> tuple[int, int]:
        return gated_deltanet_fwd_roofline(self)

from typing import Dict, Optional, Tuple

import torch

from kernels.kernel_base import Kernel
from kernels.gla import GLAFwdKernel
from perf.formulas import gla_fwd_roofline

from .op_base import Op

__all__ = ["GLAFwdOp"]


class GLAFwdOp(Op):
    """GLA (Gated Linear Attention) forward operator.

    Chunked GLA forward pass: ``(q, k, v, g) -> (o, final_state)``.

    Layout: BTHD (batch, seq_len, heads, dim).

    Input:  q   [B, S, H, DK]
            k   [B, S, H, DK]
            v   [B, S, H, DV]
            g   [B, S, H, DK]   (log-space forget gates)
    Output: o            [B, S, H, DV]
            final_state  [B, H, DK, DV]

    Backed by the TileLang ``GLAFwdKernel`` (NPU -- currently a placeholder
    stub; see ``kernels/gla.py``).

    Args:
        chunk_size: Chunk size for chunked linear attention (default 64).
        scale: Query scale factor (default -1.0 means dim_k**-0.5).
        output_final_state: Whether to return the final hidden state
            (default True).
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """

    _SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(self,
                 chunk_size: int = 64,
                 scale: float = -1.0,
                 output_final_state: bool = True,
                 *,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.scale = scale
        self.output_final_state = output_final_state
        self.tune = tune

        self.batch = None
        self.seq_len = None
        self.heads = None
        self.dim_k = None
        self.dim_v = None
        self.dtype = None

        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"gla_fwd_kernel": GLAFwdKernel}

    def _get_kernel(self, batch, seq_len, heads, dim_k, dim_v,
                    dtype, device_index) -> Kernel:
        key = (batch, seq_len, heads, dim_k, dim_v, self.chunk_size,
               self.scale, self.output_final_state, dtype, device_index,
               self.tune)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["gla_fwd_kernel"](
                batch, seq_len, heads, dim_k, dim_v, self.chunk_size,
                scale=self.scale,
                output_final_state=self.output_final_state,
                dtype=dtype, tune=self.tune)
        return self._kernel_cache[key]

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                g: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        from utils import is_available

        if not is_available():
            raise ValueError("GLAFwdOp expects an accelerator device")
        if q.ndim != 4:
            raise ValueError(
                f"q must have shape [B, S, H, DK], got {q.ndim}D")
        batch, seq_len, heads, dim_k = q.shape
        if k.shape != (batch, seq_len, heads, dim_k):
            raise ValueError("k must match q shape")
        if v.ndim != 4 or v.shape[:3] != (batch, seq_len, heads):
            raise ValueError("v must have shape [B, S, H, DV]")
        dim_v = v.shape[-1]
        if g.shape != (batch, seq_len, heads, dim_k):
            raise ValueError("g must match q shape")
        if q.dtype not in self._SUPPORTED_DTYPES:
            names = ", ".join(str(dt) for dt in self._SUPPORTED_DTYPES)
            raise ValueError(
                f"GLAFwdOp does not support dtype {q.dtype}. "
                f"Supported: [{names}]")
        for name, tensor in (("k", k), ("v", v), ("g", g)):
            if tensor.dtype != q.dtype:
                raise ValueError(
                    f"{name}.dtype must be {q.dtype}, got {tensor.dtype}")
        if seq_len % self.chunk_size != 0:
            raise ValueError(
                f"seq_len ({seq_len}) must be divisible by chunk_size "
                f"({self.chunk_size})")

        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = q.dtype

        self.kernel = self._get_kernel(
            batch, seq_len, heads, dim_k, dim_v, q.dtype, q.device.index)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        g = g.contiguous()
        o, final_state = self.kernel(q, k, v, g)
        if not self.output_final_state:
            return o, None
        return o, final_state

    def eval_roofline(self) -> tuple[int, int]:
        return gla_fwd_roofline(self)

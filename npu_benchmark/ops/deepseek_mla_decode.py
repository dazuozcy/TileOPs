from typing import Dict, Optional

import torch

from kernels.kernel_base import Kernel
from kernels.deepseek_mla_decode import MLADecodeKernel
from perf.formulas import deepseek_mla_decode_roofline

from .op_base import Op

__all__ = ["MultiHeadLatentAttentionDecodeWithKVCacheFwdOp"]


class MultiHeadLatentAttentionDecodeWithKVCacheFwdOp(Op):
    """Multi-Head Latent Attention decode with KV cache (DeepSeek-style).

    Layout: BSHD-style for KV, BHD for Q.

    Chunked MLA decode pass: ``(q, q_pe, k, k_pe) -> (o)``.

    Input:  q     [B, H, D]
            q_pe  [B, H, pe_dim]
            k     [B, N_kv, H_kv, D]
            k_pe  [B, N_kv, H_kv, pe_dim]
    Output: o     [B, H, D]

    Backed by the TileLang ``MLADecodeKernel`` (NPU -- currently a
    placeholder stub; see ``kernels/deepseek_mla_decode.py``).

    Args:
        pe_dim: Decoupled RoPE dimension.
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """

    _SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)

    def __init__(self,
                 pe_dim: int = 64,
                 *,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        super().__init__()
        self.pe_dim = pe_dim
        self.tune = tune

        self.batch = None
        self.heads = None
        self.heads_kv = None
        self.seqlen_kv = None
        self.dim = None
        self.dtype = None

        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"mla_decode_kernel": MLADecodeKernel}

    def _get_kernel(self, batch, heads, heads_kv, seqlen_kv, dim, pe_dim,
                    dtype, device_index) -> Kernel:
        key = (batch, heads, heads_kv, seqlen_kv, dim, pe_dim, dtype,
               device_index, self.tune)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["mla_decode_kernel"](
                batch, heads, heads_kv, seqlen_kv, dim, pe_dim,
                dtype=dtype, tune=self.tune)
        return self._kernel_cache[key]

    def forward(self, q: torch.Tensor, q_pe: torch.Tensor, k: torch.Tensor,
                k_pe: torch.Tensor) -> torch.Tensor:
        from utils import is_available

        if not is_available():
            raise ValueError(
                "MultiHeadLatentAttentionDecodeWithKVCacheFwdOp expects "
                "an accelerator device")
        if q.ndim != 3:
            raise ValueError(
                f"q must have shape [B, H, D], got {q.ndim}D")
        batch, heads, dim = q.shape
        if q_pe.ndim != 3 or q_pe.shape[0] != batch or q_pe.shape[1] != heads:
            raise ValueError(
                "q_pe must have shape [B, H, pe_dim] matching q")
        pe_dim_actual = q_pe.shape[2]
        if pe_dim_actual != self.pe_dim:
            raise ValueError(
                f"q_pe pe_dim ({pe_dim_actual}) != op pe_dim ({self.pe_dim})")
        if k.ndim != 4:
            raise ValueError(
                f"k must have shape [B, N_kv, H_kv, D], got {k.ndim}D")
        seqlen_kv, heads_kv = k.shape[1], k.shape[2]
        if k.shape[0] != batch or k.shape[3] != dim:
            raise ValueError(
                "k must have shape [B, N_kv, H_kv, D] matching q")
        if heads % heads_kv != 0:
            raise ValueError(
                f"heads ({heads}) must be divisible by heads_kv ({heads_kv})")
        if k_pe.shape != (batch, seqlen_kv, heads_kv, self.pe_dim):
            raise ValueError(
                "k_pe must have shape [B, N_kv, H_kv, pe_dim] matching k")
        if q.dtype not in self._SUPPORTED_DTYPES:
            names = ", ".join(str(dt) for dt in self._SUPPORTED_DTYPES)
            raise ValueError(
                f"MultiHeadLatentAttentionDecodeWithKVCacheFwdOp does not "
                f"support dtype {q.dtype}. Supported: [{names}]")
        for name, tensor in (("q_pe", q_pe), ("k", k), ("k_pe", k_pe)):
            if tensor.dtype != q.dtype:
                raise ValueError(
                    f"{name}.dtype must be {q.dtype}, got {tensor.dtype}")

        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.dtype = q.dtype

        self.kernel = self._get_kernel(
            batch, heads, heads_kv, seqlen_kv, dim, self.pe_dim,
            q.dtype, q.device.index)
        q = q.contiguous()
        q_pe = q_pe.contiguous()
        k = k.contiguous()
        k_pe = k_pe.contiguous()
        o = self.kernel(q, q_pe, k, k_pe)
        return o

    def eval_roofline(self) -> tuple[int, int]:
        return deepseek_mla_decode_roofline(self)

"""DeepSeek MLA decode kernel (NPU, target=npuir) -- STUB (empty body).

Computes Multi-Head Latent Attention decode with KV cache:

    Given q [B, H, D], q_pe [B, H, pe_dim],
          k [B, N_kv, H_kv, D], k_pe [B, N_kv, H_kv, pe_dim],
    produce o [B, H, D].

The attention score combines the latent channel (q @ k^T) and the
decoupled RoPE channel (q_pe @ k_pe^T), followed by softmax and
weighted sum over V (= k's latent channel).

Status: **STUB** -- the prim_func compiles and runs on NPU, but the kernel
body is empty (``pass``).  Output tensors are allocated with the correct
shape/dtype but their values are undefined, so correctness tests will fail.
Benchmark / JIT-compile / roofline flows execute normally.

To implement:
  1. Replace the ``pass`` in the ``T.Kernel`` block with the real
     MLA decode forward body.
  2. Everything else (factory, wrapper, Op dispatch) stays as-is.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from kernels.kernel_base import Kernel

__all__ = ["MLADecodeKernel"]


@functools.lru_cache(maxsize=32)
def _make_mla_decode_kernel(B, H, H_kv, S_kv, D, pe_dim, dtype,
                            output_dtype=None):
    """Build MLA decode kernel (STUB -- empty body).

    Args:
        B: Batch size.
        H: Number of query heads.
        H_kv: Number of KV heads (H // H_kv = num_head_groups).
        S_kv: KV sequence length.
        D: Latent dimension per head.
        pe_dim: Decoupled RoPE dimension.
        dtype: Input dtype string (float16 / bfloat16).
        output_dtype: Output dtype string; defaults to ``dtype``.

    ``block_size`` is passed at call time via the ``block_size`` argument
    to the returned ``kernel`` callable.
    """
    out_dtype = output_dtype or dtype

    @tilelang.jit(out_idx=[4], target="npuir")
    def kernel(block_size):
        @T.prim_func
        def main(
            q: T.Tensor((B, H, D), dtype),
            q_pe: T.Tensor((B, H, pe_dim), dtype),
            k: T.Tensor((B, S_kv, H_kv, D), dtype),
            k_pe: T.Tensor((B, S_kv, H_kv, pe_dim), dtype),
            o: T.Tensor((B, H, D), out_dtype),
        ):
            with T.Kernel(T.ceildiv(B * H, block_size),
                          is_npu=True) as (cid, _):
                # TODO: MLA decode forward:
                #   1. load q, q_pe, k, k_pe
                #   2. compute scores = q@k^T + q_pe@k_pe^T
                #   3. softmax(scores / sqrt(D + pe_dim))
                #   4. o = softmax @ k
                pass

        return main

    return kernel


class MLADecodeKernel(Kernel):
    """MLA decode kernel wrapper (NPU STUB -- empty body).

    Implements Multi-Head Latent Attention decode with KV cache,
    producing ``o`` [B, H, D].

    Input:  q     [B, H, D]
            q_pe  [B, H, pe_dim]
            k     [B, N_kv, H_kv, D]
            k_pe  [B, N_kv, H_kv, pe_dim]
    Output: o     [B, H, D]    (same dtype as input)

    Supported dtypes: float16, bfloat16.

    NPU tiling: each core processes ``block_size`` output elements.  There
    is no GPU-style ``threads`` / ``npt`` split because the NPU has no
    SIMT thread model.

    Status: **STUB** -- kernel body is empty; output values are undefined.
    """

    supported_archs: Optional[list[int]] = None
    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
    prof_name = "main"

    def __init__(self,
                 batch: int,
                 heads: int,
                 heads_kv: int,
                 seqlen_kv: int,
                 dim: int,
                 pe_dim: int,
                 dtype: torch.dtype = torch.float16,
                 config: Optional[dict] = None,
                 tune: bool = False):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"MLADecodeKernel only supports dtypes [{supported}], got {dtype}"
            )
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.pe_dim = pe_dim
        self.dtype = dtype
        self.dtype_str = self.dtype_to_str(dtype)

        self.kernel = _make_mla_decode_kernel(
            batch, heads, heads_kv, seqlen_kv, dim, pe_dim, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"block_size": 256}

    def forward(self, q: torch.Tensor, q_pe: torch.Tensor, k: torch.Tensor,
                k_pe: torch.Tensor):
        prim_func = self.kernel(block_size=self.config["block_size"])
        return prim_func(q, q_pe, k, k_pe)

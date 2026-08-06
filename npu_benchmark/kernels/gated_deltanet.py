"""Gated DeltaNet forward kernel (NPU, target=npuir) -- STUB (empty body).

Computes the chunked Gated DeltaNet forward pass:

    S_t = alpha_t * S_{t-1} + beta_t * v_t (x) k_t^T
    o_t = q_t @ S_t

producing the output ``o``, per-chunk boundary states ``S``, and WY
representation matrices ``Aw`` / ``Au`` for backward.

Status: **STUB** -- the prim_func compiles and runs on NPU, but the kernel
body is empty (``pass``).  Output tensors are allocated with the correct
shape/dtype but their values are undefined, so correctness tests will fail.
Benchmark / JIT-compile / roofline flows execute normally.

To implement:
  1. Replace the ``pass`` in the ``T.Kernel`` block with the real
     chunked delta-rule forward body.
  2. Everything else (factory, wrapper, Op dispatch) stays as-is.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from kernels.kernel_base import Kernel

__all__ = ["GatedDeltaNetFwdKernel"]


@functools.lru_cache(maxsize=32)
def _make_gated_deltanet_fwd_kernel(B, H, S, DK, DV, chunk_size, dtype,
                                    output_dtype=None):
    """Build Gated DeltaNet forward kernel (STUB -- empty body).

    Args:
        B: Batch size.
        H: Number of heads.
        S: Sequence length (must be divisible by chunk_size).
        DK: Key/query dimension.
        DV: Value dimension.
        chunk_size: Chunk size for chunked linear attention.
        dtype: Input dtype string (float16 / bfloat16 / float32).
        output_dtype: Output dtype string for ``o``; defaults to ``dtype``.
            ``S``, ``Aw``, ``Au`` are always float32.

    ``block_size`` is passed at call time via the ``block_size`` argument
    to the returned ``kernel`` callable.
    """
    out_dtype = output_dtype or dtype
    NC = S // chunk_size

    @tilelang.jit(out_idx=[5, 6, 7, 8], target="npuir")
    def kernel(block_size):
        @T.prim_func
        def main(
            q: T.Tensor((B, H, S, DK), dtype),
            k: T.Tensor((B, H, S, DK), dtype),
            v: T.Tensor((B, H, S, DV), dtype),
            g: T.Tensor((B, H, S), dtype),
            beta: T.Tensor((B, H, S), dtype),
            o: T.Tensor((B, H, S, DV), out_dtype),
            S_state: T.Tensor((B, H, NC + 1, DK, DV), "float32"),
            Aw: T.Tensor((B, H, S, chunk_size), "float32"),
            Au: T.Tensor((B, H, S, chunk_size), "float32"),
        ):
            with T.Kernel(T.ceildiv(B * H * S, block_size),
                          is_npu=True) as (cid, _):
                # TODO: chunked gated delta-rule forward:
                #   1. prepare_wy_repr(k, g, beta) -> (Aw, Au)
                #   2. intra-chunk attention with delta rule
                #   3. inter-chunk state contribution
                #   4. write o, S_state, Aw, Au
                pass

        return main

    return kernel


class GatedDeltaNetFwdKernel(Kernel):
    """Gated DeltaNet forward kernel wrapper (NPU STUB -- empty body).

    Implements the chunked Gated DeltaNet forward pass producing
    ``(o, S, Aw, Au)``.

    Input:  q    [B, H, S, DK]
            k    [B, H, S, DK]
            v    [B, H, S, DV]
            g    [B, H, S]
            beta [B, H, S]
    Output: o        [B, H, S, DV]      (same dtype as input)
            S_state  [B, H, NC+1, DK, DV]  (float32)
            Aw       [B, H, S, chunk_size] (float32)
            Au       [B, H, S, chunk_size] (float32)

    Supported dtypes: float16, bfloat16, float32.

    NPU tiling: each core processes ``block_size`` output elements.  There
    is no GPU-style ``threads`` / ``npt`` split because the NPU has no
    SIMT thread model.

    Status: **STUB** -- kernel body is empty; output values are undefined.
    """

    supported_archs: Optional[list[int]] = None
    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
    prof_name = "main"

    def __init__(self,
                 batch: int,
                 heads: int,
                 seq_len: int,
                 chunk_size: int,
                 dim_k: int,
                 dim_v: int,
                 dtype: torch.dtype = torch.float16,
                 config: Optional[dict] = None,
                 tune: bool = False):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"GatedDeltaNetFwdKernel only supports dtypes [{supported}], got {dtype}"
            )
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.chunk_size = chunk_size
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype
        self.dtype_str = self.dtype_to_str(dtype)

        self.kernel = _make_gated_deltanet_fwd_kernel(
            batch, heads, seq_len, dim_k, dim_v, chunk_size, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"block_size": 256}

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                g: torch.Tensor, beta: torch.Tensor):
        prim_func = self.kernel(block_size=self.config["block_size"])
        return prim_func(q, k, v, g, beta)

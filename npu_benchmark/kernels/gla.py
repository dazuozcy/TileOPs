"""GLA (Gated Linear Attention) forward kernel (NPU, target=npuir) -- STUB (empty body).

Computes the chunked GLA forward pass:

    S_t = S_{t-1} * exp(g_t) + k_t (x) v_t^T
    o_t = (q_t * scale) @ S_t

producing the output ``o`` and the final hidden state ``final_state``.

Status: **STUB** -- the prim_func compiles and runs on NPU, but the kernel
body is empty (``pass``).  Output tensors are allocated with the correct
shape/dtype but their values are undefined, so correctness tests will fail.
Benchmark / JIT-compile / roofline flows execute normally.

To implement:
  1. Replace the ``pass`` in the ``T.Kernel`` block with the real
     chunked GLA forward body.
  2. Everything else (factory, wrapper, Op dispatch) stays as-is.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from kernels.kernel_base import Kernel

__all__ = ["GLAFwdKernel"]


@functools.lru_cache(maxsize=32)
def _make_gla_fwd_kernel(B, S, H, DK, DV, chunk_size, dtype,
                         output_dtype=None):
    """Build GLA forward kernel (STUB -- empty body).

    Args:
        B: Batch size.
        S: Sequence length (must be divisible by chunk_size).
        H: Number of heads.
        DK: Key/query dimension.
        DV: Value dimension.
        chunk_size: Chunk size for chunked linear attention.
        dtype: Input dtype string (float16 / bfloat16 / float32).
        output_dtype: Output dtype string; defaults to ``dtype``.

    ``block_size`` is passed at call time via the ``block_size`` argument
    to the returned ``kernel`` callable.
    """
    out_dtype = output_dtype or dtype

    @tilelang.jit(out_idx=[4, 5], target="npuir")
    def kernel(block_size):
        @T.prim_func
        def main(
            q: T.Tensor((B, S, H, DK), dtype),
            k: T.Tensor((B, S, H, DK), dtype),
            v: T.Tensor((B, S, H, DV), dtype),
            g: T.Tensor((B, S, H, DK), dtype),
            o: T.Tensor((B, S, H, DV), out_dtype),
            final_state: T.Tensor((B, H, DK, DV), out_dtype),
        ):
            with T.Kernel(T.ceildiv(B * S * H, block_size),
                          is_npu=True) as (cid, _):
                # TODO: chunked GLA forward:
                #   1. intra-chunk attention with log-space gating
                #   2. inter-chunk state contribution
                #   3. write o, final_state
                pass

        return main

    return kernel


class GLAFwdKernel(Kernel):
    """GLA forward kernel wrapper (NPU STUB -- empty body).

    Implements the chunked GLA forward pass producing ``(o, final_state)``.

    Input:  q   [B, S, H, DK]
            k   [B, S, H, DK]
            v   [B, S, H, DV]
            g   [B, S, H, DK]   (log-space forget gates)
    Output: o            [B, S, H, DV]    (same dtype as input)
            final_state  [B, H, DK, DV]   (same dtype as input)

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
                 seq_len: int,
                 heads: int,
                 dim_k: int,
                 dim_v: int,
                 chunk_size: int,
                 scale: float = -1.0,
                 output_final_state: bool = True,
                 dtype: torch.dtype = torch.float16,
                 config: Optional[dict] = None,
                 tune: bool = False):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"GLAFwdKernel only supports dtypes [{supported}], got {dtype}"
            )
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.chunk_size = chunk_size
        self.scale = scale
        self.output_final_state = output_final_state
        self.dtype = dtype
        self.dtype_str = self.dtype_to_str(dtype)

        self.kernel = _make_gla_fwd_kernel(
            batch, seq_len, heads, dim_k, dim_v, chunk_size, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"block_size": 256}

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                g: torch.Tensor):
        prim_func = self.kernel(block_size=self.config["block_size"])
        return prim_func(q, k, v, g)

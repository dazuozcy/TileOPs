"""SSD chunk scan forward kernel (NPU, target=npuir) -- STUB (empty body).

Fuses the Mamba-2 SSD chunk output: history contribution
``exp(dA[l]) * (C[l] @ prev_states)`` plus the causal intra-chunk path
over ``cb``, producing the output ``y``.

Status: **STUB** -- the prim_func compiles and runs on NPU, but the kernel
body is empty (``pass``).  The output tensor is allocated with the correct
shape/dtype but its values are undefined, so correctness tests will fail.
Benchmark / JIT-compile / roofline flows execute normally.

To implement:
  1. Replace the ``pass`` in the ``T.Kernel`` block with the real
     fused SSD chunk scan body.
  2. Everything else (factory, wrapper, Op dispatch) stays as-is.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from kernels.kernel_base import Kernel

__all__ = ["SSDChunkScanFwdKernel"]


@functools.lru_cache(maxsize=32)
def _make_ssd_chunk_scan_kernel(B, NC, Q, H, P, N, G, dtype,
                                output_dtype=None):
    """Build SSD chunk scan kernel (STUB -- empty body).

    Args:
        B: Batch size.
        NC: Number of chunks.
        Q: Chunk length.
        H: Number of heads.
        P: Head dimension (d_head).
        N: State dimension (d_state).
        G: Number of groups.
        dtype: Input dtype string (float16 / bfloat16).
        output_dtype: Output dtype string; defaults to ``float32``.

    ``block_size`` is passed at call time via the ``block_size`` argument
    to the returned ``kernel`` callable.
    """
    out_dtype = output_dtype or "float32"
    S = NC * Q

    @tilelang.jit(out_idx=[6], target="npuir")
    def kernel(block_size):
        @T.prim_func
        def main(
            x: T.Tensor((B, S, H, P), dtype),
            cb: T.Tensor((B, NC, G, Q, Q), dtype),
            dA_cumsum: T.Tensor((B, H, NC, Q), "float32"),
            C: T.Tensor((B, S, G, N), dtype),
            prev_states: T.Tensor((B, NC, H, P, N), "float32"),
            dt: T.Tensor((B, H, NC, Q), dtype),
            y: T.Tensor((B, S, H, P), out_dtype),
        ):
            with T.Kernel(T.ceildiv(B * S * H * P, block_size),
                          is_npu=True) as (cid, _):
                # TODO: fused SSD chunk scan:
                #   out[l, p] = exp(dA[l]) * (C[l] @ prev_states)
                #             + sum_{s<=l} cb[l,s] * exp(dA[l]-dA[s]) * dt[s] * x[s, p]
                pass

        return main

    return kernel


class SSDChunkScanFwdKernel(Kernel):
    """SSD chunk scan forward kernel wrapper (NPU STUB -- empty body).

    Implements the fused SSD chunk output producing ``y``.

    Input:  x            [B, S, H, P]
            cb           [B, NC, G, Q, Q]
            dA_cumsum    [B, H, NC, Q]          (float32)
            C            [B, S, G, N]
            prev_states  [B, NC, H, P, N]       (float32)
            dt           [B, H, NC, Q]
    Output: y            [B, S, H, P]           (float32)

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
                 num_chunks: int,
                 chunk_len: int,
                 n_heads: int,
                 d_head: int,
                 d_state: int,
                 n_groups: int,
                 dtype: torch.dtype = torch.float16,
                 config: Optional[dict] = None,
                 tune: bool = False):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"SSDChunkScanFwdKernel only supports dtypes [{supported}], got {dtype}"
            )
        self.batch = batch
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.dtype = dtype
        self.dtype_str = self.dtype_to_str(dtype)

        self.kernel = _make_ssd_chunk_scan_kernel(
            batch, num_chunks, chunk_len, n_heads, d_head, d_state,
            n_groups, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"block_size": 256}

    def forward(self, x: torch.Tensor, cb: torch.Tensor,
                dA_cumsum: torch.Tensor, C: torch.Tensor,
                prev_states: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        prim_func = self.kernel(block_size=self.config["block_size"])
        return prim_func(x, cb, dA_cumsum, C, prev_states, dt)

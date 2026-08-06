"""MoE grouped GEMM forward kernel (NPU, target=npuir) -- STUB (empty body).

Computes NT grouped GEMM for MoE without block_m-aligned padding:

    c[e, row, col] = a[offset_e + row, :] @ b[e, col, :]^T

for each expert ``e``, where ``offset_e`` and row counts come from
``true_offsets`` and ``true_sizes``.

Status: **STUB** -- the prim_func compiles and runs on NPU, but the kernel
body is empty (``pass``).  The output tensor is allocated with the correct
shape/dtype but its values are undefined, so correctness tests will fail.
Benchmark / JIT-compile / roofline flows execute normally.

To implement:
  1. Replace the ``pass`` in the ``T.Kernel`` block with the real
     tile-scheduled NT grouped GEMM body.
  2. Everything else (factory, wrapper, Op dispatch) stays as-is.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from kernels.kernel_base import Kernel

__all__ = ["MoeGroupedGemmNopadKernel"]


@functools.lru_cache(maxsize=32)
def _make_moe_grouped_gemm_kernel(numel, num_experts, N, K, dtype,
                                  output_dtype=None):
    """Build MoE grouped GEMM kernel (STUB -- empty body).

    Args:
        numel: Total (token, expert) pairs = tight row count.
        num_experts: Total number of experts E.
        N: Output feature dimension.
        K: Input feature dimension.
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
            a: T.Tensor((numel, K), dtype),
            b: T.Tensor((num_experts, N, K), dtype),
            true_sizes: T.Tensor((num_experts,), "int32"),
            true_offsets: T.Tensor((num_experts,), "int32"),
            c: T.Tensor((numel, N), out_dtype),
        ):
            with T.Kernel(T.ceildiv(numel, block_size),
                          is_npu=True) as (cid, _):
                # TODO: tile-scheduled NT grouped GEMM:
                #   for each (expert, row_offset) tile:
                #     c[row, col] = a[row, :] @ b[expert, col, :]^T
                pass

        return main

    return kernel


class MoeGroupedGemmNopadKernel(Kernel):
    """MoE grouped GEMM forward kernel wrapper (NPU STUB -- empty body).

    Implements ``c = grouped_gemm(a, b, true_sizes, true_offsets)`` (NT,
    no-pad, tile-scheduled).

    Input:  a             [numel, K]
            b             [num_experts, N, K]
            true_sizes    [num_experts]     (int32)
            true_offsets  [num_experts]     (int32)
    Output: c             [numel, N]

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
                 numel: int,
                 num_experts: int,
                 n: int,
                 k: int,
                 dtype: torch.dtype = torch.bfloat16,
                 config: Optional[dict] = None,
                 tune: bool = False):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"MoeGroupedGemmNopadKernel only supports dtypes [{supported}], got {dtype}"
            )
        self.numel = numel
        self.num_experts = num_experts
        self.n = n
        self.k = k
        self.dtype = dtype
        self.dtype_str = self.dtype_to_str(dtype)

        self.kernel = _make_moe_grouped_gemm_kernel(
            numel, num_experts, n, k, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"block_size": 256}

    def forward(self, a: torch.Tensor, b: torch.Tensor,
                true_sizes: torch.Tensor,
                true_offsets: torch.Tensor) -> torch.Tensor:
        prim_func = self.kernel(block_size=self.config["block_size"])
        return prim_func(a, b, true_sizes, true_offsets)

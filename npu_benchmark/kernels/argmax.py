"""Argmax forward kernel (NPU, target=npuir) — STUB (empty body).

Computes ``y[m] = argmax(x[m, :])`` over the last dimension of a 2-D
``(M, N)`` tensor, producing a 1-D ``(M,)`` int64 output of indices.

Status: **STUB** — the prim_func compiles and runs on NPU, but the kernel
body is empty (``pass``).  The output tensor is allocated with the correct
shape/dtype but its values are undefined, so correctness tests will fail.
Benchmark / JIT-compile / roofline flows execute normally.

To implement:
  1. Replace the ``pass`` in the ``T.Kernel`` block with the real
     reduction body (running max + index tracking).
  2. Everything else (factory, wrapper, Op dispatch) stays as-is.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from kernels.kernel_base import Kernel

__all__ = ["ArgmaxKernel"]

_NPU_MAX_CORE_DIM = 65535


@functools.lru_cache(maxsize=32)
def _make_argmax_kernel(M, N, dtype, output_dtype="int64"):
    """Build Argmax kernel (STUB — empty body).

    Args:
        M: Number of rows (product of non-reduction dims).
        N: Reduction dimension size.
        dtype: Input dtype string (float16 / bfloat16 / float32).
        output_dtype: Output dtype string (defaults to ``int64``).

    ``block_size`` (rows per NPU core) is passed at call time via the
    ``block_size`` argument to the returned ``kernel`` callable.
    """
    out_dtype = output_dtype

    @tilelang.jit(out_idx=[1], target="npuir")
    def kernel(block_size):
        @T.prim_func
        def main(
            x: T.Tensor((M, N), dtype),
            y: T.Tensor((M,), out_dtype),
        ):
            with T.Kernel(T.ceildiv(M, block_size), is_npu=True) as (cid, _):
                # TODO: per-row argmax reduction:
                #   running_max = -inf, running_idx = 0
                #   for n in range(N):
                #     v = x[m, n]
                #     if v > running_max:
                #       running_max = v
                #       running_idx = n
                #   y[m] = running_idx
                pass

        return main

    return kernel


class ArgmaxKernel(Kernel):
    """Argmax forward kernel wrapper (NPU STUB — empty body).

    Implements ``y = argmax(x, dim=-1)`` over the last dimension of a
    2-D ``(M, N)`` tensor, producing a 1-D ``(M,)`` int64 output.

    The Op layer flattens the input to ``(M, N)`` (moving the reduction
    dim to the last axis) before dispatching this kernel.

    Supported dtypes: float16, bfloat16, float32 (output is always int64).

    NPU tiling: each core processes ``block_size`` rows (GM → UB load,
    vector reduce, UB → GM store).  There is no GPU-style ``threads`` /
    ``npt`` split because the NPU has no SIMT thread model.

    Status: **STUB** — kernel body is empty; output values are undefined.
    """

    supported_archs: Optional[list[int]] = None
    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
    prof_name = "main"

    def __init__(self,
                 M: int,
                 N: int,
                 dtype: torch.dtype,
                 config: Optional[dict] = None,
                 tune: bool = False):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"ArgmaxKernel only supports dtypes [{supported}], got {dtype}"
            )
        self.M = M
        self.N = N
        self.dtype = dtype
        self.out_dtype = torch.int64
        self.dtype_str = self.dtype_to_str(dtype)
        self.out_dtype_str = self.dtype_to_str(torch.int64)

        self.kernel = _make_argmax_kernel(self.M, self.N, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        block_size = 1
        min_block_size = (self.M + _NPU_MAX_CORE_DIM - 1) // _NPU_MAX_CORE_DIM
        if block_size < min_block_size:
            block_size = min_block_size
        return {"block_size": block_size}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prim_func = self.kernel(block_size=self.config["block_size"])
        return prim_func(x)

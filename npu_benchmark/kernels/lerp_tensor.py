"""Tensor-weight lerp kernel — out = a + w * (b - a).

Ported from TileOPs (tileops/kernels/elementwise.py::_make_lerp_tensor_kernel).
The TileLang prim_func is backend-agnostic IR; the actual codegen target
(NPU/CUDA) is determined by tilelang at JIT time based on the device of
the input tensors.

The Op layer pre-broadcasts ``input`` / ``end`` / ``weight`` to the flat
output shape so the kernel sees three contiguous 1-D tensors of size ``N``.

Uses the register-fragment load -> compute -> fragment store strategy
(alloc_fragment, T.copy, T.Parallel) so all three inputs and the output
share the same vectorized memory access path.
"""

from __future__ import annotations

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from kernels.kernel_base import Kernel

__all__ = ["LerpTensorKernel"]


@functools.lru_cache(maxsize=32)
def _make_lerp_tensor_kernel(N, dtype, output_dtype=None, threads=256, npt=8):
    """Build Tensor-weight lerp kernel: out = a + weight * (b - a).

    Args:
        N: Number of elements (flat 1-D size, post-broadcast).
        dtype: Input dtype string (float16 / bfloat16 / float32).
        output_dtype: Output dtype string; defaults to ``dtype``.
        threads: Threads per block.
        npt: Elements per thread (vectorization factor).
    """
    out_dtype = output_dtype or dtype
    block_size = threads * npt

    @tilelang.jit(out_idx=[3])
    def kernel(threads_arg, npt_arg):
        @T.prim_func
        def main(
            a: T.Tensor((N,), dtype),
            b: T.Tensor((N,), dtype),
            w: T.Tensor((N,), dtype),
            out: T.Tensor((N,), out_dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                a_reg = T.alloc_fragment((block_size,), dtype)
                b_reg = T.alloc_fragment((block_size,), dtype)
                w_reg = T.alloc_fragment((block_size,), dtype)
                T.copy(a[bx * block_size : (bx + 1) * block_size], a_reg)
                T.copy(b[bx * block_size : (bx + 1) * block_size], b_reg)
                T.copy(w[bx * block_size : (bx + 1) * block_size], w_reg)
                for i, j in T.Parallel(threads_arg, npt_arg):
                    k = i * npt_arg + j
                    a_reg[k] = a_reg[k] + w_reg[k] * (b_reg[k] - a_reg[k])
                T.copy(a_reg, out[bx * block_size : (bx + 1) * block_size])

        return main

    return kernel


class LerpTensorKernel(Kernel):
    """Kernel wrapper for tensor-weight lerp.

    Implements the Tensor-weight overload of ``torch.lerp`` —
    ``torch.lerp(input, end, weight: Tensor)`` — where all three operands
    are float tensors of the same dtype, pre-broadcast by the Op layer to
    a flat ``N``-element view.

    Supported dtypes: float16, bfloat16, float32.
    """

    supported_archs: Optional[list[int]] = None
    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(self,
                 N: int,
                 dtype: torch.dtype,
                 config: Optional[dict] = None,
                 tune: bool = False):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"LerpTensorKernel only supports dtypes [{supported}], got {dtype}"
            )
        self.N = N
        self.dtype = dtype
        self.dtype_str = self.dtype_to_str(dtype)

        cfg = self.default_config
        self.kernel = _make_lerp_tensor_kernel(
            self.N, self.dtype_str,
            threads=cfg["threads"], npt=cfg["num_per_thread"])
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        if self.dtype == torch.float32:
            npt = 4
        else:
            npt = 8
        return {"threads": 256, "num_per_thread": npt}

    def forward(self, a: torch.Tensor, b: torch.Tensor,
                w: torch.Tensor) -> torch.Tensor:
        prim_func = self.kernel(
            threads=self.config["threads"],
            npt=self.config["num_per_thread"],
        )
        return prim_func(a, b, w)

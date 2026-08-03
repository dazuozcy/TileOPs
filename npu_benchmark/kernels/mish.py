"""Element-wise Mish activation kernel — y = x * tanh(softplus(x)).

Ported from TileOPs (tileops/kernels/elementwise.py::MishFwdKernel).
The TileLang prim_func is backend-agnostic IR; the actual codegen target
(NPU/CUDA) is determined by tilelang at JIT time based on the device of
the input tensors.

Uses the register-fragment load -> compute -> fragment store strategy
(alloc_fragment, T.copy, T.Parallel) for coalesced memory access.

Computation: mish(x) = x * tanh(log(1 + exp(x))) computed in float32
for numerical stability, then stored back to the input dtype.
"""

from __future__ import annotations

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from kernels.kernel_base import Kernel

__all__ = ["MishKernel"]


@functools.lru_cache(maxsize=32)
def _make_mish_kernel(N, dtype, output_dtype=None, threads=256, npt=8):
    """Build Mish kernel: y = x * tanh(softplus(x)).

    Args:
        N: Number of elements (flat 1-D size).
        dtype: Input dtype string (float16 / bfloat16 / float32).
        output_dtype: Output dtype string; defaults to ``dtype``.
        threads: Threads per block.
        npt: Elements per thread (vectorization factor).
    """
    out_dtype = output_dtype or dtype
    block_size = threads * npt

    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg, npt_arg):
        @T.prim_func
        def main(
            x: T.Tensor((N,), dtype),
            y: T.Tensor((N,), out_dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                x_reg = T.alloc_fragment((block_size,), dtype)
                y_reg = T.alloc_fragment((block_size,), out_dtype)
                T.copy(x[bx * block_size : (bx + 1) * block_size], x_reg)
                for i, j in T.Parallel(threads_arg, npt_arg):
                    k = i * npt_arg + j
                    xv = x_reg[k]
                    one = T.cast(1.0, "float32")
                    y_reg[k] = xv * T.tanh(T.log(one + T.exp(xv)))
                T.copy(y_reg, y[bx * block_size : (bx + 1) * block_size])

        return main

    return kernel


class MishKernel(Kernel):
    """Kernel wrapper for element-wise Mish activation.

    Implements ``y = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))``.

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
                f"MishKernel only supports dtypes [{supported}], got {dtype}"
            )
        self.N = N
        self.dtype = dtype
        self.dtype_str = self.dtype_to_str(dtype)

        cfg = self.default_config
        self.kernel = _make_mish_kernel(
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prim_func = self.kernel(
            threads=self.config["threads"],
            npt=self.config["num_per_thread"],
        )
        return prim_func(x)

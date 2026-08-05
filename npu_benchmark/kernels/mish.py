"""Element-wise Mish activation kernel — y = x * tanh(softplus(x)).

Ported from TileOPs (tileops/kernels/elementwise.py::MishFwdKernel).
The TileLang prim_func targets the Ascend NPU backend (target="npuir")
and uses NPU vector primitives (alloc_ub, vexp, vadd, vmul, vsub, vdiv,
vcast) on Unified Buffer (UB).

Computation: mish(x) = x * tanh(log(1 + exp(x))) computed in float32
for numerical stability, then stored back to the input dtype.

NPU tiling: each core processes ``block_size`` contiguous elements
(GM -> UB copy, vector compute, UB -> GM store).  ``block_size`` is a
runtime argument to the JIT kernel — there is no GPU-style
``threads`` / ``npt`` split because the NPU has no SIMT thread model.
"""

from __future__ import annotations

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from kernels.kernel_base import Kernel

__all__ = ["MishKernel"]

_NPU_MAX_CORE_DIM = 65535


@functools.lru_cache(maxsize=32)
def _make_mish_kernel(N, dtype, output_dtype=None):
    """Build Mish kernel: y = x * tanh(softplus(x)).

    Args:
        N: Number of elements (flat 1-D size).
        dtype: Input dtype string (float16 / bfloat16 / float32).
        output_dtype: Output dtype string; defaults to ``dtype``.

    ``block_size`` (elements per NPU core) is passed at call time via
    the ``block_size`` argument to the returned ``kernel`` callable —
    it is not a factory parameter.
    """
    out_dtype = output_dtype or dtype

    @tilelang.jit(out_idx=[1], target="npuir")
    def kernel(block_size):
        @T.prim_func
        def main(
            x: T.Tensor((N,), dtype),
            y: T.Tensor((N,), out_dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_size), is_npu=True) as (cid, _):
                # Tail-block handling: valid element count for this block.
                tail_size = T.min(block_size, N - cid * block_size)

                if dtype != "float32":
                    # --- float16 / bfloat16 path: float32 intermediate ---
                    # alloc_ub places buffers in UB (scope="shared"); alloc_shared
                    # maps to L1/cbuf whose GM store path is unsupported for 1D
                    # vector kernels (REVIEW warn-6, DESIGN.md §2.3 fallback).
                    x_ub = T.alloc_ub((block_size,), dtype)         # original dtype input
                    x_f32 = T.alloc_ub((block_size,), "float32")    # upcast input
                    t1 = T.alloc_ub((block_size,), "float32")       # exp(x)
                    t2 = T.alloc_ub((block_size,), "float32")       # 1 + exp(x)
                    t2sq = T.alloc_ub((block_size,), "float32")     # t2^2
                    num = T.alloc_ub((block_size,), "float32")      # t2^2 - 1
                    den = T.alloc_ub((block_size,), "float32")      # t2^2 + 1
                    tanh_sp = T.alloc_ub((block_size,), "float32")  # tanh(softplus(x))
                    y_f32 = T.alloc_ub((block_size,), "float32")    # mish(x) in float32
                    y_ub = T.alloc_ub((block_size,), out_dtype)     # downcast output

                    # GM -> UB (copy only valid elements)
                    T.copy(x[cid * block_size : cid * block_size + tail_size],
                           x_ub[0:tail_size])

                    # Upcast to float32 for numerical stability
                    T.vcast(x_ub, x_f32, round_mode="rint")

                    # Core mish computation in float32.
                    # Uses the algebraic identity
                    #   tanh(softplus(x)) = (t2^2 - 1) / (t2^2 + 1),  t2 = 1 + exp(x)
                    # which is mathematically equal to tanh(ln(1 + exp(x))) but avoids
                    # T.vtanh: on 1-D UB buffers T.vtanh lowers to a 7th-order Taylor
                    # polynomial that diverges for |t| > ~1.57 (softplus(x) > ~1.57
                    # for x > ~1.57), producing large precision errors. vdiv lowers
                    # to the hardware VDivOp which is exact (verified up to x=5).
                    T.vexp(x_f32, t1)             # t1 = exp(x)
                    T.vadd(t1, 1.0, t2)           # t2 = 1 + exp(x)
                    T.vmul(t2, t2, t2sq)          # t2sq = t2^2
                    T.vsub(t2sq, 1.0, num)        # num  = t2^2 - 1
                    T.vadd(t2sq, 1.0, den)        # den  = t2^2 + 1
                    T.vdiv(num, den, tanh_sp)     # tanh(softplus(x)) = num / den
                    T.vmul(x_f32, tanh_sp, y_f32) # y    = x * tanh(softplus(x))

                    # Downcast back to original dtype
                    T.vcast(y_f32, y_ub, round_mode="round")

                    # UB -> GM (copy only valid elements)
                    T.copy(y_ub[0:tail_size],
                           y[cid * block_size : cid * block_size + tail_size])
                else:
                    # --- float32 path: direct computation ---
                    x_ub = T.alloc_ub((block_size,), "float32")
                    t1 = T.alloc_ub((block_size,), "float32")       # exp(x)
                    t2 = T.alloc_ub((block_size,), "float32")       # 1 + exp(x)
                    t2sq = T.alloc_ub((block_size,), "float32")     # t2^2
                    num = T.alloc_ub((block_size,), "float32")      # t2^2 - 1
                    den = T.alloc_ub((block_size,), "float32")      # t2^2 + 1
                    tanh_sp = T.alloc_ub((block_size,), "float32")  # tanh(softplus(x))
                    y_ub = T.alloc_ub((block_size,), "float32")     # mish(x)

                    # GM -> UB
                    T.copy(x[cid * block_size : cid * block_size + tail_size],
                           x_ub[0:tail_size])

                    # Core mish computation in float32 (algebraic identity, see above)
                    T.vexp(x_ub, t1)             # t1 = exp(x)
                    T.vadd(t1, 1.0, t2)          # t2 = 1 + exp(x)
                    T.vmul(t2, t2, t2sq)         # t2sq = t2^2
                    T.vsub(t2sq, 1.0, num)       # num  = t2^2 - 1
                    T.vadd(t2sq, 1.0, den)       # den  = t2^2 + 1
                    T.vdiv(num, den, tanh_sp)    # tanh(softplus(x)) = num / den
                    T.vmul(x_ub, tanh_sp, y_ub)  # y    = x * tanh(softplus(x))

                    # UB -> GM
                    T.copy(y_ub[0:tail_size],
                           y[cid * block_size : cid * block_size + tail_size])

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

        self.kernel = _make_mish_kernel(self.N, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        if self.dtype == torch.float32:
            block_size = 1024
        else:
            block_size = 2048
        # NPU coreDim (block count) limit is 65535. Ensure
        # ceildiv(N, block_size) <= 65535 by growing block_size when N is large.
        min_block_size = (self.N + _NPU_MAX_CORE_DIM - 1) // _NPU_MAX_CORE_DIM
        if block_size < min_block_size:
            block_size = ((min_block_size + 255) // 256) * 256
        return {"block_size": block_size}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prim_func = self.kernel(block_size=self.config["block_size"])
        return prim_func(x)

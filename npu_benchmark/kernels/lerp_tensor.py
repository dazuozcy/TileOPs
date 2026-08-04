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

_NPU_MAX_CORE_DIM = 65535


@functools.lru_cache(maxsize=32)
def _make_lerp_tensor_kernel(N, dtype, output_dtype=None, threads=256, npt=8):
    """Build Tensor-weight lerp kernel: out = a + weight * (b - a).

    Args:
        N: Number of elements (flat 1-D size, post-broadcast).
        dtype: Input dtype string (float16 / bfloat16 / float32).
        output_dtype: Output dtype string; defaults to ``dtype``.
        threads: Threads per block (factory cache key; the runtime value is
            passed to ``kernel`` via ``threads_arg``).
        npt: Elements per thread (factory cache key; the runtime value is
            passed to ``kernel`` via ``npt_arg``).
    """
    out_dtype = output_dtype or dtype

    @tilelang.jit(out_idx=[3], target="npuir")
    def kernel(threads_arg, npt_arg):
        block_size = threads_arg * npt_arg

        @T.prim_func
        def main(
            a: T.Tensor((N,), dtype),
            b: T.Tensor((N,), dtype),
            w: T.Tensor((N,), dtype),
            out: T.Tensor((N,), out_dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_size), is_npu=True) as (cid, _):
                tail_size = T.min(block_size, N - cid * block_size)

                if dtype != "float32":
                    # --- float16 / bfloat16 path: float32 intermediate ---
                    a_ub = T.alloc_ub((block_size,), dtype)
                    b_ub = T.alloc_ub((block_size,), dtype)
                    w_ub = T.alloc_ub((block_size,), dtype)
                    a_f32 = T.alloc_ub((block_size,), "float32")
                    b_f32 = T.alloc_ub((block_size,), "float32")
                    w_f32 = T.alloc_ub((block_size,), "float32")
                    tmp_f32 = T.alloc_ub((block_size,), "float32")
                    out_f32 = T.alloc_ub((block_size,), "float32")
                    out_ub = T.alloc_ub((block_size,), out_dtype)

                    T.copy(a[cid * block_size : cid * block_size + tail_size],
                           a_ub[0:tail_size])
                    T.copy(b[cid * block_size : cid * block_size + tail_size],
                           b_ub[0:tail_size])
                    T.copy(w[cid * block_size : cid * block_size + tail_size],
                           w_ub[0:tail_size])

                    T.vcast(a_ub, a_f32, round_mode="rint")
                    T.vcast(b_ub, b_f32, round_mode="rint")
                    T.vcast(w_ub, w_f32, round_mode="rint")

                    T.vsub(b_f32, a_f32, tmp_f32)
                    T.vmul(w_f32, tmp_f32, tmp_f32)
                    T.vadd(a_f32, tmp_f32, out_f32)

                    T.vcast(out_f32, out_ub, round_mode="round")
                    T.copy(out_ub[0:tail_size],
                           out[cid * block_size : cid * block_size + tail_size])
                else:
                    # --- float32 path: direct computation ---
                    a_ub = T.alloc_ub((block_size,), "float32")
                    b_ub = T.alloc_ub((block_size,), "float32")
                    w_ub = T.alloc_ub((block_size,), "float32")
                    tmp_ub = T.alloc_ub((block_size,), "float32")
                    out_ub = T.alloc_ub((block_size,), "float32")

                    T.copy(a[cid * block_size : cid * block_size + tail_size],
                           a_ub[0:tail_size])
                    T.copy(b[cid * block_size : cid * block_size + tail_size],
                           b_ub[0:tail_size])
                    T.copy(w[cid * block_size : cid * block_size + tail_size],
                           w_ub[0:tail_size])

                    T.vsub(b_ub, a_ub, tmp_ub)
                    T.vmul(w_ub, tmp_ub, tmp_ub)
                    T.vadd(a_ub, tmp_ub, out_ub)

                    T.copy(out_ub[0:tail_size],
                           out[cid * block_size : cid * block_size + tail_size])

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
        # NPU coreDim (block count) limit is 65535. Ensure
        # ceildiv(N, block_size) <= 65535 by growing npt when N is large.
        min_block_size = (self.N + _NPU_MAX_CORE_DIM - 1) // _NPU_MAX_CORE_DIM
        block_size = 256 * npt
        if block_size < min_block_size:
            npt = max(npt, (min_block_size + 255) // 256)
        return {"threads": 256, "num_per_thread": npt}

    def forward(self, a: torch.Tensor, b: torch.Tensor,
                w: torch.Tensor) -> torch.Tensor:
        prim_func = self.kernel(
            threads_arg=self.config["threads"],
            npt_arg=self.config["num_per_thread"],
        )
        return prim_func(a, b, w)

"""LogSumExp forward kernel (NPU, target=npuir) — PLACEHOLDER STUB.

Computes ``y[m] = log(sum(exp(x[m, :])))`` over the last dimension of a
2-D ``(M, N)`` tensor, producing a 1-D ``(M,)`` output.

Status: **PLACEHOLDER** — the TileLang Ascend backend does not yet provide
the reduction primitives (intra-UB vector reduce, cross-tile accumulation)
needed for an efficient NPU-native logsumexp.  ``forward()`` raises
``NotImplementedError`` so the Op / manifest / workload / bench / test
layers can be wired up without a compilable kernel.

To implement:
  1. Uncomment and fill in the ``_make_logsumexp_kernel`` factory below
     with NPU vector primitives (``alloc_ub``, ``vexp``, ``vadd``,
     ``vmax``, etc.) or a sequential per-core reduction.
  2. Set ``self.kernel = _make_logsumexp_kernel(...)`` in ``__init__``.
  3. Remove the ``raise NotImplementedError`` in ``forward()`` and call
     the compiled prim_func.
  4. Drop the ``pytest.mark.skip`` markers in the test / bench files.

Intended prim_func signature::

    @tilelang.jit(out_idx=[1], target="npuir")
    def kernel(block_size):
        @T.prim_func
        def main(
            x: T.Tensor((M, N), dtype),
            y: T.Tensor((M,), out_dtype),
        ):
            with T.Kernel(T.ceildiv(M, block_size), is_npu=True) as (cid, _):
                # Per-row online logsumexp:
                #   load row tile -> running max + rescaled sum -> y[m] = log(sum) + max
                ...
        return main
"""

from __future__ import annotations

from typing import Optional

import torch

from kernels.kernel_base import Kernel

__all__ = ["LogSumExpKernel"]

_NPU_MAX_CORE_DIM = 65535


# ---------- Kernel factory (PLACEHOLDER — uncomment when implementing) ----------
#
# import functools
# import tilelang
# import tilelang.language as T
#
# @functools.lru_cache(maxsize=32)
# def _make_logsumexp_kernel(M, N, dtype, output_dtype=None):
#     out_dtype = output_dtype or dtype
#
#     @tilelang.jit(out_idx=[1], target="npuir")
#     def kernel(block_size):
#         @T.prim_func
#         def main(
#             x: T.Tensor((M, N), dtype),
#             y: T.Tensor((M,), out_dtype),
#         ):
#             with T.Kernel(T.ceildiv(M, block_size), is_npu=True) as (cid, _):
#                 # TODO: implement online logsumexp reduction per row.
#                 # Algorithm:
#                 #   running_max = -inf, running_sum = 0
#                 #   for tile in N:
#                 #     x_ub = load x[m, off:off+tile]
#                 #     tile_max = vmax(x_ub)
#                 #     running_sum = running_sum * exp(running_max - tile_max) + sum(exp(x_ub - tile_max))
#                 #     running_max = max(running_max, tile_max)
#                 #   y[m] = running_max + log(running_sum)
#                 pass
#         return main
#
#     return kernel


class LogSumExpKernel(Kernel):
    """LogSumExp forward kernel wrapper (NPU PLACEHOLDER).

    Implements ``y = logsumexp(x, dim=-1)`` over the last dimension of a
    2-D ``(M, N)`` tensor, producing a 1-D ``(M,)`` output.

    The Op layer flattens the input to ``(M, N)`` (moving the reduction
    dim to the last axis) before dispatching this kernel.

    Supported dtypes: float16, bfloat16, float32.

    NPU tiling: each core processes ``block_size`` rows (GM → UB load,
    vector reduce, UB → GM store).  There is no GPU-style ``threads`` /
    ``npt`` split because the NPU has no SIMT thread model.

    Status: **PLACEHOLDER** — ``forward()`` raises ``NotImplementedError``.
    """

    supported_archs: Optional[list[int]] = None
    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

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
                f"LogSumExpKernel only supports dtypes [{supported}], got {dtype}"
            )
        self.M = M
        self.N = N
        self.dtype = dtype
        self.dtype_str = self.dtype_to_str(dtype)

        # self.kernel = _make_logsumexp_kernel(self.M, self.N, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        block_size = 1
        min_block_size = (self.M + _NPU_MAX_CORE_DIM - 1) // _NPU_MAX_CORE_DIM
        if block_size < min_block_size:
            block_size = min_block_size
        return {"block_size": block_size}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "LogSumExpKernel NPU implementation is a placeholder. "
            "The TileLang Ascend backend does not yet provide the "
            "reduction primitives needed for an NPU-native logsumexp. "
            "See kernels/logsumexp.py for the stub and implementation guide."
        )

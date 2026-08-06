"""Vector norm kernel (NPU, target=npuir) — STUB (empty body).

Computes ``y[m] = ||x[m, :]||_p`` over the last dimension of a 2-D
``(M, N)`` tensor, producing a 1-D ``(M,)`` output.  Supports L1, L2,
and infinity norms via the ``op_kind`` parameter.

Status: **STUB** — the prim_func compiles and runs on NPU, but the kernel
body is empty (``pass``).  The output tensor is allocated with the correct
shape/dtype but its values are undefined, so correctness tests will fail.
Benchmark / JIT-compile / roofline flows execute normally.

To implement:
  1. Replace the ``pass`` in the ``T.Kernel`` block with the real
     reduction body.
  2. Everything else (factory, wrapper, Op dispatch) stays as-is.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from kernels.kernel_base import Kernel

__all__ = ["VectorNormKernel"]

_NPU_MAX_CORE_DIM = 65535


@functools.lru_cache(maxsize=32)
def _make_vector_norm_kernel(M, N, op_kind, dtype, output_dtype=None):
    """Build vector-norm kernel (STUB — empty body).

    Args:
        M: Number of rows (product of non-reduction dims).
        N: Reduction dimension size.
        op_kind: One of ``"l1"``, ``"l2"``, ``"inf"``.
        dtype: Input dtype string (float16 / bfloat16 / float32).
        output_dtype: Output dtype string; defaults to ``dtype``.

    ``block_size`` (rows per NPU core) is passed at call time via the
    ``block_size`` argument to the returned ``kernel`` callable.
    """
    out_dtype = output_dtype or dtype

    @tilelang.jit(out_idx=[1], target="npuir")
    def kernel(block_size):
        @T.prim_func
        def main(
            x: T.Tensor((M, N), dtype),
            y: T.Tensor((M,), out_dtype),
        ):
            with T.Kernel(T.ceildiv(M, block_size), is_npu=True) as (cid, _):
                # TODO: per-row vector norm reduction.
                # op_kind == "l1":  y[m] = sum(abs(x[m, :]))
                # op_kind == "l2":  y[m] = sqrt(sum(x[m, :] * x[m, :]))
                # op_kind == "inf": y[m] = max(abs(x[m, :]))
                pass

        return main

    return kernel


class VectorNormKernel(Kernel):
    """Vector norm kernel wrapper (NPU STUB — empty body).

    Implements ``y[m] = ||x[m, :]||_p`` over the last dimension of a 2-D
    ``(M, N)`` tensor, producing a 1-D ``(M,)`` output.

    The Op layer flattens the input to ``(M, N)`` (moving the reduction
    dim to the last axis) before dispatching this kernel.

    Supported dtypes: float16, bfloat16, float32.

    NPU tiling: each core processes ``block_size`` rows (GM → UB load,
    vector reduce, UB → GM store).  There is no GPU-style ``threads`` /
    ``npt`` split because the NPU has no SIMT thread model.

    Args:
        M: Number of rows (product of non-reduction dims).
        N: Reduction dimension size.
        op_kind: One of ``"l1"``, ``"l2"``, ``"inf"``.
        dtype: Data type (float16, bfloat16, or float32).
        config: Optional kernel configuration dict.
        tune: Whether to autotune (default False).

    Status: **STUB** — kernel body is empty; output values are undefined.
    """

    supported_archs: Optional[list[int]] = None
    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
    _VALID_KINDS = ("l1", "l2", "inf")

    def __init__(self,
                 M: int,
                 N: int,
                 op_kind: str,
                 dtype: torch.dtype,
                 config: Optional[dict] = None,
                 tune: bool = False):
        super().__init__()
        if op_kind not in self._VALID_KINDS:
            raise ValueError(
                f"VectorNormKernel op_kind must be one of {self._VALID_KINDS}, "
                f"got {op_kind!r}"
            )
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"VectorNormKernel only supports dtypes [{supported}], got {dtype}"
            )
        self.M = M
        self.N = N
        self.op_kind = op_kind
        self.dtype = dtype
        self.dtype_str = self.dtype_to_str(dtype)

        self.kernel = _make_vector_norm_kernel(
            self.M, self.N, self.op_kind, self.dtype_str)
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

"""MaxPool3d forward kernel (NPU, target=npuir) — STUB (empty body).

Computes ``output = max_pool3d(input, kernel_size, stride, padding,
dilation, ceil_mode)``, matching ``torch.nn.functional.max_pool3d``.

Status: **STUB** — the prim_func compiles and runs on NPU, but the kernel
body is empty (``pass``).  The output tensor is allocated with the correct
shape/dtype but its values are undefined, so correctness tests will fail.
Benchmark / JIT-compile / roofline flows execute normally.

To implement:
  1. Replace the ``pass`` in the ``T.Kernel`` block with the real
     pooling body (tiled window max-reduce).
  2. Everything else (factory, wrapper, Op dispatch) stays as-is.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from kernels.kernel_base import Kernel

__all__ = ["MaxPool3dKernel"]


@functools.lru_cache(maxsize=32)
def _make_max_pool3d_kernel(N, C, D, H, W, kD, kH, kW,
                            stride_d, stride_h, stride_w,
                            pad_d, pad_h, pad_w,
                            dilation_d, dilation_h, dilation_w,
                            ceil_mode, dtype, output_dtype=None):
    """Build MaxPool3d kernel (STUB — empty body).

    Args:
        N: Batch size.
        C: Channels.
        D, H, W: Input spatial dims.
        kD, kH, kW: Kernel spatial dims.
        stride_d, stride_h, stride_w: Stride.
        pad_d, pad_h, pad_w: Padding.
        dilation_d, dilation_h, dilation_w: Dilation.
        ceil_mode: Use ceil-mode output size computation.
        dtype: Input dtype string (float16 / bfloat16 / float32).
        output_dtype: Output dtype string; defaults to ``dtype``.

    ``block_size`` is passed at call time via the ``block_size`` argument
    to the returned ``kernel`` callable.
    """
    out_dtype = output_dtype or dtype

    def _out_dim(inp, k, pad, dilation, stride):
        if ceil_mode:
            return (inp + 2 * pad - dilation * (k - 1) - 1 + stride - 1) // stride + 1
        return (inp + 2 * pad - dilation * (k - 1) - 1) // stride + 1

    out_D = _out_dim(D, kD, pad_d, dilation_d, stride_d)
    out_H = _out_dim(H, kH, pad_h, dilation_h, stride_h)
    out_W = _out_dim(W, kW, pad_w, dilation_w, stride_w)

    @tilelang.jit(out_idx=[1], target="npuir")
    def kernel(block_size):
        @T.prim_func
        def main(
            input: T.Tensor((N, C, D, H, W), dtype),
            output: T.Tensor((N, C, out_D, out_H, out_W), out_dtype),
        ):
            with T.Kernel(T.ceildiv(N * C * out_D * out_H * out_W, block_size),
                          is_npu=True) as (cid, _):
                # TODO: tiled max-pool — for each (n, c, od, oh, ow) in tile:
                #   running_max = -inf
                #   for kd in range(kD):
                #     for kh in range(kH):
                #       for kw in range(kW):
                #         id = od*stride_d - pad_d + kd*dilation_d
                #         ih = oh*stride_h - pad_h + kh*dilation_h
                #         iw = ow*stride_w - pad_w + kw*dilation_w
                #         if 0 <= id < D and 0 <= ih < H and 0 <= iw < W:
                #           running_max = max(running_max, input[n,c,id,ih,iw])
                #   output[n, c, od, oh, ow] = running_max
                pass

        return main

    return kernel


class MaxPool3dKernel(Kernel):
    """MaxPool3d forward kernel wrapper (NPU STUB — empty body).

    Implements ``output = max_pool3d(input, kernel_size, stride, padding,
    dilation)`` matching ``torch.nn.functional.max_pool3d``.

    Input:  input  [N, C, D, H, W]
    Output: output [N, C, out_D, out_H, out_W]

    Supported dtypes: float16, bfloat16, float32.

    NPU tiling: each core processes ``block_size`` output elements.  There
    is no GPU-style ``threads`` / ``npt`` split because the NPU has no
    SIMT thread model.

    Args:
        n: Batch size.
        c: Channels.
        d, h, w: Input spatial dims.
        kD, kH, kW: Kernel spatial dims.
        stride_d, stride_h, stride_w: Stride.
        pad_d, pad_h, pad_w: Padding.
        dilation_d, dilation_h, dilation_w: Dilation.
        ceil_mode: Use ceil-mode output size computation.
        dtype: Data type.
        config: Optional kernel configuration dict.
        tune: Whether to autotune (default False).

    Status: **STUB** — kernel body is empty; output values are undefined.
    """

    supported_archs: Optional[list[int]] = None
    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
    prof_name = "main"

    def __init__(self,
                 n: int,
                 c: int,
                 d: int,
                 h: int,
                 w: int,
                 kD: int,
                 kH: int,
                 kW: int,
                 stride_d: int,
                 stride_h: int,
                 stride_w: int,
                 pad_d: int,
                 pad_h: int,
                 pad_w: int,
                 dilation_d: int,
                 dilation_h: int,
                 dilation_w: int,
                 ceil_mode: bool = False,
                 dtype: torch.dtype = torch.float16,
                 config: Optional[dict] = None,
                 tune: bool = False):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"MaxPool3dKernel only supports dtypes [{supported}], got {dtype}"
            )
        self.n = n
        self.c = c
        self.d = d
        self.h = h
        self.w = w
        self.kD = kD
        self.kH = kH
        self.kW = kW
        self.stride_d = stride_d
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.pad_d = pad_d
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.dilation_d = dilation_d
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.ceil_mode = ceil_mode
        self.dtype = dtype
        self.dtype_str = self.dtype_to_str(dtype)

        def _out_dim(inp, k, pad, dilation, stride):
            if ceil_mode:
                return (inp + 2 * pad - dilation * (k - 1) - 1 + stride - 1) // stride + 1
            return (inp + 2 * pad - dilation * (k - 1) - 1) // stride + 1

        self.out_d = _out_dim(d, kD, pad_d, dilation_d, stride_d)
        self.out_h = _out_dim(h, kH, pad_h, dilation_h, stride_h)
        self.out_w = _out_dim(w, kW, pad_w, dilation_w, stride_w)

        self.kernel = _make_max_pool3d_kernel(
            n, c, d, h, w, kD, kH, kW,
            stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w,
            dilation_d, dilation_h, dilation_w,
            ceil_mode, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"block_size": 256}

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        prim_func = self.kernel(block_size=self.config["block_size"])
        return prim_func(input)

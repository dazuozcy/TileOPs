"""AvgPool2d forward kernel (NPU, target=npuir) — STUB (empty body).

Computes ``output = avg_pool2d(input, kernel_size, stride, padding,
ceil_mode, count_include_pad)``, matching ``torch.nn.functional.avg_pool2d``.

Status: **STUB** — the prim_func compiles and runs on NPU, but the kernel
body is empty (``pass``).  The output tensor is allocated with the correct
shape/dtype but its values are undefined, so correctness tests will fail.
Benchmark / JIT-compile / roofline flows execute normally.

To implement:
  1. Replace the ``pass`` in the ``T.Kernel`` block with the real
     pooling body (tiled window reduce + divide).
  2. Everything else (factory, wrapper, Op dispatch) stays as-is.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from kernels.kernel_base import Kernel

__all__ = ["AvgPool2dKernel"]


@functools.lru_cache(maxsize=32)
def _make_avg_pool2d_kernel(N, C, H, W, kH, kW,
                            stride_h, stride_w, pad_h, pad_w,
                            ceil_mode, count_include_pad,
                            dtype, output_dtype=None):
    """Build AvgPool2d kernel (STUB — empty body).

    Args:
        N: Batch size.
        C: Channels.
        H, W: Input spatial dims.
        kH, kW: Kernel spatial dims.
        stride_h, stride_w: Stride.
        pad_h, pad_w: Padding.
        ceil_mode: Use ceil-mode output size computation.
        count_include_pad: Include padding in the divisor.
        dtype: Input dtype string (float16 / bfloat16 / float32).
        output_dtype: Output dtype string; defaults to ``dtype``.

    ``block_size`` is passed at call time via the ``block_size`` argument
    to the returned ``kernel`` callable.
    """
    out_dtype = output_dtype or dtype

    def _out_dim(inp, k, pad, stride):
        if ceil_mode:
            return (inp + 2 * pad - k + stride - 1) // stride + 1
        return (inp + 2 * pad - k) // stride + 1

    out_H = _out_dim(H, kH, pad_h, stride_h)
    out_W = _out_dim(W, kW, pad_w, stride_w)

    @tilelang.jit(out_idx=[1], target="npuir")
    def kernel(block_size):
        @T.prim_func
        def main(
            input: T.Tensor((N, C, H, W), dtype),
            output: T.Tensor((N, C, out_H, out_W), out_dtype),
        ):
            with T.Kernel(T.ceildiv(N * C * out_H * out_W, block_size),
                          is_npu=True) as (cid, _):
                # TODO: tiled avg-pool — for each (n, c, oh, ow) in the tile:
                #   acc = 0; count = 0
                #   for kh in range(kH):
                #     for kw in range(kW):
                #       ih = oh*stride_h - pad_h + kh
                #       iw = ow*stride_w - pad_w + kw
                #       if 0 <= ih < H and 0 <= iw < W:
                #         acc += input[n, c, ih, iw]; count += 1
                #   divisor = count if count_include_pad else count
                #   output[n, c, oh, ow] = acc / divisor
                pass

        return main

    return kernel


class AvgPool2dKernel(Kernel):
    """AvgPool2d forward kernel wrapper (NPU STUB — empty body).

    Implements ``output = avg_pool2d(input, kernel_size, stride, padding)``
    matching ``torch.nn.functional.avg_pool2d``.

    Input:  input  [N, C, H, W]
    Output: output [N, C, out_H, out_W]

    Supported dtypes: float16, bfloat16, float32.

    NPU tiling: each core processes ``block_size`` output elements.  There
    is no GPU-style ``threads`` / ``npt`` split because the NPU has no
    SIMT thread model.

    Args:
        n: Batch size.
        c: Channels.
        h, w: Input spatial dims.
        kH, kW: Kernel spatial dims.
        stride_h, stride_w: Stride.
        pad_h, pad_w: Padding.
        ceil_mode: Use ceil-mode output size computation.
        count_include_pad: Include padding in the divisor.
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
                 h: int,
                 w: int,
                 kH: int,
                 kW: int,
                 stride_h: int,
                 stride_w: int,
                 pad_h: int,
                 pad_w: int,
                 ceil_mode: bool = False,
                 count_include_pad: bool = True,
                 dtype: torch.dtype = torch.float16,
                 config: Optional[dict] = None,
                 tune: bool = False):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"AvgPool2dKernel only supports dtypes [{supported}], got {dtype}"
            )
        self.n = n
        self.c = c
        self.h = h
        self.w = w
        self.kH = kH
        self.kW = kW
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.dtype = dtype
        self.dtype_str = self.dtype_to_str(dtype)

        def _out_dim(inp, k, pad, stride):
            if ceil_mode:
                return (inp + 2 * pad - k + stride - 1) // stride + 1
            return (inp + 2 * pad - k) // stride + 1

        self.out_h = _out_dim(h, kH, pad_h, stride_h)
        self.out_w = _out_dim(w, kW, pad_w, stride_w)

        self.kernel = _make_avg_pool2d_kernel(
            n, c, h, w, kH, kW, stride_h, stride_w, pad_h, pad_w,
            ceil_mode, count_include_pad, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"block_size": 256}

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        prim_func = self.kernel(block_size=self.config["block_size"])
        return prim_func(input)

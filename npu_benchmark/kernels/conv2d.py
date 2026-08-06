"""Conv2d forward kernel (NPU, target=npuir) — STUB (empty body).

Computes ``output = conv2d(input, weight, stride, padding, dilation,
groups)`` (bias=None), matching ``torch.nn.functional.conv2d``.

Status: **STUB** — the prim_func compiles and runs on NPU, but the kernel
body is empty (``pass``).  The output tensor is allocated with the correct
shape/dtype but its values are undefined, so correctness tests will fail.
Benchmark / JIT-compile / roofline flows execute normally.

To implement:
  1. Replace the ``pass`` in the ``T.Kernel`` block with the real
     convolution body (im2col + GEMM, or direct tiled convolution).
  2. Everything else (factory, wrapper, Op dispatch) stays as-is.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from kernels.kernel_base import Kernel

__all__ = ["Conv2dKernel"]


@functools.lru_cache(maxsize=32)
def _make_conv2d_kernel(N, C_in, H, W, C_out, C_in_g,
                        kH, kW, stride_h, stride_w,
                        pad_h, pad_w, dilation_h, dilation_w,
                        groups, dtype, output_dtype=None):
    """Build Conv2d kernel (STUB — empty body).

    Args:
        N: Batch size.
        C_in: Input channels.
        H, W: Input spatial dims.
        C_out: Output channels.
        C_in_g: Input channels per group (``C_in // groups``).
        kH, kW: Kernel spatial dims.
        stride_h, stride_w: Stride.
        pad_h, pad_w: Padding.
        dilation_h, dilation_w: Dilation.
        groups: Group count.
        dtype: Input dtype string (float16 / bfloat16 / float32).
        output_dtype: Output dtype string; defaults to ``dtype``.

    ``block_size`` is passed at call time via the ``block_size`` argument
    to the returned ``kernel`` callable.
    """
    out_dtype = output_dtype or dtype
    out_H = (H + 2 * pad_h - dilation_h * (kH - 1) - 1) // stride_h + 1
    out_W = (W + 2 * pad_w - dilation_w * (kW - 1) - 1) // stride_w + 1

    @tilelang.jit(out_idx=[2], target="npuir")
    def kernel(block_size):
        @T.prim_func
        def main(
            input: T.Tensor((N, C_in, H, W), dtype),
            weight: T.Tensor((C_out, C_in_g, kH, kW), dtype),
            output: T.Tensor((N, C_out, out_H, out_W), out_dtype),
        ):
            with T.Kernel(T.ceildiv(N * C_out * out_H * out_W, block_size),
                          is_npu=True) as (cid, _):
                # TODO: tiled convolution — im2col + GEMM or direct tiled conv.
                # For each (n, c_out, oh, ow) in the tile:
                #   acc = 0
                #   for g, ci, kh, kw:
                #     ih = oh * stride_h - pad_h + kh * dilation_h
                #     iw = ow * stride_w - pad_w + kw * dilation_w
                #     if 0 <= ih < H and 0 <= iw < W:
                #       acc += input[n, g*C_in_g+ci, ih, iw] * weight[...]
                #   output[n, c_out, oh, ow] = acc
                pass

        return main

    return kernel


class Conv2dKernel(Kernel):
    """Conv2d forward kernel wrapper (NPU STUB — empty body).

    Implements ``output = conv2d(input, weight)`` (bias=None) matching
    ``torch.nn.functional.conv2d``.

    Input:  input  [N, C_in, H, W]
            weight [C_out, C_in_g, kH, kW]
    Output: output [N, C_out, out_H, out_W]

    Supported dtypes: float16, bfloat16, float32.

    NPU tiling: each core processes ``block_size`` output elements.  There
    is no GPU-style ``threads`` / ``npt`` split because the NPU has no
    SIMT thread model.

    Args:
        n: Batch size.
        c_in: Input channels.
        h, w: Input spatial dims.
        c_out: Output channels.
        c_in_g: Input channels per group (``c_in // groups``).
        kernel_h, kernel_w: Kernel spatial dims.
        stride_h, stride_w: Stride.
        pad_h, pad_w: Padding.
        dilation_h, dilation_w: Dilation.
        groups: Group count.
        dtype: Data type.
        has_bias: Whether a bias term is expected (always False here).
        config: Optional kernel configuration dict.
        tune: Whether to autotune (default False).

    Status: **STUB** — kernel body is empty; output values are undefined.
    """

    supported_archs: Optional[list[int]] = None
    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(self,
                 n: int,
                 c_in: int,
                 h: int,
                 w: int,
                 c_out: int,
                 c_in_g: int,
                 kernel_h: int,
                 kernel_w: int,
                 stride_h: int,
                 stride_w: int,
                 pad_h: int,
                 pad_w: int,
                 dilation_h: int,
                 dilation_w: int,
                 groups: int = 1,
                 dtype: torch.dtype = torch.float16,
                 has_bias: bool = False,
                 config: Optional[dict] = None,
                 tune: bool = False):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"Conv2dKernel only supports dtypes [{supported}], got {dtype}"
            )
        self.n = n
        self.c_in = c_in
        self.h = h
        self.w = w
        self.c_out = c_out
        self.c_in_g = c_in_g
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.groups = groups
        self.dtype = dtype
        self.has_bias = has_bias
        self.dtype_str = self.dtype_to_str(dtype)

        self.out_h = (h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
        self.out_w = (w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

        self.kernel = _make_conv2d_kernel(
            n, c_in, h, w, c_out, c_in_g,
            kernel_h, kernel_w, stride_h, stride_w,
            pad_h, pad_w, dilation_h, dilation_w,
            groups, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"block_size": 256}

    def forward(self, input: torch.Tensor,
                weight: torch.Tensor) -> torch.Tensor:
        prim_func = self.kernel(block_size=self.config["block_size"])
        return prim_func(input, weight)

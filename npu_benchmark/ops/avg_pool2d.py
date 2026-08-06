from typing import Dict, Optional, Tuple

import torch

from kernels.kernel_base import Kernel
from kernels.avg_pool2d import AvgPool2dKernel
from perf.formulas import avg_pool2d_fwd_roofline

from .op_base import Op

__all__ = ["AvgPool2dFwdOp"]


def _pair(value) -> Tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"expected int or 2-element tuple/list, got {value!r}")


def _avg_pool2d_out_dim(input_size, kernel_size, padding, stride, ceil_mode):
    if ceil_mode:
        return (input_size + 2 * padding - kernel_size + stride - 1) // stride + 1
    return (input_size + 2 * padding - kernel_size) // stride + 1


class AvgPool2dFwdOp(Op):
    """AvgPool2d forward operator: output = avg_pool2d(input, kernel_size, ...).

    Conforms to ``torch.nn.functional.avg_poold2d``.  Dispatches the
    TileLang ``AvgPool2dKernel`` (NPU — currently a placeholder stub).

    Input:  input  [N, C, H, W]
    Output: output [N, C, out_H, out_W]

    Args:
        kernel_size: Window size (int or 2-tuple).
        stride: Stride (int or 2-tuple, defaults to ``kernel_size``).
        padding: Padding (int or 2-tuple, default 0).
        ceil_mode: Use ceil-mode output size (default False).
        count_include_pad: Include padding in the divisor (default True).
        divisor_override: Optional fixed divisor (default None).
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """

    _SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 ceil_mode: bool = False,
                 count_include_pad: bool = True,
                 divisor_override: Optional[int] = None,
                 *,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride) if stride is not None else self.kernel_size
        self.padding = _pair(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.tune = tune

        self.dtype = None
        self.n = None
        self.c = None
        self.h = None
        self.w = None
        self.out_h = None
        self.out_w = None

        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"avg_pool2d_kernel": AvgPool2dKernel}

    def _get_kernel(self, n, c, h, w, kH, kW, dtype, device_index) -> Kernel:
        key = (n, c, h, w, kH, kW, self.stride, self.padding,
               self.ceil_mode, self.count_include_pad, dtype, device_index,
               self.tune)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["avg_pool2d_kernel"](
                n=n, c=c, h=h, w=w, kH=kH, kW=kW,
                stride_h=self.stride[0], stride_w=self.stride[1],
                pad_h=self.padding[0], pad_w=self.padding[1],
                ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
                dtype=dtype, tune=self.tune)
        return self._kernel_cache[key]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        from utils import is_available

        if not is_available():
            raise ValueError("AvgPool2dFwdOp expects an accelerator device")
        if input.ndim != 4:
            raise ValueError(
                f"AvgPool2d expects 4D input [N, C, H, W], got {input.ndim}D")
        if input.dtype not in self._SUPPORTED_DTYPES:
            names = ", ".join(str(dt) for dt in self._SUPPORTED_DTYPES)
            raise ValueError(
                f"AvgPool2dFwdOp does not support dtype {input.dtype}. "
                f"Supported: [{names}]"
            )

        n, c, h, w = input.shape
        kH, kW = self.kernel_size

        out_h = _avg_pool2d_out_dim(h, kH, self.padding[0],
                                    self.stride[0], self.ceil_mode)
        out_w = _avg_pool2d_out_dim(w, kW, self.padding[1],
                                    self.stride[1], self.ceil_mode)

        self.dtype = input.dtype
        self.n = n
        self.c = c
        self.h = h
        self.w = w
        self.out_h = out_h
        self.out_w = out_w

        self.kernel = self._get_kernel(
            n, c, h, w, kH, kW, input.dtype, input.device.index)
        return self.kernel(input)

    def eval_roofline(self) -> tuple[int, int]:
        return avg_pool2d_fwd_roofline(self)

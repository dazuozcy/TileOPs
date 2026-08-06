from typing import Dict, Optional, Tuple

import torch

from kernels.kernel_base import Kernel
from kernels.conv2d import Conv2dKernel
from perf.formulas import conv2d_fwd_roofline

from .op_base import Op

__all__ = ["Conv2dFwdOp"]


def _pair(value) -> Tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"expected int or 2-element tuple/list, got {value!r}")


def _conv_out_dim(input_size, kernel_size, stride, padding, dilation):
    return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


class Conv2dFwdOp(Op):
    """Conv2d forward operator: output = conv2d(input, weight).

    Conforms to ``torch.nn.functional.conv2d`` (bias=None).  Dispatches
    the TileLang ``Conv2dKernel`` (NPU — currently a placeholder stub).

    Input:  input  [N, C_in, H, W]
            weight [C_out, C_in_g, kH, kW]
    Output: output [N, C_out, out_H, out_W]

    Args:
        stride: Stride (int or 2-tuple, default 1).
        padding: Padding (int or 2-tuple, default 0).
        dilation: Dilation (int or 2-tuple, default 1).
        groups: Group count (default 1).
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """

    _SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(self,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups: int = 1,
                 *,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        super().__init__()
        if not isinstance(groups, int) or groups <= 0:
            raise ValueError("groups must be a positive int")
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.tune = tune

        self.dtype = None
        self.n = None
        self.c_in = None
        self.h = None
        self.w = None
        self.c_out = None
        self.c_in_g = None
        self.kernel_size = None
        self.out_h = None
        self.out_w = None

        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"conv2d_kernel": Conv2dKernel}

    def _get_kernel(self, n, c_in, h, w, c_out, c_in_g,
                    kernel_h, kernel_w, dtype, device_index) -> Kernel:
        key = (n, c_in, h, w, c_out, c_in_g,
               kernel_h, kernel_w, self.stride, self.padding,
               self.dilation, self.groups, dtype, device_index, self.tune)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["conv2d_kernel"](
                n=n, c_in=c_in, h=h, w=w, c_out=c_out, c_in_g=c_in_g,
                kernel_h=kernel_h, kernel_w=kernel_w,
                stride_h=self.stride[0], stride_w=self.stride[1],
                pad_h=self.padding[0], pad_w=self.padding[1],
                dilation_h=self.dilation[0], dilation_w=self.dilation[1],
                groups=self.groups, dtype=dtype, tune=self.tune)
        return self._kernel_cache[key]

    def forward(self, input: torch.Tensor,
                weight: torch.Tensor) -> torch.Tensor:
        from utils import is_available

        if not is_available():
            raise ValueError("Conv2dFwdOp expects an accelerator device")
        if input.ndim != 4:
            raise ValueError(
                f"Conv2d expects 4D input [N, C_in, H, W], got {input.ndim}D")
        if weight.ndim != 4:
            raise ValueError(
                f"Conv2d expects 4D weight [C_out, C_in_g, kH, kW], "
                f"got {weight.ndim}D")
        if input.dtype not in self._SUPPORTED_DTYPES:
            names = ", ".join(str(dt) for dt in self._SUPPORTED_DTYPES)
            raise ValueError(
                f"Conv2dFwdOp does not support dtype {input.dtype}. "
                f"Supported: [{names}]"
            )
        if weight.dtype != input.dtype:
            raise ValueError(
                f"weight.dtype must match input.dtype {input.dtype}, "
                f"got {weight.dtype}")

        n, c_in, h, w = input.shape
        c_out, c_in_g, kernel_h, kernel_w = weight.shape

        if c_in % self.groups != 0:
            raise ValueError("c_in must be divisible by groups")
        if c_out % self.groups != 0:
            raise ValueError("c_out must be divisible by groups")
        if c_in_g != c_in // self.groups:
            raise ValueError(
                f"expected weight.shape[1]={c_in // self.groups}, got {c_in_g}")

        out_h = _conv_out_dim(h, kernel_h, self.stride[0],
                              self.padding[0], self.dilation[0])
        out_w = _conv_out_dim(w, kernel_w, self.stride[1],
                              self.padding[1], self.dilation[1])

        self.dtype = input.dtype
        self.n = n
        self.c_in = c_in
        self.h = h
        self.w = w
        self.c_out = c_out
        self.c_in_g = c_in_g
        self.kernel_size = (kernel_h, kernel_w)
        self.out_h = out_h
        self.out_w = out_w

        self.kernel = self._get_kernel(
            n, c_in, h, w, c_out, c_in_g,
            kernel_h, kernel_w, input.dtype, input.device.index)
        return self.kernel(input, weight)

    def eval_roofline(self) -> tuple[int, int]:
        return conv2d_fwd_roofline(self)

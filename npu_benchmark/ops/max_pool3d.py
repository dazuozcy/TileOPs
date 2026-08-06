from typing import Dict, Optional, Tuple

import torch

from kernels.kernel_base import Kernel
from kernels.max_pool3d import MaxPool3dKernel
from perf.formulas import max_pool3d_fwd_roofline

from .op_base import Op

__all__ = ["MaxPool3dFwdOp"]


def _triple(value) -> Tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    if isinstance(value, (tuple, list)) and len(value) == 3:
        return (int(value[0]), int(value[1]), int(value[2]))
    raise ValueError(f"expected int or 3-element tuple/list, got {value!r}")


def _max_pool3d_out_dim(input_size, kernel_size, padding, stride, dilation,
                        ceil_mode):
    if ceil_mode:
        return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1
                + stride - 1) // stride + 1
    return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


class MaxPool3dFwdOp(Op):
    """MaxPool3d forward operator: output = max_pool3d(input, kernel_size, ...).

    Conforms to ``torch.nn.functional.max_pool3d``.  Dispatches the
    TileLang ``MaxPool3dKernel`` (NPU — currently a placeholder stub).

    Input:  input  [N, C, D, H, W]
    Output: output [N, C, out_D, out_H, out_W]

    Args:
        kernel_size: Window size (int or 3-tuple).
        stride: Stride (int or 3-tuple, defaults to ``kernel_size``).
        padding: Padding (int or 3-tuple, default 0).
        dilation: Dilation (int or 3-tuple, default 1).
        ceil_mode: Use ceil-mode output size (default False).
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """

    _SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 dilation=1,
                 ceil_mode: bool = False,
                 *,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        super().__init__()
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride) if stride is not None else self.kernel_size
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.ceil_mode = ceil_mode
        self.tune = tune

        self.dtype = None
        self.n = None
        self.c = None
        self.d = None
        self.h = None
        self.w = None
        self.out_d = None
        self.out_h = None
        self.out_w = None

        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"max_pool3d_kernel": MaxPool3dKernel}

    def _get_kernel(self, n, c, d, h, w, kD, kH, kW,
                    dtype, device_index) -> Kernel:
        key = (n, c, d, h, w, kD, kH, kW, self.stride, self.padding,
               self.dilation, self.ceil_mode, dtype, device_index, self.tune)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["max_pool3d_kernel"](
                n=n, c=c, d=d, h=h, w=w, kD=kD, kH=kH, kW=kW,
                stride_d=self.stride[0], stride_h=self.stride[1],
                stride_w=self.stride[2],
                pad_d=self.padding[0], pad_h=self.padding[1],
                pad_w=self.padding[2],
                dilation_d=self.dilation[0], dilation_h=self.dilation[1],
                dilation_w=self.dilation[2],
                ceil_mode=self.ceil_mode, dtype=dtype, tune=self.tune)
        return self._kernel_cache[key]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        from utils import is_available

        if not is_available():
            raise ValueError("MaxPool3dFwdOp expects an accelerator device")
        if input.ndim != 5:
            raise ValueError(
                f"MaxPool3d expects 5D input [N, C, D, H, W], got {input.ndim}D")
        if input.dtype not in self._SUPPORTED_DTYPES:
            names = ", ".join(str(dt) for dt in self._SUPPORTED_DTYPES)
            raise ValueError(
                f"MaxPool3dFwdOp does not support dtype {input.dtype}. "
                f"Supported: [{names}]"
            )

        n, c, d, h, w = input.shape
        kD, kH, kW = self.kernel_size

        out_d = _max_pool3d_out_dim(d, kD, self.padding[0],
                                    self.stride[0], self.dilation[0],
                                    self.ceil_mode)
        out_h = _max_pool3d_out_dim(h, kH, self.padding[1],
                                    self.stride[1], self.dilation[1],
                                    self.ceil_mode)
        out_w = _max_pool3d_out_dim(w, kW, self.padding[2],
                                    self.stride[2], self.dilation[2],
                                    self.ceil_mode)

        self.dtype = input.dtype
        self.n = n
        self.c = c
        self.d = d
        self.h = h
        self.w = w
        self.out_d = out_d
        self.out_h = out_h
        self.out_w = out_w

        self.kernel = self._get_kernel(
            n, c, d, h, w, kD, kH, kW, input.dtype, input.device.index)
        return self.kernel(input)

    def eval_roofline(self) -> tuple[int, int]:
        return max_pool3d_fwd_roofline(self)

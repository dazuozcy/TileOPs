from typing import Dict, Optional

import torch

from kernels.kernel_base import Kernel
from kernels.argmax import ArgmaxKernel
from perf.formulas import argmax_fwd_roofline

from .op_base import Op

__all__ = ["ArgmaxFwdOp"]


class ArgmaxFwdOp(Op):
    """Argmax operator: y = argmax(x, dim, keepdim).

    Conforms to ``torch.argmax``. The Op layer moves the reduction
    dimension to the last axis, reshapes the input to a 2-D ``(M, N)``
    tensor (where ``N`` is the reduction-dim size and ``M`` is the
    product of all other dims), and dispatches the ``ArgmaxKernel``.
    The output is an int64 index tensor.

    Input:  x [...]
    Output: output [...]  (int64 indices; reduced shape depends on ``keepdim``)

    Backed by the TileLang ``ArgmaxKernel`` (NPU — currently a
    placeholder stub; see ``kernels/argmax.py``).

    Args:
        dim: Reduction dimension (default -1).
        keepdim: Retain reduced dimension as size 1 (default False).
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """

    _SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(self,
                 dim: int = -1,
                 keepdim: bool = False,
                 *,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.tune = tune

        self.dtype = None
        self.out_dtype = torch.int64
        self.M = None
        self.N = None
        self.out_shape = None

        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"argmax_fwd": ArgmaxKernel}

    def _get_kernel(self, M, N, dtype, device_index) -> Kernel:
        key = (M, N, dtype, device_index, self.tune)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["argmax_fwd"](
                M, N, dtype, tune=self.tune)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from utils import is_available

        if not is_available():
            raise ValueError("ArgmaxFwdOp expects an accelerator device")
        if x.dtype not in self._SUPPORTED_DTYPES:
            names = ", ".join(str(dt) for dt in self._SUPPORTED_DTYPES)
            raise ValueError(
                f"ArgmaxFwdOp does not support dtype {x.dtype}. "
                f"Supported: [{names}]"
            )
        if x.ndim == 0:
            raise ValueError("Input tensor must be at least 1D")

        orig_shape = tuple(x.shape)
        dim = self.dim
        if dim < 0:
            dim += x.ndim
        if dim < 0 or dim >= x.ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[{-x.ndim}, {x.ndim - 1}], but got {self.dim})"
            )

        N = x.shape[dim]
        M = 1
        for i, s in enumerate(x.shape):
            if i != dim:
                M *= s

        # Move reduction dim to last and reshape to (M, N)
        if dim != x.ndim - 1:
            x = x.movedim(dim, -1)
        x = x.contiguous().reshape(M, N)

        self.dtype = x.dtype
        self.M = M
        self.N = N

        # Compute output shape
        if self.keepdim:
            self.out_shape = tuple(
                1 if i == dim else orig_shape[i] for i in range(len(orig_shape)))
        else:
            self.out_shape = tuple(
                orig_shape[i] for i in range(len(orig_shape)) if i != dim)

        self.kernel = self._get_kernel(
            M, N, x.dtype, x.device.index)
        y = self.kernel(x)  # (M,) int64

        # Reshape output
        if len(self.out_shape) == 0:
            return y.squeeze()
        return y.view(*self.out_shape)

    def eval_roofline(self) -> tuple[int, int]:
        return argmax_fwd_roofline(self)

from typing import Dict, Optional

import torch

from kernels.kernel_base import Kernel
from kernels.lerp_tensor import LerpTensorKernel
from kernels.lerp_tensor_torch import LerpTensorTorchKernel
from perf.formulas import lerp_tensor_roofline

from .op_base import Op

__all__ = ["LerpTensorOp"]


class LerpTensorOp(Op):
    """Tensor-weight lerp op: out = input + weight * (end - input).

    Conforms to the Tensor-weight overload of ``torch.lerp`` —
    ``torch.lerp(input, end, weight: Tensor)`` where ``weight`` is a
    Tensor that broadcasts together with ``input`` and ``end`` to the
    output shape. The Op layer expands the three inputs to the broadcast
    shape and dispatches the flat ``LerpTensorKernel`` on
    ``N_total = product(broadcast_shape)`` elements.

    Input:  input  [...]
            end    [...]
            weight [...]
    Output: output [...]  (broadcast shape)

    By default uses ``LerpTensorTorchKernel`` (PyTorch-based, NPU-ready).
    Pass ``kernel_map={"lerp_tensor_kernel": LerpTensorKernel}`` to use
    the TileLang kernel (requires a backend that supports register
    fragments, e.g. CUDA or a future TileLang NPU backend).
    """

    _SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(self,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        super().__init__()
        self.tune = tune

        self.dtype = None
        self.N_total = None
        self.out_shape = None

        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"lerp_tensor_kernel": LerpTensorTorchKernel}

    def _get_kernel(self, N_total, dtype, device_index) -> Kernel:
        key = (N_total, dtype, device_index, self.tune)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["lerp_tensor_kernel"](
                N_total, dtype, tune=self.tune)
        return self._kernel_cache[key]

    @staticmethod
    def _expand_flat(t: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """Expand ``t`` to ``target_shape`` and return a contiguous flat view."""
        if tuple(t.shape) != tuple(target_shape):
            t = t.expand(target_shape)
        return t.contiguous().view(-1)

    def forward(self, input: torch.Tensor, end: torch.Tensor,
                weight: torch.Tensor) -> torch.Tensor:
        from utils import is_available

        if not is_available():
            raise ValueError("LerpTensorOp expects an accelerator device")
        if not (input.dtype == end.dtype == weight.dtype):
            raise ValueError("input, end, weight must have the same dtype")
        if input.dtype not in self._SUPPORTED_DTYPES:
            names = ", ".join(str(dt) for dt in self._SUPPORTED_DTYPES)
            raise ValueError(
                f"LerpTensorOp does not support dtype {input.dtype}. "
                f"Supported: [{names}]"
            )

        out_shape = torch.broadcast_shapes(input.shape, end.shape, weight.shape)
        N_total = 1
        for d in out_shape:
            N_total *= d

        a_flat = self._expand_flat(input, out_shape)
        b_flat = self._expand_flat(end, out_shape)
        w_flat = self._expand_flat(weight, out_shape)

        self.dtype = input.dtype
        self.N_total = N_total
        self.out_shape = out_shape

        self.kernel = self._get_kernel(
            N_total, input.dtype, input.device.index)
        out_flat = self.kernel(a_flat, b_flat, w_flat)
        return out_flat.view(out_shape)

    def eval_roofline(self) -> tuple[int, int]:
        return lerp_tensor_roofline(self)

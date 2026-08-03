from typing import Dict, Optional

import torch

from kernels.kernel_base import Kernel
from kernels.mish import MishKernel
from kernels.mish_torch import MishTorchKernel
from perf.formulas import mish_fwd_roofline

from .op_base import Op

__all__ = ["MishFwdOp"]


class MishFwdOp(Op):
    """Element-wise Mish activation op: y = x * tanh(softplus(x)).

    Conforms to ``torch.nn.functional.mish``. The Op layer flattens the
    input to a 1-D tensor of ``N_total = product(input.shape)`` elements
    and dispatches the flat ``MishKernel``.

    Input:  input [...]
    Output: output [...]  (same shape as input)

    By default uses ``MishTorchKernel`` (PyTorch-based, NPU-ready).
    Pass ``kernel_map={"mish_kernel": MishKernel}`` to use the TileLang
    kernel (requires a backend that supports register fragments, e.g.
    CUDA or a future TileLang NPU backend).
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
        return {"mish_kernel": MishTorchKernel}

    def _get_kernel(self, N_total, dtype, device_index) -> Kernel:
        key = (N_total, dtype, device_index, self.tune)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["mish_kernel"](
                N_total, dtype, tune=self.tune)
        return self._kernel_cache[key]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        from utils import is_available

        if not is_available():
            raise ValueError("MishFwdOp expects an accelerator device")
        if input.dtype not in self._SUPPORTED_DTYPES:
            names = ", ".join(str(dt) for dt in self._SUPPORTED_DTYPES)
            raise ValueError(
                f"MishFwdOp does not support dtype {input.dtype}. "
                f"Supported: [{names}]"
            )

        out_shape = tuple(input.shape)
        N_total = input.numel()

        x_flat = input.contiguous().view(-1)

        self.dtype = input.dtype
        self.N_total = N_total
        self.out_shape = out_shape

        self.kernel = self._get_kernel(
            N_total, input.dtype, input.device.index)
        out_flat = self.kernel(x_flat)
        return out_flat.view(out_shape)

    def eval_roofline(self) -> tuple[int, int]:
        return mish_fwd_roofline(self)

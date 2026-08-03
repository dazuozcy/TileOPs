"""PyTorch-based tensor-weight lerp kernel — NPU-ready fallback.

The TileLang lerp kernel (kernels/lerp_tensor.py) uses register-fragment
primitives (alloc_fragment, T.copy) that may not yet be supported by the
TileLang Ascend backend.  This module provides a PyTorch-based
implementation that runs on any device (NPU, CUDA, CPU), so the
benchmark framework is fully functional end-to-end.

When the TileLang NPU backend matures to support the needed primitives,
switch the Op's default kernel to ``LerpTensorKernel`` (the TileLang
implementation) — no other changes needed.
"""

from __future__ import annotations

from typing import Optional

import torch

from kernels.kernel_base import Kernel

__all__ = ["LerpTensorTorchKernel"]


class LerpTensorTorchKernel(Kernel):
    """Tensor-weight lerp using torch.lerp — runs on NPU/CUDA/CPU.

    This kernel computes ``out = a + w * (b - a)`` on flat 1-D tensors,
    matching the TileLang kernel's I/O contract:

      Input:  a [N], b [N], w [N]  (pre-broadcast & flattened by the Op)
      Output: out [N]
    """

    def __init__(self,
                 N: int,
                 dtype: torch.dtype,
                 config: Optional[dict] = None,
                 tune: bool = False):
        super().__init__()
        self.N = N
        self.dtype = dtype
        self.config = self.default_config

    @property
    def default_config(self) -> dict:
        return {"impl": "torch.lerp"}

    def forward(self, a: torch.Tensor, b: torch.Tensor,
                w: torch.Tensor) -> torch.Tensor:
        return torch.lerp(a, b, w)

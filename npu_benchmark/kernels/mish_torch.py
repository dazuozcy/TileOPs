"""PyTorch-based Mish activation kernel — NPU-ready fallback.

The TileLang Mish kernel (kernels/mish.py) uses register-fragment
primitives (alloc_fragment, T.copy) that may not yet be supported by the
TileLang Ascend backend.  This module provides a PyTorch-based
implementation that runs on any device (NPU, CUDA, CPU), so the
benchmark framework is fully functional end-to-end.

When the TileLang NPU backend matures to support the needed primitives,
switch the Op's default kernel to ``MishKernel`` (the TileLang
implementation) — no other changes needed.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from kernels.kernel_base import Kernel

__all__ = ["MishTorchKernel"]


class MishTorchKernel(Kernel):
    """Mish activation using torch — runs on NPU/CUDA/CPU.

    Computes ``y = x * tanh(softplus(x))`` on a flat 1-D tensor,
    matching the TileLang kernel's I/O contract:

      Input:  x [N]  (flattened by the Op)
      Output: y [N]
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
        return {"impl": "torch.mish"}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))

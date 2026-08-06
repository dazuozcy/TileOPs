from typing import Dict, Optional, Union
from math import inf

import torch

from kernels.kernel_base import Kernel
from kernels.vector_norm import VectorNormKernel
from perf.formulas import (
    l1_norm_fwd_roofline,
    l2_norm_fwd_roofline,
    inf_norm_fwd_roofline,
)

from .op_base import Op

__all__ = ["L1NormFwdOp", "L2NormFwdOp", "InfNormFwdOp"]


class _VectorNormBaseOp(Op):
    """Base class for vector-norm ops (L1, L2, inf).

    Conforms to ``torch.linalg.vector_norm(x, ord, dim, keepdim)``. The
    Op layer moves the reduction dimension to the last axis, reshapes
    the input to a 2-D ``(M, N)`` tensor (where ``N`` is the
    reduction-dim size and ``M`` is the product of all other dims), and
    dispatches the ``VectorNormKernel``.

    Subclasses set ``_op_kind`` (``"l1"`` / ``"l2"`` / ``"inf"``) and
    ``_required_ord`` and override ``eval_roofline()``.

    Args:
        dim: Reduction dimension (default -1).
        keepdim: Retain reduced dimension as size 1 (default False).
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """

    _op_kind: str
    _required_ord: Union[int, float]
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
        self.M = None
        self.N = None
        self.out_shape = None

        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"vector_norm": VectorNormKernel}

    def _get_kernel(self, M, N, dtype, device_index) -> Kernel:
        key = (M, N, dtype, device_index, self._op_kind, self.tune)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["vector_norm"](
                M, N, self._op_kind, dtype, tune=self.tune)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from utils import is_available

        if not is_available():
            raise ValueError(f"{type(self).__name__} expects an accelerator device")
        if x.dtype not in self._SUPPORTED_DTYPES:
            names = ", ".join(str(dt) for dt in self._SUPPORTED_DTYPES)
            raise ValueError(
                f"{type(self).__name__} does not support dtype {x.dtype}. "
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

        self.kernel = self._get_kernel(M, N, x.dtype, x.device.index)
        y = self.kernel(x)  # (M,)

        # Reshape output
        if len(self.out_shape) == 0:
            return y.squeeze()
        return y.view(*self.out_shape)


class L1NormFwdOp(_VectorNormBaseOp):
    """L1 norm operator: y = vector_norm(x, ord=1, dim, keepdim).

    Computes ``sum(abs(x))`` along ``dim``.  Backed by the TileLang
    ``VectorNormKernel`` (NPU — currently a placeholder stub).
    """

    _op_kind = "l1"
    _required_ord = 1

    def __init__(self,
                 dim: int = -1,
                 keepdim: bool = False,
                 *,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        super().__init__(dim=dim, keepdim=keepdim,
                         kernel_map=kernel_map, tune=tune)

    def eval_roofline(self) -> tuple[int, int]:
        return l1_norm_fwd_roofline(self)


class L2NormFwdOp(_VectorNormBaseOp):
    """L2 norm operator: y = vector_norm(x, ord=2, dim, keepdim).

    Computes ``sqrt(sum(x^2))`` along ``dim``.  Backed by the TileLang
    ``VectorNormKernel`` (NPU — currently a placeholder stub).
    """

    _op_kind = "l2"
    _required_ord = 2

    def __init__(self,
                 dim: int = -1,
                 keepdim: bool = False,
                 *,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        super().__init__(dim=dim, keepdim=keepdim,
                         kernel_map=kernel_map, tune=tune)

    def eval_roofline(self) -> tuple[int, int]:
        return l2_norm_fwd_roofline(self)


class InfNormFwdOp(_VectorNormBaseOp):
    """Infinity norm operator: y = vector_norm(x, ord=inf, dim, keepdim).

    Computes ``max(abs(x))`` along ``dim``.  Backed by the TileLang
    ``VectorNormKernel`` (NPU — currently a placeholder stub).
    """

    _op_kind = "inf"
    _required_ord = inf

    def __init__(self,
                 dim: int = -1,
                 keepdim: bool = False,
                 *,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        super().__init__(dim=dim, keepdim=keepdim,
                         kernel_map=kernel_map, tune=tune)

    def eval_roofline(self) -> tuple[int, int]:
        return inf_norm_fwd_roofline(self)

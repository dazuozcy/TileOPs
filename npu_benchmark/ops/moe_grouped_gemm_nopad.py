from typing import Dict, Optional

import torch

from kernels.kernel_base import Kernel
from kernels.moe_grouped_gemm_nopad import MoeGroupedGemmNopadKernel
from perf.formulas import moe_grouped_gemm_nopad_fwd_roofline

from .op_base import Op

__all__ = ["MoeGroupedGemmNopadFwdOp"]


class MoeGroupedGemmNopadFwdOp(Op):
    """MoE grouped GEMM forward operator (no-pad variant).

    NT grouped GEMM for MoE without block_m-aligned padding.  Accepts
    tight ``A[numel, K]`` inputs (no padding between experts), producing
    tight ``C[numel, N]`` outputs.

    Input:  a             [numel, K]
            b             [num_experts, N, K]
            true_sizes    [num_experts]     (int32)
            true_offsets  [num_experts]     (int32)
    Output: c             [numel, N]

    Backed by the TileLang ``MoeGroupedGemmNopadKernel`` (NPU -- currently
    a placeholder stub; see ``kernels/moe_grouped_gemm_nopad.py``).

    Args:
        numel: Total (token, expert) pairs = tight row count.
        num_experts: Total number of experts E.
        n: Output feature dimension N.
        k: Input feature dimension K.
        dtype: Activation and weight dtype (default bfloat16).
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """

    _SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)

    def __init__(self,
                 numel: int,
                 num_experts: int,
                 n: int,
                 k: int,
                 dtype: torch.dtype = torch.bfloat16,
                 *,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        super().__init__()
        if dtype not in self._SUPPORTED_DTYPES:
            names = ", ".join(str(dt) for dt in self._SUPPORTED_DTYPES)
            raise ValueError(
                f"MoeGroupedGemmNopadFwdOp does not support dtype {dtype}. "
                f"Supported: [{names}]")
        self.numel = numel
        self.num_experts = num_experts
        self.n = n
        self.k = k
        self.dtype = dtype
        self.tune = tune

        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"moe_grouped_gemm_kernel": MoeGroupedGemmNopadKernel}

    def _get_kernel(self) -> Kernel:
        key = (self.numel, self.num_experts, self.n, self.k,
               self.dtype, self.tune)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["moe_grouped_gemm_kernel"](
                self.numel, self.num_experts, self.n, self.k,
                dtype=self.dtype, tune=self.tune)
        return self._kernel_cache[key]

    def forward(self, a: torch.Tensor, b: torch.Tensor,
                true_sizes: torch.Tensor,
                true_offsets: torch.Tensor) -> torch.Tensor:
        from utils import is_available

        if not is_available():
            raise ValueError("MoeGroupedGemmNopadFwdOp expects an accelerator device")
        if a.ndim != 2:
            raise ValueError(
                f"a must have shape [numel, K], got {a.ndim}D")
        if a.shape != (self.numel, self.k):
            raise ValueError(
                f"a shape {tuple(a.shape)} does not match expected "
                f"({self.numel}, {self.k})")
        if b.shape != (self.num_experts, self.n, self.k):
            raise ValueError(
                f"b shape {tuple(b.shape)} does not match expected "
                f"({self.num_experts}, {self.n}, {self.k})")
        if a.dtype != self.dtype:
            raise ValueError(
                f"a.dtype must be {self.dtype}, got {a.dtype}")
        if b.dtype != self.dtype:
            raise ValueError(
                f"b.dtype must be {self.dtype}, got {b.dtype}")
        if true_sizes.shape != (self.num_experts,):
            raise ValueError(
                f"true_sizes shape {tuple(true_sizes.shape)} does not match "
                f"({self.num_experts},)")
        if true_offsets.shape != (self.num_experts,):
            raise ValueError(
                f"true_offsets shape {tuple(true_offsets.shape)} does not match "
                f"({self.num_experts},)")

        self.kernel = self._get_kernel()
        a = a.contiguous()
        b = b.contiguous()
        true_sizes = true_sizes.contiguous()
        true_offsets = true_offsets.contiguous()
        return self.kernel(a, b, true_sizes, true_offsets)

    def eval_roofline(self) -> tuple[int, int]:
        return moe_grouped_gemm_nopad_fwd_roofline(self)

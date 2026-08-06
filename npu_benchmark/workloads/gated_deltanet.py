import torch

from workloads.workload_base import WorkloadBase
from utils import device_str


class GatedDeltaNetFwdWorkload(WorkloadBase):

    def __init__(self, q_shape, k_shape, v_shape, g_shape, beta_shape,
                 chunk_size: int, dtype: torch.dtype):
        self.q_shape = tuple(q_shape)
        self.k_shape = tuple(k_shape)
        self.v_shape = tuple(v_shape)
        self.g_shape = tuple(g_shape)
        self.beta_shape = tuple(beta_shape)
        self.chunk_size = chunk_size
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                  torch.Tensor, torch.Tensor]:
        q = torch.randn(*self.q_shape, dtype=self.dtype, device=device_str())
        k = torch.randn(*self.k_shape, dtype=self.dtype, device=device_str())
        v = torch.randn(*self.v_shape, dtype=self.dtype, device=device_str())
        g = torch.randn(*self.g_shape, dtype=self.dtype, device=device_str())
        beta = torch.rand(*self.beta_shape, dtype=self.dtype, device=device_str())
        beta = beta.clamp(0, 1)
        return q, k, v, g, beta

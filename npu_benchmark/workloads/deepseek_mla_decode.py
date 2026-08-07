import torch

from workloads.workload_base import WorkloadBase
from utils import device_str


class MlaDecodeWorkload(WorkloadBase):

    def __init__(self, q_shape, q_pe_shape, kv_shape, k_pe_shape,
                 pe_dim: int, dtype: torch.dtype):
        self.q_shape = tuple(q_shape)
        self.q_pe_shape = tuple(q_pe_shape)
        self.kv_shape = tuple(kv_shape)
        self.k_pe_shape = tuple(k_pe_shape)
        self.pe_dim = pe_dim
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor,
                                  torch.Tensor, torch.Tensor]:
        q = torch.randn(*self.q_shape, dtype=self.dtype, device=device_str())
        q_pe = torch.randn(*self.q_pe_shape, dtype=self.dtype,
                           device=device_str())
        k = torch.randn(*self.kv_shape, dtype=self.dtype, device=device_str())
        k_pe = torch.randn(*self.k_pe_shape, dtype=self.dtype,
                           device=device_str())
        return q, q_pe, k, k_pe

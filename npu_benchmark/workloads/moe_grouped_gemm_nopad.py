import torch

from workloads.workload_base import WorkloadBase
from utils import device_str


class MoeGroupedGemmNopadFwdWorkload(WorkloadBase):

    def __init__(self, numel: int, num_experts: int, n: int, k: int,
                 dtype: torch.dtype):
        self.numel = numel
        self.num_experts = num_experts
        self.n = n
        self.k = k
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor,
                                  torch.Tensor, torch.Tensor]:
        a = torch.randn(self.numel, self.k, dtype=self.dtype, device=device_str())
        b = torch.randn(self.num_experts, self.n, self.k,
                        dtype=self.dtype, device=device_str()) * 0.02

        # Evenly distribute numel across experts.
        base = self.numel // self.num_experts
        remainder = self.numel % self.num_experts
        true_sizes = torch.full((self.num_experts,), base,
                                dtype=torch.int32, device=device_str())
        true_sizes[:remainder] += 1
        true_offsets = torch.cumsum(
            torch.cat([torch.zeros(1, dtype=torch.int32, device=device_str()),
                       true_sizes[:-1]]), dim=0).to(torch.int32)

        return a, b, true_sizes, true_offsets

import torch

from workloads.workload_base import WorkloadBase
from utils import device_str


class LerpTensorWorkload(WorkloadBase):

    def __init__(self, input_shape, dtype: torch.dtype):
        self.input_shape = tuple(input_shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input = torch.randn(*self.input_shape, dtype=self.dtype, device=device_str())
        end = torch.randn(*self.input_shape, dtype=self.dtype, device=device_str())
        weight = torch.randn(*self.input_shape, dtype=self.dtype, device=device_str())
        return input, end, weight

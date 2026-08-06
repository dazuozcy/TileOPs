import torch

from workloads.workload_base import WorkloadBase
from utils import device_str


class Conv2dWorkload(WorkloadBase):

    def __init__(self,
                 input_shape,
                 C_out: int,
                 kH: int,
                 kW: int,
                 dtype: torch.dtype,
                 stride=(1, 1),
                 padding=(0, 0),
                 dilation=(1, 1),
                 groups: int = 1):
        self.input_shape = tuple(input_shape)
        self.C_out = C_out
        self.kH = kH
        self.kW = kW
        self.dtype = dtype
        self.stride = tuple(stride)
        self.padding = tuple(padding)
        self.dilation = tuple(dilation)
        self.groups = groups

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        N, C_in, H, W = self.input_shape
        C_in_g = C_in // self.groups
        input = torch.randn(N, C_in, H, W, dtype=self.dtype, device=device_str())
        weight = torch.randn(self.C_out, C_in_g, self.kH, self.kW,
                             dtype=self.dtype, device=device_str())
        return input, weight

import torch

from workloads.workload_base import WorkloadBase
from utils import device_str


class AvgPool2dWorkload(WorkloadBase):

    def __init__(self,
                 input_shape,
                 kernel_size,
                 dtype: torch.dtype,
                 stride=None,
                 padding=0,
                 ceil_mode: bool = False,
                 count_include_pad: bool = True):
        self.input_shape = tuple(input_shape)
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def gen_inputs(self) -> tuple[torch.Tensor,]:
        N, C, H, W = self.input_shape
        input = torch.randn(N, C, H, W, dtype=self.dtype, device=device_str())
        return (input,)

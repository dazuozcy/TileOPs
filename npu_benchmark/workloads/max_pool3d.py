import torch

from workloads.workload_base import WorkloadBase
from utils import device_str


class MaxPool3dWorkload(WorkloadBase):

    def __init__(self,
                 input_shape,
                 kernel_size,
                 dtype: torch.dtype,
                 stride=None,
                 padding=0,
                 dilation=1,
                 ceil_mode: bool = False):
        self.input_shape = tuple(input_shape)
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def gen_inputs(self) -> tuple[torch.Tensor,]:
        N, C, D, H, W = self.input_shape
        input = torch.randn(N, C, D, H, W, dtype=self.dtype, device=device_str())
        return (input,)

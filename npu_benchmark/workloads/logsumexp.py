import torch

from workloads.workload_base import WorkloadBase
from utils import device_str


class LogSumExpWorkload(WorkloadBase):

    def __init__(self, x_shape, dtype: torch.dtype,
                 dim: int = -1, keepdim: bool = False):
        self.x_shape = tuple(x_shape)
        self.dtype = dtype
        self.dim = dim
        self.keepdim = keepdim

    def gen_inputs(self) -> tuple[torch.Tensor,]:
        x = torch.randn(*self.x_shape, dtype=self.dtype, device=device_str())
        return (x,)

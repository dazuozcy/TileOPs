import torch

from workloads.workload_base import WorkloadBase
from utils import device_str


class SSDChunkScanFwdWorkload(WorkloadBase):

    def __init__(self, x_shape, cb_shape, dA_cumsum_shape, C_shape,
                 prev_states_shape, dt_shape, dtype: torch.dtype):
        self.x_shape = tuple(x_shape)
        self.cb_shape = tuple(cb_shape)
        self.dA_cumsum_shape = tuple(dA_cumsum_shape)
        self.C_shape = tuple(C_shape)
        self.prev_states_shape = tuple(prev_states_shape)
        self.dt_shape = tuple(dt_shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                  torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(*self.x_shape, dtype=self.dtype, device=device_str())
        cb = torch.randn(*self.cb_shape, dtype=self.dtype, device=device_str())
        dA_cumsum = torch.randn(*self.dA_cumsum_shape, dtype=torch.float32,
                                device=device_str()).cumsum(dim=-1)
        C = torch.randn(*self.C_shape, dtype=self.dtype, device=device_str())
        prev_states = torch.randn(*self.prev_states_shape, dtype=torch.float32,
                                  device=device_str())
        dt = torch.rand(*self.dt_shape, dtype=self.dtype, device=device_str())
        dt = dt.clamp(0.01, 1.0)
        return x, cb, dA_cumsum, C, prev_states, dt

"""Benchmark for the SSD chunk scan forward op on Ascend NPU.

The NPU ``SSDChunkScanFwdKernel`` is currently a stub (empty body).  The
kernel compiles and runs (producing undefined output) so the benchmark
timing / roofline flow executes normally.
"""

from __future__ import annotations

import pytest
import torch

from benchmarks.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workload_field_params,
)
from manifest import load_workloads
from ops import SSDChunkScanFwdOp
from workloads.ssd_chunk_scan import SSDChunkScanFwdWorkload

_TUNE = True
_OP_NAME = "SSDChunkScanFwdOp"
_PARAMS = workload_field_params(
    load_workloads(_OP_NAME),
    ("x_shape", "cb_shape", "dA_cumsum_shape", "C_shape",
     "prev_states_shape", "dt_shape", "dtype"),
)


@pytest.mark.parametrize(
    "x_shape, cb_shape, dA_cumsum_shape, C_shape, "
    "prev_states_shape, dt_shape, dtype",
    _PARAMS,
)
def test_ssd_chunk_scan_bench(x_shape, cb_shape, dA_cumsum_shape, C_shape,
                              prev_states_shape, dt_shape,
                              dtype: torch.dtype) -> None:
    test = SSDChunkScanFwdWorkload(x_shape, cb_shape, dA_cumsum_shape,
                                   C_shape, prev_states_shape, dt_shape,
                                   dtype)
    inputs = test.gen_inputs()

    op = SSDChunkScanFwdOp(tune=_TUNE)
    bm = ManifestBenchmark(_OP_NAME, op, test)

    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="kernel")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])

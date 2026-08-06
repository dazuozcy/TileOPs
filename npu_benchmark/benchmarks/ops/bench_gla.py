"""Benchmark for the GLA forward op on Ascend NPU.

The NPU ``GLAFwdKernel`` is currently a stub (empty body).  The kernel
compiles and runs (producing undefined output) so the benchmark timing
/ roofline flow executes normally.
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
from ops import GLAFwdOp
from workloads.gla import GLAFwdWorkload

_TUNE = True
_OP_NAME = "GLAFwdOp"
_PARAMS = workload_field_params(
    load_workloads(_OP_NAME),
    ("q_shape", "k_shape", "v_shape", "g_shape",
     "chunk_size", "scale", "dtype"),
)


@pytest.mark.parametrize(
    "q_shape, k_shape, v_shape, g_shape, chunk_size, scale, dtype",
    _PARAMS,
)
def test_gla_bench(q_shape, k_shape, v_shape, g_shape,
                   chunk_size, scale, dtype: torch.dtype) -> None:
    test = GLAFwdWorkload(q_shape, k_shape, v_shape, g_shape,
                          chunk_size, dtype, scale=scale)
    inputs = test.gen_inputs()

    op = GLAFwdOp(chunk_size=chunk_size, scale=scale,
                  output_final_state=True, tune=_TUNE)
    bm = ManifestBenchmark(_OP_NAME, op, test)

    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="kernel")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])

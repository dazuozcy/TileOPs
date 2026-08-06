"""Benchmark for the MoE grouped GEMM forward op on Ascend NPU.

The NPU ``MoeGroupedGemmNopadKernel`` is currently a stub (empty body).
The kernel compiles and runs (producing undefined output) so the benchmark
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
from ops import MoeGroupedGemmNopadFwdOp
from workloads.moe_grouped_gemm_nopad import MoeGroupedGemmNopadFwdWorkload

_TUNE = True
_OP_NAME = "MoeGroupedGemmNopadFwdOp"
_PARAMS = workload_field_params(
    load_workloads(_OP_NAME),
    ("numel", "num_experts", "n", "k", "dtype"),
)


@pytest.mark.parametrize(
    "numel, num_experts, n, k, dtype",
    _PARAMS,
)
def test_moe_grouped_gemm_nopad_bench(numel, num_experts, n, k,
                                      dtype: torch.dtype) -> None:
    test = MoeGroupedGemmNopadFwdWorkload(numel, num_experts, n, k, dtype)
    inputs = test.gen_inputs()

    op = MoeGroupedGemmNopadFwdOp(numel, num_experts, n, k, dtype=dtype,
                                 tune=_TUNE)
    bm = ManifestBenchmark(_OP_NAME, op, test)

    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="kernel")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])

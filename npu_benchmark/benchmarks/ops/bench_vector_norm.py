"""Benchmarks for vector-norm ops (L1, L2, inf) on Ascend NPU.

The NPU ``VectorNormKernel`` is currently a stub (empty body).  The
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
from ops import L1NormFwdOp, L2NormFwdOp, InfNormFwdOp
from workloads.vector_norm import VectorNormWorkload

_TUNE = True
_PARAMS = workload_field_params(
    load_workloads("L1NormFwdOp"),
    ("x_shape", "dtype", "dim", "keepdim"),
)


@pytest.mark.parametrize("x_shape, dtype, dim, keepdim", _PARAMS)
def test_l1_norm_bench(x_shape, dtype: torch.dtype,
                       dim: int, keepdim: bool) -> None:
    test = VectorNormWorkload(x_shape, dtype, dim=dim, keepdim=keepdim)
    inputs = test.gen_inputs()
    op = L1NormFwdOp(dim=dim, keepdim=keepdim, tune=_TUNE)
    bm = ManifestBenchmark("L1NormFwdOp", op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="kernel")


@pytest.mark.parametrize("x_shape, dtype, dim, keepdim", _PARAMS)
def test_l2_norm_bench(x_shape, dtype: torch.dtype,
                       dim: int, keepdim: bool) -> None:
    test = VectorNormWorkload(x_shape, dtype, dim=dim, keepdim=keepdim)
    inputs = test.gen_inputs()
    op = L2NormFwdOp(dim=dim, keepdim=keepdim, tune=_TUNE)
    bm = ManifestBenchmark("L2NormFwdOp", op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="kernel")


@pytest.mark.parametrize("x_shape, dtype, dim, keepdim", _PARAMS)
def test_inf_norm_bench(x_shape, dtype: torch.dtype,
                        dim: int, keepdim: bool) -> None:
    test = VectorNormWorkload(x_shape, dtype, dim=dim, keepdim=keepdim)
    inputs = test.gen_inputs()
    op = InfNormFwdOp(dim=dim, keepdim=keepdim, tune=_TUNE)
    bm = ManifestBenchmark("InfNormFwdOp", op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="kernel")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])

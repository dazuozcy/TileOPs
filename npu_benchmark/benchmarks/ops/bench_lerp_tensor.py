"""Benchmark for the tensor-weight lerp op on Ascend NPU.

Workload shapes and dtypes come from the manifest; roofline FLOP and byte
counts come from the op's ``eval_roofline()`` via
:class:`ManifestBenchmark`.
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
from ops import LerpTensorOp
from workloads.lerp_tensor import LerpTensorWorkload

_TUNE = True
_LERP_TENSOR_OP = "LerpTensorOp"
_LERP_TENSOR_PARAMS = workload_field_params(
    load_workloads(_LERP_TENSOR_OP),
    ("input_shape", "dtype"),
)


@pytest.mark.parametrize(
    "input_shape, dtype",
    _LERP_TENSOR_PARAMS,
)
def test_lerp_tensor_bench(input_shape, dtype: torch.dtype) -> None:
    test = LerpTensorWorkload(input_shape, dtype)
    inputs = test.gen_inputs()

    op = LerpTensorOp(tune=_TUNE)
    bm = ManifestBenchmark(_LERP_TENSOR_OP, op, test)

    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="kernel")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])

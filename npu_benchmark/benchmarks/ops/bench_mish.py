"""Benchmark for the Mish activation op on Ascend NPU.

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
from ops import MishFwdOp
from workloads.mish import MishWorkload

_TUNE = True
_MISH_OP = "MishFwdOp"
_MISH_PARAMS = workload_field_params(
    load_workloads(_MISH_OP),
    ("input_shape", "dtype"),
)


@pytest.mark.parametrize(
    "input_shape, dtype",
    _MISH_PARAMS,
)
def test_mish_bench(input_shape, dtype: torch.dtype) -> None:
    test = MishWorkload(input_shape, dtype)
    inputs = test.gen_inputs()

    op = MishFwdOp(tune=_TUNE)
    bm = ManifestBenchmark(_MISH_OP, op, test)

    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="kernel")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])

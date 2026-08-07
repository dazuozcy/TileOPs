"""Benchmark for the MLA decode op on Ascend NPU.

The NPU ``MLADecodeKernel`` is currently a stub (empty body).  The kernel
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
from ops import MultiHeadLatentAttentionDecodeWithKVCacheFwdOp
from workloads.deepseek_mla_decode import MlaDecodeWorkload

_TUNE = True
_OP_NAME = "MultiHeadLatentAttentionDecodeWithKVCacheFwdOp"
_PARAMS = workload_field_params(
    load_workloads(_OP_NAME),
    ("q_shape", "q_pe_shape", "kv_shape", "k_pe_shape", "pe_dim", "dtype"),
)


@pytest.mark.parametrize(
    "q_shape, q_pe_shape, kv_shape, k_pe_shape, pe_dim, dtype",
    _PARAMS,
)
def test_mla_decode_bench(q_shape, q_pe_shape, kv_shape, k_pe_shape,
                          pe_dim, dtype: torch.dtype) -> None:
    test = MlaDecodeWorkload(q_shape, q_pe_shape, kv_shape, k_pe_shape,
                             pe_dim, dtype)
    inputs = test.gen_inputs()

    op = MultiHeadLatentAttentionDecodeWithKVCacheFwdOp(pe_dim=pe_dim,
                                                        tune=_TUNE)
    bm = ManifestBenchmark(_OP_NAME, op, test)

    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="kernel")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])

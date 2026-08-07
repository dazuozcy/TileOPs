# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

# NPUIR implementation of topk_selector_fwd_func (Stage 4 optimized v4).
#
# v4 optimizations over v3:
#   1. Combined single-pass processing: eliminates the separate chunk max
#      serial scan and chunk-level early-exit check. Instead, every element
#      is processed in a single pass with an implicit element-level early-exit
#      (score > root comparison is O(1), heap operations only when competitive).
#      This removes O(interval) chunk max scanning overhead.
#   2. Inherits v3's T.copy batch loading and v2's min-heap candidate buffer.
#
# Complexity per row:
#   - Fill phase: O(topk * log topk)
#   - Single-pass scanning: O(interval) comparisons + O(competitive * log topk) heap ops
#   - No separate chunk max scan (saved O(interval) comparisons)
#
# Complexity per row:
#   - Fill phase: O(topk * log topk)
#   - Scanning: O(interval)
#   - Replacements: O(num_replacements * log topk)
#   For random data: num_replacements ~ topk * ln(interval/topk)
#   Total: O(topk * log topk + interval + topk * ln(interval/topk) * log topk)
#
# Migration compliance preserved:
#   - topk_selector_fwd_func (@tilelang.jit) name/signature unchanged
#   - _topk_selector_kernel_main (@T.prim_func) params unchanged
#   - _topk_selector_kernel factory + @functools.lru_cache preserved

import os

os.environ["TILELANG_ASCEND_MODE"] = "Developer"

import argparse
import functools
import time
from typing import Optional
import itertools

import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

from kernels.kernel_base import Kernel

__all__ = ["TopkSelectorKernel"]

# ---------------------------------------------------------------------------
# convert_to_uint16 / convert_to_uint32 (retained for migration compliance)
# ---------------------------------------------------------------------------

def convert_to_uint16(x):
    hval = T.Cast("float16", x)
    bits_uint = T.reinterpret("uint16", hval)
    bits_uint = T.if_then_else(
        x < 0,
        ~bits_uint & T.Cast("uint16", 0xFFFF),
        bits_uint | T.Cast("uint16", 0x8000),
    )
    return bits_uint >> 8


def convert_to_uint32(x):
    bits_uint = T.reinterpret("uint32", T.Cast("float32", x))
    bits_uint = T.if_then_else(
        x < 0,
        ~bits_uint & T.Cast("uint32", 0xFFFFFFFF),
        bits_uint | T.Cast("uint32", 0x80000000),
    )
    return bits_uint


# ---------------------------------------------------------------------------
# Golden reference (PyTorch CPU implementation)
# ---------------------------------------------------------------------------

def golden_topk_selector(index_score, starts, ends, topk):
    batch, seq_len, seq_len_kv, kv_group = index_score.shape
    index_out = torch.full(
        (batch, seq_len, kv_group, topk), -1, dtype=torch.int32
    )
    for b in range(batch):
        for s in range(seq_len):
            s_start = int(starts[b, s].item())
            s_end = int(ends[b, s].item())
            valid_count = max(0, min(s_end, seq_len_kv) - max(0, s_start))
            for g in range(kv_group):
                if valid_count == 0:
                    continue
                scores = index_score[b, s, :, g].clone().float()
                mask = torch.zeros(seq_len_kv, dtype=torch.bool)
                lo = max(0, s_start)
                hi = min(s_end, seq_len_kv)
                mask[lo:hi] = True
                scores[~mask] = float("-inf")
                k = min(topk, valid_count)
                _, topk_indices = torch.topk(scores, k, dim=0)
                index_out[b, s, g, :k] = topk_indices.to(torch.int32)
    return index_out


def set_compare(output, output_ref):
    ref_np = output_ref.cpu().to(torch.int32).numpy()
    trt_np = output.cpu().to(torch.int32).numpy()
    set_ref = set(ref_np.flatten().tolist()) - {-1}
    set_trt = set(trt_np.flatten().tolist()) - {-1}
    if len(set_ref) == 0 and len(set_trt) == 0:
        return True
    if len(set_ref) == 0:
        return False
    intersection = set_ref & set_trt
    ratio = len(intersection) / len(set_ref)
    if ratio != 1.0:
        print(
            f"  [MISMATCH] set_ref={sorted(set_ref)}, "
            f"set_trt={sorted(set_trt)}, ratio={ratio:.4f}"
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Kernel factory (Stage 4 v2: min-heap running top-k)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=32)
def _topk_selector_kernel(batch, seq_len, seq_len_kv, kv_group, topk, in_dtype, out_dtype):
    """Factory that returns a JIT-compiled topk_selector_fwd_func.

    Stage 4 v4: min-heap with T.copy loading and combined single-pass.
    Eliminates separate chunk max scan; processes elements directly.
    """

    @tilelang.jit(target="npuir", out_idx=[1])
    def topk_selector_fwd_func(
        RADIX=1 << 8, BLOCK_SIZE=1024, SMEM_INPUT_SIZE=4096, block_m=32
    ):
        batch_sym = T.symbolic("batch")
        seq_len_kv_sym = T.symbolic("seq_len_kv")

        # HEAPIFY_MAX_STEPS: safe upper bound for heapify up/down.
        # log2(topk) + 1 steps suffice; 32 covers topk up to 2^31.
        HEAPIFY_MAX_STEPS = 32

        @T.prim_func
        def _topk_selector_kernel_main(
            index_score: T.Tensor[
                (batch_sym, seq_len, seq_len_kv_sym, kv_group), in_dtype
            ],
            index: T.Tensor[
                (batch_sym, seq_len, kv_group, topk), out_dtype
            ],
            starts: T.Tensor[(batch_sym, seq_len), out_dtype],
            ends: T.Tensor[(batch_sym, seq_len), out_dtype],
        ):
            with T.Kernel(batch_sym * seq_len * kv_group, is_npu=True) as (cid, _):
                g = cid % kv_group
                seq_row = (cid // kv_group) % seq_len
                bx = cid // (seq_len * kv_group)

                # --- Load interval bounds ---
                l_start = starts[bx, seq_row]
                l_end = ends[bx, seq_row]

                # --- UB buffer allocation ---
                s_output_idx = T.alloc_shared([topk], "int32")
                score_ub = T.alloc_shared([BLOCK_SIZE], "float32")
                # Min-heap: root (index 0) is the smallest of the top-k
                cand_val = T.alloc_shared([topk], "float32")
                cand_idx = T.alloc_shared([topk], "int32")

                # --- Scalar variables ---
                l_interval = T.alloc_shared([1], "int32")
                l_chunk_start = T.alloc_shared([1], "int32")
                l_chunk_size = T.alloc_shared([1], "int32")
                l_next_fill = T.alloc_shared([1], "int32")
                l_pos = T.alloc_shared([1], "int32")

                # Heap operation temporaries
                l_heap_pos = T.alloc_shared([1], "int32")
                l_parent = T.alloc_shared([1], "int32")
                l_left = T.alloc_shared([1], "int32")
                l_right = T.alloc_shared([1], "int32")
                l_smallest = T.alloc_shared([1], "int32")
                l_tmp_val = T.alloc_shared([1], "float32")
                l_tmp_idx = T.alloc_shared([1], "int32")
                l_done = T.alloc_shared([1], "int32")

                # --- Init ---
                for k in T.serial(topk):
                    s_output_idx[k] = -1
                    cand_val[k] = -T.infinity("float32")
                    cand_idx[k] = -1

                l_next_fill[0] = 0
                l_interval[0] = T.max(l_end - l_start, 0)

                # === Chunked processing with min-heap (combined single-pass) ===
                # No separate chunk max scan; elements processed directly with
                # implicit element-level early-exit (O(1) root comparison).
                for chunk in T.serial(T.ceildiv(l_interval[0], BLOCK_SIZE)):
                    l_chunk_start[0] = l_start + chunk * BLOCK_SIZE
                    l_chunk_size[0] = T.min(BLOCK_SIZE, l_end - l_chunk_start[0])

                    # --- Load chunk scores via T.copy (batch GM->UB) ---
                    T.copy(
                        index_score[bx, seq_row, l_chunk_start[0]:l_chunk_start[0] + l_chunk_size[0], g],
                        score_ub[0:l_chunk_size[0]],
                    )

                    # --- Single-pass element processing ---
                    for i in T.serial(l_chunk_size[0]):
                        if l_next_fill[0] < topk:
                            # === Fill phase: insert at next position ===
                            l_pos[0] = l_next_fill[0]
                            cand_val[l_pos[0]] = score_ub[i]
                            cand_idx[l_pos[0]] = l_chunk_start[0] + i
                            l_next_fill[0] = l_pos[0] + 1

                            # --- Heapify up ---
                            l_heap_pos[0] = l_pos[0]
                            l_done[0] = 0
                            for step in T.serial(HEAPIFY_MAX_STEPS):
                                if l_done[0] == 0:
                                    if l_heap_pos[0] > 0:
                                        l_parent[0] = (l_heap_pos[0] - 1) // 2
                                        if cand_val[l_heap_pos[0]] < cand_val[l_parent[0]]:
                                            # Swap
                                            l_tmp_val[0] = cand_val[l_heap_pos[0]]
                                            l_tmp_idx[0] = cand_idx[l_heap_pos[0]]
                                            cand_val[l_heap_pos[0]] = cand_val[l_parent[0]]
                                            cand_idx[l_heap_pos[0]] = cand_idx[l_parent[0]]
                                            cand_val[l_parent[0]] = l_tmp_val[0]
                                            cand_idx[l_parent[0]] = l_tmp_idx[0]
                                            l_heap_pos[0] = l_parent[0]
                                        else:
                                            l_done[0] = 1
                                    else:
                                        l_done[0] = 1
                        else:
                            # === Replace phase: compare with root ===
                            if score_ub[i] > cand_val[0]:
                                # Replace root
                                cand_val[0] = score_ub[i]
                                cand_idx[0] = l_chunk_start[0] + i

                                # --- Heapify down ---
                                l_heap_pos[0] = 0
                                l_done[0] = 0
                                for step in T.serial(HEAPIFY_MAX_STEPS):
                                    if l_done[0] == 0:
                                        l_left[0] = 2 * l_heap_pos[0] + 1
                                        l_right[0] = 2 * l_heap_pos[0] + 2
                                        l_smallest[0] = l_heap_pos[0]

                                        if l_left[0] < topk:
                                            if cand_val[l_left[0]] < cand_val[l_smallest[0]]:
                                                l_smallest[0] = l_left[0]

                                        if l_right[0] < topk:
                                            if cand_val[l_right[0]] < cand_val[l_smallest[0]]:
                                                l_smallest[0] = l_right[0]

                                        if l_smallest[0] != l_heap_pos[0]:
                                            # Swap
                                            l_tmp_val[0] = cand_val[l_heap_pos[0]]
                                            l_tmp_idx[0] = cand_idx[l_heap_pos[0]]
                                            cand_val[l_heap_pos[0]] = cand_val[l_smallest[0]]
                                            cand_idx[l_heap_pos[0]] = cand_idx[l_smallest[0]]
                                            cand_val[l_smallest[0]] = l_tmp_val[0]
                                            cand_idx[l_smallest[0]] = l_tmp_idx[0]
                                            l_heap_pos[0] = l_smallest[0]
                                        else:
                                            l_done[0] = 1

                # --- Write output: copy heap indices to output buffer ---
                for k in T.serial(topk):
                    s_output_idx[k] = cand_idx[k]
                T.copy(s_output_idx, index[bx, seq_row, g, 0], size=[topk])

        return _topk_selector_kernel_main

    return topk_selector_fwd_func


class TopkSelectorKernel(Kernel):
    """Kernel wrapper for the radix top-k selector.

    The supported_archs list is empty (no enforcement) because NPU arch
    detection differs from CUDA SM versioning.  Add arch checks here when
    needed.
    """

    supported_archs: Optional[list[int]] = None
    prof_name = "_topk_selector_kernel_main"

    def __init__(self,
                 batch: int,
                 seq_len: int,
                 seq_len_kv: int,
                 kv_group: int,
                 topk: int,
                 in_dtype: torch.dtype,
                 out_dtype: torch.dtype,
                 config: Optional[dict] = None,
                 tune: bool = False):
        super().__init__()
        self.batch = batch
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.kv_group = kv_group
        self.topk = topk
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.in_dtype_str = self.dtype_to_str(in_dtype)
        self.out_dtype_str = self.dtype_to_str(out_dtype)

        self.kernel = _topk_selector_kernel(
            self.batch, self.seq_len, self.seq_len_kv,
            self.kv_group, self.topk, self.in_dtype_str, self.out_dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "RADIX": 1 << 8,
            "BLOCK_SIZE": 1024,
            "SMEM_INPUT_SIZE": 4096,
            "block_m": 32,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        RADIX = [1 << 8]
        BLOCK_SIZE = [1024]
        SMEM_INPUT_SIZE = [4096]
        block_m = [32]
        _configs = list(itertools.product(RADIX, BLOCK_SIZE, SMEM_INPUT_SIZE, block_m))
        return [{'RADIX': c[0], 'BLOCK_SIZE': c[1],
                 'SMEM_INPUT_SIZE': c[2], 'block_m': c[3]} for c in _configs]

    def forward(self, index_score: torch.Tensor, starts: torch.Tensor,
                ends: torch.Tensor) -> torch.Tensor:
        prim_func = self.kernel(
            RADIX=self.config["RADIX"],
            BLOCK_SIZE=self.config["BLOCK_SIZE"],
            SMEM_INPUT_SIZE=self.config["SMEM_INPUT_SIZE"],
            block_m=self.config["block_m"],
        )
        return prim_func(index_score, starts, ends)


# ---------------------------------------------------------------------------
# Test harness (identical to v1 / baseline)
# ---------------------------------------------------------------------------

def run_case(
    batch, seq_len, seq_len_kv, kv_group, topk,
    starts_vals=None, ends_vals=None, tag="L0",
    score_fn=None, expect_warn=False,
):
    device = "npu"

    if score_fn is not None:
        score_cpu = score_fn(batch, seq_len, seq_len_kv, kv_group)
    else:
        score_cpu = torch.randn(
            batch, seq_len, seq_len_kv, kv_group, dtype=torch.float32
        )

    if starts_vals is not None:
        starts_cpu = torch.tensor(starts_vals, dtype=torch.int32)
    else:
        starts_cpu = torch.zeros(batch, seq_len, dtype=torch.int32)
    if ends_vals is not None:
        ends_cpu = torch.tensor(ends_vals, dtype=torch.int32)
    else:
        ends_cpu = torch.full(
            (batch, seq_len), seq_len_kv, dtype=torch.int32
        )

    score_npu = score_cpu.to(device)
    starts_npu = starts_cpu.to(device)
    ends_npu = ends_cpu.to(device)

    jit_func = _topk_selector_kernel(
        batch, seq_len, seq_len_kv, kv_group, topk, "float32", "int32"
    )
    prim_func = jit_func()

    try:
        output = prim_func(score_npu, starts_npu, ends_npu)
    except Exception as e:
        if expect_warn:
            print(f"  [{tag}] WARN (recorded): kernel exception: {e}")
            return True
        print(f"  [{tag}] FAIL: kernel exception: {e}")
        return False

    golden = golden_topk_selector(score_cpu, starts_cpu, ends_cpu, topk)

    ok = set_compare(output, golden)
    if ok:
        print(f"  [{tag}] PASS: shape=({batch},{seq_len},{seq_len_kv},{kv_group}) topk={topk}")
    else:
        print(f"  [{tag}] FAIL: shape=({batch},{seq_len},{seq_len_kv},{kv_group}) topk={topk}")
        b = 0
        s = 0
        g = 0
        print(f"    output[{b},{s},{g}] = {output[b, s, g].cpu().tolist()}")
        print(f"    golden[{b},{s},{g}] = {golden[b, s, g].tolist()}")
    return ok


def run_L0():
    print("=== L0: Gate tests ===")
    results = []

    results.append(run_case(4, 256, 1024, 1, 32, tag="L0-1"))
    results.append(run_case(4, 256, 1024, 1, 32, tag="L0-2"))
    results.append(run_case(1, 1, 32, 1, 1, tag="L0-3"))
    results.append(run_case(1, 1, 32, 1, 32, tag="L0-4"))
    results.append(run_case(
        2, 4, 128, 1, 16,
        starts_vals=[[10, 20, 0, 5]] * 2,
        ends_vals=[[80, 90, 64, 100]] * 2,
        tag="L0-5",
    ))

    def neg_scores(b, s, skv, g):
        return (torch.randn(b, s, skv, g, dtype=torch.float32) * 2.0 - 1.0)
    results.append(run_case(1, 1, 64, 1, 8, score_fn=neg_scores, tag="L0-6"))

    def tied_scores(b, s, skv, g):
        t = torch.zeros(b, s, skv, g, dtype=torch.float32)
        for i in range(skv):
            t[0, 0, i, 0] = float(i % 8)
        return t
    results.append(run_case(1, 1, 64, 1, 8, score_fn=tied_scores, tag="L0-7"))

    passed = sum(results)
    total = len(results)
    print(f"--- L0: {passed}/{total} passed ---\n")
    return passed == total


def run_L1():
    print("=== L1: Larger shapes ===")
    results = []

    results.append(run_case(1, 4, 4096, 1, 64, tag="L1-1"))
    results.append(run_case(2, 16, 1024, 1, 32, tag="L1-2"))
    results.append(run_case(1, 2, 2048, 1, 128, tag="L1-3"))
    results.append(run_case(
        1, 4, 4096, 1, 32,
        starts_vals=[[100, 200, 300, 400]],
        ends_vals=[[1000, 2000, 3000, 3900]],
        tag="L1-4",
    ))

    passed = sum(results)
    total = len(results)
    print(f"--- L1: {passed}/{total} passed ---\n")
    return passed == total


def run_L2():
    print("=== L2: Edge cases (warn only, non-blocking) ===")

    try:
        run_case(
            1, 2, 64, 1, 8,
            starts_vals=[[10, 20]],
            ends_vals=[[10, 20]],
            tag="L2-1-empty",
            expect_warn=True,
        )
    except Exception as e:
        print(f"  [L2-1] WARN (recorded): {e}")

    try:
        run_case(
            1, 1, 32, 1, 64,
            starts_vals=[[5]],
            ends_vals=[[10]],
            tag="L2-2-topk-gt-interval",
            expect_warn=True,
        )
    except Exception as e:
        print(f"  [L2-2] WARN (recorded): {e}")

    print("--- L2: completed (warnings non-blocking) ---\n")


def run_boundary():
    print("=== Boundary: Special values (warn only, non-blocking) ===")

    def zero_scores(b, s, skv, g):
        return torch.zeros(b, s, skv, g, dtype=torch.float32)
    try:
        run_case(1, 1, 64, 1, 8, score_fn=zero_scores, tag="Boundary-zeros", expect_warn=True)
    except Exception as e:
        print(f"  [Boundary-zeros] WARN: {e}")

    def same_scores(b, s, skv, g):
        return torch.full((b, s, skv, g), 3.14, dtype=torch.float32)
    try:
        run_case(1, 1, 64, 1, 8, score_fn=same_scores, tag="Boundary-same", expect_warn=True)
    except Exception as e:
        print(f"  [Boundary-same] WARN: {e}")

    def extreme_scores(b, s, skv, g):
        return torch.randn(b, s, skv, g, dtype=torch.float32) * 1e10
    try:
        run_case(1, 1, 64, 1, 8, score_fn=extreme_scores, tag="Boundary-extreme", expect_warn=True)
    except Exception as e:
        print(f"  [Boundary-extreme] WARN: {e}")

    try:
        device = "npu"
        kv_group = 1
        topk = 8
        batch, seq_len, seq_len_kv = 1, 1, 64
        score_cpu = torch.zeros(batch, seq_len, seq_len_kv, kv_group, dtype=torch.float32)
        for i in range(seq_len_kv):
            score_cpu[0, 0, i, 0] = float(i % 4)
        starts_cpu = torch.zeros(batch, seq_len, dtype=torch.int32)
        ends_cpu = torch.full((batch, seq_len), seq_len_kv, dtype=torch.int32)
        score_npu = score_cpu.to(device)
        starts_npu = starts_cpu.to(device)
        ends_npu = ends_cpu.to(device)
        jit_func = _topk_selector_kernel(batch, seq_len, seq_len_kv, kv_group, topk, "float32", "int32")
        output = jit_func()(score_npu, starts_npu, ends_npu)
        golden = golden_topk_selector(score_cpu, starts_cpu, ends_cpu, topk)
        out_vals = set()
        for idx in output[0, 0, 0].cpu().tolist():
            if idx >= 0:
                out_vals.add(score_cpu[0, 0, idx, 0].item())
        ref_vals = set()
        for idx in golden[0, 0, 0].tolist():
            if idx >= 0:
                ref_vals.add(score_cpu[0, 0, idx, 0].item())
        if out_vals == ref_vals:
            print(f"  [Boundary-ties-gt-topk] PASS (value-based): out_vals={out_vals}, ref_vals={ref_vals}")
        else:
            print(f"  [Boundary-ties-gt-topk] WARN (value-based): out_vals={out_vals}, ref_vals={ref_vals}")
    except Exception as e:
        print(f"  [Boundary-ties-gt-topk] WARN: {e}")

    print("--- Boundary: completed (warnings non-blocking) ---\n")


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

BENCH_CASES = [
    ("bench-1", 4, 256, 1024, 1, 32, "representative"),
    ("bench-2", 1, 4, 4096, 1, 64, "medium"),
    ("bench-3", 1, 2, 2048, 1, 128, "large-topk"),
    ("bench-4", 1, 32768, 65536, 1, 1024, "manifest-large"),
    ("bench-5", 1, 32768, 65536, 1, 2048, "manifest-large"),
]


def benchmark_kernel(name, batch, seq_len, seq_len_kv, kv_group, topk,
                     warmup=3, repeat=5):
    device = "npu"
    score_npu = torch.randn(
        batch, seq_len, seq_len_kv, kv_group, dtype=torch.float32, device=device
    )
    starts_npu = torch.zeros(batch, seq_len, dtype=torch.int32, device=device)
    ends_npu = torch.full(
        (batch, seq_len), seq_len_kv, dtype=torch.int32, device=device
    )

    jit_func = _topk_selector_kernel(
        batch, seq_len, seq_len_kv, kv_group, topk, "float32", "int32"
    )
    prim_func = jit_func()

    for _ in range(warmup):
        prim_func(score_npu, starts_npu, ends_npu)
    torch.npu.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        prim_func(score_npu, starts_npu, ends_npu)
    torch.npu.synchronize()
    end = time.perf_counter()

    latency_us = (end - start) / repeat * 1e6
    throughput = (batch * seq_len * kv_group) / (latency_us * 1e-6)

    score_cpu = score_npu.cpu()
    starts_cpu = starts_npu.cpu()
    ends_cpu = ends_npu.cpu()
    golden = golden_topk_selector(score_cpu, starts_cpu, ends_cpu, topk)
    output = prim_func(score_npu, starts_npu, ends_npu)
    ok = set_compare(output, golden)

    status = "PASS" if ok else "FAIL"
    print(
        f"  [{name}] {status}: shape=({batch},{seq_len},{seq_len_kv},{kv_group}) "
        f"topk={topk}  latency={latency_us:.1f} us  "
        f"throughput={throughput:.0f} rows/s"
    )
    return latency_us, ok


def run_benchmarks():
    print("=== Benchmarks (bench-1 ~ bench-5) ===")
    results = []
    for name, b, s, skv, g, k, desc in BENCH_CASES:
        try:
            lat, ok = benchmark_kernel(name, b, s, skv, g, k)
            results.append((name, lat, ok, desc))
        except Exception as e:
            print(f"  [{name}] ERROR: {e}")
            results.append((name, -1, False, desc))
    print()
    return results


def run_pytorch_comparison():
    print("=== PyTorch torch.topk comparison (full-tensor, NPU) ===")
    device = "npu"
    for name, b, s, skv, g, k, desc in BENCH_CASES[:3]:
        score_npu = torch.randn(b, s, skv, g, dtype=torch.float32, device=device)
        for _ in range(3):
            torch.topk(score_npu, k, dim=2)
        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(5):
            torch.topk(score_npu, k, dim=2)
        torch.npu.synchronize()
        end = time.perf_counter()
        lat_us = (end - start) / 5 * 1e6
        print(f"  [{name}] torch.topk: latency={lat_us:.1f} us  ({desc})")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all", "bench"])
    args, _ = parser.parse_known_args()

    if args.level == "L0":
        ok = run_L0()
        if ok:
            print("\033[92mAll L0 checks passed!\033[0m")
        else:
            print("\033[91mSome L0 checks FAILED!\033[0m")
            raise SystemExit(1)
    elif args.level == "bench":
        run_benchmarks()
        run_pytorch_comparison()
    else:
        l0_ok = run_L0()
        l1_ok = run_L1()
        run_L2()
        run_boundary()
        run_benchmarks()
        run_pytorch_comparison()
        if l0_ok and l1_ok:
            print("\033[92mAll checks passed (L2/Boundary warnings non-blocking)!\033[0m")
        else:
            print("\033[91mSome L0/L1 checks FAILED!\033[0m")
            raise SystemExit(1)


if __name__ == "__main__":
    main()

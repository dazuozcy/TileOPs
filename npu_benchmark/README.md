# NPU Benchmark

Standalone benchmark framework for TileLang kernels on Ascend NPU.

Extracted from the TileOPs project — does not depend on any TileOPs interfaces.

## Architecture

```
manifest (YAML, authoritative spec)
   │  load_workloads()
   ▼
workloads (gen_inputs → NPU tensors)
   │
   ▼
bench test (parametrize from manifest)
   ├── Op → dispatches Kernel (TileLang JIT)
   └── ManifestBenchmark.profile() → bench_kernel() (event timing)
        │
        ├── op.eval_roofline() → perf.formulas (FLOPs / bytes)
        │
        ▼
   BenchmarkReport.record() → conftest → profile_run.log (markdown)
```

### Layer-by-layer

| Layer | File | Responsibility |
|-------|------|----------------|
| **Manifest** | `manifest/*.yaml` | Op signature, workloads, roofline, source paths |
| **Manifest loader** | `manifest/__init__.py` | Merge YAML files, `load_workloads()` |
| **Workload** | `workloads/*.py` | `gen_inputs()` — create NPU tensors |
| **Kernel** | `kernels/*.py` | TileLang JIT prim_func + config/autotune |
| **Op** | `ops/*.py` | Input validation, kernel dispatch/cache, `eval_roofline()` |
| **Perf** | `perf/formulas.py` | Roofline FLOP/byte formulas |
| **Benchmark** | `benchmarks/benchmark_base.py` | `bench_kernel()`, `ManifestBenchmark`, `BenchmarkReport` |
| **Conftest** | `benchmarks/conftest.py` | Session hooks, JUnit properties, report dump |
| **Device** | `utils/device.py` | NPU/CUDA abstraction |

### Timing protocol

`bench_kernel()` follows a SOL-ExecBench-style protocol:
1. L2 cache flush (configurable buffer) before each iteration.
2. 10 warmup iterations (untimed).
3. 3 trials × 50 repeats, each timed with backend events.
4. Median of trial means reported (robust to outliers).
5. Inputs cloned from a 3-element pool (fresh addresses per iteration).

NPU timing uses `torch.npu.Event(enable_timing=True)` (via `utils.device.timing_event()`).
Set `NPU_BENCHMARK_FORCE_CUDA=1` to use CUDA events for local NVIDIA development.

### msprof op profiling

Event-based timing includes host-side launch overhead and scheduling
latency.  For **kernel-only** latency, switch to `msprof op` mode:

```bash
make bench-msprof
# or:
NPU_BENCH_TIMING=msprof python -m pytest benchmarks/
```

This runs each kernel via:

```
msprof op --kernel-name=<auto> --output=<tmp> --launch-count=50 --warm-up=10 \
    python <generated_script>.py
```

The generated script reconstructs the Op + Workload in a fresh process,
launches the kernel `warm_up + launch_count` times, then exits.  msprof
collects per-launch data and writes `OpBasicInfo_*.csv`; the framework
parses the `Task Duration(us)` column and reports the **median** in
`profile_run.log` (shown as `timing: msprof` in the report table).

**How it works:**
1. The Op is called once in-process to bind shape/dtype for roofline
   calculation.
2. A standalone Python script is generated that reconstructs the Op
   (with `tune=False`, i.e. default config) and Workload.
3. `msprof op` wraps the script as a subprocess.
4. `OpBasicInfo_*.csv` is parsed for `Task Duration(us)`.

**Kernel name auto-detection:** each `Kernel` subclass declares a
`prof_name` class attribute (default `"main"`, matching the
`@T.prim_func def main(...)` convention).  Override via
`NPU_BENCH_MSPROF_KERNEL_NAME`.

**Configuration env vars:**

| Variable | Default | Description |
|----------|---------|-------------|
| `NPU_BENCH_TIMING` | `events` | `msprof` to switch profiling method |
| `NPU_BENCH_MSPROF_LAUNCH_COUNT` | `10` | Timed launches collected |
| `NPU_BENCH_MSPROF_WARM_UP` | `5` | Warm-up launches skipped |
| `NPU_BENCH_MSPROF_KERNEL_NAME` | auto | Override `--kernel-name` |
| `NPU_BENCH_MSPROF_OUTPUT` | temp dir | Fixed output directory |
| `NPU_BENCH_MSPROF_KEEP_OUTPUT` | `1` | `0` to auto-delete temp dir |

## Usage

### Install

```bash
cd npu_benchmark
pip install -r requirements.txt
# For Ascend: also install torch_npu matching your torch/CANN version
```

### Run benchmarks

```bash
make bench           # all workloads (event timing)
make bench-smoke     # first workload per op (fast sanity)
make bench-msprof    # all workloads (msprof op timing, kernel-only latency)
```

### Run correctness tests

```bash
make test
```

### Output

- `profile_run.log` — markdown report with latency, TFLOPS, bandwidth, and
  kernel-vs-torch ratio for each workload.

## Adding a new kernel

1. **Manifest**: add an entry to `manifest/<family>.yaml` (or a new YAML file).
2. **Workload**: create `workloads/<op>.py` implementing `gen_inputs()`.
3. **Kernel**: create `kernels/<op>.py` with a `Kernel` subclass.
4. **Op**: create `ops/<op>.py` with an `Op` subclass (dispatch + `eval_roofline()`).
5. **Roofline**: add the formula to `perf/formulas.py`.
6. **Bench**: create `benchmarks/ops/bench_<op>.py` (parametrize from manifest).
7. **Test** (optional): create `tests/ops/test_<op>.py` for correctness.

See `TopkSelectorOp` for a complete reference.

## Current ops

- **TopkSelectorOp** — radix-based top-k index selection over `[B, S, S_kv, G]`.
- **LerpTensorOp** — tensor-weight linear interpolation `out = input + weight * (end - input)`.
- **MishFwdOp** — element-wise Mish activation `y = x * tanh(softplus(x))`.
- **LogSumExpFwdOp** — row-wise `logsumexp` reduction `y = log(sum(exp(x, dim)))` (NPU kernel placeholder).

### NPU tiling model

NPU kernels use a **block_size** tiling parameter — the number of elements
processed per NPU core (GM → UB copy, vector compute, UB → GM store).
There is no GPU-style `threads` / `npt` split because the NPU has no SIMT
thread model: each core executes vector instructions (`vexp`, `vadd`,
`vmul`, `vsub`, `vdiv`, `vcast`) over a contiguous tile in Unified Buffer.

The `block_size` is a runtime argument to the JIT kernel, chosen by
`default_config` based on dtype and `N` (grown when needed to keep
`ceildiv(N, block_size)` within the NPU coreDim limit of 65535).

### Kernel implementations

| Kernel | File | Backend | Notes |
|--------|------|---------|-------|
| `TopkSelectorKernel` | `kernels/topk_selector.py` | TileLang | Radix top-k — uses CUDA SIMT primitives (`alloc_shared`, `sync_threads`, `atomic_add`) not supported by the TileLang Ascend backend |
| `LerpTensorKernel` | `kernels/lerp_tensor.py` | TileLang (NPU) | NPU-native — uses `alloc_ub` + vector primitives (`vcast`, `vsub`, `vmul`, `vadd`) with `block_size` tiling |
| `MishKernel` | `kernels/mish.py` | TileLang (NPU) | NPU-native — uses `alloc_ub` + vector primitives (`vexp`, `vadd`, `vmul`, `vsub`, `vdiv`, `vcast`) with `block_size` tiling |
| `LogSumExpKernel` | `kernels/logsumexp.py` | TileLang (NPU) | PLACEHOLDER stub — `forward()` raises `NotImplementedError`; NPU reduction primitives not yet available |

All ops are backed by their TileLang kernel. The Op's `kernel_map`
parameter may still override the default kernel class.

### NPU backend limitation

The TileLang Ascend backend supports NPU vector primitives (`alloc_ub`,
`vexp`, `vadd`, `vmul`, `vsub`, `vdiv`, `vcast`, `T.copy` GM↔UB) used by
the elementwise kernels (`MishKernel`, `LerpTensorKernel`).

It does **not** yet support CUDA SIMT primitives:
- `T.alloc_shared()` / `T.alloc_local()` — segfault
- `T.sync_threads()` / `T.block_barrier()` / `T.subblock_barrier()` — segfault
- `T.atomic_add()` on shared memory — unsupported

Kernels that rely on the CUDA SIMT model (shared memory histograms,
thread-level synchronization, intra-block atomics) — such as
`TopkSelectorKernel` — cannot be compiled on NPU until these primitives
are implemented.

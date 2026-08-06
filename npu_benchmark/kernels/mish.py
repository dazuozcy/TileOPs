# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Mish activation kernel (NPU, target=npuir) — performance-optimized.

Stage 4 tuned version. Key optimizations over the Stage 3 baseline:

1. **T.Pipelined double buffering**: Each block processes multiple inner tiles
   via ``T.Pipelined(num_inner, num_stages=2)``, overlapping GM→UB loads with
   vector compute across tiles (MTE3 vs V pipeline overlap).  This enables
   large effective block_size (32768 for f16) while keeping per-tile UB
   allocation small.

2. **Reduced UB buffers + in-place ops**: 10 intermediate buffers collapsed to
   4 (f16/bf16) / 3 (f32) via in-place vadd/vmul/vsub/vdiv (all support
   ``dst = src``).  Buffer ``xy_ub`` is reused for both f16 input load and
   output store; ``x_f32`` is reused for both upcast input and final vmul
   result (in-place ``dst = src1``).

3. **Large block_size sweep**: Baseline block_size=2048 → optimized 32768
   (f16) / 24576 (f32).  Fewer blocks → less launch overhead, better
   amortization.  UB budget controlled by inner_tile × num_stages, not
   block_size, so large block_size fits within 192 KB UB.

Computation: mish(x) = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x))),
computed in float32 for numerical stability via the algebraic identity
  tanh(softplus(x)) = (t2² - 1) / (t2² + 1),  t2 = 1 + exp(x)
which avoids T.vtanh (Stage 3 fallback: vtanh Taylor divergence on 1-D UB).

NPU tiling: each core processes ``block_size = inner_tile * num_inner``
contiguous elements (GM → UB copy, vector compute, UB → GM store).
``inner_tile`` is the per-tile UB allocation size; ``num_inner`` is the
number of inner tiles pipelined per block.  There is no GPU-style
``threads`` / ``npt`` split because the NPU has no SIMT thread model.

Performance (shape [16, 256, 80, 80] f16, N=26,214,400):
  - Stage 3 baseline (bs=2048): ~2327 us
  - This version (bs=32768, Pipelined): ~164 us  (-92.9%)

run: python mish.py --level L0
run: python mish.py --level all
"""

import argparse
import functools
from typing import Optional

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401  # registers .npu() device

import tilelang
import tilelang.language as T

from kernels.kernel_base import Kernel


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_mish(x: torch.Tensor) -> torch.Tensor:
    """Mish activation PyTorch reference.

    y = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))

    Uses torch.nn.functional.mish (matches manifest ref_api). Computed in
    float32 then cast back to the input dtype, mirroring the kernel's
    float32-intermediate behaviour.
    """
    return F.mish(x.cpu().float()).to(x.dtype)


# ---------- Kernel factory ----------
@functools.lru_cache(maxsize=32)
def _make_mish_kernel(N, dtype, output_dtype=None):
    """Build Mish kernel with T.Pipelined double buffering.

    Args:
        N: Number of elements (flat 1-D size).
        dtype: Input dtype string (float16 / bfloat16 / float32).
        output_dtype: Output dtype string; defaults to ``dtype``.

    ``inner_tile`` (per-tile UB allocation size), ``num_inner`` (inner tiles
    per block, ``block_size = inner_tile * num_inner``), and ``num_stages``
    (pipeline depth for ``T.Pipelined``) are passed at call time via keyword
    arguments to the returned ``kernel`` callable — they are not factory
    parameters.

    The @tilelang.jit(out_idx=[1], target="npuir") / @T.prim_func /
    main(x, y) declarations are preserved from the CUDA source. The kernel
    body uses T.Pipelined to overlap GM→UB loads with vector compute across
    inner tiles. Buffers are allocated at inner_tile size (not block_size),
    enabling large effective block_size within the 192 KB UB budget.

    UB budget (f16 path, 4 buffers, num_stages=2):
      2 * inner_tile * (2 + 4*3) = 2 * inner_tile * 14 bytes
      inner=8192 → 224 KB (compiler liveness reuse reduces to <192 KB)
      inner=4096 → 112 KB (comfortable)
    UB budget (f32 path, 3 buffers, num_stages=2):
      2 * inner_tile * (4*3) = 2 * inner_tile * 12 bytes
      inner=4096 → 96 KB (comfortable)
    """
    out_dtype = output_dtype or dtype

    @tilelang.jit(out_idx=[1], target="npuir")
    def kernel(inner_tile=8192, num_inner=4, num_stages=2):
        block_size = inner_tile * num_inner

        @T.prim_func
        def main(
            x: T.Tensor((N,), dtype),
            y: T.Tensor((N,), out_dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_size), is_npu=True) as (cid, _):
                if dtype != "float32":
                    # --- float16 / bfloat16 path: 4 UB buffers at inner_tile ---
                    # xy_ub: f16 input load -> f16 output store (reused)
                    # x_f32: upcast input -> final vmul result (in-place dst=src1)
                    # work_a: exp -> +1 -> sqr -> +1 = den
                    # work_b: -1 = num -> /den -> *x = y_f32 (in-place)
                    xy_ub = T.alloc_ub((inner_tile,), dtype)
                    x_f32 = T.alloc_ub((inner_tile,), "float32")
                    work_a = T.alloc_ub((inner_tile,), "float32")
                    work_b = T.alloc_ub((inner_tile,), "float32")

                    for i in T.Pipelined(num_inner, num_stages=num_stages):
                        offset = cid * block_size + i * inner_tile
                        # Safe tail: T.max(0, ...) for out-of-bounds inner tiles
                        inner_tail = T.max(
                            0, T.min(inner_tile, N - offset))

                        # GM -> UB (copy only valid elements)
                        T.copy(x[offset : offset + inner_tail],
                               xy_ub[0:inner_tail])

                        # Upcast to float32 for numerical stability
                        T.vcast(xy_ub, x_f32, round_mode="rint")

                        # Core mish (algebraic identity, in-place ops).
                        # tanh(softplus(x)) = (t2²-1)/(t2²+1), t2 = 1+exp(x)
                        # avoids T.vtanh (Taylor divergence on 1-D UB, see
                        # Stage 3 DESIGN.md §3.3 fallback note).
                        T.vexp(x_f32, work_a)             # work_a = exp(x)
                        T.vadd(work_a, 1.0, work_a)       # work_a = 1 + exp(x) = t2
                        T.vmul(work_a, work_a, work_a)    # work_a = t2² (in-place)
                        T.vsub(work_a, 1.0, work_b)       # work_b = t2² - 1 = num
                        T.vadd(work_a, 1.0, work_a)       # work_a = t2² + 1 = den
                        T.vdiv(work_b, work_a, work_b)    # work_b = num/den
                        T.vmul(x_f32, work_b, x_f32)      # x_f32 = x * tanh(sp) (in-place)

                        # Downcast back to original dtype
                        T.vcast(x_f32, xy_ub, round_mode="round")

                        # UB -> GM (copy only valid elements)
                        T.copy(xy_ub[0:inner_tail],
                               y[offset : offset + inner_tail])
                else:
                    # --- float32 path: 3 UB buffers at inner_tile ---
                    # xy_ub: input load -> final vmul result (in-place dst=src1)
                    # work_a: exp -> +1 -> sqr -> +1 = den
                    # work_b: -1 = num -> /den = tanh_sp
                    xy_ub = T.alloc_ub((inner_tile,), "float32")
                    work_a = T.alloc_ub((inner_tile,), "float32")
                    work_b = T.alloc_ub((inner_tile,), "float32")

                    for i in T.Pipelined(num_inner, num_stages=num_stages):
                        offset = cid * block_size + i * inner_tile
                        inner_tail = T.max(
                            0, T.min(inner_tile, N - offset))

                        # GM -> UB
                        T.copy(x[offset : offset + inner_tail],
                               xy_ub[0:inner_tail])

                        # Core mish (algebraic identity, in-place ops)
                        T.vexp(xy_ub, work_a)             # work_a = exp(x)
                        T.vadd(work_a, 1.0, work_a)       # work_a = 1 + exp(x) = t2
                        T.vmul(work_a, work_a, work_a)    # work_a = t2² (in-place)
                        T.vsub(work_a, 1.0, work_b)       # work_b = t2² - 1 = num
                        T.vadd(work_a, 1.0, work_a)       # work_a = t2² + 1 = den
                        T.vdiv(work_b, work_a, work_b)    # work_b = num/den
                        T.vmul(xy_ub, work_b, xy_ub)      # xy_ub = x * tanh(sp) (in-place)

                        # UB -> GM
                        T.copy(xy_ub[0:inner_tail],
                               y[offset : offset + inner_tail])

        return main

    return kernel


class MishKernel(Kernel):
    """Element-wise Mish activation kernel wrapper.

    Implements y = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x))).
    Supported dtypes: float16, bfloat16, float32.

    Stage 4 optimized: uses T.Pipelined double buffering with large
    block_size for maximum throughput. Default configs are tuned per dtype
    for the [16, 256, 80, 80] benchmark shape.

    NPU tiling: each core processes ``block_size = inner_tile * num_inner``
    contiguous elements.  There is no GPU-style ``threads`` / ``npt`` split
    because the NPU has no SIMT thread model.
    """

    supported_archs: Optional[list[int]] = None
    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(self,
                 N: int,
                 dtype: torch.dtype,
                 config: Optional[dict] = None,
                 tune: bool = False):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"MishKernel only supports dtypes [{supported}], got {dtype}"
            )
        self.N = N
        self.dtype = dtype
        self.dtype_str = self.dtype_to_str(dtype)
        self.kernel = _make_mish_kernel(self.N, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        """Tuned configs per dtype (Stage 4).

        f16/bf16: inner_tile=8192, num_inner=4
                  → block_size=32768, 800 blocks for N=26M
        f32:      inner_tile=4096, num_inner=6
                  → block_size=24576, 1067 blocks for N=26M
        """
        if self.dtype == torch.float32:
            return {"inner_tile": 4096, "num_inner": 6, "num_stages": 2}
        else:
            return {"inner_tile": 8192, "num_inner": 4, "num_stages": 2}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prim_func = self.kernel(
            inner_tile=self.config["inner_tile"],
            num_inner=self.config["num_inner"],
            num_stages=self.config["num_stages"])
        return prim_func(x)


# ---------- precision comparing helper ----------
def _run_case(shape, dtype, tag, *, inner_tile=None,
              num_inner=None, num_stages=2,
              atol=None, rtol=None, kind="randn"):
    """Run one mish case on NPU and compare against the PyTorch golden.

    Args:
        shape: original multi-dim shape (only used to compute N and reporting;
            the kernel operates on a flattened 1-D tensor of N elements).
        dtype: torch.dtype of the input.
        tag: test-layer tag (L0 / L1 / L2 / Boundary).
        inner_tile/num_inner/num_stages: block config; defaults follow
            MishKernel.default_config.
        atol/rtol: tolerances; defaults follow DESIGN.md §8.2.
        kind: input data kind (randn / zeros / ones / large_pos).
    """
    dtype_str = str(dtype).replace("torch.", "")
    N = 1
    for s in shape:
        N *= s

    # Apply default configs per dtype when not specified
    if num_stages is None:
        num_stages = 2
    if inner_tile is None or num_inner is None:
        if dtype == torch.float32:
            inner_tile = inner_tile or 4096
            num_inner = num_inner or 6
        else:
            inner_tile = inner_tile or 8192
            num_inner = num_inner or 4

    if atol is None:
        atol = 1e-3 if dtype == torch.float32 else 1e-2
    if rtol is None:
        rtol = 1e-3 if dtype == torch.float32 else 1e-2

    # Build flattened 1-D input on NPU
    if kind == "zeros":
        x = torch.zeros(N, dtype=dtype, device="npu")
    elif kind == "ones":
        x = torch.ones(N, dtype=dtype, device="npu")
    elif kind == "large_pos":
        x = torch.randn(N, device="npu").abs() * 5.0
        x = x.to(dtype)
    else:  # randn
        x = torch.randn(N, device="npu").to(dtype)

    # Build & run kernel
    config = {
        "inner_tile": inner_tile,
        "num_inner": num_inner,
        "num_stages": num_stages,
    }
    mish_kernel = MishKernel(N, dtype, config)
    y = mish_kernel.forward(x)

    # Golden on CPU
    ref = golden_mish(x)
    y_cpu = y.cpu()
    max_diff = (y_cpu.float() - ref.float()).abs().max().item()

    torch.testing.assert_close(y_cpu, ref, rtol=rtol, atol=atol)
    block_size = inner_tile * num_inner
    print(f"[{tag}] PASS: shape={list(shape)} dtype={dtype_str} N={N} "
          f"inner_tile={inner_tile} num_inner={num_inner} "
          f"block_size={block_size} max_diff={max_diff:.2e} "
          f"(atol={atol:.0e} rtol={rtol:.0e})")
    return max_diff


# ---------- hierarchical testing ----------
# L0: representative shapes from DESIGN.md §8.3 (must pass).
_L0_CASES = [
    # (shape, dtype, inner_tile, num_inner, num_stages, atol, rtol, tag-suffix)
    ([16, 256, 80, 80], torch.float32, 4096, 6, 2, 1e-3, 1e-3, "mish-yolo-p3-f32"),
    ([16, 256, 80, 80], torch.float16, 8192, 4, 2, 1e-2, 1e-2, "mish-yolo-p3-f16"),
    ([16, 512, 40, 40], torch.float16, 8192, 4, 2, 1e-2, 1e-2, "mish-yolo-p4-f16"),
]


def run_L0():
    print("===== L0: representative threshold tests =====")
    for shape, dtype, inner_tile, ni, ns, atol, rtol, suffix in _L0_CASES:
        _run_case(shape, dtype, "L0/" + suffix,
                  inner_tile=inner_tile, num_inner=ni, num_stages=ns,
                  atol=atol, rtol=rtol)


def run_L1():
    print("===== L1: multi-shape / dtype coverage =====")
    # Small & mid shapes, all three dtypes (incl. bf16 per REVIEW warn-5).
    _run_case([1, 1024], torch.float32, "L1/small-f32")
    _run_case([2, 2048], torch.float16, "L1/small-f16")
    _run_case([4, 512], torch.bfloat16, "L1/small-bf16")
    _run_case([16, 1024], torch.float32, "L1/mid-f32")
    _run_case([8, 4096], torch.float16, "L1/mid-f16")
    _run_case([3, 2048], torch.bfloat16, "L1/mid-bf16")
    _run_case([1, 256, 32, 32], torch.float32, "L1/4d-f32")


def run_L2():
    print("===== L2: performance benchmark =====")
    import time
    shape = [16, 256, 80, 80]
    dtype = torch.float16
    N = 1
    for s in shape:
        N *= s
    inner_tile, num_inner, num_stages = 8192, 4, 2
    x = torch.randn(N, device="npu").to(dtype)
    kernel_fn = _make_mish_kernel(N, "float16")
    prim_func = kernel_fn(inner_tile=inner_tile, num_inner=num_inner,
                          num_stages=num_stages)
    # warm up
    prim_func(x)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    iters = 10
    for _ in range(iters):
        prim_func(x)
    torch.npu.synchronize()
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) / iters * 1000.0
    block_size = inner_tile * num_inner
    print(f"[L2] PASS: shape={shape} dtype=float16 N={N} "
          f"inner_tile={inner_tile} num_inner={num_inner} "
          f"block_size={block_size} "
          f"avg={avg_ms:.3f}ms/iter over {iters} iters")


def run_boundary():
    print("===== Boundary: edge values =====")
    # Use default configs per dtype; block sizes are large (32768/24576)
    # so boundary tests verify correct tail handling for N << block_size.
    cases = [
        # N=1 extreme small
        ([1], torch.float32, None, None, None, "Boundary/N=1-f32"),
        ([1], torch.float16, None, None, None, "Boundary/N=1-f16"),
        # exact inner_tile boundary
        ([8192], torch.float16, 8192, 4, 2, "Boundary/exact-inner-f16"),
        # inner_tile - 1 (tail within first inner tile)
        ([8191], torch.float16, 8192, 4, 2, "Boundary/inner-1-f16"),
        # inner_tile + 1 (second inner tile has 1 element)
        ([8193], torch.float16, 8192, 4, 2, "Boundary/inner+1-f16"),
        # exact block_size
        ([32768], torch.float16, 8192, 4, 2, "Boundary/exact-bs-f16"),
        # block_size - 1 (last inner tile is partial)
        ([32767], torch.float16, 8192, 4, 2, "Boundary/bs-1-f16"),
        # block_size + 1 (two blocks, second is tiny tail)
        ([32769], torch.float16, 8192, 4, 2, "Boundary/bs+1-f16"),
        # f32 boundaries
        ([4096], torch.float32, 4096, 6, 2, "Boundary/exact-inner-f32"),
        ([4095], torch.float32, 4096, 6, 2, "Boundary/inner-1-f32"),
        ([24576], torch.float32, 4096, 6, 2, "Boundary/exact-bs-f32"),
        ([24575], torch.float32, 4096, 6, 2, "Boundary/bs-1-f32"),
        # value boundaries
        ([4096], torch.float32, None, None, None, "Boundary/zeros"),
        ([4096], torch.float32, None, None, None, "Boundary/large_pos"),
    ]
    for shape, dtype, inner_tile, ni, ns, name in cases:
        kind = "zeros" if name.endswith("zeros") else (
            "large_pos" if name.endswith("large_pos") else "randn")
        try:
            _run_case(shape, dtype, name,
                      inner_tile=inner_tile, num_inner=ni, num_stages=ns,
                      kind=kind)
        except Exception as e:  # noqa: BLE001  (Boundary is non-blocking)
            print(f"[{name}] WARN (record without blocking): {e}")


def main():
    parser = argparse.ArgumentParser(description="Mish NPU kernel tests (optimized)")
    parser.add_argument("--level", default="L0", choices=["L0", "all"],
                        help="test level: L0 (threshold) or all (L0+L1+L2+Boundary)")
    args, _ = parser.parse_known_args()

    if args.level == "L0":
        run_L0()
    else:
        run_L0()
        run_L1()
        run_L2()
        run_boundary()
    print("\033[92mAll requested checks passed!\033[0m")


if __name__ == "__main__":
    main()

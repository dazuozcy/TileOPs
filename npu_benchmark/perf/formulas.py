"""Roofline formulas for performance upper-bound estimation.

Each function takes an Op instance (with shape/dtype attributes bound during
forward()) and returns (flops, bytes).
"""

from __future__ import annotations

from typing import Callable

_DTYPE_BYTES = {
    "float64": 8,
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
}


def _dtype_itemsize(dtype) -> int:
    if isinstance(dtype, str):
        return _DTYPE_BYTES.get(dtype, 4)
    return _DTYPE_BYTES.get(str(dtype).split(".")[-1], 4)


def topk_selector_roofline(op) -> tuple[int, int]:
    batch = int(op.batch)
    seq_len = int(op.seq_len)
    seq_len_kv = int(op.seq_len_kv)
    kv_group = int(op.kv_group)
    topk = int(op.topk)
    in_elem = _dtype_itemsize(getattr(op, "in_dtype", "float32"))
    out_elem = _dtype_itemsize(getattr(op, "out_dtype", "int32"))
    comparisons = batch * seq_len * kv_group * seq_len_kv
    nbytes = comparisons * in_elem + batch * seq_len * 2 * out_elem
    nbytes += batch * seq_len * kv_group * topk * out_elem
    return int(comparisons), int(nbytes)


def lerp_tensor_roofline(op) -> tuple[int, int]:
    """Roofline for ``LerpTensorOp`` (Tensor-weight ``torch.lerp``).

    Per output element: 3 flops (sub + mul + add); 3 reads + 1 write at
    post-broadcast ``N_total``.
    """
    n_total = int(op.N_total)
    elem_bytes = _dtype_itemsize(getattr(op, "dtype", "float32"))
    return 3 * n_total, 4 * n_total * elem_bytes


def mish_fwd_roofline(op) -> tuple[int, int]:
    """Roofline for ``MishFwdOp`` (``torch.nn.functional.mish``).

    Per output element: 4 flops (softplus = exp + log1p = 2;
    tanh = 1; final mul = 1); 1 read + 1 write at ``N_total``.
    """
    n_total = int(op.N_total)
    elem_bytes = _dtype_itemsize(getattr(op, "dtype", "float32"))
    return 4 * n_total, 2 * n_total * elem_bytes


def logsumexp_fwd_roofline(op) -> tuple[int, int]:
    """Roofline for ``LogSumExpFwdOp`` (``torch.logsumexp``).

    Per row of length ``N``: max (N) + sub (N) + exp (N) + sum (N) +
    log (≈1) ≈ 4N flops.  Reads ``M * N`` input elements, writes ``M``
    output elements.
    """
    M = int(op.M)
    N = int(op.N)
    elem_bytes = _dtype_itemsize(getattr(op, "dtype", "float32"))
    return 4 * M * N, (M * N + M) * elem_bytes


def l1_norm_fwd_roofline(op) -> tuple[int, int]:
    """Roofline for ``L1NormFwdOp`` (``torch.linalg.vector_norm(ord=1)``).

    Per row of length ``N``: abs (N) + add (N) = 2N flops.
    Reads ``M * N`` input elements, writes ``M`` output elements.
    """
    M = int(op.M)
    N = int(op.N)
    elem_bytes = _dtype_itemsize(getattr(op, "dtype", "float32"))
    return 2 * M * N, (M * N + M) * elem_bytes


def l2_norm_fwd_roofline(op) -> tuple[int, int]:
    """Roofline for ``L2NormFwdOp`` (``torch.linalg.vector_norm(ord=2)``).

    Per row of length ``N``: square (N) + add (N) + sqrt (1) = 2N + 1 flops.
    Reads ``M * N`` input elements, writes ``M`` output elements.
    """
    M = int(op.M)
    N = int(op.N)
    elem_bytes = _dtype_itemsize(getattr(op, "dtype", "float32"))
    return 2 * M * N + M, (M * N + M) * elem_bytes


def inf_norm_fwd_roofline(op) -> tuple[int, int]:
    """Roofline for ``InfNormFwdOp`` (``torch.linalg.vector_norm(ord=inf)``).

    Per row of length ``N``: abs (N) + max (N) = 2N flops.
    Reads ``M * N`` input elements, writes ``M`` output elements.
    """
    M = int(op.M)
    N = int(op.N)
    elem_bytes = _dtype_itemsize(getattr(op, "dtype", "float32"))
    return 2 * M * N, (M * N + M) * elem_bytes


def conv2d_fwd_roofline(op) -> tuple[int, int]:
    """Roofline for ``Conv2dFwdOp`` (``torch.nn.functional.conv2d``).

    FLOPs: 2 * N * C_out * out_H * out_W * C_in_g * kH * kW (one MAC =
    2 flops: multiply + add).
    Bytes: read input (N*C_in*H*W) + read weight (C_out*C_in_g*kH*kW) +
    write output (N*C_out*out_H*out_W).
    """
    n = int(op.n)
    c_in = int(op.c_in)
    h = int(op.h)
    w = int(op.w)
    c_out = int(op.c_out)
    c_in_g = int(op.c_in_g)
    kh, kw = op.kernel_size
    out_h = int(op.out_h)
    out_w = int(op.out_w)
    elem_bytes = _dtype_itemsize(getattr(op, "dtype", "float32"))
    flops = 2 * n * c_out * out_h * out_w * c_in_g * kh * kw
    nbytes = (n * c_in * h * w + c_out * c_in_g * kh * kw
              + n * c_out * out_h * out_w) * elem_bytes
    return int(flops), int(nbytes)


def argmax_fwd_roofline(op) -> tuple[int, int]:
    """Roofline for ``ArgmaxFwdOp`` (``torch.argmax``).

    Per row of length ``N``: N comparisons to locate the max.  Reads
    ``M * N`` input elements, writes ``M`` int64 index outputs.
    """
    M = int(op.M)
    N = int(op.N)
    in_elem = _dtype_itemsize(getattr(op, "dtype", "float32"))
    out_elem = _dtype_itemsize(getattr(op, "out_dtype", "int64"))
    return M * N, (M * N * in_elem + M * out_elem)


def avg_pool2d_fwd_roofline(op) -> tuple[int, int]:
    """Roofline for ``AvgPool2dFwdOp`` (``torch.nn.functional.avg_pool2d``).

    Per output element: ``kH * kW`` adds + 1 div = ``kH*kW + 1`` flops.
    Reads input (N*C*H*W) + writes output (N*C*out_H*out_W).
    """
    n = int(op.n)
    c = int(op.c)
    h = int(op.h)
    w = int(op.w)
    kh, kw = op.kernel_size
    out_h = int(op.out_h)
    out_w = int(op.out_w)
    elem_bytes = _dtype_itemsize(getattr(op, "dtype", "float32"))
    flops = n * c * out_h * out_w * (kh * kw + 1)
    nbytes = (n * c * h * w + n * c * out_h * out_w) * elem_bytes
    return int(flops), int(nbytes)


def max_pool3d_fwd_roofline(op) -> tuple[int, int]:
    """Roofline for ``MaxPool3dFwdOp`` (``torch.nn.functional.max_pool3d``).

    Per output element: ``kD * kH * kW`` comparisons.  Reads input
    (N*C*D*H*W) + writes output (N*C*out_D*out_H*out_W).
    """
    n = int(op.n)
    c = int(op.c)
    d = int(op.d)
    h = int(op.h)
    w = int(op.w)
    kd, kh, kw = op.kernel_size
    out_d = int(op.out_d)
    out_h = int(op.out_h)
    out_w = int(op.out_w)
    elem_bytes = _dtype_itemsize(getattr(op, "dtype", "float32"))
    flops = n * c * out_d * out_h * out_w * kd * kh * kw
    nbytes = (n * c * d * h * w + n * c * out_d * out_h * out_w) * elem_bytes
    return int(flops), int(nbytes)


def gated_deltanet_fwd_roofline(op) -> tuple[int, int]:
    """Roofline for ``GatedDeltaNetFwdOp``.

    FLOPs: 4 * B * H * S * DK * DV (state update + query-state matmul
    per token, each involving ~2*DK*DV flops).
    Bytes: read q, k, v, g, beta + write o, S, Aw, Au.
    """
    B = int(op.batch)
    H = int(op.heads)
    S = int(op.seq_len)
    DK = int(op.dim_k)
    DV = int(op.dim_v)
    chunk_size = int(op.chunk_size)
    NC = S // chunk_size
    elem_bytes = _dtype_itemsize(getattr(op, "dtype", "float32"))
    flops = 4 * B * H * S * DK * DV
    nbytes = B * H * S * (2 * DK + DV + 2) * elem_bytes  # q, k, v, g, beta
    nbytes += B * H * S * DV * elem_bytes                  # o
    nbytes += B * H * (NC + 1) * DK * DV * 4               # S (float32)
    nbytes += 2 * B * H * S * chunk_size * 4               # Aw, Au (float32)
    return int(flops), int(nbytes)


def gla_fwd_roofline(op) -> tuple[int, int]:
    """Roofline for ``GLAFwdOp`` (Gated Linear Attention forward).

    FLOPs: 4 * B * S * H * DK * DV (state update + query-state matmul).
    Bytes: read q, k, v, g + write o, final_state.
    """
    B = int(op.batch)
    S = int(op.seq_len)
    H = int(op.heads)
    DK = int(op.dim_k)
    DV = int(op.dim_v)
    elem_bytes = _dtype_itemsize(getattr(op, "dtype", "float32"))
    flops = 4 * B * S * H * DK * DV
    nbytes = B * S * H * (2 * DK + DV + DK) * elem_bytes  # q, k, v, g
    nbytes += B * S * H * DV * elem_bytes                   # o
    nbytes += B * H * DK * DV * elem_bytes                  # final_state
    return int(flops), int(nbytes)


def ssd_chunk_scan_fwd_roofline(op) -> tuple[int, int]:
    """Roofline for ``SSDChunkScanFwdOp``.

    FLOPs: intra-chunk (cb @ dt @ x) + inter-chunk (C @ prev_states).
    Bytes: read x, cb, dA_cumsum, C, prev_states, dt + write y.
    """
    B = int(op.batch)
    NC = int(op.num_chunks)
    Q = int(op.chunk_len)
    H = int(op.n_heads)
    P = int(op.d_head)
    N = int(op.d_state)
    G = int(op.n_groups)
    S = NC * Q
    elem_bytes = _dtype_itemsize(getattr(op, "dtype", "float16"))
    flops = 2 * B * NC * Q * Q * P + 2 * B * NC * H * P * N
    nbytes = (
        B * S * H * P * elem_bytes          # x
        + B * NC * G * Q * Q * elem_bytes   # cb
        + B * H * NC * Q * 4               # dA_cumsum (float32)
        + B * S * G * N * elem_bytes        # C
        + B * NC * H * P * N * 4           # prev_states (float32)
        + B * H * NC * Q * elem_bytes       # dt
        + B * S * H * P * 4                # y (float32)
    )
    return int(flops), int(nbytes)


def moe_grouped_gemm_nopad_fwd_roofline(op) -> tuple[int, int]:
    """Roofline for ``MoeGroupedGemmNopadFwdOp`` (NT grouped GEMM).

    FLOPs: 2 * numel * N * K (one MAC = 2 flops).
    Bytes: read a [numel, K] + b [E, N, K] + true_sizes + true_offsets
           + write c [numel, N].
    """
    numel = int(op.numel)
    num_experts = int(op.num_experts)
    n = int(op.n)
    k = int(op.k)
    elem_bytes = _dtype_itemsize(getattr(op, "dtype", "bfloat16"))
    flops = 2 * numel * n * k
    nbytes = (numel * k + num_experts * n * k + numel * n) * elem_bytes
    nbytes += 2 * num_experts * 4  # true_sizes + true_offsets (int32)
    return int(flops), int(nbytes)


ROOFLINE_REGISTRY: dict[str, Callable] = {
    "topk_selector_roofline": topk_selector_roofline,
    "lerp_tensor_roofline": lerp_tensor_roofline,
    "mish_fwd_roofline": mish_fwd_roofline,
    "logsumexp_fwd_roofline": logsumexp_fwd_roofline,
    "l1_norm_fwd_roofline": l1_norm_fwd_roofline,
    "l2_norm_fwd_roofline": l2_norm_fwd_roofline,
    "inf_norm_fwd_roofline": inf_norm_fwd_roofline,
    "conv2d_fwd_roofline": conv2d_fwd_roofline,
    "argmax_fwd_roofline": argmax_fwd_roofline,
    "avg_pool2d_fwd_roofline": avg_pool2d_fwd_roofline,
    "max_pool3d_fwd_roofline": max_pool3d_fwd_roofline,
    "gated_deltanet_fwd_roofline": gated_deltanet_fwd_roofline,
    "gla_fwd_roofline": gla_fwd_roofline,
    "ssd_chunk_scan_fwd_roofline": ssd_chunk_scan_fwd_roofline,
    "moe_grouped_gemm_nopad_fwd_roofline": moe_grouped_gemm_nopad_fwd_roofline,
}

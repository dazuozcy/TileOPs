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


ROOFLINE_REGISTRY: dict[str, Callable] = {
    "topk_selector_roofline": topk_selector_roofline,
    "lerp_tensor_roofline": lerp_tensor_roofline,
    "mish_fwd_roofline": mish_fwd_roofline,
    "logsumexp_fwd_roofline": logsumexp_fwd_roofline,
    "l1_norm_fwd_roofline": l1_norm_fwd_roofline,
    "l2_norm_fwd_roofline": l2_norm_fwd_roofline,
    "inf_norm_fwd_roofline": inf_norm_fwd_roofline,
    "conv2d_fwd_roofline": conv2d_fwd_roofline,
}

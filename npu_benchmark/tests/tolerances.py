"""Dtype-aware tolerances for floating-point correctness tests.

Each dtype has a different number of mantissa bits, so a single fixed
(rtol, atol) pair is either too loose for float32 (hiding real bugs) or
too tight for float16 / bfloat16 (causing spurious failures).

Values are derived from each dtype's machine epsilon
(``2 ^ -(mantissa_bits)``) with a small multiplier to absorb non-associative
reordering between the kernel and the torch reference:

    float32  — 23 mantissa bits → eps ≈ 1.2e-7  → rtol 1e-5,  atol 1e-5
    float16  — 10 mantissa bits → eps ≈ 9.8e-4  → rtol 1e-3,  atol 1e-3
    bfloat16 —  7 mantissa bits → eps ≈ 7.8e-3  → rtol 1.6e-2, atol 1e-2

These match the PyTorch ``torch.testing.assert_close`` defaults, which
are well-calibrated per dtype.
"""

from __future__ import annotations

import torch

_TOLERANCES: dict[torch.dtype, tuple[float, float]] = {
    torch.float64: (1e-7, 1e-7),
    torch.float32: (1.3e-6, 1e-5),
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (1.6e-2, 1e-2),
}


def dtype_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    """Return ``(rtol, atol)`` appropriate for *dtype*.

    Raises ``KeyError`` if the dtype is not in the table — callers should
    only use this for floating-point dtypes declared in the manifest.
    """
    return _TOLERANCES[dtype]

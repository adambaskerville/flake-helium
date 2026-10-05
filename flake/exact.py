"""Exact arbitrary-precision pieces: radial moments, Gram matrices, inverse Cholesky, storage.

Everything here runs in gmpy2/MPFR at a chosen bit precision (`set_bits`).  These are the
small, horribly conditioned objects of the method; they are computed exactly, once.
"""

from __future__ import annotations

from functools import lru_cache

import gmpy2
import numpy as np
from gmpy2 import mpfr

Z = 2        # nuclear charge
LIMBS = 8    # exact matrices are stored as unevaluated sums of 8 float64s (~424 bits)


def set_bits(bits: int) -> None:
    gmpy2.get_context().precision = int(bits)
    radial_moment.cache_clear()


@lru_cache(maxsize=None)
def radial_moment(n: int, p: int):
    """int_0^inf s^n (ln s)^p e^{-2s} ds for integer n (0 if n < 0) and any p >= 0.

    n!/2^(n+1) * B_p(k_1..k_p): complete Bell polynomial of the cumulants
    k_1 = psi(n+1) - ln 2 and k_j = psi^(j-1)(n+1), with the polygammas at integer
    argument written as harmonic sums and zeta values (closed forms, any precision).
    """
    if n < 0:
        return mpfr(0)
    kappa = [None, gmpy2.fsum(mpfr(1) / k for k in range(1, n + 1)) - gmpy2.const_euler() - gmpy2.log(mpfr(2))]
    for j in range(2, p + 1):
        tail = gmpy2.zeta(mpfr(j)) - gmpy2.fsum(mpfr(k) ** -j for k in range(1, n + 1))
        kappa.append((-1) ** j * gmpy2.fac(j - 1) * tail)
    bell = [mpfr(1)]
    for t in range(p):
        bell.append(gmpy2.fsum(gmpy2.comb(t, i) * bell[t - i] * kappa[i + 1] for i in range(t + 1)))
    return mpfr(gmpy2.fac(n)) / mpfr(2) ** (n + 1) * bell[p]


def angular_coords(order: int) -> list:
    """Angular functions xi^x eta^b (x = m + n, b = n even), ordered by x then b."""
    return [(x, b) for x in range(order + 1) for b in range(0, x + 1, 2)]


def radial_rows(lmax: int, logs: int, log3_lmax: int | None = None) -> list:
    """Radial functions s^l (ln s)^p e^{-s} as (l, p), in whitening order.

    All p <= 1 first (interleaved by l), then one block per log power 2..logs.  Log
    powers >= 3 stop at l <= log3_lmax (higher ones are numerically dependent).
    """
    rows = [(l, p) for l in range(lmax + 1) for p in (0, 1)]
    for p in range(2, logs + 1):
        top = lmax if (p < 3 or log3_lmax is None) else min(lmax, log3_lmax)
        rows += [(l, p) for l in range(top + 1)]
    return rows


def radial_gram(rows) -> np.ndarray:
    g = np.empty((len(rows), len(rows)), dtype=object)
    for i, (l1, p1) in enumerate(rows):
        for j, (l2, p2) in enumerate(rows):
            g[i, j] = radial_moment(l1 + l2 + 5, p1 + p2)      # s^5 from the measure
    return g


def angular_gram(order: int) -> np.ndarray:
    """int_0^1 int_-1^1 xi^(x+x'+2) (1 - xi^2 eta^2) eta^(b+b') d eta d xi."""
    coords = angular_coords(order)
    g = np.empty((len(coords), len(coords)), dtype=object)
    for i, (x1, b1) in enumerate(coords):
        for j, (x2, b2) in enumerate(coords):
            x, b = x1 + x2, b1 + b2
            g[i, j] = mpfr(2) / ((b + 1) * (x + 3)) - mpfr(2) / ((b + 3) * (x + 5))
    return g


def inverse_cholesky(gram: np.ndarray) -> np.ndarray:
    """Upper triangular U with U^T G U = I (G an object array of mpfr)."""
    n = gram.shape[0]
    lower = np.empty((n, n), dtype=object)
    lower.fill(mpfr(0))
    for i in range(n):
        for j in range(i + 1):
            value = gram[i, j] - lower[i, :j].dot(lower[j, :j]) if j else gram[i, j]
            if i == j:
                if not value > 0:
                    raise RuntimeError(f"Gram not positive definite at row {i}: raise the precision")
                lower[i, i] = gmpy2.sqrt(value)
            else:
                lower[i, j] = value / lower[j, j]
    inverse = np.empty((n, n), dtype=object)
    inverse.fill(mpfr(0))
    for col in range(n):
        for i in range(col, n):
            acc = mpfr(1) if i == col else mpfr(0)
            if i > col:
                acc -= lower[i, col:i].dot(inverse[col:i, col])
            inverse[i, col] = acc / lower[i, i]
    return inverse.T.copy()


def to_limbs(matrix: np.ndarray) -> np.ndarray:
    """mpfr array -> float64 array (..., LIMBS) whose last axis sums (exactly) to the value."""
    out = np.zeros(matrix.shape + (LIMBS,))
    for index, value in np.ndenumerate(matrix):
        rest = value
        for k in range(LIMBS):
            limb = float(rest)
            out[index + (k,)] = limb
            rest -= limb
    return out


def from_limbs(values: np.ndarray) -> np.ndarray:
    """Exact sum of the limbs along the last axis, as an object array of mpfr."""
    values = np.asarray(values, dtype=np.float64)
    flat = values.reshape(-1, values.shape[-1])
    out = np.empty(flat.shape[0], dtype=object)
    for i, row in enumerate(flat):
        out[i] = gmpy2.fsum(mpfr(float(x)) for x in row)
    return out.reshape(values.shape[:-1])

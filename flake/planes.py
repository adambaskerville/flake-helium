"""~280-bit exact arithmetic at float64 BLAS speed (error-free digit-plane splitting).

Matrix products (the Kronecker apply)
-------------------------------------
Each operand row is scaled by a power of two to WIDTH = 294 bits and split into NP = 14
signed slices of BETA = 21 bits.  A slice product summed over k <= 2^11 terms is below
2^53, so an ordinary float64 GEMM computes it exactly; slice pairs with s + t > KEEP
(weight < 2^-220 relative) are skipped; groups are recombined with exact integer shifts.
Vectors are fixed point: Python ints at scale 2^-T.

Vector algebra (the eigensolver)
--------------------------------
A vector of N reals at resolution 2^-T is held as int64 digits a[d], base 2^B:
value = sum_d a[d] 2^(B(E-d)), with E = 3 integer digits above the units digit so that
EVERY digit stays balanced in [-2^19, 2^19) for |value| < 2^79 (H.v reaches ~1e18 on the
whitened log rows).  Dot products and linear combinations are exact float64 GEMMs over
digit pairs plus integer recombination.
"""

from __future__ import annotations

import gmpy2
import numpy as np

T = 280                       # vector fixed-point scale: value = int * 2^-T

# ------------------------------------------------------------- matrix products
BETA, NP = 21, 14
WIDTH = BETA * NP
KEEP = NP - 1
MAT_FRAC = 400                # matrix fixed-point scale before row normalisation
_MASK = (1 << BETA) - 1
_BITLEN = np.frompyfunc(lambda v: abs(v).bit_length(), 1, 1)
_ABS = np.frompyfunc(abs, 1, 1)


def _row_planes(ints: np.ndarray):
    """Row-normalise an int matrix to WIDTH bits; return (planes (NP, m, k) float64, shifts)."""
    m, k = ints.shape
    shift = WIDTH - np.maximum(_BITLEN(ints).max(axis=1).astype(np.int64), 1)
    vals = np.empty((m, k), dtype=object)
    for i in range(m):
        sh = int(shift[i])
        vals[i] = (ints[i] << sh) if sh >= 0 else ((ints[i] + (1 << (-sh - 1))) >> -sh)
    sign = np.where((vals < 0).astype(bool), -1.0, 1.0)
    mag = _ABS(vals)
    planes = np.empty((NP, m, k))
    for s in range(NP):
        planes[s] = sign * ((mag >> (BETA * (NP - 1 - s))) & _MASK).astype(np.float64)
    return planes, shift


def prepare(mat) -> dict:
    """Slices of an mpfr (object) matrix or a float64 limb array (n, k, LIMBS)."""
    if isinstance(mat, np.ndarray) and mat.dtype != object and mat.ndim == 3:
        ints = np.zeros(mat.shape[:-1], dtype=object)
        for k in range(mat.shape[-1]):
            ints = ints + np.vectorize(int, otypes=[object])(np.rint(np.ldexp(mat[..., k], MAT_FRAC)))
    else:
        scale = gmpy2.mpfr(2) ** MAT_FRAC
        ints = np.vectorize(lambda v: int(gmpy2.rint(gmpy2.mpfr(v) * scale)), otypes=[object])(mat)
    planes, shift = _row_planes(ints)
    return {"planes": planes, "shift": shift}


def x_mt(x: np.ndarray, mat: dict) -> np.ndarray:
    """Fixed-point X M^T  (X: (m, k) Python ints at 2^-T; M prepared, (n, k))."""
    xp, xs = _row_planes(x)
    m, n = x.shape[0], mat["planes"].shape[1]
    groups = [np.zeros((m, n), dtype=np.int64) for _ in range(KEEP + 1)]
    for t in range(KEEP + 1):
        ns = KEEP + 1 - t                                            # only pairs s + t <= KEEP
        prod = (xp[:ns].reshape(ns * m, -1) @ mat["planes"][t].T).reshape(ns, m, n)   # exact
        for s in range(ns):
            groups[s + t] += prod[s].astype(np.int64)
    z = np.zeros((m, n), dtype=object)
    for g, grp in enumerate(groups):
        z = z + grp.astype(object) * (1 << (BETA * (2 * NP - 2 - g)))
    sh = xs[:, None].astype(object) + mat["shift"][None, :].astype(object) + MAT_FRAC
    pos = sh > 0
    half = np.where(pos, np.left_shift(np.ones_like(z), np.where(pos, sh - 1, 0)), 0)
    return np.where(pos, np.right_shift(z + half, np.where(pos, sh, 0)), np.left_shift(z, np.where(pos, 0, -sh)))


class KronApply:
    """Prepared H C = sum_A M_A (C A^T) on the leading (n_r, n_a) block of a build."""

    def __init__(self, kron: dict, n_r: int, n_a: int, which: str = "h"):
        by_a: dict = {}
        for (rkey, akey), coef in kron[which].items():
            r = kron["rad"][rkey][:n_r, :n_r] * coef
            by_a[akey] = r if akey not in by_a else by_a[akey] + r
        self.shape = (n_r, n_a)
        self.parts = [(prepare(kron["ang"][akey][:n_a, :n_a]), prepare(m)) for akey, m in by_a.items()]

    def __call__(self, c: np.ndarray) -> np.ndarray:
        out = np.zeros(self.shape, dtype=object)
        for a_prep, m_prep in self.parts:
            out = out + x_mt(x_mt(c, a_prep).T.copy(), m_prep).T
        return out


# ------------------------------------------------------------- vector digits
B, E = 20, 3
FRAC = T // B
D = E + 1 + FRAC              # 18 digits
HALF = 1 << (B - 1)
_DMASK = (1 << B) - 1
CHUNK = 1 << 12               # dot-product chunk: 2^19 * 2^19 * 2^12 < 2^53


def normalize(a: np.ndarray) -> np.ndarray:
    for d in range(a.shape[0] - 1, 0, -1):
        carry = (a[d] + HALF) >> B
        a[d] -= carry << B
        a[d - 1] += carry
    return a


def from_pyints(v) -> np.ndarray:
    v = np.asarray(v, dtype=object)
    a = np.empty((D,) + v.shape, dtype=np.int64)
    for d in range(D - 1, 0, -1):
        a[d] = np.bitwise_and(v, _DMASK).astype(np.int64)
        v = np.right_shift(v, B)
    a[0] = v.astype(np.int64)
    return normalize(a)


def to_pyints(a: np.ndarray) -> np.ndarray:
    out = a[0].astype(object) << (B * (D - 1))
    for d in range(1, D):
        out = out + (a[d].astype(object) << (B * (D - 1 - d)))
    return out


def from_float(v: np.ndarray) -> np.ndarray:
    a = np.zeros((D,) + v.shape, dtype=np.int64)
    r = np.array(v, dtype=np.float64)
    for d in range(E, D):
        a[d] = np.rint(r).astype(np.int64)
        r = (r - a[d]) * float(1 << B)
        if not r.any():
            break
    return normalize(a)


def to_float(a: np.ndarray) -> np.ndarray:
    """All digits summed smallest first: residuals of 1e-25 keep full relative precision."""
    out = np.zeros(a.shape[1:], dtype=np.float64)
    for d in range(D - 1, -1, -1):
        out = out + a[d].astype(np.float64) * 2.0 ** (B * (E - d))
    return out


def combine(vecs, coefs) -> np.ndarray:
    """sum_k coef_k vec_k (coefs: mpfr/float; vecs: (D, N) digit arrays), rounded to 2^-T."""
    scale = gmpy2.mpfr(2) ** T
    c = from_pyints(np.array([int(gmpy2.rint(gmpy2.mpfr(x) * scale)) for x in coefs], dtype=object)).astype(np.float64)
    stack = np.stack(vecs)
    G = 2                                               # guard digits
    out = np.zeros((D + G + E,) + stack.shape[2:], dtype=np.int64)   # slot = coef digit + vector digit
    for i in range(D):
        prod = c @ stack[:, i, :].astype(np.float64)
        top = D + G + E - i
        out[i:i + min(top, D)] += prod[:min(top, D)].astype(np.int64)
    normalize(out)
    assert not out[:E].any(), "combine overflow: |result| >= 2^79"
    guard = out[D + E] * (1 << B) + out[D + E + 1]
    out[D + E - 1] += (guard + (1 << (2 * B - 1))) >> (2 * B)
    return normalize(out[E:E + D].copy())


def dots(x: np.ndarray, vecs) -> list:
    """[x . v for v in vecs] exactly (to 2^-2T-ish), as mpfr."""
    stack = np.stack(vecs)
    k, _, n = stack.shape
    flat = stack.reshape(k * D, n)
    acc = np.zeros((k * D, D), dtype=np.int64)
    for s in range(0, n, CHUNK):
        acc += (flat[:, s:s + CHUNK].astype(np.float64) @ x[:, s:s + CHUNK].astype(np.float64).T).astype(np.int64)
    acc = acc.reshape(k, D, D)
    denom = gmpy2.mpfr(2) ** (2 * B * (D - 1 - E))
    out = []
    for m in range(k):
        total = 0
        for i in range(D):
            for j in range(D):
                if i + j - 2 * E <= FRAC + 3:
                    total += int(acc[m, i, j]) << (B * (2 * (D - 1) - i - j))
        out.append(gmpy2.mpfr(total) / denom)
    return out

"""The helium Hamiltonian as a short sum of Kronecker products in an exactly whitened basis.

In box coordinates (s, xi = u/s, eta = t/u) every basis function is
s^L (ln s)^P e^{-s} * xi^X eta^B, so each derivative "bank" is a short sum of
(radial operator) (x) (angular operator):

    value = I (x) I
    ds    = R_ds (x) I  -  S_L (x) diag(X),     R_ds = S_L diag(L) + S_LP diag(P) - I
    dt    = S_L (x) A_dt,   A_dt: (X,B) -> (X-1,B-1) times B
    du    = S_L (x) A_du,   A_du: (X,B) -> (X-1,B)   times X-B

and every weight of the gradient-form energy is a monomial s^dl xi^dx eta^db, whose
integral factorises.  Hence

    H = sum_j c_j R_j (x) A_j   (27 terms: 6 radial, 26 angular matrices),

stored here already transformed by the exact inverse-Cholesky factors U_r, U_a, so that
S = I.  White matrix elements depend only on the functions involved (the factors are
triangular), so one build serves every smaller space and every mask.
"""

from __future__ import annotations

import ast
import multiprocessing as mproc
from pathlib import Path
from time import perf_counter

import numpy as np
from gmpy2 import mpfr

from .exact import (Z, angular_coords, angular_gram, from_limbs, inverse_cholesky, radial_gram,
                    radial_moment, radial_rows, set_bits, to_limbs)

BANKS = {                       # derivative bank -> [(radial op, angular op, coefficient)]
    "value": [("I", "I", 1)],
    "ds": [("Rds", "I", 1), ("SL", "Dx", -1)],
    "dt": [("SL", "Adt", 1)],
    "du": [("SL", "Adu", 1)],
}
Q = {                           # weight -> [(coefficient, dl, dx, db)]: sum of c s^dl xi^dx eta^db
    "ov": [(1, 3, 1, 0), (-1, 3, 3, 2)],                       # u (s^2 - t^2)
    "cs": [(-1, 3, 2, 2), (1, 3, 2, 0)],                       # s (u^2 - t^2)
    "ct": [(1, 3, 1, 1), (-1, 3, 3, 1)],                       # t (s^2 - u^2)
    "pot": [(-4 * Z, 2, 1, 0), (1, 2, 0, 0), (-1, 2, 2, 2)],   # s^2 - t^2 - 4 Z s u
}
H_PAIRS = [("ds", "ov", "ds"), ("dt", "ov", "dt"), ("du", "ov", "du"),
           ("ds", "cs", "du"), ("du", "cs", "ds"), ("dt", "ct", "du"),
           ("du", "ct", "dt"), ("value", "pot", "value")]
S_PAIRS = [("value", "ov", "value")]


def terms(pairs) -> dict:
    """{(radial_key, angular_key): coefficient}, radial_key = (rb, dl, rk), angular_key = (ab, dx, db, ak)."""
    out: dict = {}
    for bra, q, ket in pairs:
        for rb, ab, cb in BANKS[bra]:
            for cm, dl, dx, db in Q[q]:
                for rk, ak, ck in BANKS[ket]:
                    key = ((rb, dl, rk), (ab, dx, db, ak))
                    out[key] = out.get(key, 0) + cb * cm * ck
    return {k: v for k, v in out.items() if v}


# ------------------------------------------------------------------ radial side
def _radial_ops(lmax: int, npow: int) -> dict:
    """Radial bank operators on box rows (L+1)*npow + P, L in -1..lmax (one halo row for derivatives)."""
    n = npow * (lmax + 2)
    zero = lambda: np.array([[mpfr(0)] * n for _ in range(n)], dtype=object)
    ident, sl, slp, dl = zero(), zero(), zero(), zero()
    for L in range(-1, lmax + 1):
        for P in range(npow):
            i = (L + 1) * npow + P
            ident[i, i] = mpfr(1)
            dl[i, i] = mpfr(L)
            if L >= 0:
                sl[L * npow + P, i] = mpfr(1)                   # s^L -> s^(L-1)
                if P >= 1:
                    slp[L * npow + P - 1, i] = mpfr(P)          # d/ds (ln s)^P
    return {"I": ident, "SL": sl, "Rds": sl.dot(dl) + slp - ident}


def _radial_master(lmax: int, npow: int, dl: int) -> np.ndarray:
    n = npow * (lmax + 2)
    m = np.empty((n, n), dtype=object)
    for i in range(n):
        L1, P1 = i // npow - 1, i % npow
        for j in range(n):
            L2, P2 = j // npow - 1, j % npow
            m[i, j] = radial_moment(L1 + L2 + dl + 2, P1 + P2)  # +2: Jacobian s^2
    return m


def radial_white(rows, rkeys) -> dict:
    lmax, npow = max(l for l, _ in rows), 1 + max(p for _, p in rows)
    ur = inverse_cholesky(radial_gram(rows))
    ubox = np.array([[mpfr(0)] * len(rows) for _ in range(npow * (lmax + 2))], dtype=object)
    for i, (l, p) in enumerate(rows):
        ubox[(l + 1) * npow + p, :] = ur[i, :]
    ops, masters, out = _radial_ops(lmax, npow), {}, {}
    for rb, dl, rk in rkeys:
        if dl not in masters:
            masters[dl] = _radial_master(lmax, npow, dl)
        out[(rb, dl, rk)] = (ops[rb].dot(ubox)).T.dot(masters[dl]).dot(ops[rk].dot(ubox))
    return out


# ----------------------------------------------------------------- angular side
_A: dict = {}


def _angular_setup(order: int) -> None:
    ua = inverse_cholesky(angular_gram(order))
    coords = angular_coords(order)
    nx, nb = order + 2, order + 1
    u = np.empty((nx, nb, len(coords)), dtype=object)
    u.fill(mpfr(0))
    for i, (x, b) in enumerate(coords):
        u[x + 1, b, :] = ua[i, :]
    _A["u"] = u


def _ang_op(name: str, v: np.ndarray) -> np.ndarray:
    """Angular bank operator on box columns v (nx, nb, ncols); box index xi = X + 1."""
    if name == "I":
        return v
    out = np.empty_like(v)
    out.fill(mpfr(0))
    nx, nb = v.shape[0], v.shape[1]
    if name == "Dx":
        for xi in range(nx):
            out[xi] = v[xi] * (xi - 1)
    elif name == "Adt":
        for xi in range(1, nx):
            for b in range(1, nb):
                out[xi - 1, b - 1] = out[xi - 1, b - 1] + v[xi, b] * b
    elif name == "Adu":
        for xi in range(1, nx):
            for b in range(nb):
                if (xi - 1) - b:
                    out[xi - 1, b] = out[xi - 1, b] + v[xi, b] * ((xi - 1) - b)
    else:
        raise ValueError(name)
    return out


def _ang_master(v: np.ndarray, dx: int, db: int) -> np.ndarray:
    nx, nb = v.shape[0], v.shape[1]
    xs = np.arange(-1, nx - 1)
    xm = np.empty((nx, nx), dtype=object)
    for i, a in enumerate(xs):
        for j, c in enumerate(xs):
            s = a + c + dx
            xm[i, j] = mpfr(1) / (s + 2) if s >= 0 else mpfr(0)     # int xi^(s+1) dxi
    bm = np.empty((nb, nb), dtype=object)
    for i in range(nb):
        for j in range(nb):
            s = i + j + db
            bm[i, j] = mpfr(2) / (s + 1) if s % 2 == 0 else mpfr(0)  # int eta^s d eta
    t = np.tensordot(xm, v, axes=([1], [0]))
    return np.tensordot(t, bm, axes=([1], [1])).transpose(0, 2, 1)


def _angular_term(key):
    ab, dx, db, ak = key
    started = perf_counter()
    u = _A["u"]
    ket = _ang_master(_ang_op(ak, u), dx, db)
    bra = _ang_op(ab, u)
    nx, nb, na = u.shape
    bra2, ket2 = bra.reshape(nx * nb, na), ket.reshape(nx * nb, na)
    rows = [i for i in range(nx * nb) if any(x != 0 for x in bra2[i])]
    return key, bra2[rows].T.dot(ket2[rows]), perf_counter() - started


# ------------------------------------------------------------------------ build
def build(order: int, radial: int, logs: int, log3_lmax: int | None, out: Path,
          bits_radial: int = 3000, bits_angular: int = 640, workers: int = 6, verbose: bool = True) -> None:
    """Build the white Kronecker operator: angular order `order`, radial degree `radial`, (ln s)^p for p <= logs."""
    say = print if verbose else (lambda *a, **k: None)
    started = perf_counter()
    th, ts = terms(H_PAIRS), terms(S_PAIRS)
    rkeys = sorted({k[0] for k in list(th) + list(ts)})
    akeys = sorted({k[1] for k in list(th) + list(ts)})
    rows = radial_rows(radial, max(logs, 1), log3_lmax)
    say(f"{len(th)} H terms, {len(ts)} S terms: {len(rkeys)} radial matrices ({len(rows)} rows), "
        f"{len(akeys)} angular matrices ({len(angular_coords(order))} rows)", flush=True)
    set_bits(bits_radial)
    rad = radial_white(rows, rkeys)
    say(f"radial done ({perf_counter() - started:.0f}s)", flush=True)
    set_bits(bits_angular)
    _angular_setup(order)
    ang = {}
    with mproc.get_context("fork").Pool(workers) as pool:
        for key, mat, sec in pool.imap_unordered(_angular_term, akeys):
            ang[key] = mat
            say(f"  angular {key} ({sec:.0f}s)", flush=True)
    arrays = {f"R{i}": to_limbs(rad[k]) for i, k in enumerate(rkeys)}
    arrays.update({f"A{i}": to_limbs(ang[k]) for i, k in enumerate(akeys)})
    np.savez(out, order=order, radial_rows=np.array(rows),
             rkeys=np.array([repr(k) for k in rkeys]), akeys=np.array([repr(k) for k in akeys]),
             hterms=np.array([repr((k, v)) for k, v in th.items()]),
             sterms=np.array([repr((k, v)) for k, v in ts.items()]), **arrays)
    say(f"saved {out} ({perf_counter() - started:.0f}s)", flush=True)


def load(path: Path, bits: int = 800) -> dict:
    """Radial matrices as mpfr (small), angular matrices kept as float64 limb arrays (large)."""
    set_bits(bits)
    z = np.load(path)
    rk = [ast.literal_eval(k) for k in z["rkeys"]]
    ak = [ast.literal_eval(k) for k in z["akeys"]]
    return {"order": int(z["order"]),
            "radial_rows": [tuple(int(v) for v in row) for row in z["radial_rows"]],
            "rad": {k: from_limbs(z[f"R{i}"]) for i, k in enumerate(rk)},
            "ang": {k: np.asarray(z[f"A{i}"]) for i, k in enumerate(ak)},
            "h": dict(ast.literal_eval(t) for t in z["hterms"]),
            "s": dict(ast.literal_eval(t) for t in z["sterms"])}

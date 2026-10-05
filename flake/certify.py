"""Independent exact certification of a solver state.

The white grid C defines the explicit function with monomial coefficients M = U_r C U_a^T.
U_r, U_a are recomputed here from the closed-form Grams (never taken from the Kronecker
build), and the Rayleigh quotient N/D of the gradient-form energy functional is evaluated
directly from box-coordinate master integrals in arbitrary precision.  Because Psi is an
explicit function, N/D is a rigorous upper bound on the ground-state energy.
"""

from __future__ import annotations

import json
import multiprocessing as mproc
from pathlib import Path

import gmpy2
import numpy as np
from gmpy2 import mpfr

from .exact import Z, angular_coords, angular_gram, inverse_cholesky, radial_gram, radial_moment, set_bits
from .solver import load_state


class Box:
    """Coefficients c[L+1, X+1, B, P] of s^(L-X) u^(X-B) t^B (ln s)^P e^{-s} (one halo cell for derivatives)."""

    def __init__(self, lmax: int, xmax: int, npow: int):
        self.nl, self.nx, self.nb, self.np = lmax + 2, xmax + 2, xmax + 1, npow
        self._cache: dict = {}

    def zeros(self) -> np.ndarray:
        out = np.empty((self.nl, self.nx, self.nb, self.np), dtype=object)
        out.fill(mpfr(0))
        return out

    def _angular(self, dx: int, db: int):
        if ("a", dx, db) not in self._cache:
            xs = np.arange(-1, self.nx - 1)
            xm = np.array([[mpfr(1) / (a + c + dx + 2) if a + c + dx >= 0 else mpfr(0) for c in xs] for a in xs],
                          dtype=object)
            bm = np.array([[mpfr(2) / (a + c + db + 1) if (a + c + db) % 2 == 0 else mpfr(0)
                            for c in range(self.nb)] for a in range(self.nb)], dtype=object)
            self._cache[("a", dx, db)] = (xm, bm)
        return self._cache[("a", dx, db)]

    def _radial(self, dl: int):
        if ("r", dl) not in self._cache:
            n = self.nl * self.np
            rm = np.empty((n, n), dtype=object)
            for i in range(n):
                for j in range(n):
                    rm[i, j] = radial_moment(i // self.np + j // self.np - 2 + dl + 2, i % self.np + j % self.np)
            self._cache[("r", dl)] = rm
        return self._cache[("r", dl)]

    def master(self, f, g, dl: int, dx: int, db: int):
        """sum_ij f_i g_j K(dl, dx, db)_ij."""
        xm, bm = self._angular(dx, db)
        t = np.tensordot(g, bm, axes=([2], [1]))                  # (L', X', P', B)
        t = np.tensordot(t, xm, axes=([1], [1]))                  # (L', P', B, X)
        u = self._radial(dl).dot(t.reshape(self.nl * self.np, self.nb * self.nx))
        return gmpy2.fsum((f.transpose(0, 3, 2, 1).reshape(u.shape) * u).ravel())

    def derivatives(self, value):
        ds, dt, du = self.zeros(), self.zeros(), self.zeros()
        for li, xi, b, p in np.argwhere(np.vectorize(lambda v: v != 0)(value)):
            c = value[li, xi, b, p]
            a, m = (li - 1) - (xi - 1), (xi - 1) - b             # powers of s and u
            if a:
                ds[li - 1, xi, b, p] += a * c
            if p:
                ds[li - 1, xi, b, p - 1] += p * c
            ds[li, xi, b, p] -= c
            if b:
                dt[li - 1, xi - 1, b - 1, p] += b * c
            if m:
                du[li - 1, xi - 1, b, p] += m * c
        return ds, dt, du


# (bra field, ket field, coefficient, dl, dx, db, target): the gradient-form functional
_TASKS = [("v", "v", 1, 3, 1, 0, "s"), ("v", "v", -1, 3, 3, 2, "s")]
for _f in ("ds", "dt", "du"):
    _TASKS += [(_f, _f, 1, 3, 1, 0, "h"), (_f, _f, -1, 3, 3, 2, "h")]
_TASKS += [("ds", "du", -2, 3, 2, 2, "h"), ("ds", "du", 2, 3, 2, 0, "h"),
           ("dt", "du", 2, 3, 1, 1, "h"), ("dt", "du", -2, 3, 3, 1, "h"),
           ("v", "v", -4 * Z, 2, 1, 0, "h"), ("v", "v", 1, 2, 0, 0, "h"), ("v", "v", -1, 2, 2, 2, "h")]
_C: dict = {}


def _task(t):
    f, g, coef, dl, dx, db, target = t
    return target, coef * _C["box"].master(_C[f], _C[g], dl, dx, db)


def rayleigh(box: Box, value, workers: int = 6):
    """(<Psi|H|Psi>, <Psi|Psi>) for a coefficient box."""
    ds, dt, du = box.derivatives(value)
    _C.update(box=box, v=value, ds=ds, dt=dt, du=du)
    with mproc.get_context("fork").Pool(workers) as pool:
        parts = pool.map(_task, _TASKS, chunksize=1)
    return gmpy2.fsum(v for t, v in parts if t == "h"), gmpy2.fsum(v for t, v in parts if t == "s")


def monomial_box(grid, rows, order: int) -> tuple:
    """Box of the explicit function encoded by a white grid (rows = radial (l, p) labels)."""
    ur = inverse_cholesky(radial_gram(rows))
    ua = inverse_cholesky(angular_gram(order))
    m = ur.dot(grid).dot(ua.T)
    box = Box(max(l for l, _ in rows), order, 1 + max(p for _, p in rows))
    value = box.zeros()
    for r, (l, p) in enumerate(rows):
        for j, (x, b) in enumerate(angular_coords(order)):
            if m[r, j] != 0:
                value[l + 1, x + 1, b, p] = m[r, j]
    return box, value


def certify(run_dir: Path, bits: int = 2000, workers: int = 6) -> dict:
    set_bits(bits)
    st = load_state(run_dir)
    scale = mpfr(2) ** 280
    grid = np.vectorize(lambda z: mpfr(int(z)) / scale, otypes=[object])(st["grid"])
    box, value = monomial_box(grid, st["radial_rows"], st["order"])
    h, s = rayleigh(box, value, workers)
    e = h / s
    out = {"run": str(run_dir), "certified_energy": format(e, ".80f"),
           "certified_minus_solver": format(e - mpfr(st["energy"]), ".3e"), "norm_minus_1": format(s - 1, ".3e"),
           "N": int(st["mask"].sum()), "bits": bits}
    (Path(run_dir) / "certificate.json").write_text(json.dumps(out, indent=1) + "\n")
    return out

"""Davidson for the lowest eigenvalue of the white Kronecker Hamiltonian on any masked space.

Space: grid of (radial row (l, p)) x (angular (x, b)).  Schwartz shells l + x <= order;
optionally extra radial degree l + x <= radial for low angular order x < x_cut; optional
(ln s)^p rows (p >= 2) for x < log_x_cut and l >= log_lmin.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import gmpy2
import mpmath as mp
import numpy as np
from gmpy2 import mpfr
from scipy.linalg import cho_factor, cho_solve

from . import planes as P
from .exact import angular_coords, set_bits

DPS = 110   # projected eigenproblem precision (decimal digits)


def build_mask(kron: dict, order: int, radial: int | None = None, x_cut: int = 10, logs: int = 1,
               log_x_cut: int = 20, log_lmin: int = 0):
    """(radial rows used, angular coords, mask).  Radial rows are a prefix of the build's ordering."""
    radial = order if radial is None else radial
    ang = angular_coords(order)
    assert order <= kron["order"], f"space order {order} > build angular order {kron['order']}"
    rows_all = kron["radial_rows"]
    assert radial <= max(l for l, _ in rows_all), "space radial degree exceeds the build"
    mask = np.zeros((len(rows_all), len(ang)), dtype=bool)
    for r, (l, p) in enumerate(rows_all):
        if p > logs:
            continue
        for a, (x, _) in enumerate(ang):
            cap = radial if x < x_cut else order
            if l + x <= cap and (p <= 1 or (x < log_x_cut and l >= log_lmin)):
                mask[r, a] = True
    n_r = int(np.flatnonzero(mask.any(axis=1)).max()) + 1          # prefix keeps the white chart
    return rows_all[:n_r], ang, mask[:n_r]


class Precond:
    """Block-Jacobi: exact float64 blocks of H over groups of angular-order sectors (<= block_max rows).

    Blocks are factored by Cholesky of H_b - theta + tau diag(H_b), tau escalated from 0: float64
    assembly of blocks with norms ~1e18 (whitened log rows) is indefinite at the ~n*eps scaled level.
    """

    def __init__(self, kron: dict, mask: np.ndarray, ang, block_max: int = 8000, floor: float = 1e-10):
        def flt(v):
            v = np.asarray(v)
            return v[..., 0] + v[..., 1] if v.dtype != object and v.ndim == 3 else np.array(v, dtype=float)
        n_r, n_a = mask.shape
        rad = {k: flt(v)[:n_r, :n_r] for k, v in kron["rad"].items()}
        angm = {k: flt(v[:n_a, :n_a]) for k, v in kron["ang"].items()}
        xarr = np.array([x for x, _ in ang])
        groups, size = [], 0
        for x in sorted(set(xarr)):
            nx = int(mask[:, xarr == x].sum())
            if not groups or size + nx > block_max:
                groups.append(set())
                size = 0
            groups[-1].add(x)
            size += nx
        self.blocks = []
        for grp in groups:
            a_idx = [a for a, (x, _) in enumerate(ang) if x in grp and mask[:, a].any()]
            if not a_idx:
                continue
            r_idx = sorted({r for a in a_idx for r in np.flatnonzero(mask[:, a])})
            sub = [(r, a) for a in a_idx for r in r_idx if mask[r, a]]
            ri, ai = {r: i for i, r in enumerate(r_idx)}, {a: i for i, a in enumerate(a_idx)}
            flat = [ai[a] * len(r_idx) + ri[r] for r, a in sub]
            blk = np.zeros((len(a_idx) * len(r_idx),) * 2)
            for (rkey, akey), coef in kron["h"].items():
                blk += coef * np.kron(angm[akey][np.ix_(a_idx, a_idx)], rad[rkey][np.ix_(r_idx, r_idx)])
            blk = blk[np.ix_(flat, flat)]
            self.blocks.append((np.array([r for r, _ in sub]), np.array([a for _, a in sub]), 0.5 * (blk + blk.T)))
        self.floor, self.theta = floor, None

    def _factor(self, theta: float) -> None:
        self.theta, self.fac, taus = theta, [], []
        for _, _, blk in self.blocks:
            shifted = blk - theta * np.eye(len(blk))
            dg, ii = np.diag(blk).clip(min=0), np.diag_indices(len(blk))
            for tau in (0, 1e-12, 1e-11, 1e-10, 1e-9):
                m = shifted.copy()
                m[ii] += tau * dg
                try:
                    self.fac.append(("chol", cho_factor(m, overwrite_a=True)))
                    taus.append(tau)
                    break
                except np.linalg.LinAlgError:
                    pass
            else:                                          # block genuinely below theta: floored eigen-inverse
                w, v = np.linalg.eigh(shifted)
                w = np.where(np.abs(w) < self.floor, np.sign(w) * self.floor + (w == 0) * self.floor, w)
                self.fac.append(("eig", (w, v)))
                taus.append("eig")
        print(f"  preconditioner factored at theta={theta:.12f}: tau per block {taus}", flush=True)

    def __call__(self, res: np.ndarray, theta: float) -> np.ndarray:
        """~ -(H - theta)^-1 res."""
        if self.theta is None or abs(theta - self.theta) > 1e-6:
            self._factor(theta)
        out = np.zeros_like(res)
        for (rr, aa, _), (kind, f) in zip(self.blocks, self.fac):
            b = res[rr, aa]
            out[rr, aa] = -(cho_solve(f, b) if kind == "chol" else f[1] @ ((f[1].T @ b) / f[0]))
        return out


class Space:
    """Digit-plane vectors on the masked coordinates of the (n_r, n_a) grid."""

    def __init__(self, mask: np.ndarray, apply):
        self.mask, self.apply_grid = mask, apply
        self.rows, self.cols = np.nonzero(mask)

    def gather(self, grid) -> np.ndarray:
        return P.from_pyints(grid[self.rows, self.cols])

    def scatter(self, v: np.ndarray) -> np.ndarray:
        grid = np.zeros(self.mask.shape, dtype=object)
        grid[self.rows, self.cols] = P.to_pyints(v)
        return grid

    def apply(self, v: np.ndarray) -> np.ndarray:
        return self.gather(self.apply_grid(self.scatter(v)))

    def to_float_grid(self, v: np.ndarray) -> np.ndarray:
        g = np.zeros(self.mask.shape)
        g[self.rows, self.cols] = P.to_float(v)
        return g

    def from_float_grid(self, g: np.ndarray) -> np.ndarray:
        return P.from_float(g[self.rows, self.cols])


def save_state(out: Path, sp: Space, x: np.ndarray, rows, order: int, theta, meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "state.npz", digits=x, mask=sp.mask, radial_rows=np.array(rows), order=order,
             energy=mp.nstr(theta, 100))
    (out / "progress.json").write_text(json.dumps(meta, indent=1) + "\n")


def load_state(path: Path) -> dict:
    z = np.load(Path(path) / "state.npz")
    mask = z["mask"]
    grid = np.zeros(mask.shape, dtype=object)
    r, c = np.nonzero(mask)
    grid[r, c] = P.to_pyints(z["digits"])
    return {"grid": grid, "mask": mask, "order": int(z["order"]), "energy": str(z["energy"]),
            "radial_rows": [tuple(int(v) for v in row) for row in z["radial_rows"]]}


def seed_grid(seed: Path, rows, ang, mask) -> np.ndarray:
    """Embed a previous state by (l, p) and (x, b) labels (exact within one build's radial chart)."""
    s = load_state(seed)
    n = min(len(s["radial_rows"]), len(rows))
    if s["radial_rows"][:n] != list(rows[:n]):
        print("  WARNING: seed comes from a different radial build; the embedded function is only "
              "approximately the same (expect a worse starting energy)", flush=True)
    rmap = {lab: i for i, lab in enumerate(rows)}
    amap = {lab: j for j, lab in enumerate(ang)}
    sang = angular_coords(s["order"])
    grid = np.zeros(mask.shape, dtype=object)
    for i, lab in enumerate(s["radial_rows"]):
        if lab not in rmap:
            continue
        for j, alab in enumerate(sang):
            jj = amap.get(alab)
            if jj is not None and mask[rmap[lab], jj]:
                grid[rmap[lab], jj] = s["grid"][i, j]
    return grid


def _orthonormalize(t: np.ndarray, V: list):
    """Exact Gram-Schmidt against V, then normalise; repeat while much norm is lost
    (dividing by a small norm would amplify the 2^-280 orthogonality error)."""
    for _ in range(4):
        for _ in range(2):
            t = P.combine([t] + V, [mpfr(1)] + [-c for c in P.dots(t, V)])
        tn = gmpy2.sqrt(P.dots(t, [t])[0])
        if tn < mpfr(10) ** -70:
            return None                                 # correction already in the subspace
        t = P.combine([t], [1 / tn])
        if tn > 0.5:
            return t
    return t


def davidson(kron: dict, out: Path, order: int, radial: int | None = None, x_cut: int = 10, logs: int = 1,
             log_x_cut: int = 20, log_lmin: int = 0, seed: Path | None = None, max_iterations: int = 220,
             max_subspace: int = 60, keep: int = 10, block_max: int = 8000,
             target_residual: float = 1e-30, energy_tolerance: float = 1e-64, min_iterations: int = 5) -> dict:
    started = perf_counter()
    set_bits(800)
    mp.mp.dps = DPS                                    # own the precision; callers may change it
    rows, ang, mask = build_mask(kron, order, radial, x_cut, logs, log_x_cut, log_lmin)
    sp = Space(mask, P.KronApply(kron, mask.shape[0], mask.shape[1], "h"))
    pre = Precond(kron, mask, ang, block_max)
    print(f"space: order {order}, radial {radial or order} (x < {x_cut}), logs {logs} (x < {log_x_cut}, "
          f"l >= {log_lmin}): N = {int(mask.sum())}, grid {mask.shape[0]}x{mask.shape[1]}, "
          f"{len(pre.blocks)} preconditioner blocks; setup {perf_counter() - started:.0f}s", flush=True)
    if seed is not None:
        s = sp.gather(seed_grid(seed, rows, ang, mask))
    else:                                              # the white function ~ e^{-s}; fine for small spaces
        s = np.zeros((P.D, int(mask.sum())), dtype=np.int64)
        s[P.E, 0] = 1
    V = [P.combine([s], [1 / gmpy2.sqrt(P.dots(s, [s])[0])])]
    HV = [sp.apply(V[0])]
    G = [[mp.mpf(str(P.dots(V[0], HV)[0]))]]
    history, e_prev, small, restarted = [], None, 0, False
    for it in range(1, max_iterations + 1):
        k = len(V)
        w, U = mp.eigsy(mp.matrix(G))
        idx = min(range(k), key=lambda i: w[i])
        theta = w[idx]
        y = [mpfr(str(U[i, idx])) for i in range(k)]
        x, hx = P.combine(V, y), P.combine(HV, y)
        r = P.combine([hx, x], [mpfr(1), -mpfr(str(theta))])
        rnorm = gmpy2.sqrt(P.dots(r, [r])[0])
        de = None if e_prev is None else theta - e_prev
        e_prev = theta
        history.append({"iteration": it, "energy": mp.nstr(theta, 90), "residual": float(rnorm), "k": k})
        print(f"iteration {it}: k={k} E={mp.nstr(theta, 64)} residual={float(rnorm):.3e}"
              + ("" if de is None else f" dE={mp.nstr(de, 4)}") + f" ({perf_counter() - started:.0f}s)", flush=True)
        if de is not None and not restarted:
            small = small + 1 if abs(de) < energy_tolerance else 0
        restarted = False
        if float(rnorm) < target_residual or (small >= 3 and it > min_iterations):
            break
        u = pre(sp.to_float_grid(r), float(theta))
        t = _orthonormalize(sp.from_float_grid(u / np.linalg.norm(u)), V)
        if t is None:
            break
        if k >= max_subspace:                              # thick restart: Ritz vector + last `keep` vectors
            kept = list(zip(V[-keep:], HV[-keep:])) if keep else []
            xn = gmpy2.sqrt(P.dots(x, [x])[0])
            V, HV = [P.combine([x], [1 / xn])], [P.combine([hx], [1 / xn])]
            for v, hv in kept:                             # H v carried by the same combinations: no new apply
                for _ in range(2):
                    c = [-z for z in P.dots(v, V)]
                    v, hv = P.combine([v] + V, [mpfr(1)] + c), P.combine([hv] + HV, [mpfr(1)] + c)
                vn = gmpy2.sqrt(P.dots(v, [v])[0])
                if vn > 1e-3:
                    V.append(P.combine([v], [1 / vn]))
                    HV.append(P.combine([hv], [1 / vn]))
            G = [[mp.mpf(str(z)) for z in P.dots(h, V)] for h in HV]
            restarted = True
            t = _orthonormalize(t, V)
            if t is None:
                break
        v_new = t
        h_new = sp.apply(v_new)
        col = [mp.mpf(str(c)) for c in P.dots(h_new, V + [v_new])]
        for i in range(len(V)):
            G[i].append(col[i])
        G.append(col)
        V.append(v_new)
        HV.append(h_new)
    meta = {"order": order, "radial": radial or order, "x_cut": x_cut, "logs": logs, "log_x_cut": log_x_cut,
            "log_lmin": log_lmin, "N": int(mask.sum()), "energy": mp.nstr(theta, 100),
            "elapsed_s": perf_counter() - started, "history": history}
    save_state(out, sp, x, rows, order, theta, meta)
    return meta

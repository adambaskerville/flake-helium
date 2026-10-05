"""flake: certified high-precision helium ground-state energies on a laptop.

  flake build   --order 30 --radial 30 --out f30.npz          # exact white Kronecker operator
  flake solve   --kron f30.npz --order 30 --out runs/f30      # Davidson on any masked space
  flake certify runs/f30                                      # independent exact Rayleigh quotient
  flake check                                                 # fast self-checks of every layer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _check() -> None:
    import gmpy2
    import mpmath as mp
    import numpy as np
    from gmpy2 import mpfr

    from . import planes as P
    from .certify import monomial_box, rayleigh
    from .exact import radial_moment, set_bits
    from .operator import build, load
    from .solver import build_mask

    # 1. radial moments vs numerical quadrature
    set_bits(200)
    with mp.workdps(40):
        for n, p in ((0, 0), (3, 1), (5, 2), (7, 3)):
            ref = mp.quad(lambda s: s ** n * mp.log(s) ** p * mp.exp(-2 * s), [0, 1, mp.inf])
            assert abs(mp.mpf(str(radial_moment(n, p))) - ref) < mp.mpf(10) ** -30, (n, p)
    print("ok  radial moments (closed form) match quadrature")

    # 2. digit-plane vector algebra is exact, including large integer parts
    rng = np.random.default_rng(0)
    vals = [np.array([int(v) for v in rng.integers(-(1 << 62), 1 << 62, 3000)], dtype=object) << (P.T - 62)
            for _ in range(2)]
    vals[1] = vals[1] * 10 ** 15
    vecs = [P.from_pyints(v) for v in vals]
    assert all((P.to_pyints(a) == v).all() for a, v in zip(vecs, vals))
    exact = mpfr(sum(int(a) * int(b) for a, b in zip(*vals))) / mpfr(2) ** (2 * P.T)
    assert abs(P.dots(vecs[0], [vecs[1]])[0] - exact) <= abs(exact) * mpfr(2) ** -250
    got = P.to_pyints(P.combine(vecs, [mpfr("0.3"), mpfr("-1.7")]))
    ci = [int(gmpy2.rint(mpfr(c) * mpfr(2) ** P.T)) for c in ("0.3", "-1.7")]
    ref = [(int(a) * ci[0] + int(b) * ci[1] + (1 << (P.T - 1))) >> P.T for a, b in zip(*vals)]
    assert max(abs(int(g) - r) for g, r in zip(got, ref)) <= 4
    print("ok  digit-plane dots and combinations exact (values up to 1e15)")

    # 3. Kronecker operator (fast apply) == independent Box quadratic form, with a log tower
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "k.npz"
        build(6, 9, 2, None, path, bits_radial=800, bits_angular=800, workers=2, verbose=False)
        kron = load(path, 800)
    rows, ang, mask = build_mask(kron, 6, radial=9, x_cut=3, logs=2, log_x_cut=4)
    c = np.zeros(mask.shape, dtype=object)
    for r, a in zip(*np.nonzero(mask)):
        c[r, a] = int(rng.integers(-(1 << 40), 1 << 40)) << (P.T - 40)
    h_fast = sum(int(x) * int(y) for x, y in zip(c[mask], P.KronApply(kron, *mask.shape, "h")(c)[mask]))
    s_fast = sum(int(x) * int(y) for x, y in zip(c[mask], P.KronApply(kron, *mask.shape, "s")(c)[mask]))
    set_bits(800)
    grid = np.vectorize(lambda z: mpfr(int(z)) / mpfr(2) ** P.T, otypes=[object])(c)
    box, value = monomial_box(grid, rows, 6)
    h, s = rayleigh(box, value, workers=2)
    e_fast = mpfr(h_fast) / mpfr(s_fast)
    rel = abs(e_fast - h / s) / abs(h / s)
    assert rel < mpfr(10) ** -60, float(rel)
    print(f"ok  Kronecker apply agrees with the independent certifier (relative difference {float(rel):.1e})")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="flake", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build the exact white Kronecker operator")
    b.add_argument("--order", type=int, required=True, help="largest angular order x = m + n")
    b.add_argument("--radial", type=int, required=True, help="largest radial degree l")
    b.add_argument("--logs", type=int, default=1, help="largest power p of ln s (Schwartz: 1)")
    b.add_argument("--log3-lmax", type=int, default=80, help="cap l for (ln s)^p, p >= 3")
    b.add_argument("--bits", type=int, default=3000, help="precision of the radial whitening")
    b.add_argument("--angular-bits", type=int, default=640, help="precision of the angular whitening")
    b.add_argument("--workers", type=int, default=6)
    b.add_argument("--out", type=Path, required=True)

    s = sub.add_parser("solve", help="lowest eigenvalue on a (masked) space")
    s.add_argument("--kron", type=Path, required=True)
    s.add_argument("--order", type=int, required=True, help="Schwartz shells l + x <= order")
    s.add_argument("--radial", type=int, default=None, help="l + x <= radial for x < x-cut (default: order)")
    s.add_argument("--x-cut", type=int, default=10)
    s.add_argument("--logs", type=int, default=1, help="largest (ln s) power used (<= build)")
    s.add_argument("--log-x-cut", type=int, default=20, help="(ln s)^p, p >= 2, only for x < this")
    s.add_argument("--log-lmin", type=int, default=0, help="(ln s)^p, p >= 2, only for l >= this")
    s.add_argument("--seed", type=Path, default=None, help="previous run dir (same build) to start from")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--max-iterations", type=int, default=220)
    s.add_argument("--max-subspace", type=int, default=60)
    s.add_argument("--keep", type=int, default=10, help="basis vectors kept at a restart")
    s.add_argument("--block-max", type=int, default=8000, help="preconditioner block size (rows)")
    s.add_argument("--target-residual", type=float, default=1e-30)
    s.add_argument("--energy-tolerance", type=float, default=1e-64)
    s.add_argument("--certify", action="store_true", help="certify the result afterwards")

    c = sub.add_parser("certify", help="independent exact Rayleigh quotient of a solved state")
    c.add_argument("run", type=Path)
    c.add_argument("--bits", type=int, default=2000)
    c.add_argument("--workers", type=int, default=6)

    sub.add_parser("check", help="fast self-checks of every layer (seconds)")
    a = p.parse_args(argv)

    if a.cmd == "build":
        from .operator import build
        build(a.order, a.radial, a.logs, a.log3_lmax, a.out, a.bits, a.angular_bits, a.workers)
    elif a.cmd == "solve":
        from .operator import load
        from .solver import davidson
        meta = davidson(load(a.kron), a.out, a.order, a.radial, a.x_cut, a.logs, a.log_x_cut, a.log_lmin,
                        a.seed, a.max_iterations, a.max_subspace, a.keep, a.block_max,
                        a.target_residual, a.energy_tolerance)
        print(f"E = {meta['energy'][:70]}  (N = {meta['N']}, {meta['elapsed_s']:.0f}s) -> {a.out}")
        if a.certify:
            from .certify import certify
            print(json.dumps(certify(a.out), indent=1))
    elif a.cmd == "certify":
        from .certify import certify
        print(json.dumps(certify(a.run, a.bits, a.workers), indent=1))
    else:
        _check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

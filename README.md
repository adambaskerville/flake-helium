# FLAKE: helium to 60 significant figures on a laptop

**FLAKE** (Fock-Log Anisotropic Kronecker Eigensolver) computes certified variational upper bounds on the
non-relativistic, infinite-nuclear-mass ground-state energy of helium. On a MacBook Pro
(Apple M4 Pro, 24 GB) it gives

```
E0 = -2.90372437703411959831115924519440444669692531111729033904234...  hartree
```

That is **60 significant figures** (59 decimal places). The best previous calculation
(Schwartz, 2006) is correct to 44. The wrote a comprehensive [blog post](https://adambaskerville.github.io/posts/flake-Helium-Deep-Dive/) on how this works (be warned, it is long!).

The whole method ended up only being ~ 1,100 lines of Python on top of NumPy, SciPy, gmpy2 and mpmath.
It needs no GPU, no compiled extensions and no multiprecision linear-algebra library. I wrote a bunch of highly optimised eigen3 C++ code which was not needed, but I will add this into a separate repository at some point as there are some goodies in there (at least I think so anyway).

## Install

```bash
pip install -e .          # Python >= 3.10; installs numpy, scipy, gmpy2, mpmath
flake check                # self-checks of every layer, a few seconds
python tests/test_flake.py # adds an end-to-end F10 check against an independent code (~1 min)
```

## Quick start: Schwartz's F30 space in about six minutes

```bash
flake build   --order 30 --radial 30 --out f30.npz              # exact operator, ~30 s
flake solve   --kron f30.npz --order 30 --out runs/f30 --certify  # ~5 min
```

This prints `E = -2.90372437703411959831115924518995689434265920368...` (N = 5,712), then
the certificate, an independent exact Rayleigh quotient that agrees with the solver to about 1e-83.

## The four commands

| command | what it does |
|---|---|
| `flake build` | Builds the Hamiltonian as 27 Kronecker products in an exactly orthonormalised ("white") basis. `--order` is the largest angular order, `--radial` the largest radial degree, `--logs` the largest power of ln s. It runs once, and the result serves every smaller space. |
| `flake solve` | Davidson for the lowest eigenvalue on any sub-space of a build. `--order`: Schwartz shells l + x ≤ order. `--radial`: extra radial degree for low angular order x < `--x-cut`. `--logs`: (ln s)^p rows for x < `--log-x-cut`. `--seed`: start from a previous run in the same build. `--certify`: certify at the end. |
| `flake certify` | Independent certification of a run: recomputes the orthonormalisation exactly, maps the result back to an explicit function, and evaluates its Rayleigh quotient at 2000 bits. The result is a rigorous upper bound. |
| `flake check` | Fast self-checks: closed-form integrals against quadrature, exactness of the digit-plane arithmetic, and the Kronecker operator against the certifier. |

Each run directory holds `state.npz` (the exact coefficient vector and the space),
`progress.json` (settings and iteration history) and, after certification, `certificate.json`.

## Reproducing the record

[`examples/reproduce_record.sh`](examples/reproduce_record.sh) runs the full chain in one build:

1. the isotropic Schwartz ladder F30 → F62, each rung seeded by the last;
2. the radial extension for low angular order, with the (ln s)² tower;
3. angular order 62 → 66 → 70, then radial degree 118, and the (ln s)³ variant.

The chain takes roughly a day on an M4 Pro and peaks at about 11 GB of memory. The final
certified bounds should match:

| space | N | certified energy |
|---|---|---|
| Ω = 70, radial 110, (ln s)³ | 80,437 | −2.903724377034119598311159245194404446696925311117290339042345757212… |
| Ω = 70, radial 118, (ln s)² | 74,247 | −2.903724377034119598311159245194404446696925311117290339042346702569… |

## How it works, in five lines

1. **Box coordinates.** In s = r₁ + r₂, ξ = r₁₂/s and η = (r₁ − r₂)/r₁₂, the triangular domain becomes a box and every Schwartz function factorises into radial × angular, so S = S_rad ⊗ S_ang exactly.
2. **Kronecker Hamiltonian.** The gradient-form energy has polynomial weights, so H = Σ₂₇ c_j R_j ⊗ A_j. It is applied as Σ R_j C A_jᵀ on a coefficient grid and never formed.
3. **Exact whitening.** Inverse Cholesky factors of the two small Grams are computed in 3000-bit arithmetic, so S = I and no cancellation is left in the big computation.
4. **Digit-plane BLAS.** About 280-bit fixed-point products are done as exact float64 GEMMs on 21-bit slices (the Ozaki scheme), and dot products are exact integer digit arithmetic.
5. **Davidson + certification.** The preconditioner uses exact float64 blocks of merged angular sectors with a regularised Cholesky. Every result is then re-evaluated by independent exact code.

## Layout

```
flake/exact.py      closed-form integrals, Grams, exact inverse Cholesky, limb storage
flake/operator.py   the 27-term Kronecker Hamiltonian in the white basis (build / load)
flake/planes.py     exact digit-plane matrix products and vector algebra on float64 BLAS
flake/solver.py     spaces (masks), preconditioner, Davidson, seeding, run state
flake/certify.py    independent exact Rayleigh quotient (the certificate)
flake/cli.py        the `flake` command
```

## Performance notes

- BLAS threads: the apply is dominated by float64 GEMMs, so let Accelerate/OpenBLAS use all cores. When running two jobs at once, limit each one, for example `VECLIB_MAXIMUM_THREADS=6`.
- Memory is dominated by the preconditioner blocks (`--block-max`, default 8000 rows, 0.5 GB each) and the build (about 3 GB at angular order 70, radial degree 120).
- Seeds must come from the **same build**. The orthonormal radial functions depend on the build's radial ordering, so a state from another build is only approximately the same function. `solve` warns when that happens.

## Validation

- `flake check`: closed-form radial moments agree with quadrature to 1e-30. The digit-plane arithmetic is exact, including values around 1e15. The fast Kronecker energy agrees with the independent certifier to about 1e-86 (relative), on a space with a (ln s)² tower.
- F10 (322 functions) agrees with an earlier, completely independent oct-double C++ implementation to all 41 published digits.
- The certifier reproduces the production F56 certificate to all 80 printed digits. Seeded with that state, the solver starts exactly at its energy.

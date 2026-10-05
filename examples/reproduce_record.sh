#!/bin/sh
# Reproduce the 60-significant-figure helium bound (roughly a day on an Apple M4 Pro, <= ~11 GB).
set -e
K=kron_a70_r120_l3.npz
R=runs

# 1. One exact build: angular order 70, radial degree 120, (ln s)^p up to p = 3 (p = 3 only for l <= 80).
[ -f $K ] || flake build --order 70 --radial 120 --logs 3 --log3-lmax 80 --out $K

# 2. Isotropic Schwartz ladder, each rung seeded by the previous one.
flake solve --kron $K --order 30 --out $R/f30
prev=$R/f30
for o in 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62; do
  flake solve --kron $K --order $o --seed $prev --out $R/f$o
  prev=$R/f$o
done

# 3. Extra radial degree for low angular order (x < 10), with the (ln s)^2 tower for x < 20.
flake solve --kron $K --order 62 --radial 90  --logs 2 --seed $R/f62  --out $R/a62_r90
flake solve --kron $K --order 62 --radial 110 --logs 2 --seed $R/a62_r90 --out $R/b62 --certify

# 4. Angular order 62 -> 66 -> 70, then radial degree 118 and the (ln s)^3 variant.
flake solve --kron $K --order 66 --radial 110 --logs 2 --seed $R/b62 --out $R/b66
flake solve --kron $K --order 70 --radial 110 --logs 2 --seed $R/b66 --out $R/b70 --certify
flake solve --kron $K --order 70 --radial 118 --logs 2 --seed $R/b70 --out $R/b70_r118 --certify
flake solve --kron $K --order 70 --radial 110 --logs 3 --seed $R/b70 --out $R/b70_l3 --certify

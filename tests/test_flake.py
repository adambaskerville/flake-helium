"""End-to-end checks: every layer, then F10 against an independent oct-double implementation.

Run with `python -m pytest tests` or `python tests/test_flake.py` (about a minute).
"""

import json
import tempfile
from decimal import Decimal
from pathlib import Path

from flake.certify import certify
from flake.cli import _check
from flake.operator import build, load
from flake.solver import davidson

# F10 (322 functions) from the earlier, completely independent oct-double C++ code
F10_REFERENCE = Decimal("-2.90372437703306886109375820976962659811947")


def test_layers():
    _check()


def test_f10_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        build(10, 10, 1, None, tmp / "k.npz", workers=2, verbose=False)
        meta = davidson(load(tmp / "k.npz"), tmp / "f10", order=10)
        cert = certify(tmp / "f10", workers=2)
    assert meta["N"] == 322
    assert abs(Decimal(cert["certified_energy"]) - F10_REFERENCE) < Decimal("1e-40")
    assert abs(float(cert["certified_minus_solver"])) < 1e-75
    print(json.dumps(cert, indent=1))


if __name__ == "__main__":
    test_layers()
    test_f10_end_to_end()
    print("all tests passed")

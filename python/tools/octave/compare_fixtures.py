#!/usr/bin/env python3
"""Compare a regenerated fixture directory with the committed one.

Array values are compared with a tolerance so a different BLAS build does not
fail on roundoff. The default pytest suite instead checks MANIFEST.json hashes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fixture_io import EXPECTED_MATS, load_fixture

RTOL = 1e-8
ATOL = 1e-8


def _compare_file(reference: Path, candidate: Path) -> None:
    left = load_fixture(reference)
    right = load_fixture(candidate)
    if set(left) != set(right):
        raise SystemExit(f"{reference.name}: keys differ {sorted(set(left) ^ set(right))}")
    for key in sorted(left):
        a = left[key]
        b = right[key]
        if a.shape != b.shape:
            raise SystemExit(f"{reference.name}:{key} shape {a.shape} != {b.shape}")
        try:
            np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)
        except AssertionError as exc:
            raise SystemExit(f"{reference.name}:{key}\n{exc}") from exc


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args(argv)
    for name in EXPECTED_MATS:
        ref = args.reference / name
        cand = args.candidate / name
        if not ref.is_file() or not cand.is_file():
            raise SystemExit(f"missing {name}")
        _compare_file(ref, cand)
    print(f"matched {len(EXPECTED_MATS)} fixtures within rtol={RTOL} atol={ATOL}")


if __name__ == "__main__":
    main()

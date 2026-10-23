#!/usr/bin/env python3
"""Regenerate the Octave reference fixtures and their manifest.

Requires Octave (the committed fixtures were produced with 8.4). From the
repository root:

    python python/tools/octave/export_fixtures.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from fixture_io import EXPECTED_MATS, fingerprint


def _octave_version(executable: str) -> str:
    output = subprocess.check_output([executable, "--version"], text=True)
    for line in output.splitlines():
        if "version" not in line.lower():
            continue
        for token in line.replace(",", " ").split():
            if token[:1].isdigit():
                return token
    raise SystemExit(f"could not parse Octave version from:\n{output}")


def _write_source(directory: Path, version: str) -> None:
    text = "\n".join(
        [
            "schema: 1",
            f"octave: {version}",
            "generator: python/tools/octave/export_fixtures.m",
            "mat_format: v7",
            "rng: Octave rng(seed) before each simulator and each bug-impact trial",
            "seeds: plane 42, iros 43, rcim2020 44, rcim2022 45, meas2023 46",
            "excluded: plane_TRO2016 (Octave has no lsqnonlin)",
            "bug_line: fun/calib/line_RCIM2022.m:58",
            "bug_sim: fun/sim/simMEAS2023.m:90",
            "shims: python/tools/octave/shims/*.m.in copied to a temporary directory at runtime",
            "note: MAT headers contain a timestamp; MANIFEST.json hashes array bytes",
            "",
        ]
    )
    (directory / "SOURCE.txt").write_text(text)


def _write_manifest(directory: Path) -> None:
    files = {}
    for name in EXPECTED_MATS:
        path = directory / name
        files[name] = {"sha256": fingerprint(path), "bytes": path.stat().st_size}
    document = {"schema": 1, "files": files}
    (directory / "MANIFEST.json").write_text(json.dumps(document, indent=2) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        help="output directory (default: python/tests/fixtures)",
    )
    parser.add_argument("--octave", default="octave", help="Octave executable")
    args = parser.parse_args(argv)

    if shutil.which(args.octave) is None:
        raise SystemExit(f"Octave executable not found: {args.octave}")

    repo = Path(__file__).resolve().parents[3]
    out = (args.out or (repo / "python" / "tests" / "fixtures")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    octave_dir = Path(__file__).resolve().parent
    expression = f"addpath('{octave_dir.as_posix()}'); export_fixtures('{out.as_posix()}')"
    subprocess.run(
        [args.octave, "--no-gui", "--quiet", "--eval", expression],
        check=True,
        cwd=repo,
    )
    found = {path.name for path in out.glob("*.mat")}
    expected = set(EXPECTED_MATS)
    if found != expected:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        raise SystemExit(f"fixture set mismatch; missing={missing} extra={extra}")
    _write_manifest(out)
    _write_source(out, _octave_version(args.octave))
    print(f"wrote {len(EXPECTED_MATS)} fixtures to {out}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)

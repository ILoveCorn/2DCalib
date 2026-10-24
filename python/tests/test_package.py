from __future__ import annotations

import re
from pathlib import Path

import handeye2d


def test_version_matches_pyproject(python_root: Path) -> None:
    text = (python_root / "pyproject.toml").read_text()
    match = re.search(r'(?m)^version = "([^"]+)"', text)
    assert match is not None
    assert match.group(1) == handeye2d.__version__ == "0.0.0"


def test_package_exports_the_foundation_modules() -> None:
    exported = sorted(name for name in vars(handeye2d) if not name.startswith("_"))
    assert exported == ["calib", "fitting", "io", "metrics", "rotations", "so3"]

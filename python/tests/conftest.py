from __future__ import annotations

import sys
from pathlib import Path

import pytest

PYTHON_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PYTHON_ROOT.parent
TOOLS = PYTHON_ROOT / "tools" / "octave"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

FIXTURES = PYTHON_ROOT / "tests" / "fixtures"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def python_root() -> Path:
    return PYTHON_ROOT


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES

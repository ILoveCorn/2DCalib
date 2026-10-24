"""Column-major ``randn`` / ``rand`` draws.

MATLAB fills the first subscript fastest. A NumPy ``reshape(..., order="F")``
of one flat draw matches that order. The stream itself is still NumPy's.
"""

from __future__ import annotations

import numpy as np


def draw(rng: np.random.Generator, shape: tuple[int, ...], kind: str = "normal") -> np.ndarray:
    """Draw ``shape`` in MATLAB column-major order."""
    count = int(np.prod(shape, dtype=int))
    values = rng.normal(size=count) if kind == "normal" else rng.random(size=count)
    return values.reshape(shape, order="F")

"""Convert MATLAB page arrays to the batch-first layout used by handeye2d."""

from __future__ import annotations

import numpy as np


def vectors(array: np.ndarray) -> np.ndarray:
    """``(3, 1)`` or ``(3, k, m)`` to ``(3,)`` or ``(m, k, 3)``."""
    values = np.asarray(array, dtype=float)
    if values.ndim == 1:
        return values
    if values.shape[0] != 3:
        raise ValueError(f"expected a leading axis of 3, got {values.shape}")
    if values.ndim == 2:
        if values.shape[1] == 1:
            return values[:, 0]
        return values.T
    moved = np.moveaxis(values, 0, -1)
    # (k, m, 3) -> (m, k, 3). A singleton k is dropped.
    ordered = np.swapaxes(moved, 0, 1)
    if ordered.shape[1] == 1 and ordered.ndim == 3:
        return ordered[:, 0, :]
    return ordered


def rotations(array: np.ndarray) -> np.ndarray:
    """``(3, 3)`` or ``(3, 3, ...)`` to ``(..., 3, 3)``."""
    values = np.asarray(array, dtype=float)
    if values.shape[:2] != (3, 3):
        raise ValueError(f"expected rotation pages, got {values.shape}")
    if values.ndim == 2:
        return values
    return np.moveaxis(values, (0, 1), (-2, -1))


def points(array: np.ndarray) -> np.ndarray:
    """``(3, n, m)`` profile pages to ``(m, n, 3)``."""
    values = np.asarray(array, dtype=float)
    if values.ndim != 3 or values.shape[0] != 3:
        raise ValueError(f"expected (3, n, m) points, got {values.shape}")
    return np.transpose(values, (2, 1, 0))

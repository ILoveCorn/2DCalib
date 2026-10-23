"""Numeric checks for fixture tests.

These helpers are test oracles. They are not the handeye2d algorithm API.
"""

from __future__ import annotations

import numpy as np


def pose_error(
    translation: np.ndarray,
    rotation: np.ndarray,
    translation_est: np.ndarray,
    rotation_est: np.ndarray,
) -> tuple[float, float]:
    """Return position error in millimeters and rotation error in radians."""
    delta_t = np.reshape(np.asarray(translation, float), (3,)) - np.reshape(
        np.asarray(translation_est, float), (3,)
    )
    estimated = np.reshape(np.asarray(rotation_est, float), (3, 3))
    truth = np.reshape(np.asarray(rotation, float), (3, 3))
    cosine = float(np.clip((np.trace(estimated.T @ truth) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.linalg.norm(delta_t)), float(np.arccos(cosine))


def assert_rotations(rotation: np.ndarray, atol: float = 1e-8) -> None:
    array = np.asarray(rotation, float)
    if array.ndim == 2:
        array = array[:, :, None]
    assert array.shape[0] == 3 and array.shape[1] == 3
    flat = array.reshape(3, 3, -1)
    eye = np.eye(3)
    for index in range(flat.shape[2]):
        matrix = flat[:, :, index]
        np.testing.assert_allclose(matrix.T @ matrix, eye, atol=atol)
        np.testing.assert_allclose(np.linalg.det(matrix), 1.0, atol=atol)


def assert_finite(data: dict[str, np.ndarray]) -> None:
    for key, value in data.items():
        if np.issubdtype(value.dtype, np.number):
            assert np.isfinite(value).all(), key


def matlab_round(value: float) -> int:
    """MATLAB round (half away from zero) for the positive counts used here."""
    return int(np.floor(value + 0.5))

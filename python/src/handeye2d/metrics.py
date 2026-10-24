"""Calibration error.

``calibration_error(t, R, te, Re)`` matches ``calibrationError``: the Euclidean
position error in millimeters, and the angle of ``Re.T @ R`` in degrees, in
``[0, 180]``.
"""

from __future__ import annotations

import numpy as np

from handeye2d.rotations import rotm_to_angle


def calibration_error(
    t: np.ndarray,
    R: np.ndarray,
    te: np.ndarray,
    Re: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Position error in mm and orientation error in degrees.

    A single pose returns length-1 arrays squeezed to NumPy scalars when the
    inputs have no batch axis. Batched inputs ``t`` of shape ``(..., 3)`` and
    ``R`` of shape ``(..., 3, 3)`` return one error per pose.
    """
    truth_t = _vectors(t)
    est_t = _vectors(te)
    truth_R = _rotations(R)
    est_R = _rotations(Re)
    position = np.linalg.norm(truth_t - est_t, axis=-1)
    delta = np.swapaxes(est_R, -1, -2) @ truth_R
    orientation = np.rad2deg(rotm_to_angle(delta))
    if position.shape == ():
        return position, orientation
    return position, orientation


def _vectors(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape == (3, 1):
        return array[:, 0]
    if array.shape[-1] != 3:
        raise ValueError(f"expected a vector (..., 3), got {array.shape}")
    return array


def _rotations(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape[-2:] != (3, 3):
        raise ValueError(f"expected a rotation (..., 3, 3), got {array.shape}")
    return array

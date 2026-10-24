"""Rotation conversions used by the MATLAB code.

Sequences follow the Robotics System Toolbox intrinsic convention,
``R = R(e1) R(e2) R(e3)``, which is SciPy's uppercase sequence. Quaternions
are scalar-first ``[w, x, y, z]`` and are normalized. Axes passed to
:func:`axang_to_rotm` are normalized explicitly, because callers pass
``rand(1, 3)``.

Angles are radians. Batch axes come first: an Euler triple is ``(..., 3)`` and
a rotation matrix is ``(..., 3, 3)``.
"""

from __future__ import annotations

import numpy as np

from handeye2d.so3 import hat

_AXES = frozenset("XYZ")
_GIMBAL = 1e-8


def quat_to_rotm(q_wxyz: np.ndarray) -> np.ndarray:
    """Rotation matrix of a scalar-first quaternion. A zero quaternion is ``I``."""
    quat = np.asarray(q_wxyz, dtype=float)
    if quat.shape[-1] != 4:
        raise ValueError(f"quat_to_rotm expected (..., 4), got {quat.shape}")
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    zero = norm < np.finfo(float).eps
    quat = quat / np.where(zero, 1.0, norm)
    w = quat[..., 0]
    x = quat[..., 1]
    y = quat[..., 2]
    z = quat[..., 3]
    row0 = np.stack((1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)), -1)
    row1 = np.stack((2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)), -1)
    row2 = np.stack((2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)), -1)
    rotation = np.stack((row0, row1, row2), axis=-2)
    return np.where(zero[..., None], np.eye(3), rotation)


def eul_to_rotm(angles: np.ndarray, seq: str = "ZYX") -> np.ndarray:
    """Intrinsic Euler angles to a rotation matrix. Default sequence is ``ZYX``."""
    euler = np.asarray(angles, dtype=float)
    if euler.shape[-1] != 3:
        raise ValueError(f"eul_to_rotm expected (..., 3), got {euler.shape}")
    sequence = seq.upper()
    if len(sequence) != 3 or any(axis not in _AXES for axis in sequence):
        raise ValueError(f"sequence must be three of X, Y, Z, got {seq!r}")
    rotation = _elementary(sequence[0], euler[..., 0])
    rotation = rotation @ _elementary(sequence[1], euler[..., 1])
    return rotation @ _elementary(sequence[2], euler[..., 2])


def rotm_to_eul(rotation: np.ndarray, seq: str = "ZYX") -> np.ndarray:
    """``rotm2eul`` for ``ZYX`` (default) and ``ZXY``.

    At a gimbal lock (``hypot`` of the relevant pair at or below ``1e-8``) the
    third angle is zero, matching the Octave shim and SciPy.
    """
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(f"rotm_to_eul expected (..., 3, 3), got {rotation.shape}")
    sequence = seq.upper()
    if sequence == "ZYX":
        return _eul_zyx(rotation)
    if sequence == "ZXY":
        return _eul_zxy(rotation)
    raise ValueError(f"rotm_to_eul implements ZYX and ZXY, got {seq!r}")


def axang_to_rotm(axis: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Axis-angle to a rotation matrix. ``axis`` is normalized; a zero axis is ``I``.

    ``theta`` is the angle in radians, not the norm of ``axis``. It broadcasts
    over the leading axes of ``axis``.
    """
    axis = np.asarray(axis, dtype=float)
    angle = np.asarray(theta, dtype=float)
    if axis.shape[-1] != 3:
        raise ValueError(f"axang_to_rotm expected axis (..., 3), got {axis.shape}")
    norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    zero = norm < np.finfo(float).eps
    unit = axis / np.where(zero, 1.0, norm)
    skew = hat(unit)
    sine = np.sin(angle)
    cosine = np.cos(angle)
    rotation = (
        np.eye(3) + sine[..., None, None] * skew + (1.0 - cosine)[..., None, None] * (skew @ skew)
    )
    return np.where(zero[..., None], np.eye(3), rotation)


def rotm_to_angle(rotation: np.ndarray) -> np.ndarray:
    """Rotation angle in ``[0, π]``, the only part of ``rotm2axang`` the metrics use."""
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(f"rotm_to_angle expected (..., 3, 3), got {rotation.shape}")
    trace = np.trace(rotation, axis1=-2, axis2=-1)
    cosine = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cosine)


def _elementary(axis: str, angle: np.ndarray) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    ones = np.ones_like(cosine)
    zeros = np.zeros_like(cosine)
    if axis == "X":
        rows = (
            np.stack((ones, zeros, zeros), axis=-1),
            np.stack((zeros, cosine, -sine), axis=-1),
            np.stack((zeros, sine, cosine), axis=-1),
        )
    elif axis == "Y":
        rows = (
            np.stack((cosine, zeros, sine), axis=-1),
            np.stack((zeros, ones, zeros), axis=-1),
            np.stack((-sine, zeros, cosine), axis=-1),
        )
    else:
        rows = (
            np.stack((cosine, -sine, zeros), axis=-1),
            np.stack((sine, cosine, zeros), axis=-1),
            np.stack((zeros, zeros, ones), axis=-1),
        )
    return np.stack(rows, axis=-2)


def _eul_zyx(rotation: np.ndarray) -> np.ndarray:
    # R = Rz(z) Ry(y) Rx(x), angles [z, y, x]
    sy = -rotation[..., 2, 0]
    cy = np.hypot(rotation[..., 0, 0], rotation[..., 1, 0])
    gimbal = cy <= _GIMBAL
    z = np.where(
        gimbal,
        np.arctan2(-rotation[..., 0, 1], rotation[..., 1, 1]),
        np.arctan2(rotation[..., 1, 0], rotation[..., 0, 0]),
    )
    y = np.arctan2(sy, cy)
    x = np.where(gimbal, 0.0, np.arctan2(rotation[..., 2, 1], rotation[..., 2, 2]))
    return np.stack((z, y, x), axis=-1)


def _eul_zxy(rotation: np.ndarray) -> np.ndarray:
    # R = Rz(z) Rx(x) Ry(y), angles [z, x, y]
    cx = np.hypot(rotation[..., 0, 1], rotation[..., 1, 1])
    gimbal = cx <= _GIMBAL
    x = np.where(gimbal, np.arctan2(rotation[..., 2, 1], 0.0), np.arctan2(rotation[..., 2, 1], cx))
    z = np.where(
        gimbal,
        np.arctan2(rotation[..., 1, 0], rotation[..., 0, 0]),
        np.arctan2(-rotation[..., 0, 1], rotation[..., 1, 1]),
    )
    y = np.where(gimbal, 0.0, np.arctan2(-rotation[..., 2, 0], rotation[..., 2, 2]))
    return np.stack((z, x, y), axis=-1)

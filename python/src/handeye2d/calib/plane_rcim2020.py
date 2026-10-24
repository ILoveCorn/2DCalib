"""Plane hand-eye calibration from Sharifzadeh, RCIM 2020.

Iterative linear least squares with an SVD projection onto SO(3). A 1×1 page
in the MATLAB ``pagemtimes`` is scalar multiplication; here that is ``n' R``
scaled by the OCT ``x`` or ``z`` coordinate.

Inputs are batch-first: ``tH2Wn`` and ``pOn`` are ``(m, 3)``, ``RH2Wn`` is
``(m, 3, 3)``. The normal is the last right singular vector of the centered
world points. Its sign follows the SVD and can disagree with MATLAB.
"""

from __future__ import annotations

import time

import numpy as np

from handeye2d.calib._linalg import solve_hand_eye
from handeye2d.rotations import rotm_to_angle


def plane_rcim2020(
    tO2H: np.ndarray,
    RO2H: np.ndarray,
    tH2Wn: np.ndarray,
    RH2Wn: np.ndarray,
    pOn: np.ndarray,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return ``(tO2He, RO2He, n, time)``."""
    translation = np.asarray(tO2H, dtype=float).reshape(3).copy()
    rotation = np.asarray(RO2H, dtype=float).reshape(3, 3).copy()
    measured_t = np.asarray(tH2Wn, dtype=float)
    measured_R = np.asarray(RH2Wn, dtype=float)
    point = np.asarray(pOn, dtype=float)
    _count(measured_t, measured_R, point)
    elapsed = 0.0
    iteration = 1
    while True:
        started = time.perf_counter()
        world = np.einsum("mij,mj->mi", measured_R, point @ rotation.T + translation)
        world = world + measured_t
        normal, distance = _centered_normal(world)
        weighted = np.einsum("j,mji->mi", normal, measured_R)
        design = np.concatenate(
            (weighted * point[:, 0:1], weighted * point[:, 2:3], weighted), axis=1
        )
        target = distance - measured_t @ normal
        previous_t = translation
        previous_R = rotation
        rotation, translation = solve_hand_eye(design, target)
        elapsed += time.perf_counter() - started
        if _stop(iteration, translation, previous_t, rotation, previous_R, tol):
            break
        iteration += 1
    return translation, rotation, normal, elapsed


def _centered_normal(points: np.ndarray) -> tuple[np.ndarray, float]:
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1]
    return normal, float(normal @ center)


def _stop(
    iteration: int,
    translation: np.ndarray,
    previous_t: np.ndarray,
    rotation: np.ndarray,
    previous_R: np.ndarray,
    tol: float,
) -> bool:
    if iteration <= 1:
        return False
    angle = float(rotm_to_angle(rotation.T @ previous_R))
    step = float(np.hypot(np.linalg.norm(translation - previous_t), abs(angle)))
    return step < tol or iteration > 500


def _count(measured_t: np.ndarray, measured_R: np.ndarray, point: np.ndarray) -> None:
    if measured_t.ndim != 2 or measured_t.shape[1] != 3:
        raise ValueError(f"tH2Wn expected (m, 3), got {measured_t.shape}")
    count = int(measured_t.shape[0])
    if measured_R.shape != (count, 3, 3):
        raise ValueError(f"RH2Wn expected {(count, 3, 3)}, got {measured_R.shape}")
    if point.shape != (count, 3):
        raise ValueError(f"pOn expected {(count, 3)}, got {point.shape}")

"""Multi-plane hand-eye calibration from Carlson, IROS 2015.

Same update as ``plane_rcim2020``, with one normal per plane. Batch axes are
``(n_planes, m, ...)``: ``tH2Wn`` and ``pOn`` are ``(n_planes, m, 3)`` and
``RH2Wn`` is ``(n_planes, m, 3, 3)``. That is the MATLAB ``(3, 1, m, n_planes)``
array with the vector axis moved to the end and the plane axis moved to the front.

The design-matrix rows are ordered measurement-fast, plane-slow, matching
``reshape(..., m*n_planes, 9)`` of a column-major page array.
"""

from __future__ import annotations

import time

import numpy as np

from handeye2d.calib._linalg import solve_hand_eye
from handeye2d.rotations import rotm_to_angle


def plane_iros2015(
    tO2H: np.ndarray,
    RO2H: np.ndarray,
    tH2Wn: np.ndarray,
    RH2Wn: np.ndarray,
    pOn: np.ndarray,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return ``(tO2He, RO2He, n, time)``. ``n`` is ``(n_planes, 3)``."""
    translation = np.asarray(tO2H, dtype=float).reshape(3).copy()
    rotation = np.asarray(RO2H, dtype=float).reshape(3, 3).copy()
    measured_t = np.asarray(tH2Wn, dtype=float)
    measured_R = np.asarray(RH2Wn, dtype=float)
    point = np.asarray(pOn, dtype=float)
    n_plane, count = _shape(measured_t, measured_R, point)
    elapsed = 0.0
    iteration = 1
    normal = np.zeros((n_plane, 3))
    while True:
        started = time.perf_counter()
        world = np.einsum("ij,pmj->pmi", rotation, point) + translation
        world = np.einsum("pmij,pmj->pmi", measured_R, world) + measured_t
        center = world.mean(axis=1)
        distance = np.empty(n_plane)
        for index in range(n_plane):
            _, _, vh = np.linalg.svd(world[index] - center[index], full_matrices=False)
            normal[index] = vh[-1]
            distance[index] = float(normal[index] @ center[index])
        weighted = np.einsum("pj,pmji->pmi", normal, measured_R)
        design = np.concatenate(
            (weighted * point[..., 0:1], weighted * point[..., 2:3], weighted), axis=-1
        ).reshape(n_plane * count, 9)
        predicted = np.einsum("pmi,pi->pm", measured_t, normal)
        target = np.repeat(distance, count) - predicted.reshape(-1)
        previous_t = translation
        previous_R = rotation
        rotation, translation = solve_hand_eye(design, target)
        elapsed += time.perf_counter() - started
        if iteration > 1:
            angle = float(rotm_to_angle(rotation.T @ previous_R))
            step = float(np.hypot(np.linalg.norm(translation - previous_t), abs(angle)))
            if step < tol or iteration > 500:
                break
        iteration += 1
    return translation, rotation, normal, elapsed


def _shape(measured_t: np.ndarray, measured_R: np.ndarray, point: np.ndarray) -> tuple[int, int]:
    if measured_t.ndim != 3 or measured_t.shape[-1] != 3:
        raise ValueError(f"tH2Wn expected (n_planes, m, 3), got {measured_t.shape}")
    n_plane, count, _ = measured_t.shape
    if measured_R.shape != (n_plane, count, 3, 3):
        raise ValueError(f"RH2Wn expected {(n_plane, count, 3, 3)}, got {measured_R.shape}")
    if point.shape != (n_plane, count, 3):
        raise ValueError(f"pOn expected {(n_plane, count, 3)}, got {point.shape}")
    return n_plane, count

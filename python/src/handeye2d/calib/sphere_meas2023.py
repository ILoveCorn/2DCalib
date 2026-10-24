"""Closed-form sphere hand-eye calibration (Tang, MEAS 2023).

This is the one calibration routine pulled into the foundation milestone so
the saved real-data result can be checked. The other solvers are not in this
version. There is no ``_debug`` variant and no iteration count.

Inputs are batch-first, which is the column-major reshape of the MATLAB
arrays used by ``sphere_MEAS2023``:

* ``tH2Wn`` ``(m, mt, 3)``
* ``RH2Wn`` ``(m, 3, 3)`` — one orientation per group, not per position
* ``pcOe`` ``(m, mt, 3)`` — sphere centers in the OCT frame

``m`` is the number of orientations and ``mt`` is the number of positions of
each orientation. Both must be at least 2.
"""

from __future__ import annotations

import time

import numpy as np


def sphere_meas2023(
    tH2Wn: np.ndarray,
    RH2Wn: np.ndarray,
    pcOe: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(tO2He, RO2He, time)`` with ``tO2He`` in millimeters."""
    started = time.perf_counter()
    translation = np.asarray(tH2Wn, dtype=float)
    rotation = np.asarray(RH2Wn, dtype=float)
    center = np.asarray(pcOe, dtype=float)
    if translation.ndim != 3 or translation.shape[-1] != 3:
        raise ValueError(f"tH2Wn expected (m, mt, 3), got {translation.shape}")
    n_pose, n_position, _ = translation.shape
    if n_pose < 2 or n_position < 2:
        raise ValueError("sphere_meas2023 needs at least two orientations and two positions")
    if rotation.shape != (n_pose, 3, 3):
        raise ValueError(f"RH2Wn expected {(n_pose, 3, 3)}, got {rotation.shape}")
    if center.shape != translation.shape:
        raise ValueError(f"pcOe expected {translation.shape}, got {center.shape}")

    # MATLAB stores these as (3, mt, m). Transpose back before the column-major reshape.
    center_m = np.transpose(center, (2, 1, 0))
    translation_m = np.transpose(translation, (2, 1, 0))
    rotation_m = np.transpose(rotation, (1, 2, 0))

    center_delta = center_m[:, :1, :] - center_m[:, 1:, :]
    hand_delta = translation_m[:, 1:, :] - translation_m[:, :1, :]
    # Bp = RH^T * (t(:, 2:end) - t(:, 1))
    hand_delta = np.einsum("jim,jkm->ikm", rotation_m, hand_delta)
    design_a = np.reshape(center_delta, (3, -1), order="F")
    design_b = np.reshape(hand_delta, (3, -1), order="F")

    # [U,~,V] = svd(A*B'); R = V*U'. V from NumPy is Vh.T.
    left, _, vh = np.linalg.svd(design_a @ design_b.T, full_matrices=True)
    right = vh.T.copy()
    estimated_R = right @ left.T
    if abs(np.linalg.det(estimated_R) - 1.0) > 1e-5:
        right[:, 2] *= -1.0
        estimated_R = right @ left.T

    mapped = np.einsum("ij,jtm->itm", estimated_R, center_m)
    world = np.einsum("ijm,jtm->itm", rotation_m, mapped) + translation_m
    world_delta = world[:, :, 1:] - world[:, :, :1]
    target = np.reshape(world_delta, (-1, 1), order="F")

    # C stacks (RH(:,:,1) - RH(:,:,k)) once per position. The position index
    # varies faster than k, matching the MATLAB permute/reshape.
    difference = rotation_m[:, :, :1] - rotation_m[:, :, 1:]
    blocks = [difference[:, :, k] for k in range(n_pose - 1) for _ in range(n_position)]
    regressor = np.vstack(blocks)
    estimated_t, *_ = np.linalg.lstsq(regressor, target, rcond=None)
    elapsed = time.perf_counter() - started
    return estimated_t.reshape(3), estimated_R, float(elapsed)

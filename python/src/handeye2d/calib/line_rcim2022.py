"""Line-target hand-eye calibration from Xu, RCIM 2022.

``tH2Wn`` and ``pOn`` are ``(m, mt, 3)`` and ``RH2Wn`` is ``(m, 3, 3)``, the
batch-first form of the MATLAB ``(3, mt, m)`` arrays. ``m`` is the number of
orientations and ``mt`` is the number of positions on the line. Both ``m`` and
``mt`` must be at least 3 and 2 respectively; the orientation block needs
``mt >= 3``.

The MATLAB source at ``fun/calib/line_RCIM2022.m:58`` writes every pose into
the same columns, ``ub(:,(1:(mt-1))+(mt-1))``. This port writes pose ``i``
into columns ``(1:(mt-1))+(mt-1)*(i-1)``. The Octave fixtures were generated
from that patched indexing. Without pose noise the two versions agree; with
noise the original drops every pose but the last.
"""

from __future__ import annotations

import time

import numpy as np

from handeye2d.calib._linalg import project_so3
from handeye2d.so3 import hat


def line_rcim2022(
    tH2Wn: np.ndarray,
    RH2Wn: np.ndarray,
    pOn: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(tO2He, RO2He, time)``."""
    started = time.perf_counter()
    translation = np.asarray(tH2Wn, dtype=float)
    rotation = np.asarray(RH2Wn, dtype=float)
    point = np.asarray(pOn, dtype=float)
    n_pose, n_position = _shape(translation, rotation, point)
    sensor = _orientation(translation, rotation, point, n_pose, n_position)
    direction = _line_direction(translation, rotation, point, sensor)
    estimated_t = _translation(translation, rotation, point, sensor, direction, n_pose, n_position)
    return estimated_t, sensor, time.perf_counter() - started


def _direction_samples(
    translation: np.ndarray,
    rotation: np.ndarray,
    point: np.ndarray,
    sensor_R: np.ndarray,
) -> np.ndarray:
    """``ub`` with one block of ``mt - 1`` columns per pose.

    The original MATLAB statement keeps only the last pose. Column ``0`` of a
    later pose must not overwrite column ``0`` of pose 0.
    """
    n_pose, n_position, _ = translation.shape
    width = n_position - 1
    samples = np.zeros((3, n_pose * width))
    for index in range(n_pose):
        sensor_step = np.diff(point[index], axis=0).T
        hand_step = np.diff(translation[index], axis=0).T
        block = slice(index * width, (index + 1) * width)
        samples[:, block] = rotation[index] @ sensor_R @ sensor_step + hand_step
    return samples


def _orientation(
    translation: np.ndarray,
    rotation: np.ndarray,
    point: np.ndarray,
    n_pose: int,
    n_position: int,
) -> np.ndarray:
    span = n_position - 2
    design = np.zeros((3 * n_pose * span, 9))
    target = np.zeros(3 * n_pose * span)
    for index in range(n_pose):
        sensor_step = np.diff(point[index], axis=0).T
        hand_step = np.diff(translation[index], axis=0).T
        hand = rotation[index]
        for step in range(span):
            block = slice(3 * span * index + 3 * step, 3 * span * index + 3 * step + 3)
            first_s = sensor_step[:, step]
            second_s = sensor_step[:, step + 1]
            first_t = hand_step[:, step]
            second_t = hand_step[:, step + 1]
            design[block] = (
                np.kron(np.cross(first_s, second_s).reshape(1, 3), hand)
                + hat(second_t).T @ np.kron(first_s.reshape(1, 3), hand)
                - hat(first_t).T @ np.kron(second_s.reshape(1, 3), hand)
            )
            target[block] = -np.cross(first_t, second_t)
    solved, *_ = np.linalg.lstsq(design, target, rcond=None)
    return project_so3(np.asarray(solved, dtype=float).reshape((3, 3), order="F"))


def _line_direction(
    translation: np.ndarray,
    rotation: np.ndarray,
    point: np.ndarray,
    sensor_R: np.ndarray,
) -> np.ndarray:
    samples = _direction_samples(translation, rotation, point, sensor_R)
    _, _, vh = np.linalg.svd(samples.T, full_matrices=False)
    return vh[0]


def _translation(
    translation: np.ndarray,
    rotation: np.ndarray,
    point: np.ndarray,
    sensor_R: np.ndarray,
    direction: np.ndarray,
    n_pose: int,
    n_position: int,
) -> np.ndarray:
    width = 3 * (n_pose - 1)
    design = np.zeros((n_position * width, 3))
    target = np.zeros(n_position * width)
    skew = hat(direction)
    for position in range(n_position):
        for pose in range(n_pose - 1):
            block = slice(width * position + 3 * pose, width * position + 3 * pose + 3)
            if position == 0:
                design[block] = skew.T @ (rotation[pose + 1] - rotation[pose])
            mapped = (
                rotation[pose + 1] @ sensor_R @ point[pose + 1, position]
                - rotation[pose] @ sensor_R @ point[pose, position]
                + translation[pose + 1, position]
                - translation[pose, position]
            )
            target[block] = skew @ mapped
        if position > 0:
            design[width * position : width * (position + 1)] = design[:width]
    solved, *_ = np.linalg.lstsq(design, target, rcond=None)
    return np.asarray(solved, dtype=float).reshape(3)


def _shape(translation: np.ndarray, rotation: np.ndarray, point: np.ndarray) -> tuple[int, int]:
    if translation.ndim != 3 or translation.shape[-1] != 3:
        raise ValueError(f"tH2Wn expected (m, mt, 3), got {translation.shape}")
    n_pose, n_position, _ = translation.shape
    if n_pose < 2 or n_position < 3:
        raise ValueError("line_rcim2022 needs at least two orientations and three positions")
    if rotation.shape != (n_pose, 3, 3):
        raise ValueError(f"RH2Wn expected {(n_pose, 3, 3)}, got {rotation.shape}")
    if point.shape != translation.shape:
        raise ValueError(f"pOn expected {translation.shape}, got {point.shape}")
    return n_pose, n_position

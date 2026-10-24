"""Shared steps for the iterative plane baselines.

``project_so3`` is the MATLAB sequence ``[U,~,V] = svd(R); R = U*diag([1 1 det(U)*det(V)])*V'``.
NumPy returns ``Vh``, and ``det(Vh)`` equals ``det(V)``.
"""

from __future__ import annotations

import numpy as np


def project_so3(matrix: np.ndarray) -> np.ndarray:
    """Nearest rotation matrix, with the MATLAB singular-vector sign."""
    left, _, vh = np.linalg.svd(np.asarray(matrix, dtype=float), full_matrices=True)
    scale = np.diag([1.0, 1.0, float(np.linalg.det(left) * np.linalg.det(vh))])
    return left @ scale @ vh


def solve_hand_eye(design: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One RCIM2020 / IROS2015 update. ``design`` is ``(n, 9)`` and ``target`` is ``(n,)``."""
    weights, *_ = np.linalg.lstsq(design, target, rcond=None)
    axis_x = np.asarray(weights[:3], dtype=float)
    axis_z = np.asarray(weights[3:6], dtype=float)
    candidate = np.column_stack((axis_x, -np.cross(axis_x, axis_z), axis_z))
    rotation = project_so3(candidate)
    solved = np.concatenate((rotation[:, 0], rotation[:, 2]))
    translation, *_ = np.linalg.lstsq(design[:, 6:9], target - design[:, :6] @ solved, rcond=None)
    return rotation, np.asarray(translation, dtype=float).reshape(3)

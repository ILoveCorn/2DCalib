"""Plane and circle fits.

Points use batch-first layout. ``scanO`` is ``(m, n_points, 3)`` with the
profile in the OCT frame (the ``y`` coordinate of a profile point is zero).
``tH2W`` is ``(m, 3)`` and ``RH2W`` is ``(m, 3, 3)``.

``plane_fit`` follows ``planeFit``: transform the scans into the world frame,
then take the right singular vector of the centered points that belongs to the
smallest singular value. NumPy and MATLAB do not agree on the sign of that
vector, so callers should treat ``n`` and ``-n`` as the same plane (``d``
flips with ``n``). ``res`` is the mean absolute point-to-plane distance.
"""

from __future__ import annotations

import numpy as np


def plane_fit(
    tO2H: np.ndarray,
    RO2H: np.ndarray,
    tH2W: np.ndarray,
    RH2W: np.ndarray,
    scanO: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Return ``(n, d, res)`` for the plane ``n · x = d``."""
    translation = np.asarray(tO2H, dtype=float).reshape(3)
    rotation = np.asarray(RO2H, dtype=float).reshape(3, 3)
    hand_translation = np.asarray(tH2W, dtype=float)
    hand_rotation = np.asarray(RH2W, dtype=float)
    scan = np.asarray(scanO, dtype=float)
    if hand_translation.ndim != 2 or hand_translation.shape[1] != 3:
        raise ValueError(f"tH2W expected (m, 3), got {hand_translation.shape}")
    if hand_rotation.shape != (hand_translation.shape[0], 3, 3):
        raise ValueError(f"RH2W expected (m, 3, 3), got {hand_rotation.shape}")
    if scan.ndim != 3 or scan.shape[0] != hand_translation.shape[0] or scan.shape[2] != 3:
        raise ValueError(f"scanO expected (m, n_points, 3), got {scan.shape}")

    scan_h = np.einsum("ij,mnj->mni", rotation, scan) + translation
    scan_w = np.einsum("mij,mnj->mni", hand_rotation, scan_h) + hand_translation[:, None, :]
    points = scan_w.reshape(-1, 3)
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1]
    distance = float(normal @ centroid)
    residual = float(np.abs(points @ normal - distance).sum() / points.shape[0])
    return normal, distance, residual


def circle_fit(px: np.ndarray, pz: np.ndarray) -> tuple[float, float, float]:
    """Algebraic circle fit used by MEAS2023.

    Solves ``[x, z, 1] [a, b, c]^T = -(x² + z²)`` in the least-squares sense
    and returns ``xc = -a/2``, ``zc = -b/2``, ``r = sqrt(a² + b² - 4c) / 2``.
    This is ``A \\ b`` on a thin matrix, so it is a least-squares solve rather
    than a square one.
    """
    x = np.asarray(px, dtype=float).reshape(-1)
    z = np.asarray(pz, dtype=float).reshape(-1)
    if x.shape != z.shape or x.size < 3:
        raise ValueError("circle_fit expected at least three paired x and z coordinates")
    design = np.column_stack((x, z, np.ones(x.shape, dtype=float)))
    target = -(np.square(x) + np.square(z))
    (a, b, c), *_ = np.linalg.lstsq(design, target, rcond=None)
    center_x = float(-a / 2.0)
    center_z = float(-b / 2.0)
    radius = float(np.sqrt(a * a + b * b - 4.0 * c) / 2.0)
    return center_x, center_z, radius

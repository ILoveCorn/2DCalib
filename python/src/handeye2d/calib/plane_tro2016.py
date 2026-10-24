"""Plane hand-eye calibration from Kang, TRO 2016.

Two unconstrained nonlinear least-squares problems: Euler angles of the
hand-eye rotation, then the translation. ``scipy.optimize.least_squares``
with ``method='trf'`` stands in for ``lsqnonlin``. Tolerances are the MATLAB
defaults (``1e-6``), which are looser than SciPy's defaults. Octave has no
equivalent, so this solver is checked against the saved real-calibration
result and against noiseless recovery, not against an Octave fixture.

On one real session the rotation cost has a flat shoulder. SciPy's relative
function tolerance walks off that shoulder into a lower but distant basin
(about 11 degrees from the saved pose). ``lsqnonlin`` stays on the shoulder,
and that pose is the one that matches the measured hand-eye. The rotation
solve therefore returns the last Gauss-Newton iterate at which the
first-order optimality ``||J^T F||_∞`` was still decreasing, once that
measure has fallen below ``5e-4``.

``rotm2eul`` / ``eul2rotm`` use the default ``ZYX`` sequence. ``scanO`` is
``(m, n_points, 3)``. ``vOn`` is ``(m, 3)``.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import least_squares

from handeye2d.rotations import eul_to_rotm, rotm_to_eul

_TOL = 1e-6
# First-order optimality below this, an increase means the Euler step is
# leaving the shoulder where lsqnonlin stops. 5e-4 sits above that shoulder
# on the saved sessions and below the harmless increases seen earlier.
_SHOULDER = 5e-4


def plane_tro2016(
    tO2H: np.ndarray,
    RO2H: np.ndarray,
    tH2Wn: np.ndarray,
    RH2Wn: np.ndarray,
    vOn: np.ndarray,
    scanO: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return ``(tO2He, RO2He, n, time)``."""
    translation = np.asarray(tO2H, dtype=float).reshape(3).copy()
    rotation = np.asarray(RO2H, dtype=float).reshape(3, 3).copy()
    measured_t = np.asarray(tH2Wn, dtype=float)
    measured_R = np.asarray(RH2Wn, dtype=float)
    direction = np.asarray(vOn, dtype=float)
    scan = np.asarray(scanO, dtype=float)
    _check(measured_t, measured_R, direction, scan)
    started = time.perf_counter()
    euler0 = np.asarray(rotm_to_eul(rotation, "ZYX"), dtype=float).reshape(3)
    anchor = _direction_normal(np.einsum("mij,mj->mi", measured_R, direction @ rotation.T))

    def cost_rotation(euler: np.ndarray) -> np.ndarray:
        hand = eul_to_rotm(np.asarray(euler, dtype=float), "ZYX")
        world = np.einsum("mij,mj->mi", measured_R, direction @ hand.T)
        normal = _direction_normal(world, anchor)
        return world @ normal

    euler = _least_squares_on_shoulder(cost_rotation, euler0)
    rotation = eul_to_rotm(euler, "ZYX")
    normal = _direction_normal(np.einsum("mij,mj->mi", measured_R, direction @ rotation.T), anchor)

    def cost_translation(hand_t: np.ndarray) -> np.ndarray:
        sensor = np.einsum("ij,mnj->mni", rotation, scan) + hand_t
        world = np.einsum("mij,mnj->mni", measured_R, sensor) + measured_t[:, None, :]
        signed = world.reshape(-1, 3) @ normal
        return signed - signed.mean()

    translation_fit = least_squares(
        cost_translation,
        translation,
        method="trf",
        ftol=_TOL,
        xtol=_TOL,
        gtol=_TOL,
        max_nfev=100 * translation.size,
    )
    elapsed = time.perf_counter() - started
    return np.asarray(translation_fit.x, dtype=float).reshape(3), rotation, normal, elapsed


def _least_squares_on_shoulder(fun, x0: np.ndarray) -> np.ndarray:
    """Trust-region least squares that stays on the ``lsqnonlin`` shoulder."""
    start = np.asarray(x0, dtype=float).reshape(-1).copy()
    state = {"x": start, "opt": _optimality(fun, start)}

    def callback(*, intermediate_result) -> None:
        point = np.asarray(intermediate_result.x, dtype=float).reshape(-1).copy()
        opt = _optimality(fun, point)
        if opt > state["opt"] and state["opt"] < _SHOULDER:
            raise StopIteration
        state["x"] = point
        state["opt"] = opt

    least_squares(
        fun,
        start,
        method="trf",
        ftol=_TOL,
        xtol=_TOL,
        gtol=_TOL,
        max_nfev=100 * start.size,
        callback=callback,
    )
    return state["x"]


def _optimality(fun, point: np.ndarray) -> float:
    """``||J^T F||_∞`` with MATLAB's forward-difference step."""
    residual = np.asarray(fun(point), dtype=float).reshape(-1)
    jacobian = np.empty((residual.size, point.size))
    relative = np.sqrt(np.finfo(float).eps)
    for index in range(point.size):
        step = relative * max(abs(float(point[index])), 1.0)
        bumped = point.copy()
        bumped[index] += step
        jacobian[:, index] = (np.asarray(fun(bumped), dtype=float).reshape(-1) - residual) / step
    return float(np.linalg.norm(jacobian.T @ residual, ord=np.inf))


def _direction_normal(world_direction: np.ndarray, anchor: np.ndarray | None = None) -> np.ndarray:
    """Last right singular vector of the directions, without centering.

    The singular-vector sign is chosen to stay on the same side as ``anchor``.
    An unanchored sign flip makes the residual jump and sends ``least_squares``
    into a different basin from ``lsqnonlin``.
    """
    _, _, vh = np.linalg.svd(world_direction, full_matrices=False)
    normal = vh[-1].copy()
    if anchor is not None and float(normal @ anchor) < 0.0:
        normal = -normal
    return normal


def _check(
    measured_t: np.ndarray,
    measured_R: np.ndarray,
    direction: np.ndarray,
    scan: np.ndarray,
) -> None:
    if measured_t.ndim != 2 or measured_t.shape[1] != 3:
        raise ValueError(f"tH2Wn expected (m, 3), got {measured_t.shape}")
    count = measured_t.shape[0]
    if measured_R.shape != (count, 3, 3):
        raise ValueError(f"RH2Wn expected {(count, 3, 3)}, got {measured_R.shape}")
    if direction.shape != (count, 3):
        raise ValueError(f"vOn expected {(count, 3)}, got {direction.shape}")
    if scan.ndim != 3 or scan.shape[0] != count or scan.shape[2] != 3:
        raise ValueError(f"scanO expected (m, n_points, 3), got {scan.shape}")

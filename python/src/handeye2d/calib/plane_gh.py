"""Gauss-Helmert plane calibration.

``plane_jnt_gh``, ``plane_alt_gh``, and ``plane_jnt_gh_decoupled`` share one
loop. It follows the MATLAB control flow, including the extra inner iteration
that rebuilds the final Jacobian and then breaks before the update. The
variance-factor step keeps ``DtE`` from the last accepted step and multiplies
it by the Jacobian of that extra pass. That mix is what the Octave fixtures
record.

No iteration counts are returned. There is no ``_debug`` variant and no GPU
variant.

Inputs are batch-first: ``tH2Wn`` is ``(m, 3)``, ``RH2Wn`` is ``(m, 3, 3)``,
and ``pOn`` / ``vOn`` are ``(m, 3)``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from handeye2d.so3 import exp, hat, jr, log

Kind = Literal["joint", "alternating", "decoupled"]
_VFE_TOL = 1e-4


@dataclass(frozen=True)
class PlaneGhResult:
    """Estimates from one Gauss-Helmert solve.

    ``tH2W`` and ``RH2W`` are the corrected hand-world poses. ``sigma`` is
    ``[sigma_t, sigma_R]``. ``time`` is seconds spent in the accepted steps
    and the variance update. The extra Jacobian pass is not included, matching
    where MATLAB calls ``toc``.
    """

    tO2H: np.ndarray
    RO2H: np.ndarray
    n: np.ndarray
    d: float
    tH2W: np.ndarray
    RH2W: np.ndarray
    sigma: np.ndarray
    time: float


def plane_jnt_gh(
    tO2H: np.ndarray,
    RO2H: np.ndarray,
    n: np.ndarray,
    d: float,
    sigma: np.ndarray,
    tH2Wn: np.ndarray,
    RH2Wn: np.ndarray,
    pOn: np.ndarray,
    vOn: np.ndarray,
    iImax: int,
    iOmax: int,
    tol: float,
) -> PlaneGhResult:
    """Joint Gauss-Helmert estimate. The inner stop test uses ``‖dx(1:6)‖``."""
    return _solve("joint", tO2H, RO2H, n, d, sigma, tH2Wn, RH2Wn, pOn, vOn, iImax, iOmax, tol)


def plane_alt_gh(
    tO2H: np.ndarray,
    RO2H: np.ndarray,
    n: np.ndarray,
    d: float,
    sigma: np.ndarray,
    tH2Wn: np.ndarray,
    RH2Wn: np.ndarray,
    pOn: np.ndarray,
    vOn: np.ndarray,
    iImax: int,
    iOmax: int,
    tol: float,
) -> PlaneGhResult:
    """Alternating estimate. The plane is refit by SVD and the stop test uses ``‖dx‖``."""
    return _solve(
        "alternating",
        tO2H,
        RO2H,
        n,
        d,
        sigma,
        tH2Wn,
        RH2Wn,
        pOn,
        vOn,
        iImax,
        iOmax,
        tol,
    )


def plane_jnt_gh_decoupled(
    tO2H: np.ndarray,
    RO2H: np.ndarray,
    n: np.ndarray,
    d: float,
    sigma: np.ndarray,
    tH2Wn: np.ndarray,
    RH2Wn: np.ndarray,
    pOn: np.ndarray,
    vOn: np.ndarray,
    iImax: int,
    iOmax: int,
    tol: float,
) -> PlaneGhResult:
    """Joint estimate with the position-to-pose block ``A(1, 4:8)`` left at zero."""
    return _solve(
        "decoupled",
        tO2H,
        RO2H,
        n,
        d,
        sigma,
        tH2Wn,
        RH2Wn,
        pOn,
        vOn,
        iImax,
        iOmax,
        tol,
    )


def _solve(
    kind: Kind,
    tO2H: np.ndarray,
    RO2H: np.ndarray,
    n: np.ndarray,
    d: float,
    sigma: np.ndarray,
    tH2Wn: np.ndarray,
    RH2Wn: np.ndarray,
    pOn: np.ndarray,
    vOn: np.ndarray,
    iImax: int,
    iOmax: int,
    tol: float,
) -> PlaneGhResult:
    translation = np.asarray(tO2H, dtype=float).reshape(3).copy()
    rotation = np.asarray(RO2H, dtype=float).reshape(3, 3).copy()
    normal = np.asarray(n, dtype=float).reshape(3).copy()
    distance = float(d)
    variance = np.asarray(sigma, dtype=float).reshape(2).copy()
    measured_t = np.asarray(tH2Wn, dtype=float)
    measured_R = np.asarray(RH2Wn, dtype=float)
    point = np.asarray(pOn, dtype=float)
    direction = np.asarray(vOn, dtype=float)
    count = int(measured_t.shape[0])
    _check_batch(count, measured_t, measured_R, point, direction)

    nx = 6 if kind == "alternating" else 9
    point_hat = hat(point)
    direction_hat = hat(direction)
    corrected_t = measured_t.copy()
    corrected_R = measured_R.copy()
    residual_t = np.zeros((count, 3))
    residual_R = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
    scale = np.repeat(variance, 3) ** 2
    dx = np.zeros(nx)
    basis = np.eye(3)
    elapsed = 0.0
    inner = 1
    outer = 1
    converged = False
    weight_t: np.ndarray | None = None

    while outer <= iOmax:
        while inner == 1 or (_norm(dx) > tol and inner <= iImax) or converged:
            started = time.perf_counter()
            if kind != "alternating" and inner == 1:
                basis, normal = _plane_basis(normal)
            point_h, point_w, direction_h, direction_w = _reproject(
                translation, rotation, corrected_t, corrected_R, point, direction
            )
            if kind == "alternating":
                normal, distance = _fit_plane(point_w, direction_w)
            state = _linearize(
                kind,
                nx,
                normal,
                distance,
                basis,
                rotation,
                corrected_R,
                point_hat,
                direction_hat,
                point_h,
                point_w,
                direction_h,
                direction_w,
                residual_t,
                residual_R,
                scale,
            )
            if converged:
                break
            dx = np.linalg.solve(state.hessian, state.gradient)
            weight_t = state.weight_t
            delta = _measurement_step(state, dx, residual_t, residual_R, scale)
            translation = translation + dx[:3]
            rotation = rotation @ exp(dx[3:6])
            if kind != "alternating":
                basis = basis @ exp(np.array([dx[6], dx[7], 0.0]))
                normal = basis[:, 2].copy()
                distance = distance + float(dx[8])
            residual_t = residual_t + delta[:, :3]
            corrected_t = measured_t + residual_t
            residual_R = residual_R @ exp(delta[:, 3:])
            corrected_R = measured_R @ residual_R
            elapsed += time.perf_counter() - started
            inner += 1
            watched = dx[:6] if kind != "alternating" else dx
            converged = not (_norm(watched) > tol and inner <= iImax)

        if weight_t is None:
            raise RuntimeError("variance update ran before an accepted step")
        started = time.perf_counter()
        factor = _redundancy_factor(weight_t, state, scale, residual_t, residual_R)
        scale = factor * scale
        variance = np.sqrt(scale[np.array([0, 3])])
        elapsed += time.perf_counter() - started
        if float(np.max(np.abs(factor - 1.0))) < _VFE_TOL or outer == iOmax:
            break
        residual_t = np.zeros((count, 3))
        residual_R = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
        corrected_t = measured_t.copy()
        corrected_R = measured_R.copy()
        outer += 1
        inner = 1
        converged = False
        dx = np.zeros(nx)

    return PlaneGhResult(
        tO2H=translation,
        RO2H=rotation,
        n=normal,
        d=float(distance),
        tH2W=corrected_t,
        RH2W=corrected_R,
        sigma=variance,
        time=elapsed,
    )


@dataclass(frozen=True)
class _LinearSystem:
    jacobian_x: np.ndarray
    jacobian_l: np.ndarray
    hessian: np.ndarray
    gradient: np.ndarray
    weight_t: np.ndarray
    predicted: np.ndarray


def _linearize(
    kind: Kind,
    nx: int,
    normal: np.ndarray,
    distance: float,
    basis: np.ndarray,
    rotation: np.ndarray,
    corrected_R: np.ndarray,
    point_hat: np.ndarray,
    direction_hat: np.ndarray,
    point_h: np.ndarray,
    point_w: np.ndarray,
    direction_h: np.ndarray,
    direction_w: np.ndarray,
    residual_t: np.ndarray,
    residual_R: np.ndarray,
    scale: np.ndarray,
) -> _LinearSystem:
    count = corrected_R.shape[0]
    jacobian_x = np.zeros((count, 2, nx))
    jacobian_l = np.zeros((count, 2, 6))
    row_r = _row(normal, corrected_R)
    jacobian_x[:, 0, :3] = row_r
    pose_point = _pose_column(normal, corrected_R, rotation, point_hat)
    pose_direction = _pose_column(normal, corrected_R, rotation, direction_hat)
    if kind != "decoupled":
        jacobian_x[:, 0, 3:6] = -pose_point
    jacobian_x[:, 1, 3:6] = -pose_direction
    if kind != "alternating":
        tangent = -basis @ hat(np.array([0.0, 0.0, 1.0]))
        if kind != "decoupled":
            jacobian_x[:, 0, 6] = point_w @ tangent[:, 0]
            jacobian_x[:, 0, 7] = point_w @ tangent[:, 1]
        jacobian_x[:, 0, 8] = -1.0
        jacobian_x[:, 1, 6] = direction_w @ tangent[:, 0]
        jacobian_x[:, 1, 7] = direction_w @ tangent[:, 1]

    jacobian_l[:, 0, :3] = normal
    point_skew = hat(point_h)
    direction_skew = hat(direction_h)
    jacobian_l[:, 0, 3:] = -_row(normal, corrected_R @ point_skew)
    jacobian_l[:, 1, 3:] = -_row(normal, corrected_R @ direction_skew)
    right = jr(log(residual_R))
    jacobian_l[:, :, 3:] = jacobian_l[:, :, 3:] @ right

    constraint = np.stack((point_w @ normal - distance, direction_w @ normal), axis=-1)
    latent = _latent(residual_t, residual_R)
    weighted = jacobian_l * scale
    gram = jacobian_l @ weighted.transpose(0, 2, 1)
    gram_inv = np.linalg.inv(gram)
    predicted = (jacobian_l @ latent[..., None])[..., 0] - constraint
    weight_x = jacobian_x.transpose(0, 2, 1) @ gram_inv
    hessian = np.sum(weight_x @ jacobian_x, axis=0)
    gradient = np.sum(weight_x @ predicted[..., None], axis=0)[..., 0]
    weight_t = jacobian_l.transpose(0, 2, 1) @ gram_inv
    return _LinearSystem(jacobian_x, jacobian_l, hessian, gradient, weight_t, predicted)


def _measurement_step(
    state: _LinearSystem,
    dx: np.ndarray,
    residual_t: np.ndarray,
    residual_R: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    innovation = state.predicted - (state.jacobian_x @ dx)
    delta = (state.weight_t @ innovation[..., None])[..., 0] * scale
    delta = delta - _latent(residual_t, residual_R)
    delta[:, 3:] = (jr(log(residual_R)) @ delta[:, 3:, None])[..., 0]
    return delta


def _redundancy_factor(
    weight_t: np.ndarray,
    state: _LinearSystem,
    scale: np.ndarray,
    residual_t: np.ndarray,
    residual_R: np.ndarray,
) -> np.ndarray:
    """Diagonal redundancy weights ``dvar``, broadcast to the six covariance entries."""
    coupled_l = weight_t @ state.jacobian_l
    coupled_x = weight_t @ state.jacobian_x
    solved = _solve_against_hessian(state.hessian, coupled_x)
    removed = solved @ coupled_x.transpose(0, 2, 1)
    redundancy = scale * np.diagonal(coupled_l - removed, axis1=1, axis2=2)
    position = float(np.sum(redundancy[:, :3]))
    orientation = float(np.sum(redundancy[:, 3:]))
    angle = log(residual_R)
    position_cost = float(np.sum(residual_t**2 / scale[:3]))
    orientation_cost = float(np.sum(angle**2 / scale[3:]))
    factor = np.empty(6)
    factor[:3] = position_cost / position
    factor[3:] = orientation_cost / orientation
    return factor


def _solve_against_hessian(hessian: np.ndarray, coupled: np.ndarray) -> np.ndarray:
    """Solve ``Z @ H = coupled`` for every measurement. ``coupled`` is ``(m, 6, nx)``."""
    count, _, nx = coupled.shape
    transposed = np.moveaxis(coupled.transpose(0, 2, 1), 1, 0)
    rhs = transposed.reshape(nx, count * 6)
    solution = np.linalg.solve(hessian.T, rhs).reshape(nx, count, 6)
    return np.moveaxis(solution, 0, 2)


def _reproject(
    translation: np.ndarray,
    rotation: np.ndarray,
    corrected_t: np.ndarray,
    corrected_R: np.ndarray,
    point: np.ndarray,
    direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    point_h = point @ rotation.T + translation
    direction_h = direction @ rotation.T
    point_w = np.einsum("mij,mj->mi", corrected_R, point_h) + corrected_t
    direction_w = np.einsum("mij,mj->mi", corrected_R, direction_h)
    return point_h, point_w, direction_h, direction_w


def _pose_column(
    normal: np.ndarray,
    corrected_R: np.ndarray,
    rotation: np.ndarray,
    skew: np.ndarray,
) -> np.ndarray:
    mapped = np.einsum("ij,mjk->mik", rotation, skew)
    mapped = np.einsum("mij,mjk->mik", corrected_R, mapped)
    return _row(normal, mapped)


def _row(normal: np.ndarray, pages: np.ndarray) -> np.ndarray:
    """``n.T @ M`` for ``M`` shaped ``(m, 3, 3)`` or ``(m, 3, k)``, returning ``(m, k)``."""
    return np.einsum("j,mjk->mk", normal, pages)


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unit = normal / np.linalg.norm(normal)
    if unit[0] == 0.0 and unit[1] == 0.0:
        tangent_x = np.array([1.0, 0.0, 0.0])
    else:
        tangent_x = np.array([-unit[1], unit[0], 0.0])
        tangent_x = tangent_x / np.linalg.norm(tangent_x)
    tangent_y = hat(unit) @ tangent_x
    return np.column_stack((tangent_x, tangent_y, unit)), unit


def _fit_plane(point_w: np.ndarray, direction_w: np.ndarray) -> tuple[np.ndarray, float]:
    center = point_w.mean(axis=0)
    samples = np.concatenate((direction_w.T, (point_w - center).T), axis=1)
    left, _, _ = np.linalg.svd(samples, full_matrices=False)
    normal = left[:, -1]
    if normal[2] < 0.0:
        normal = -normal
    return normal, float(normal @ center)


def _latent(residual_t: np.ndarray, residual_R: np.ndarray) -> np.ndarray:
    return np.concatenate((residual_t, log(residual_R)), axis=-1)


def _norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def _check_batch(
    count: int,
    measured_t: np.ndarray,
    measured_R: np.ndarray,
    point: np.ndarray,
    direction: np.ndarray,
) -> None:
    if measured_t.shape != (count, 3):
        raise ValueError(f"tH2Wn expected {(count, 3)}, got {measured_t.shape}")
    if measured_R.shape != (count, 3, 3):
        raise ValueError(f"RH2Wn expected {(count, 3, 3)}, got {measured_R.shape}")
    if point.shape != (count, 3) or direction.shape != (count, 3):
        raise ValueError("pOn and vOn must both have shape (m, 3)")

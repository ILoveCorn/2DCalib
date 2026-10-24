"""Synthetic plane-target scans.

This is ``simPlane``. Random draws come from ``rng`` in the same order as the
MATLAB function: orientation noise, then each scan's image point and its
image noise, then the in-plane line positions, then position noise. NumPy's
stream is not Octave's or MATLAB's, so a fixed seed reproduces this function
and not the committed fixture. The fixture is the solver oracle; geometry
tests check this generator directly.

``Sigmat`` and ``SigmaR`` are 3×3 covariances. They must be positive definite,
because both this function and MATLAB call ``chol`` on them. A zero ``sigmaO``
still consumes the image-noise draw, then multiplies it by zero.

Returns batch-first arrays. ``scanO`` has 20 points per scan, as in MATLAB.
Profile geometry is vectorized. A single call still draws image points and
image noise one scan at a time, so an existing seed is unchanged.
``sim_plane_batch`` draws several independent sessions at once for the
noise-level plane-fit loop; that bulk stream is not ``n`` sequential calls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from handeye2d.rotations import eul_to_rotm
from handeye2d.so3 import exp

_POINTS_PER_SCAN = 20
_NOISE_REPEATS = 10


@dataclass(frozen=True)
class PlaneSimulation:
    tH2Wn: np.ndarray
    RH2Wn: np.ndarray
    pOn: np.ndarray
    vOn: np.ndarray
    scanO: np.ndarray


def sim_plane(
    pP: np.ndarray,
    nP: np.ndarray,
    lP: float,
    xMax: float,
    zMax: float,
    tO2H: np.ndarray,
    RO2H: np.ndarray,
    Sigmat: np.ndarray,
    SigmaR: np.ndarray,
    sigmaO: float,
    alphaMax: float,
    betaMax: float,
    mAlpha: int,
    mBeta: int,
    rng: np.random.Generator,
) -> PlaneSimulation:
    """Draw one plane session."""
    center = np.asarray(pP, dtype=float).reshape(3)
    normal = np.asarray(nP, dtype=float).reshape(3)
    hand_t = np.asarray(tO2H, dtype=float).reshape(3)
    hand_R = np.asarray(RO2H, dtype=float).reshape(3, 3)
    position_chol = np.linalg.cholesky(np.asarray(Sigmat, dtype=float))
    orientation_chol = np.linalg.cholesky(np.asarray(SigmaR, dtype=float))
    image_std = float(sigmaO)
    image_chol = np.zeros((2, 2)) if image_std == 0.0 else image_std * np.eye(2)
    count = int(mAlpha) * int(mBeta)

    axis_x, axis_y, plane_R, sensor_R0 = _plane_frame(normal)
    sensor_R, clean_R = _sensor_orientations(
        sensor_R0, hand_R, alphaMax, betaMax, int(mAlpha), int(mBeta)
    )
    orientation_noise = orientation_chol @ _normal(rng, (3, count))
    # Image point, then that scan's image noise, one scan at a time. The
    # geometry below is vectorized; the draws stay in this order so a seed
    # still reproduces a single call.
    image_x = np.empty(count)
    image_z = np.empty(count)
    scan_noise = np.empty((count, 2, _POINTS_PER_SCAN))
    for index in range(count):
        image_x[index] = xMax * rng.random()
        image_z[index] = zMax * rng.random()
        draw = _normal(rng, (2, _POINTS_PER_SCAN, _NOISE_REPEATS))
        scan_noise[index] = np.einsum("ij,j...->i...", image_chol, draw).mean(axis=2)
    line_uv = _normal(rng, (2, count), kind="uniform")
    position_noise = (position_chol @ _normal(rng, (3, count))).T
    return _assemble(
        center,
        float(lP),
        xMax,
        zMax,
        hand_t,
        hand_R,
        axis_x,
        axis_y,
        plane_R,
        sensor_R,
        clean_R,
        orientation_noise,
        image_x,
        image_z,
        scan_noise,
        line_uv,
        position_noise,
    )


def sim_plane_batch(
    n: int,
    pP: np.ndarray,
    nP: np.ndarray,
    lP: float,
    xMax: float,
    zMax: float,
    tO2H: np.ndarray,
    RO2H: np.ndarray,
    Sigmat: np.ndarray,
    SigmaR: np.ndarray,
    sigmaO: float,
    alphaMax: float,
    betaMax: float,
    mAlpha: int,
    mBeta: int,
    rng: np.random.Generator,
) -> PlaneSimulation:
    """Draw ``n`` independent plane sessions in one vectorized pass.

    Each session has the same distribution as ``sim_plane``. The random
    numbers are drawn in bulk (all sessions, then each kind of noise), so a
    seed of this function is not a seed of ``n`` sequential ``sim_plane``
    calls. ``tH2Wn`` is ``(n, m, 3)``, ``RH2Wn`` is ``(n, m, 3, 3)``, and
    ``scanO`` is ``(n, m, 20, 3)``.
    """
    count_sessions = int(n)
    if count_sessions < 1:
        raise ValueError("sim_plane_batch needs at least one session")
    center = np.asarray(pP, dtype=float).reshape(3)
    normal = np.asarray(nP, dtype=float).reshape(3)
    hand_t = np.asarray(tO2H, dtype=float).reshape(3)
    hand_R = np.asarray(RO2H, dtype=float).reshape(3, 3)
    position_chol = np.linalg.cholesky(np.asarray(Sigmat, dtype=float))
    orientation_chol = np.linalg.cholesky(np.asarray(SigmaR, dtype=float))
    image_std = float(sigmaO)
    image_chol = np.zeros((2, 2)) if image_std == 0.0 else image_std * np.eye(2)
    count = int(mAlpha) * int(mBeta)
    axis_x, axis_y, plane_R, sensor_R0 = _plane_frame(normal)
    sensor_R, clean_R = _sensor_orientations(
        sensor_R0, hand_R, alphaMax, betaMax, int(mAlpha), int(mBeta)
    )
    orientation_noise = np.einsum(
        "ij,njk->nik", orientation_chol, _normal(rng, (count_sessions, 3, count))
    )
    image_x = xMax * _normal(rng, (count_sessions, count), kind="uniform")
    image_z = zMax * _normal(rng, (count_sessions, count), kind="uniform")
    image_draw = _normal(rng, (count_sessions, 2, _POINTS_PER_SCAN, _NOISE_REPEATS, count))
    scan_noise = np.einsum("ij,nj...->ni...", image_chol, image_draw).mean(axis=3)
    scan_noise = np.moveaxis(scan_noise, -1, 1)
    line_uv = _normal(rng, (count_sessions, 2, count), kind="uniform")
    position_noise = np.einsum(
        "ij,njk->nik", position_chol, _normal(rng, (count_sessions, 3, count))
    )
    position_noise = np.moveaxis(position_noise, 1, -1)
    return _assemble_batch(
        center,
        float(lP),
        xMax,
        zMax,
        hand_t,
        hand_R,
        axis_x,
        axis_y,
        plane_R,
        sensor_R,
        clean_R,
        orientation_noise,
        image_x,
        image_z,
        scan_noise,
        line_uv,
        position_noise,
    )


def _sensor_orientations(
    sensor_R0: np.ndarray,
    hand_R: np.ndarray,
    alpha_max: float,
    beta_max: float,
    m_alpha: int,
    m_beta: int,
) -> tuple[np.ndarray, np.ndarray]:
    beta, alpha = np.meshgrid(
        np.linspace(-beta_max, beta_max, m_beta),
        np.linspace(-alpha_max, alpha_max, m_alpha),
        indexing="ij",
    )
    count = m_alpha * m_beta
    euler = np.column_stack((np.zeros(count), alpha.ravel(order="F"), beta.ravel(order="F")))
    sensor_R = np.einsum("ij,mjk->mik", sensor_R0, eul_to_rotm(euler, "ZYX"))
    return sensor_R, sensor_R @ hand_R.T


def _assemble(
    center: np.ndarray,
    length: float,
    x_max: float,
    z_max: float,
    hand_t: np.ndarray,
    hand_R: np.ndarray,
    axis_x: np.ndarray,
    axis_y: np.ndarray,
    plane_R: np.ndarray,
    sensor_R: np.ndarray,
    clean_R: np.ndarray,
    orientation_noise: np.ndarray,
    image_x: np.ndarray,
    image_z: np.ndarray,
    scan_noise: np.ndarray,
    line_uv: np.ndarray,
    position_noise: np.ndarray,
) -> PlaneSimulation:
    noisy_R = clean_R @ exp(orientation_noise.T)
    scan = _scans(sensor_R, axis_x, axis_y, image_x, image_z, scan_noise, x_max, z_max)
    observed_t, observed_v = _fit_profiles(scan)
    line_plane = np.zeros((3, image_x.shape[0]))
    line_plane[:2] = (length / 2.0) * (2.0 * line_uv - 1.0)
    world = (plane_R @ line_plane).T + center
    image_point = np.column_stack((image_x, np.zeros(image_x.shape[0]), image_z))
    mapped = image_point @ hand_R.T + hand_t
    clean_t = world - np.einsum("mij,mj->mi", clean_R, mapped)
    return PlaneSimulation(
        tH2Wn=clean_t + position_noise,
        RH2Wn=noisy_R,
        pOn=observed_t,
        vOn=observed_v,
        scanO=scan,
    )


def _assemble_batch(
    center: np.ndarray,
    length: float,
    x_max: float,
    z_max: float,
    hand_t: np.ndarray,
    hand_R: np.ndarray,
    axis_x: np.ndarray,
    axis_y: np.ndarray,
    plane_R: np.ndarray,
    sensor_R: np.ndarray,
    clean_R: np.ndarray,
    orientation_noise: np.ndarray,
    image_x: np.ndarray,
    image_z: np.ndarray,
    scan_noise: np.ndarray,
    line_uv: np.ndarray,
    position_noise: np.ndarray,
) -> PlaneSimulation:
    """``orientation_noise`` is ``(n, 3, m)`` and image arrays lead with ``n``."""
    noisy_R = clean_R[None] @ exp(np.moveaxis(orientation_noise, 1, -1))
    n_session, count = image_x.shape
    slopes = _intersection_slopes(sensor_R, axis_x, axis_y)
    left, right = _segment_limits_vec(
        image_x, image_z, np.broadcast_to(slopes, image_x.shape), x_max, z_max
    )
    grid = np.linspace(0.0, 1.0, _POINTS_PER_SCAN)
    sample_x = left[..., None] + (right - left)[..., None] * grid
    sample_z = image_z[..., None] + slopes[None, :, None] * (sample_x - image_x[..., None])
    scan = np.zeros((n_session, count, _POINTS_PER_SCAN, 3))
    scan[..., 0] = sample_x + scan_noise[:, :, 0, :]
    scan[..., 2] = sample_z + scan_noise[:, :, 1, :]
    observed_t, observed_v = _fit_profiles(scan.reshape(n_session * count, _POINTS_PER_SCAN, 3))
    observed_t = observed_t.reshape(n_session, count, 3)
    observed_v = observed_v.reshape(n_session, count, 3)
    line_plane = np.zeros((n_session, 3, count))
    line_plane[:, :2] = (length / 2.0) * (2.0 * line_uv - 1.0)
    world = np.einsum("ij,njk->nki", plane_R, line_plane) + center
    image_point = np.stack((image_x, np.zeros_like(image_x), image_z), axis=-1)
    mapped = image_point @ hand_R.T + hand_t
    clean_t = world - np.einsum("mij,nmj->nmi", clean_R, mapped)
    return PlaneSimulation(
        tH2Wn=clean_t + position_noise,
        RH2Wn=noisy_R,
        pOn=observed_t,
        vOn=observed_v,
        scanO=scan,
    )


def _scans(
    sensor_R: np.ndarray,
    axis_x: np.ndarray,
    axis_y: np.ndarray,
    image_x: np.ndarray,
    image_z: np.ndarray,
    scan_noise: np.ndarray,
    x_max: float,
    z_max: float,
) -> np.ndarray:
    slopes = _intersection_slopes(sensor_R, axis_x, axis_y)
    left, right = _segment_limits_vec(image_x, image_z, slopes, x_max, z_max)
    grid = np.linspace(0.0, 1.0, _POINTS_PER_SCAN)
    sample_x = left[:, None] + (right - left)[:, None] * grid
    sample_z = image_z[:, None] + slopes[:, None] * (sample_x - image_x[:, None])
    scan = np.zeros((image_x.shape[0], _POINTS_PER_SCAN, 3))
    scan[:, :, 0] = sample_x + scan_noise[:, 0, :]
    scan[:, :, 2] = sample_z + scan_noise[:, 1, :]
    return scan


def _plane_frame(
    normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if normal[0] == 0.0 and normal[1] == 0.0:
        axis_x = np.array([1.0, 0.0, 0.0])
    else:
        axis_x = np.array([-normal[1], normal[0], 0.0])
        axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = np.cross(normal, axis_x)
    plane_R = np.column_stack((axis_x, axis_y, normal))
    sensor_R0 = np.column_stack((-axis_x, axis_y, -normal))
    return axis_x, axis_y, plane_R, sensor_R0


def _intersection_slopes(
    sensor_R: np.ndarray, axis_x: np.ndarray, axis_y: np.ndarray
) -> np.ndarray:
    """Profile slope in the OCT frame, one per scan. Same branches as the scalar form."""
    sensor_y = sensor_R[:, :, 1]
    along_x = sensor_y @ axis_x
    along_y = sensor_y @ axis_y
    ratio = np.divide(along_x, along_y, out=np.zeros_like(along_x), where=np.abs(along_y) >= 1e-10)
    direction_w = axis_x - ratio[:, None] * axis_y
    small_x = np.abs(along_x) < 1e-10
    small_y = ~small_x & (np.abs(along_y) < 1e-10)
    direction_w = np.where(small_x[:, None], axis_x, direction_w)
    direction_w = np.where(small_y[:, None], axis_y, direction_w)
    direction_o = np.einsum("mji,mj->mi", sensor_R, direction_w)
    return direction_o[:, 2] / direction_o[:, 0]


def _segment_limits_vec(
    x0: np.ndarray,
    z0: np.ndarray,
    slope: np.ndarray,
    x_max: float,
    z_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vector form of the FOV clip. ``x0``, ``z0``, and ``slope`` broadcast together."""
    z_left = z0 - slope * x0
    x_left = np.zeros_like(x0, dtype=float)
    left_low = z_left < 0.0
    left_high = z_left > z_max
    safe = np.where(np.abs(slope) < 1e-15, 1.0, slope)
    x_left = np.where(left_low, x0 - z0 / safe, x_left)
    x_left = np.where(left_high, x0 + (z_max - z0) / safe, x_left)
    z_right = z0 + slope * (x_max - x0)
    x_right = np.full_like(x0, x_max, dtype=float)
    x_right = np.where(z_right < 0.0, x0 - z0 / safe, x_right)
    x_right = np.where(z_right > z_max, x0 + (z_max - z0) / safe, x_right)
    return x_left, x_right


def _fit_profiles(scan: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sample_x = scan[..., 0]
    sample_z = scan[..., 2]
    position = np.column_stack(
        (sample_x.mean(axis=1), np.zeros(scan.shape[0]), sample_z.mean(axis=1))
    )
    x_mean = sample_x.mean(axis=1, keepdims=True)
    z_mean = sample_z.mean(axis=1, keepdims=True)
    variance = np.sum((sample_x - x_mean) ** 2, axis=1)
    slope = np.sum((sample_x - x_mean) * (sample_z - z_mean), axis=1) / variance
    theta = np.arctan(slope)
    direction = np.column_stack((np.cos(theta), np.zeros(scan.shape[0]), np.sin(theta)))
    return position, direction


def _normal(rng: np.random.Generator, shape: tuple[int, ...], kind: str = "normal") -> np.ndarray:
    """MATLAB column-major ``randn`` / ``rand`` of the requested shape."""
    count = int(np.prod(shape))
    draw = rng.normal(size=count) if kind == "normal" else rng.random(size=count)
    return draw.reshape(shape, order="F")

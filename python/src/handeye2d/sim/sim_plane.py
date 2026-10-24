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
    beta, alpha = np.meshgrid(
        np.linspace(-betaMax, betaMax, int(mBeta)),
        np.linspace(-alphaMax, alphaMax, int(mAlpha)),
        indexing="ij",
    )
    euler = np.column_stack(
        (
            np.zeros(count),
            alpha.ravel(order="F"),
            beta.ravel(order="F"),
        )
    )
    sensor_R = np.einsum("ij,mjk->mik", sensor_R0, eul_to_rotm(euler, "ZYX"))
    clean_R = sensor_R @ hand_R.T
    orientation_noise = orientation_chol @ _normal(rng, (3, count))
    noisy_R = clean_R @ exp(orientation_noise.T)

    image_x = np.empty(count)
    image_z = np.empty(count)
    scan = np.zeros((count, _POINTS_PER_SCAN, 3))
    for index in range(count):
        image_x[index] = xMax * rng.random()
        image_z[index] = zMax * rng.random()
        slope = _intersection_slope(sensor_R[index], axis_x, axis_y)
        left, right = _segment_limits(image_x[index], image_z[index], slope, xMax, zMax)
        sample_x = np.linspace(left, right, _POINTS_PER_SCAN)
        sample_z = image_z[index] + slope * (sample_x - image_x[index])
        draw = _normal(rng, (2, _POINTS_PER_SCAN, _NOISE_REPEATS))
        noise = np.einsum("ij,j...->i...", image_chol, draw).mean(axis=2)
        scan[index, :, 0] = sample_x + noise[0]
        scan[index, :, 2] = sample_z + noise[1]

    observed_t, observed_v = _fit_profiles(scan)
    line_uv = _normal(rng, (2, count), kind="uniform")
    line_plane = np.zeros((3, count))
    line_plane[:2] = (float(lP) / 2.0) * (2.0 * line_uv - 1.0)
    world = (plane_R @ line_plane).T + center
    image_point = np.column_stack((image_x, np.zeros(count), image_z))
    mapped = image_point @ hand_R.T + hand_t
    clean_t = world - np.einsum("mij,mj->mi", clean_R, mapped)
    position_noise = (position_chol @ _normal(rng, (3, count))).T
    return PlaneSimulation(
        tH2Wn=clean_t + position_noise,
        RH2Wn=noisy_R,
        pOn=observed_t,
        vOn=observed_v,
        scanO=scan,
    )


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


def _intersection_slope(sensor_R: np.ndarray, axis_x: np.ndarray, axis_y: np.ndarray) -> float:
    sensor_y = sensor_R[:, 1]
    along_x = float(sensor_y @ axis_x)
    along_y = float(sensor_y @ axis_y)
    if abs(along_x) < 1e-10:
        direction_w = axis_x
    elif abs(along_y) < 1e-10:
        direction_w = axis_y
    else:
        direction_w = axis_x - (along_x / along_y) * axis_y
    direction_o = sensor_R.T @ direction_w
    return float(direction_o[2] / direction_o[0])


def _segment_limits(
    x0: float, z0: float, slope: float, x_max: float, z_max: float
) -> tuple[float, float]:
    z_left = z0 - slope * x0
    if z_left < 0.0:
        x_left = x0 - z0 / slope
    elif z_left > z_max:
        x_left = x0 + (z_max - z0) / slope
    else:
        x_left = 0.0
    z_right = z0 + slope * (x_max - x0)
    if z_right < 0.0:
        x_right = x0 - z0 / slope
    elif z_right > z_max:
        x_right = x0 + (z_max - z0) / slope
    else:
        x_right = x_max
    return x_left, x_right


def _fit_profiles(scan: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sample_x = scan[..., 0]
    sample_z = scan[..., 2]
    position = np.column_stack(
        (sample_x.mean(axis=1), np.zeros(scan.shape[0]), sample_z.mean(axis=1))
    )
    direction = np.empty_like(position)
    for index in range(scan.shape[0]):
        slope = float(np.polyfit(sample_x[index], sample_z[index], 1)[0])
        theta = np.arctan(slope)
        direction[index] = (np.cos(theta), 0.0, np.sin(theta))
    return position, direction


def _normal(rng: np.random.Generator, shape: tuple[int, ...], kind: str = "normal") -> np.ndarray:
    """MATLAB column-major ``randn`` / ``rand`` of the requested shape."""
    count = int(np.prod(shape))
    draw = rng.normal(size=count) if kind == "normal" else rng.random(size=count)
    return draw.reshape(shape, order="F")

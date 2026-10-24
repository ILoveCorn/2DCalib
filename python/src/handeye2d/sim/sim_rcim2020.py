"""Single-plane simulator from Sharifzadeh, RCIM 2020.

Nothing in ``evaluation/`` calls this function. It is ported so the API matches
``fun/sim``. Random draws follow the MATLAB order: all orientation noise, then
each scan's image point and averaged image noise, then the in-plane positions,
then position noise. A zero standard deviation builds a zero factor and still
consumes the corresponding ``randn``.

Euler angles are ``linspace`` samples of yaw, pitch, and roll, converted with
``ZXY``. ``lOn`` is ``(m, 6)``, position then direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from handeye2d.rotations import eul_to_rotm
from handeye2d.sim._draw import draw
from handeye2d.so3 import exp

_POINTS_PER_SCAN = 20
_NOISE_REPEATS = 10


@dataclass(frozen=True)
class Rcim2020Simulation:
    tH2Wn: np.ndarray
    RH2Wn: np.ndarray
    lOn: np.ndarray


def sim_rcim2020(
    pP: np.ndarray,
    nP: np.ndarray,
    lP: float,
    xMax: float,
    zMax: float,
    tO2H: np.ndarray,
    RO2H: np.ndarray,
    et: float,
    eR: float,
    eO: float,
    m: int,
    rng: np.random.Generator,
) -> Rcim2020Simulation:
    """Draw one RCIM2020 plane session. ``m`` is the number of measurements."""
    center = np.asarray(pP, dtype=float).reshape(3)
    normal = np.asarray(nP, dtype=float).reshape(3)
    normal = normal / np.linalg.norm(normal)
    hand_t = np.asarray(tO2H, dtype=float).reshape(3)
    hand_R = np.asarray(RO2H, dtype=float).reshape(3, 3)
    count = int(m)
    position_chol = _isotropic(float(et), 3, 3.0)
    orientation_chol = _isotropic(float(eR), 3, 3.0)
    image_chol = _isotropic(float(eO), 2, 2.0)
    _, _, plane_R, sensor_r0 = _plane_frame(normal)
    yaw = np.linspace(0.0, 2.0 * np.pi, count)
    pitch = np.linspace(-np.pi / 4.0, np.pi / 4.0, count)
    roll = np.linspace(-np.pi / 4.0, np.pi / 4.0, count)
    euler = np.column_stack((yaw, pitch, roll))
    clean_R = np.einsum("ij,mjk->mik", sensor_r0, eul_to_rotm(euler, "ZXY")) @ hand_R.T
    orientation_noise = orientation_chol @ draw(rng, (3, count))
    noisy_R = clean_R @ exp(orientation_noise.T)

    image = np.zeros((count, 3))
    direction = np.zeros((count, 3))
    for index in range(count):
        image_x = xMax * rng.random()
        image_z = zMax * rng.random()
        image[index] = (image_x, 0.0, image_z)
        slope = float(np.tan(roll[index]))
        left, right = _segment_limits(image_x, image_z, slope, xMax, zMax)
        sample_x = np.linspace(left, right, _POINTS_PER_SCAN)
        sample_z = image_z + slope * (sample_x - image_x)
        noise_draw = draw(rng, (2, _POINTS_PER_SCAN, _NOISE_REPEATS))
        noise = np.einsum("ij,j...->i...", image_chol, noise_draw).mean(axis=2)
        sample_x = sample_x + noise[0]
        sample_z = sample_z + noise[1]
        fitted = float(np.polyfit(sample_x, sample_z, 1)[0])
        theta = np.arctan(fitted)
        direction[index] = (np.cos(theta), 0.0, np.sin(theta))

    line_uv = draw(rng, (2, count), kind="uniform")
    line_plane = np.zeros((3, count))
    line_plane[:2] = (float(lP) / 2.0) * (2.0 * line_uv - 1.0)
    world = (plane_R @ line_plane).T + center
    mapped = image @ hand_R.T + hand_t
    clean_t = world - np.einsum("mij,mj->mi", clean_R, mapped)
    position_noise = (position_chol @ draw(rng, (3, count))).T
    return Rcim2020Simulation(
        tH2Wn=clean_t + position_noise,
        RH2Wn=noisy_R,
        lOn=np.concatenate((image, direction), axis=1),
    )


def _isotropic(std: float, size: int, divisor: float) -> np.ndarray:
    if std == 0.0:
        return np.zeros((size, size))
    return (std / np.sqrt(divisor)) * np.eye(size)


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
    sensor_r0 = np.column_stack((-axis_x, axis_y, -normal))
    return axis_x, axis_y, plane_R, sensor_r0


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

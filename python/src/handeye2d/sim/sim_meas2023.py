"""Sphere-target simulator for Tang, MEAS 2023.

``fun/sim/simMEAS2023.m:90`` clips the sample arc with ``zMax``. This port
clips it with ``xMax``, which is the lateral field of view. The Octave
fixtures use the same correction. A partial arc that stops at ``zMax`` changes
the simulated point distribution and, once image noise is present, the
calibration error.

Random draws follow the MATLAB order: one ``randSO3`` sample, orientation
noise, the two circle-center uniforms, image noise, then position noise.
``tH2Wn`` and ``pcOe`` are ``(m, mt, 3)``. ``RH2Wn`` is the hand orientation
before the per-position noise, shape ``(m, 3, 3)``. NumPy's stream is not
Octave's, so a seed does not reproduce ``sim_meas2023.mat``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from handeye2d.fitting import circle_fit
from handeye2d.rotations import eul_to_rotm
from handeye2d.sim._draw import draw
from handeye2d.so3 import exp, random_rotvec

_POINTS_PER_SCAN = 30


@dataclass(frozen=True)
class SphereSimulation:
    tH2Wn: np.ndarray
    RH2Wn: np.ndarray
    pcOe: np.ndarray


def sim_meas2023(
    pS: np.ndarray,
    rS: float,
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
    mt: int,
    rng: np.random.Generator,
) -> SphereSimulation:
    """Draw one sphere session. ``rS`` is the radius in millimeters."""
    sphere = np.asarray(pS, dtype=float).reshape(3)
    radius = float(rS)
    hand_t = np.asarray(tO2H, dtype=float).reshape(3)
    hand_R = np.asarray(RO2H, dtype=float).reshape(3, 3)
    n_pose = int(mAlpha) * int(mBeta)
    n_position = int(mt)
    position_chol = np.linalg.cholesky(np.asarray(Sigmat, dtype=float))
    orientation_chol = np.linalg.cholesky(np.asarray(SigmaR, dtype=float))
    image_std = float(sigmaO)
    image_chol = np.zeros((2, 2)) if image_std == 0.0 else image_std * np.eye(2)

    sensor_r0 = exp(random_rotvec(rng))
    beta, alpha = np.meshgrid(
        np.linspace(-betaMax, betaMax, int(mBeta)),
        np.linspace(-alphaMax, alphaMax, int(mAlpha)),
        indexing="ij",
    )
    euler = np.column_stack((np.zeros(n_pose), alpha.ravel(order="F"), beta.ravel(order="F")))
    clean_R = np.einsum("ij,mjk->mik", sensor_r0, eul_to_rotm(euler, "ZYX")) @ hand_R.T
    orientation = np.einsum("ij,jtk->itk", orientation_chol, draw(rng, (3, n_position, n_pose)))
    rotvec = np.moveaxis(orientation, 0, -1).transpose(1, 0, 2)
    noisy_R = clean_R[:, None] @ exp(rotvec)

    center_x = xMax / 2.0 + 0.25 * xMax * (2.0 * rng.random(n_position) - 1.0)
    center_y = np.linspace(-0.8 * radius, 0.8 * radius, n_position)
    circle_radius = np.sqrt(radius**2 - center_y**2)
    center_z = circle_radius + (zMax - circle_radius) * rng.random(n_position)
    center_z = np.where(circle_radius >= zMax, circle_radius, center_z)
    clean = np.zeros((n_pose, n_position, _POINTS_PER_SCAN, 3))
    for index in range(n_position):
        left, right = sample_limits(float(center_x[index]), float(circle_radius[index]), xMax)
        sample_x = np.linspace(left, right, _POINTS_PER_SCAN)
        offset = sample_x - center_x[index]
        sample_z = center_z[index] - np.sqrt(np.maximum(circle_radius[index] ** 2 - offset**2, 0.0))
        clean[:, index, :, 0] = sample_x
        clean[:, index, :, 2] = sample_z
    image_noise = np.einsum(
        "ij,j...->i...", image_chol, draw(rng, (2, _POINTS_PER_SCAN, n_position, n_pose))
    )
    observed = clean.copy()
    # draw is (2, n_points, mt, m). Profiles are stored (m, mt, n_points).
    observed[..., 0] += np.transpose(image_noise[0], (2, 1, 0))
    observed[..., 2] += np.transpose(image_noise[1], (2, 1, 0))

    fitted = _fit_centers(observed, center_y, radius)
    clean_center = np.broadcast_to(
        np.stack((center_x, center_y, center_z), axis=-1), (n_pose, n_position, 3)
    ).copy()
    center_h = np.einsum("ij,rtj->rti", hand_R, clean_center) + hand_t
    clean_t = sphere - np.einsum("rtij,rtj->rti", noisy_R, center_h)
    position_noise = np.einsum("ij,jtk->itk", position_chol, draw(rng, (3, n_position, n_pose)))
    return SphereSimulation(
        tH2Wn=clean_t + np.moveaxis(position_noise, 0, -1).transpose(1, 0, 2),
        RH2Wn=clean_R,
        pcOe=fitted,
    )


def sample_limits(center_x: float, radius: float, x_max: float) -> tuple[float, float]:
    """Arc endpoints inside the lateral field of view.

    The right edge uses ``x_max``. The MATLAB file uses ``zMax`` there.
    """
    left = max(center_x - radius, 0.0) + 1e-4
    right = min(center_x + radius, x_max) - 1e-4
    return left, right


def _fit_centers(observed: np.ndarray, center_y: np.ndarray, radius: float) -> np.ndarray:
    n_pose, n_position = observed.shape[:2]
    fitted = np.zeros((n_pose, n_position, 3))
    for pose in range(n_pose):
        for position in range(n_position):
            profile = observed[pose, position]
            fit_x, fit_z, fit_radius = circle_fit(profile[:, 0], profile[:, 2])
            if fit_radius > radius:
                fit_radius = radius
            fit_y = float(np.sign(center_y[position]) * np.sqrt(radius**2 - fit_radius**2))
            fitted[pose, position] = (fit_x, fit_y, fit_z)
    return fitted

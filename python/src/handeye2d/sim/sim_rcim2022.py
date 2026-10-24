"""Line-target simulator for Xu, RCIM 2022.

Random draws follow the MATLAB order: orientation noise, in-plane position,
image point, image noise, then position noise. The returned ``RH2Wn`` is the
hand orientation before the per-position noise; that noise is folded into
``tH2Wn``. ``tH2Wn`` and ``pOn`` are ``(m, mt, 3)`` and ``RH2Wn`` is ``(m, 3, 3)``.

``Sigmat`` and ``SigmaR`` must be positive definite. A zero ``sigmaO`` still
consumes the image-noise draw.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from handeye2d.rotations import eul_to_rotm
from handeye2d.sim._draw import draw
from handeye2d.so3 import exp


@dataclass(frozen=True)
class LineSimulation:
    tH2Wn: np.ndarray
    RH2Wn: np.ndarray
    pOn: np.ndarray


def sim_rcim2022(
    pL: np.ndarray,
    vL: np.ndarray,
    lL: float,
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
) -> LineSimulation:
    """Draw one line-target session."""
    origin = np.asarray(pL, dtype=float).reshape(3)
    direction = np.asarray(vL, dtype=float).reshape(3)
    hand_t = np.asarray(tO2H, dtype=float).reshape(3)
    hand_R = np.asarray(RO2H, dtype=float).reshape(3, 3)
    n_pose = int(mAlpha) * int(mBeta)
    n_position = int(mt)
    position_chol = np.linalg.cholesky(np.asarray(Sigmat, dtype=float))
    orientation_chol = np.linalg.cholesky(np.asarray(SigmaR, dtype=float))
    image_std = float(sigmaO)
    image_chol = np.zeros((2, 2)) if image_std == 0.0 else image_std * np.eye(2)

    sensor_r0 = _sensor_frame(direction)
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

    along = draw(rng, (n_position, n_pose), kind="uniform")
    world = origin + direction * float(lL) * along.T[..., None]
    image_x = xMax * draw(rng, (n_position, n_pose), kind="uniform").T
    image_z = zMax * draw(rng, (n_position, n_pose), kind="uniform").T
    clean = np.stack((image_x, np.zeros((n_pose, n_position)), image_z), axis=-1)
    image_noise = np.einsum("ij,jtk->itk", image_chol, draw(rng, (2, n_position, n_pose)))
    observed = clean.copy()
    observed[..., 0] += image_noise[0].T
    observed[..., 2] += image_noise[1].T

    mapped = np.einsum("ij,rtj->rti", hand_R, clean) + hand_t
    clean_t = world - np.einsum("rtij,rtj->rti", noisy_R, mapped)
    position_noise = np.einsum("ij,jtk->itk", position_chol, draw(rng, (3, n_position, n_pose)))
    return LineSimulation(
        tH2Wn=clean_t + np.moveaxis(position_noise, 0, -1).transpose(1, 0, 2),
        RH2Wn=clean_R,
        pOn=observed,
    )


def _sensor_frame(direction: np.ndarray) -> np.ndarray:
    if direction[0] == 0.0 and direction[1] == 0.0:
        axis_x = np.array([1.0, 0.0, 0.0])
    else:
        axis_x = np.array([-direction[1], direction[0], 0.0])
        axis_x = axis_x / np.linalg.norm(axis_x)
    axis_z = np.cross(axis_x, direction)
    return np.column_stack((axis_x, direction, axis_z))

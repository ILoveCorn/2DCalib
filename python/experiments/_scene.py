"""Random scenes in the same draw order as the MATLAB experiment scripts.

NumPy's generator is not MATLAB's ``rng``. The order below is what makes one
trial's methods share one plane cloud; it does not reproduce a MATLAB seed.
"""

from __future__ import annotations

import numpy as np

from experiments._common import LENGTH, N_PLANES
from handeye2d.rotations import axang_to_rotm
from handeye2d.so3 import exp, random_rotvec


def draw_hand_eye(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """``tO2H = 50*(2*rand-1)``, ``RO2H = ExpSO3(randSO3())``."""
    translation = 50.0 * (2.0 * rng.random(3) - 1.0)
    rotation = exp(random_rotvec(rng))
    return translation, rotation


def draw_unit_uniform(rng: np.random.Generator) -> np.ndarray:
    direction = rng.random(3)
    return direction / np.linalg.norm(direction)


def draw_sigma0(rng: np.random.Generator, sigmat: float, sigma_r: float) -> np.ndarray:
    """``[kt * sigmat, kR * sigmaR]`` with both factors uniform on ``[0.75, 1.25]``."""
    factor_t = 0.75 + 0.5 * rng.random()
    factor_r = 0.75 + 0.5 * rng.random()
    return np.array([factor_t * sigmat, factor_r * sigma_r])


def draw_plane_init(rng: np.random.Generator, normal: np.ndarray) -> tuple[np.ndarray, float]:
    """Perturbed normal. The plane center is the origin, so ``d0`` is 0."""
    guess = normal + 0.05 * (2.0 * rng.random(3) - 1.0)
    guess = guess / np.linalg.norm(guess)
    return guess, 0.0


def draw_box_init(
    rng: np.random.Generator,
    translation: np.ndarray,
    rotation: np.ndarray,
    dt: float,
    dtheta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Position and angle offsets are uniform on ``[-dt, dt]`` and ``[-dtheta, dtheta]``."""
    initial_t = translation + dt * (2.0 * rng.random(3) - 1.0)
    axis = rng.random(3)
    angle = dtheta * (2.0 * rng.random() - 1.0)
    initial_R = rotation @ axang_to_rotm(axis, np.asarray(angle))
    return initial_t, initial_R


def draw_fixed_init(
    rng: np.random.Generator,
    translation: np.ndarray,
    rotation: np.ndarray,
    dt: float,
    dtheta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed magnitudes along a random direction and about a random axis."""
    direction = rng.normal(size=3)
    direction = direction / np.linalg.norm(direction)
    axis = rng.normal(size=3)
    axis = axis / np.linalg.norm(axis)
    initial_t = translation + dt * direction
    initial_R = rotation @ axang_to_rotm(axis, np.asarray(dtheta))
    return initial_t, initial_R


def draw_multiplane_normals(rng: np.random.Generator) -> np.ndarray:
    """Rows are the MATLAB columns of ``ExpSO3(randSO3())``."""
    return exp(random_rotvec(rng)).T


def multiplane_centers() -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((N_PLANES, 3)), np.full(N_PLANES, LENGTH)

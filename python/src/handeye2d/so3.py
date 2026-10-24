"""SO(3) utilities.

``hat`` / ``exp`` / ``log`` / ``jr`` cover both the scalar MATLAB functions and
the ``page*`` variants. Vectors are ``(..., 3)`` and matrices are ``(..., 3, 3)``,
so the batch axes come first.

``ExpSO3`` treats every nonzero angle as exact and ``pageExpSO3`` switches to a
Taylor coefficient below ``1e-10``. ``JrSO3`` / ``invJrSO3`` use ``1e-15`` and
return the identity, while ``pageJrSO3`` uses ``1e-10`` and the Taylor
coefficients. This port uses ``1e-10`` everywhere, with those Taylor
coefficients, so a single rotation and a batch of one agree.
"""

from __future__ import annotations

import numpy as np

_SMALL_ANGLE = 1e-10


def hat(vector: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix of a vector with last axis ``(3,)``."""
    v = np.asarray(vector, dtype=float)
    if v.shape[-1] != 3:
        raise ValueError(f"hat expected (..., 3), got {v.shape}")
    x = v[..., 0]
    y = v[..., 1]
    z = v[..., 2]
    zero = np.zeros(v.shape[:-1], dtype=float)
    return np.stack(
        (
            np.stack((zero, -z, y), axis=-1),
            np.stack((z, zero, -x), axis=-1),
            np.stack((-y, x, zero), axis=-1),
        ),
        axis=-2,
    )


def exp(rotvec: np.ndarray) -> np.ndarray:
    """Exponential map. ``rotvec`` is ``(..., 3)`` and the result is ``(..., 3, 3)``.

    Angles below ``1e-10`` use ``sin(θ)/θ → 1`` and ``(1-cosθ)/θ² → 1/2``.
    """
    w = np.asarray(rotvec, dtype=float)
    theta = np.linalg.norm(w, axis=-1, keepdims=True)
    skew = hat(w)
    skew2 = skew @ skew
    small = theta < _SMALL_ANGLE
    theta_safe = np.where(small, 1.0, theta)
    c1 = np.where(small, 1.0, np.sin(theta_safe) / theta_safe)
    c2 = np.where(small, 0.5, (1.0 - np.cos(theta_safe)) / np.square(theta_safe))
    return np.eye(3) + c1[..., None] * skew + c2[..., None] * skew2


def log(rotation: np.ndarray) -> np.ndarray:
    """Logarithm map using the clamped-``acos`` form of ``pageLogSO3``.

    ``rotation`` is ``(..., 3, 3)`` and the result is ``(..., 3)``. The
    coefficient is ``θ / (2 sin θ)``, and ``1/2`` when ``θ < 1e-10``. This is
    the same expression as the MATLAB code, including near angle ``π``, where
    ``sin θ`` is almost zero.
    """
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(f"log expected (..., 3, 3), got {rotation.shape}")
    trace = np.trace(rotation, axis1=-2, axis2=-1)
    cosine = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cosine)
    vee = np.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        axis=-1,
    )
    small = theta < _SMALL_ANGLE
    theta_safe = np.where(small, 1.0, theta)
    coeff = np.where(small, 0.5, theta / (2.0 * np.sin(theta_safe)))
    return coeff[..., None] * vee


def jr(rotvec: np.ndarray) -> np.ndarray:
    """Right Jacobian of SO(3). ``rotvec`` is ``(..., 3)`` and the result is ``(..., 3, 3)``.

    Below ``1e-10`` the coefficients are the limits ``1/2`` and ``1/6``.
    ``Jr(w) @ w = w``, and ``exp(w + dw) ≈ exp(w) @ exp(jr(w) @ dw)``.
    """
    w = np.asarray(rotvec, dtype=float)
    theta = np.linalg.norm(w, axis=-1, keepdims=True)
    skew = hat(w)
    skew2 = skew @ skew
    small = theta < _SMALL_ANGLE
    theta_safe = np.where(small, 1.0, theta)
    c1 = np.where(small, 0.5, (1.0 - np.cos(theta_safe)) / np.square(theta_safe))
    c2 = np.where(
        small,
        1.0 / 6.0,
        (theta_safe - np.sin(theta_safe)) / theta_safe**3,
    )
    return np.eye(3) - c1[..., None] * skew + c2[..., None] * skew2


def jr_inv(rotvec: np.ndarray) -> np.ndarray:
    """Inverse of :func:`jr`.

    Below ``1e-10`` the ``hat(w)²`` coefficient uses its limit ``1/12``
    (``invJrSO3`` returns the identity below ``1e-15``; the two agree to
    well under ``1e-10`` because ``hat(w)`` itself is that small).
    """
    w = np.asarray(rotvec, dtype=float)
    theta = np.linalg.norm(w, axis=-1, keepdims=True)
    skew = hat(w)
    skew2 = skew @ skew
    small = theta < _SMALL_ANGLE
    theta_safe = np.where(small, 1.0, theta)
    coeff = np.where(
        small,
        1.0 / 12.0,
        1.0 / np.square(theta_safe)
        - (1.0 + np.cos(theta_safe)) / (2.0 * theta_safe * np.sin(theta_safe)),
    )
    return np.eye(3) + 0.5 * skew + coeff[..., None] * skew2


def random_rotvec(rng: np.random.Generator) -> np.ndarray:
    """One rotation vector from the Haar measure on SO(3), as ``randSO3``.

    The candidate is uniform in the open ball of radius ``π`` (spherical
    coordinates, radius ``π U^(1/3)``). It is kept when a fresh uniform draw
    is below ``(2 - 2 cos‖w‖) / ‖w‖²``. ``rng`` is a ``numpy.random.Generator``;
    NumPy's stream is not MATLAB's.
    """
    while True:
        azimuth = 2.0 * np.pi * rng.random()
        elevation = np.arcsin(2.0 * rng.random() - 1.0)
        radius = np.pi * rng.random() ** (1.0 / 3.0)
        cosine = np.cos(elevation)
        rotvec = np.array(
            [
                radius * cosine * np.cos(azimuth),
                radius * cosine * np.sin(azimuth),
                radius * np.sin(elevation),
            ]
        )
        norm = float(np.linalg.norm(rotvec))
        if norm < 1e-15:
            return rotvec
        density = (2.0 - 2.0 * np.cos(norm)) / norm**2
        if rng.random() < density:
            return rotvec

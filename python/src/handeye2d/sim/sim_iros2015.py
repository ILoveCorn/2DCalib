"""Multi-plane sessions built by calling ``sim_plane`` once per plane.

``mAlpha`` and ``mBeta`` are reduced with MATLAB ``round(m / sqrt(nPlane))``
before each call. Random draws stay on one generator, in plane order, so the
consumption order matches ``simIROS2015``.

Returns batch-first arrays with the plane axis first:
``tH2Wn`` and ``pOn`` are ``(n_planes, m_i, 3)``, ``RH2Wn`` is
``(n_planes, m_i, 3, 3)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from handeye2d.sim.sim_plane import sim_plane


@dataclass(frozen=True)
class IrosSimulation:
    tH2Wn: np.ndarray
    RH2Wn: np.ndarray
    pOn: np.ndarray


def sim_iros2015(
    nPlane: int,
    pPs: np.ndarray,
    nPs: np.ndarray,
    lPs: np.ndarray,
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
) -> IrosSimulation:
    """Draw one multi-plane session."""
    centers = np.asarray(pPs, dtype=float).reshape(int(nPlane), 3)
    normals = np.asarray(nPs, dtype=float).reshape(int(nPlane), 3)
    lengths = np.asarray(lPs, dtype=float).reshape(int(nPlane))
    per_alpha = _matlab_round(int(mAlpha) / np.sqrt(int(nPlane)))
    per_beta = _matlab_round(int(mBeta) / np.sqrt(int(nPlane)))
    translations = []
    rotations = []
    points = []
    for index in range(int(nPlane)):
        drawn = sim_plane(
            centers[index],
            normals[index],
            float(lengths[index]),
            xMax,
            zMax,
            tO2H,
            RO2H,
            Sigmat,
            SigmaR,
            sigmaO,
            alphaMax,
            betaMax,
            per_alpha,
            per_beta,
            rng,
        )
        translations.append(drawn.tH2Wn)
        rotations.append(drawn.RH2Wn)
        points.append(drawn.pOn)
    return IrosSimulation(
        tH2Wn=np.stack(translations),
        RH2Wn=np.stack(rotations),
        pOn=np.stack(points),
    )


def _matlab_round(value: float) -> int:
    """MATLAB ``round``: half away from zero."""
    sign = 1.0 if value >= 0.0 else -1.0
    return int(np.floor(abs(value) + 0.5) * sign)

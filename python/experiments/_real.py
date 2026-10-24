"""Paths and the saved real-calibration arrays shared by the two drivers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

METHODS = (
    "1 - Proposed",
    "2 - TRO-2016",
    "3 - RCIM-2020",
    "4 - MEAS-2023",
    "5 - Measured",
)
SIGMA0 = np.array([0.1, 0.01])
IIMAX = 30
IOMAX = 50
TOL = 1e-5
MT = 3
DIAMETER = 14.9981
MEASURE_MAT = "measure_260219_1144.mat"
REFERENCE_MAT = "real_calibration_260219_1146.mat"
N_PLANE_METHODS = 3


@dataclass(frozen=True)
class RealCalibration:
    """MATLAB layout: translation ``(3, n, 5)``, rotation ``(3, 3, n, 5)``, time ``(n, 5)``."""

    translation: np.ndarray
    rotation: np.ndarray
    time: np.ndarray
    plane_files: tuple[str, ...]
    sphere_files: tuple[str, ...]


def repo_root() -> Path:
    """Repository root. This file lives in ``python/experiments``."""
    root = Path(__file__).resolve().parents[2]
    plane = root / "data" / "oct" / "plane"
    if not plane.is_dir():
        raise FileNotFoundError(f"plane sessions are not under {root}")
    return root


def session_names(directory: Path, pattern: str) -> tuple[str, ...]:
    """Sorted YAML names. The saved result uses this order after the rename."""
    return tuple(sorted(path.name for path in directory.glob(pattern)))


def matlab_cells(value: np.ndarray) -> list[str]:
    """Flatten a MATLAB cell array of char vectors."""
    rows = []
    for item in np.asarray(value, dtype=object).ravel():
        if isinstance(item, np.ndarray):
            rows.append("".join(np.asarray(item).ravel().astype(str)))
        else:
            rows.append(str(item))
    return rows


def measured_poses(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Fiducial ``tO2H (3, n)`` and ``RO2H (3, 3, n)`` from the measure mat."""
    loaded = loadmat(root / "data" / MEASURE_MAT, squeeze_me=False)
    translation = np.asarray(loaded["tO2H"], dtype=float)
    rotation = np.asarray(loaded["RO2H"], dtype=float)
    if translation.ndim == 3 and translation.shape[1] == 1:
        translation = translation[:, 0, :]
    if translation.ndim != 2 or translation.shape[0] != 3:
        raise ValueError(f"tO2H expected (3, n) or (3, 1, n), got {loaded['tO2H'].shape}")
    if rotation.shape != (3, 3, translation.shape[1]):
        raise ValueError(f"RO2H expected (3, 3, n), got {rotation.shape}")
    return translation, rotation


def save_calibration(path: Path, result: RealCalibration) -> None:
    """Write the arrays ``real_calibration.m`` saves, with current plane file names."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        alltO2He=result.translation,
        allRO2He=result.rotation,
        Time=result.time,
        planeFiles=np.asarray(result.plane_files),
        sphereFiles=np.asarray(result.sphere_files),
        testedMethods=np.asarray(METHODS),
        sigma0=np.asarray(SIGMA0, dtype=float),
        iImax=np.int32(IIMAX),
        iOmax=np.int32(IOMAX),
        tol=np.float64(TOL),
        D=np.float64(DIAMETER),
    )


def load_calibration(path: Path) -> RealCalibration:
    """Load a driver ``.npz`` or the saved MATLAB ``.mat``.

    Legacy ``oct_checkerboard_*`` names in the mat are mapped to ``oct_plane_*``.
    """
    path = Path(path)
    if path.suffix == ".mat":
        loaded = loadmat(path, squeeze_me=False)
        planes = tuple(
            name.replace("oct_checkerboard", "oct_plane")
            for name in matlab_cells(loaded["planeFiles"])
        )
        spheres = tuple(matlab_cells(loaded["sphereFiles"]))
        return RealCalibration(
            translation=np.asarray(loaded["alltO2He"], dtype=float),
            rotation=np.asarray(loaded["allRO2He"], dtype=float),
            time=np.asarray(loaded["Time"], dtype=float),
            plane_files=planes,
            sphere_files=spheres,
        )
    loaded = np.load(path)
    return RealCalibration(
        translation=np.asarray(loaded["alltO2He"], dtype=float),
        rotation=np.asarray(loaded["allRO2He"], dtype=float),
        time=np.asarray(loaded["Time"], dtype=float),
        plane_files=tuple(str(name) for name in np.asarray(loaded["planeFiles"]).ravel()),
        sphere_files=tuple(str(name) for name in np.asarray(loaded["sphereFiles"]).ravel()),
    )


def reference_calibration(root: Path | None = None) -> RealCalibration:
    """The committed ``result/real_calibration_260219_1146.mat``."""
    root = repo_root() if root is None else Path(root)
    return load_calibration(root / "result" / REFERENCE_MAT)

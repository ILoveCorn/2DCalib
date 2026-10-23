"""Load Octave v7 fixtures and fingerprint their numeric arrays.

MAT-file headers include a timestamp, so integrity checks hash array bytes
rather than the file itself.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.io import loadmat

EXPECTED_MATS = (
    "bug_impact.mat",
    "line_rcim2022.mat",
    "plane_alt_gh.mat",
    "plane_iros2015.mat",
    "plane_jnt_gh.mat",
    "plane_jnt_gh_decoupled.mat",
    "plane_rcim2020.mat",
    "shim_reference.mat",
    "sim_iros2015.mat",
    "sim_meas2023.mat",
    "sim_plane.mat",
    "sim_rcim2020.mat",
    "sim_rcim2022.mat",
    "sphere_meas2023.mat",
)


def load_fixture(path: Path) -> dict[str, np.ndarray]:
    raw = loadmat(Path(path), squeeze_me=False, struct_as_record=False, mat_dtype=True)
    return {key: np.asarray(value) for key, value in raw.items() if not key.startswith("__")}


def fingerprint(path: Path) -> str:
    data = load_fixture(path)
    digest = hashlib.sha256()
    for key in sorted(data):
        array = np.ascontiguousarray(data[key])
        digest.update(key.encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def scalar(value: np.ndarray) -> float:
    return float(np.reshape(np.asarray(value), ()))

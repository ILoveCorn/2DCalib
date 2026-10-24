"""OCT YAML readers.

These follow the OCT branches of ``loadPlaneYAML``, ``loadTestYAML``, and
``loadMEAS2023YAML``. The LiDAR branch is not ported. Paths may point anywhere;
nothing is resolved through MATLAB's search path.

Positions in the files are meters and are returned in millimeters. Quaternions
are scalar-first. Profile points stay in the OCT frame, with ``y = 0``.

Batch axes come first:

* ``load_plane_yaml`` returns ``tH2Wn (m, 3)``, ``RH2Wn (m, 3, 3)``,
  ``pOn (m, 3)``, ``vOn (m, 3)``, ``scanO (m, n_points, 3)``. ``pOn`` and
  ``vOn`` are the mean and the degree-1 ``polyfit`` of each original profile.
  ``scanO`` is a separate resample to ``min(5000, shortest scan)`` with
  MATLAB's ``round(linspace)`` (half away from zero), which is not ``np.round``.
* ``load_test_yaml`` keeps each scan at its own length: a tuple of ``(n_i, 3)``
  arrays. ``center_sign_y`` is ``-sign_y`` from ``pose.center_side``, or ``0``
  when that field is absent.
* ``load_meas2023_yaml`` groups consecutive captures, ``mt`` per orientation,
  in the same order as MATLAB's column-major ``reshape``. ``RH2Wn`` is the
  first pose of each group. A fitted radius above the sphere radius is clipped
  before recovering ``y``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from handeye2d.fitting import circle_fit
from handeye2d.rotations import quat_to_rotm

_MAX_OCT_POINTS = 5000


def load_plane_yaml(
    filename: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one plane session."""
    captures = _captures(filename)
    positions = []
    quaternions = []
    scans = []
    for capture in captures:
        scans.append(_profile(capture))
        positions.append(_position_mm(capture))
        quaternions.append(_quaternion(capture))
    # loadPlaneYAML fits the intersection on the raw profile, then resamples
    # scanO down to the shortest capture. The two steps are not the same points.
    intersection, direction = _line_from_profiles(scans)
    count = _shortest(scans, _MAX_OCT_POINTS)
    scan = np.stack([_resample(points, count) for points in scans], axis=0)
    return (
        np.stack(positions, axis=0),
        quat_to_rotm(np.stack(quaternions, axis=0)),
        intersection,
        direction,
        scan,
    )


def load_test_yaml(
    filename: str | Path,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...], np.ndarray]:
    """Load one session and keep the raw profiles."""
    captures = _captures(filename)
    positions = []
    quaternions = []
    scans = []
    signs = []
    for capture in captures:
        positions.append(_position_mm(capture))
        quaternions.append(_quaternion(capture))
        scans.append(_profile(capture))
        center_side = capture.get("pose", {}).get("center_side")
        if isinstance(center_side, dict) and "sign_y" in center_side:
            signs.append(-float(center_side["sign_y"]))
        else:
            signs.append(0.0)
    return (
        np.stack(positions, axis=0),
        quat_to_rotm(np.stack(quaternions, axis=0)),
        tuple(scans),
        np.asarray(signs, dtype=float),
    )


def load_meas2023_yaml(
    filename: str | Path,
    mt: int,
    diameter: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[tuple[np.ndarray, ...], ...]]:
    """Load a sphere session the way MEAS2023 consumes it.

    ``mt`` is the number of positions per orientation and ``diameter`` is the
    sphere diameter in millimeters. Returns ``tH2Wn (m_r, mt, 3)``,
    ``RH2Wn (m_r, 3, 3)``, ``pcOe (m_r, mt, 3)``, and ``scanO`` grouped as
    ``scanO[i_r][i_t]`` with shape ``(n_points, 3)``.
    """
    if mt < 1:
        raise ValueError("mt must be positive")
    translation, rotation, scans, signs = load_test_yaml(filename)
    count = translation.shape[0]
    if count % mt != 0:
        raise ValueError(f"{count} captures do not divide into groups of {mt}")
    n_orientation = count // mt
    translation = translation.reshape(n_orientation, mt, 3)
    rotation = rotation[::mt]
    grouped = tuple(
        tuple(scans[i_r * mt + i_t] for i_t in range(mt)) for i_r in range(n_orientation)
    )
    sign = signs.reshape(n_orientation, mt)
    radius_limit = float(diameter) / 2.0
    centers = np.empty((n_orientation, mt, 3), dtype=float)
    for i_r in range(n_orientation):
        for i_t in range(mt):
            profile = grouped[i_r][i_t]
            center_x, center_z, radius = circle_fit(profile[:, 0], profile[:, 2])
            if radius > radius_limit:
                radius = radius_limit
            center_y = sign[i_r, i_t] * np.sqrt(radius_limit**2 - radius**2)
            centers[i_r, i_t] = (center_x, center_y, center_z)
    return translation, rotation, centers, grouped


def _captures(filename: str | Path) -> list[dict]:
    path = Path(filename)
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    with path.open(encoding="utf-8") as handle:
        document = yaml.load(handle, Loader=loader)
    captures = document["captures"]
    if not captures:
        raise ValueError(f"{path} has no captures")
    return captures


def _position_mm(capture: dict) -> np.ndarray:
    position = capture["pose"]["position"]
    return 1000.0 * np.array([position["x"], position["y"], position["z"]], dtype=float)


def _quaternion(capture: dict) -> np.ndarray:
    orientation = capture["pose"]["orientation"]
    return np.array(
        [orientation["w"], orientation["x"], orientation["y"], orientation["z"]],
        dtype=float,
    )


def _profile(capture: dict) -> np.ndarray:
    points = capture.get("points")
    if not isinstance(points, dict) or "x" not in points or "z" not in points:
        raise ValueError(
            "OCT captures require points.x and points.z; the LiDAR branch is not ported"
        )
    x = np.asarray(points["x"], dtype=float).reshape(-1)
    z = np.asarray(points["z"], dtype=float).reshape(-1)
    if x.shape != z.shape:
        raise ValueError("points.x and points.z have different lengths")
    return np.column_stack((x, np.zeros(x.shape, dtype=float), z))


def _shortest(scans: list[np.ndarray], limit: int) -> int:
    return min(limit, min(scan.shape[0] for scan in scans))


def _resample(points: np.ndarray, count: int) -> np.ndarray:
    if points.shape[0] <= count:
        return points
    return points[_matlab_indices(points.shape[0], count)]


def _matlab_indices(length: int, count: int) -> np.ndarray:
    """0-based ``round(linspace(1, length, count))`` with MATLAB half-up rounding."""
    grid = np.linspace(1.0, float(length), count)
    return np.floor(grid + 0.5).astype(int) - 1


def _line_from_profiles(scans: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """``pOn`` and ``vOn`` from each original profile, before resampling."""
    position = np.zeros((len(scans), 3))
    direction = np.zeros((len(scans), 3))
    for index, scan in enumerate(scans):
        x = scan[:, 0]
        z = scan[:, 2]
        position[index, 0] = float(np.mean(x))
        position[index, 2] = float(np.mean(z))
        slope = float(np.polyfit(x, z, 1)[0])
        theta = np.arctan(slope)
        direction[index, 0] = float(np.cos(theta))
        direction[index, 2] = float(np.sin(theta))
    return position, direction

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from handeye2d import io
from handeye2d.io import _matlab_indices


def _capture(
    x: list[float],
    z: list[float],
    position: tuple[float, float, float],
    sign: int | None,
):
    pose: dict = {
        "position": {"x": position[0], "y": position[1], "z": position[2]},
        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }
    if sign is not None:
        pose["center_side"] = {"sign_y": sign}
    return {"points": {"x": x, "z": z}, "pose": pose}


def test_matlab_round_linspace_is_half_away_from_zero() -> None:
    # linspace(1, 6, 4) = 1, 8/3, 13/3, 6 → 1, 3, 4, 6.
    np.testing.assert_array_equal(_matlab_indices(6, 4), [0, 2, 3, 5])
    # np.round would banker's-round 0.5 differently; MATLAB round(1.5) is 2.
    np.testing.assert_array_equal(_matlab_indices(2, 2), [0, 1])


def test_plane_loader_converts_units_and_fits_the_line(tmp_path: Path) -> None:
    # z = 2x, two scans shorter than the 5000-point cap.
    x = [0.0, 1.0, 2.0, 3.0]
    z = [0.0, 2.0, 4.0, 6.0]
    document = {
        "captures": [
            _capture(x, z, (0.01, -0.02, 0.03), None),
            _capture(x, z, (0.02, -0.02, 0.03), None),
        ]
    }
    path = tmp_path / "oct_plane_tiny.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    translation, rotation, position, direction, scan = io.load_plane_yaml(path)
    np.testing.assert_allclose(translation, [[10.0, -20.0, 30.0], [20.0, -20.0, 30.0]])
    np.testing.assert_allclose(rotation, np.broadcast_to(np.eye(3), (2, 3, 3)))
    np.testing.assert_allclose(position[:, 1], 0.0)
    np.testing.assert_allclose(scan[..., 1], 0.0)
    theta = np.arctan(2.0)
    np.testing.assert_allclose(direction[0], [np.cos(theta), 0.0, np.sin(theta)])
    assert scan.shape == (2, 4, 3)


def test_plane_loader_resamples_with_matlab_rounding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(io, "_MAX_OCT_POINTS", 4)
    long_x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    long_z = [0.0, 0.0, 0.0, 0.0, 0.0, 6.0]
    short_x = [0.0, 1.0, 2.0, 3.0]
    document = {
        "captures": [
            _capture(long_x, long_z, (0.0, 0.0, 0.0), None),
            _capture(short_x, short_x, (0.0, 0.0, 0.0), None),
        ]
    }
    path = tmp_path / "oct_plane_resample.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    _, _, position, direction, scan = io.load_plane_yaml(path)
    assert scan.shape == (2, 4, 3)
    np.testing.assert_allclose(scan[0, :, 0], [0.0, 2.0, 3.0, 5.0])
    np.testing.assert_allclose(scan[1, :, 0], short_x)
    # The line uses every original sample. Resampling would move this mean.
    np.testing.assert_allclose(position[0], [2.5, 0.0, 1.0])
    np.testing.assert_allclose(direction[0, 1], 0.0)


def test_meas_loader_groups_column_major_and_clips_the_radius(tmp_path: Path) -> None:
    # Two orientations, mt = 2. The second profile is a circle wider than the
    # sphere, so its fitted radius is clipped and the center lies in the scan plane.
    small = _circle_points(center_x=1.0, radius=1.0)
    wide = _circle_points(center_x=0.0, radius=3.0)
    document = {
        "captures": [
            _capture(small[0], small[1], (0.001, 0.0, 0.0), sign=-1),
            _capture(wide[0], wide[1], (0.002, 0.0, 0.0), sign=1),
            _capture(small[0], small[1], (0.003, 0.0, 0.0), sign=1),
            _capture(wide[0], wide[1], (0.004, 0.0, 0.0), sign=-1),
        ]
    }
    path = tmp_path / "oct_sphere_tiny.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    diameter = 4.0
    translation, rotation, centers, scans = io.load_meas2023_yaml(path, mt=2, diameter=diameter)
    assert translation.shape == (2, 2, 3)
    assert rotation.shape == (2, 3, 3)
    assert centers.shape == (2, 2, 3)
    np.testing.assert_allclose(translation[:, 0, 0], [1.0, 3.0])
    np.testing.assert_allclose(centers[0, 0, 0], 1.0, atol=1e-8)
    np.testing.assert_allclose(centers[0, 0, 2], 0.0, atol=1e-8)
    # sign_y was -1, so center_sign_y is +1.
    np.testing.assert_allclose(centers[0, 0, 1], np.sqrt((diameter / 2.0) ** 2 - 1.0), atol=1e-8)
    np.testing.assert_allclose(centers[0, 1, :2], [0.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(centers[0, 1, 2], 0.0, atol=1e-8)
    assert scans[1][0].shape[1] == 3
    assert len(scans) == 2 and len(scans[0]) == 2


def test_loader_rejects_a_lidar_capture(tmp_path: Path) -> None:
    document = {
        "captures": [{"pose": {"position": {"x": 0, "y": 0, "z": 0}}, "averaged_scan": [1.0]}]
    }
    path = tmp_path / "lidar_tiny.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="LiDAR"):
        io.load_plane_yaml(path)


def _circle_points(center_x: float, radius: float) -> tuple[list[float], list[float]]:
    theta = np.linspace(0.0, 2.0, 16)
    return ((center_x + radius * np.cos(theta)).tolist(), (radius * np.sin(theta)).tolist())

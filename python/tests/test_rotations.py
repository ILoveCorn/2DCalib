from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from fixture_io import load_fixture
from handeye2d import rotations
from layout import rotations as batch_rotations


def _angle_delta(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.abs((left - right + np.pi) % (2.0 * np.pi) - np.pi)


def test_euler_matches_the_shim_and_scipy(fixtures_dir: Path) -> None:
    data = load_fixture(fixtures_dir / "shim_reference.mat")
    for sequence, angles_key, matrix_key in (
        ("ZYX", "eul_zyx_in", "eul_zyx_R"),
        ("ZXY", "eul_zxy_in", "eul_zxy_R"),
    ):
        angles = np.asarray(data[angles_key], float)
        got = rotations.eul_to_rotm(angles, sequence)
        expected = batch_rotations(data[matrix_key])
        np.testing.assert_allclose(got, expected, atol=1e-12)
        scipy_rotation = Rotation.from_euler(sequence, angles).as_matrix()
        np.testing.assert_allclose(got, scipy_rotation, atol=1e-12)
        # Default sequence is ZYX, which is what plane_TRO2016 calls.
        if sequence == "ZYX":
            np.testing.assert_allclose(rotations.eul_to_rotm(angles), got)


def test_axis_angle_normalizes_and_matches_the_shim(fixtures_dir: Path) -> None:
    data = load_fixture(fixtures_dir / "shim_reference.mat")
    axang = np.asarray(data["axang_in"], float)
    got = rotations.axang_to_rotm(axang[:, :3], axang[:, 3])
    np.testing.assert_allclose(got, batch_rotations(data["axang_R"]), atol=1e-12)
    np.testing.assert_allclose(rotations.axang_to_rotm(np.zeros(3), 1.2), np.eye(3))


def test_rotm_to_angle_is_the_shim_angle(fixtures_dir: Path) -> None:
    data = load_fixture(fixtures_dir / "shim_reference.mat")
    matrices = batch_rotations(data["rotm_in"])
    angle = rotations.rotm_to_angle(matrices)
    np.testing.assert_allclose(angle, np.asarray(data["rotm_axang"], float)[:, 3], atol=1e-12)
    assert np.all((angle >= 0.0) & (angle <= np.pi))


def test_rotm_to_eul_matches_the_shim(fixtures_dir: Path) -> None:
    data = load_fixture(fixtures_dir / "shim_reference.mat")
    matrices = batch_rotations(data["rotm_in"])
    for sequence, key in (("ZYX", "rotm_eul_zyx"), ("ZXY", "rotm_eul_zxy")):
        euler = rotations.rotm_to_eul(matrices, sequence)
        np.testing.assert_allclose(euler, np.asarray(data[key], float), atol=1e-12)
        rebuilt = rotations.eul_to_rotm(euler, sequence)
        np.testing.assert_allclose(rebuilt, matrices, atol=1e-8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            reference = Rotation.from_matrix(matrices).as_euler(sequence)
        np.testing.assert_allclose(_angle_delta(euler, reference), 0.0, atol=1e-8)


def test_quaternion_is_scalar_first_and_normalized(fixtures_dir: Path) -> None:
    data = load_fixture(fixtures_dir / "shim_reference.mat")
    quat = np.asarray(data["quat_wxyz"], float)
    got = rotations.quat_to_rotm(quat)
    np.testing.assert_allclose(got, batch_rotations(data["quat_R"]), atol=1e-12)
    np.testing.assert_allclose(rotations.quat_to_rotm(np.array([2.0, 0.0, 0.0, 0.0])), np.eye(3))
    np.testing.assert_allclose(rotations.quat_to_rotm(np.zeros(4)), np.eye(3))

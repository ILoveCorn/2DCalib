from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import scipy.linalg
from scipy.spatial.transform import Rotation

from fixture_io import load_fixture


def _as_batch(rotation: np.ndarray) -> np.ndarray:
    """MATLAB (3, 3, n) or (3, 3, ...) to SciPy (n, 3, 3)."""
    array = np.asarray(rotation, float)
    if array.ndim == 2:
        array = array[:, :, None]
    return np.moveaxis(array, (0, 1), (-2, -1)).reshape(-1, 3, 3)


def _angle_delta(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.abs((left - right + np.pi) % (2.0 * np.pi) - np.pi)


def _pagemtimes(
    left: np.ndarray,
    right: np.ndarray,
    op_left: str = "none",
    op_right: str = "none",
) -> np.ndarray:
    a = np.asarray(left, float)
    b = np.asarray(right, float)
    if op_left != "none":
        a = np.swapaxes(a, 0, 1)
    if op_right != "none":
        b = np.swapaxes(b, 0, 1)
    a = np.moveaxis(a, (0, 1), (-2, -1))
    b = np.moveaxis(b, (0, 1), (-2, -1))
    if a.shape[-2:] == (1, 1) or b.shape[-2:] == (1, 1):
        product = a * b
    else:
        product = np.matmul(a, b)
    return np.moveaxis(product, (-2, -1), (0, 1))


def _pagemrdivide(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Solve X @ B = A page-wise with scipy.linalg.solve."""
    a = np.moveaxis(np.asarray(left, float), (0, 1), (-2, -1))
    b = np.moveaxis(np.asarray(right, float), (0, 1), (-2, -1))
    page_dims = max(a.ndim, b.ndim) - 2
    if a.ndim - 2 < page_dims:
        a = a.reshape((1,) * (page_dims - (a.ndim - 2)) + a.shape)
    if b.ndim - 2 < page_dims:
        b = b.reshape((1,) * (page_dims - (b.ndim - 2)) + b.shape)
    a = np.broadcast_to(a, np.broadcast_shapes(a.shape[:-2], b.shape[:-2]) + a.shape[-2:])
    b = np.broadcast_to(b, a.shape[:-2] + b.shape[-2:])
    out = np.empty(a.shape, dtype=float)
    for index in np.ndindex(a.shape[:-2]):
        solved = scipy.linalg.solve(b[index].T, a[index].T, assume_a="gen")
        out[index] = solved.T
    return np.moveaxis(out, (-2, -1), (0, 1))


def test_euler_matches_scipy(fixtures_dir: Path) -> None:
    data = load_fixture(fixtures_dir / "shim_reference.mat")
    for sequence, angles_key, matrix_key in (
        ("ZYX", "eul_zyx_in", "eul_zyx_R"),
        ("ZXY", "eul_zxy_in", "eul_zxy_R"),
    ):
        expected = Rotation.from_euler(sequence, data[angles_key]).as_matrix()
        got = _as_batch(data[matrix_key])
        np.testing.assert_allclose(got, expected, atol=1e-10)


def test_axis_angle_matches_scipy(fixtures_dir: Path) -> None:
    data = load_fixture(fixtures_dir / "shim_reference.mat")
    axang = np.asarray(data["axang_in"], float)
    axis = axang[:, :3]
    norm = np.linalg.norm(axis, axis=1, keepdims=True)
    safe = np.where(norm < 1e-15, 1.0, norm)
    rotvec = np.where(norm < 1e-15, 0.0, axis / safe * axang[:, 3:4])
    expected = Rotation.from_rotvec(rotvec).as_matrix()
    np.testing.assert_allclose(_as_batch(data["axang_R"]), expected, atol=1e-10)


def test_rotm2axang_reconstructs_the_matrix(fixtures_dir: Path) -> None:
    data = load_fixture(fixtures_dir / "shim_reference.mat")
    original = _as_batch(data["rotm_in"])
    axang = np.asarray(data["rotm_axang"], float)
    axis = axang[:, :3]
    theta = axang[:, 3]
    rebuilt = Rotation.from_rotvec(axis * theta[:, None]).as_matrix()
    np.testing.assert_allclose(rebuilt, original, atol=1e-8)
    trace_theta = np.arccos(np.clip((np.trace(original, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0))
    np.testing.assert_allclose(theta, trace_theta, atol=1e-8)


def test_rotm2eul_matches_scipy_angles(fixtures_dir: Path) -> None:
    data = load_fixture(fixtures_dir / "shim_reference.mat")
    original = _as_batch(data["rotm_in"])
    for sequence, key in (("ZYX", "rotm_eul_zyx"), ("ZXY", "rotm_eul_zxy")):
        euler = np.asarray(data[key], float)
        rebuilt = Rotation.from_euler(sequence, euler).as_matrix()
        np.testing.assert_allclose(rebuilt, original, atol=1e-8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            reference = Rotation.from_matrix(original).as_euler(sequence)
        np.testing.assert_allclose(_angle_delta(euler, reference), 0.0, atol=1e-8)


def test_quaternion_matches_scipy_scalar_first(fixtures_dir: Path) -> None:
    data = load_fixture(fixtures_dir / "shim_reference.mat")
    quat_wxyz = np.asarray(data["quat_wxyz"], float)
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    expected = Rotation.from_quat(quat_xyzw).as_matrix()
    np.testing.assert_allclose(_as_batch(data["quat_R"]), expected, atol=1e-10)


def test_pagemtimes_matches_numpy(fixtures_dir: Path) -> None:
    data = load_fixture(fixtures_dir / "shim_reference.mat")
    cases = (
        ("pm_a_plain", "pm_b_plain", "pm_c_plain", "none", "none"),
        ("pm_a_scalar", "pm_b_scalar", "pm_c_scalar", "none", "none"),
        ("pm_a_bcast", "pm_b_bcast", "pm_c_bcast", "none", "none"),
        ("pm_a_trans", "pm_b_trans", "pm_c_trans", "transpose", "none"),
        ("pm_a_4d", "pm_b_4d", "pm_c_4d", "none", "none"),
        ("pm_a_rscalar", "pm_b_rscalar", "pm_c_rscalar", "none", "none"),
    )
    for left, right, product, op_left, op_right in cases:
        expected = _pagemtimes(data[left], data[right], op_left, op_right)
        np.testing.assert_allclose(data[product], expected, atol=1e-12)


def test_pagemrdivide_matches_scipy_solve(fixtures_dir: Path) -> None:
    data = load_fixture(fixtures_dir / "shim_reference.mat")
    for left, right, product in (
        ("pr_a", "pr_b", "pr_c"),
        ("pr_a_bcast", "pr_b_bcast", "pr_c_bcast"),
    ):
        expected = _pagemrdivide(data[left], data[right])
        np.testing.assert_allclose(data[product], expected, atol=1e-10)

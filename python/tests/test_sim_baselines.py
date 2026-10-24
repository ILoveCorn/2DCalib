"""Geometry checks for the baseline simulators.

Octave's random stream is not NumPy's. Orientations that do not depend on
that stream are compared to the committed fixtures. The solvers are checked
on fresh noiseless draws.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fixture_io import load_fixture, scalar
from handeye2d.calib import (
    line_rcim2022,
    plane_iros2015,
    plane_rcim2020,
    plane_tro2016,
    sphere_meas2023,
)
from handeye2d.metrics import calibration_error
from handeye2d.rotations import axang_to_rotm, eul_to_rotm
from handeye2d.sim import sim_iros2015, sim_meas2023, sim_plane, sim_rcim2020, sim_rcim2022
from handeye2d.sim.sim_meas2023 import sample_limits
from layout import rotations, vectors

_TINY = 1e-18 * np.eye(3)


def test_iros_fixture_orientations_follow_each_plane(fixtures_dir: Path) -> None:
    data = load_fixture(fixtures_dir / "sim_iros2015.mat")
    expected = _iros_hand_rotation(data)
    saved = _page_rotations(data["RH2Wn"])
    np.testing.assert_allclose(saved, expected, atol=1e-6)


def test_rcim2020_fixture_orientations_follow_the_zxy_grid(fixtures_dir: Path) -> None:
    data = load_fixture(fixtures_dir / "sim_rcim2020.mat")
    np.testing.assert_allclose(rotations(data["RH2Wn"]), _rcim2020_hand_rotation(data), atol=1e-8)


def test_rcim2022_fixture_orientations_follow_the_line_frame(fixtures_dir: Path) -> None:
    data = load_fixture(fixtures_dir / "sim_rcim2022.mat")
    np.testing.assert_allclose(rotations(data["RH2Wn"]), _line_hand_rotation(data), atol=1e-8)


def test_simulators_are_reproducible_and_recover_the_pose() -> None:
    truth_t = np.array([12.0, -7.0, 30.0])
    axis = np.array([1.0, 2.0, 3.0])
    truth_R = axang_to_rotm(axis, np.pi / 5.0)
    # A tilted plane. A horizontal plane with this hand-eye rotation leaves
    # the RCIM2020 cross product poorly conditioned.
    normal = np.array([0.2, -0.3, 0.9])
    normal = normal / np.linalg.norm(normal)
    initial_t = truth_t + np.array([0.4, -0.3, 0.2])
    initial_R = truth_R @ axang_to_rotm(np.array([0.3, -0.2, 0.1]), 0.05)
    sweep = np.deg2rad(45.0)

    first = sim_plane(
        np.zeros(3),
        normal,
        50.0,
        5.0,
        2.0,
        truth_t,
        truth_R,
        _TINY,
        _TINY,
        0.0,
        sweep,
        sweep,
        4,
        4,
        np.random.default_rng(0),
    )
    again = sim_plane(
        np.zeros(3),
        normal,
        50.0,
        5.0,
        2.0,
        truth_t,
        truth_R,
        _TINY,
        _TINY,
        0.0,
        sweep,
        sweep,
        4,
        4,
        np.random.default_rng(0),
    )
    np.testing.assert_allclose(first.tH2Wn, again.tH2Wn)
    tro_t, tro_R, _, _ = plane_tro2016(
        initial_t, initial_R, first.tH2Wn, first.RH2Wn, first.vOn, first.scanO
    )
    _close_to_truth(truth_t, truth_R, tro_t, tro_R, 1e-3, 1e-3)
    rcim_t, rcim_R, _, _ = plane_rcim2020(
        initial_t, initial_R, first.tH2Wn, first.RH2Wn, first.pOn, 1e-5
    )
    _close_to_truth(truth_t, truth_R, rcim_t, rcim_R, 1e-2, 0.05)

    iros = sim_iros2015(
        3,
        np.zeros((3, 3)),
        np.eye(3),
        np.full(3, 50.0),
        5.0,
        2.0,
        truth_t,
        truth_R,
        _TINY,
        _TINY,
        0.0,
        sweep,
        sweep,
        6,
        6,
        # Some noiseless draws leave the standard initial guess outside the
        # basin of this iteration. This seed stays inside it.
        np.random.default_rng(0),
    )
    iros_t, iros_R, _, _ = plane_iros2015(
        initial_t, initial_R, iros.tH2Wn, iros.RH2Wn, iros.pOn, 1e-5
    )
    _close_to_truth(truth_t, truth_R, iros_t, iros_R, 1e-3, 1e-2)

    line = sim_rcim2022(
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        50.0,
        5.0,
        2.0,
        truth_t,
        truth_R,
        _TINY,
        _TINY,
        0.0,
        sweep,
        sweep,
        3,
        3,
        3,
        np.random.default_rng(2),
    )
    line_t, line_R, _ = line_rcim2022(line.tH2Wn, line.RH2Wn, line.pOn)
    _close_to_truth(truth_t, truth_R, line_t, line_R, 1e-3, 1e-3)

    sphere = sim_meas2023(
        np.array([0.0, 0.0, 40.0]),
        7.5,
        5.0,
        2.0,
        truth_t,
        truth_R,
        _TINY,
        _TINY,
        0.0,
        sweep,
        sweep,
        3,
        3,
        3,
        np.random.default_rng(3),
    )
    sphere_t, sphere_R, _ = sphere_meas2023(sphere.tH2Wn, sphere.RH2Wn, sphere.pcOe)
    _close_to_truth(truth_t, truth_R, sphere_t, sphere_R, 1e-3, 1e-3)


def test_meas2023_arc_uses_the_lateral_field_of_view() -> None:
    left, right = sample_limits(center_x=2.5, radius=7.5, x_max=5.0)
    np.testing.assert_allclose(left, 1e-4)
    np.testing.assert_allclose(right, 5.0 - 1e-4)


def test_rcim2020_stays_on_the_plane() -> None:
    normal = np.array([0.0, 0.0, 1.0])
    drawn = sim_rcim2020(
        np.zeros(3),
        normal,
        10.0,
        5.0,
        2.0,
        np.array([1.0, 2.0, 3.0]),
        np.eye(3),
        0.0,
        0.0,
        0.0,
        6,
        np.random.default_rng(4),
    )
    point = drawn.lOn[:, :3]
    world = np.einsum("mij,mj->mi", drawn.RH2Wn, point @ np.eye(3) + np.array([1.0, 2.0, 3.0]))
    world = world + drawn.tH2Wn
    np.testing.assert_allclose(world @ normal, 0.0, atol=1e-8)
    np.testing.assert_allclose(np.linalg.norm(drawn.lOn[:, 3:], axis=1), 1.0, atol=1e-12)


def _close_to_truth(truth_t, truth_R, estimate_t, estimate_R, position_tol, angle_tol) -> None:
    position, angle = calibration_error(truth_t, truth_R, estimate_t, estimate_R)
    assert float(position) < position_tol
    assert float(angle) < angle_tol


def _iros_hand_rotation(data: dict[str, np.ndarray]) -> np.ndarray:
    n_plane = int(scalar(data["nPlane"]))
    per_alpha = int(np.floor(scalar(data["mAlpha"]) / np.sqrt(n_plane) + 0.5))
    per_beta = int(np.floor(scalar(data["mBeta"]) / np.sqrt(n_plane) + 0.5))
    frames = []
    normals = np.asarray(data["nPs"], dtype=float)
    for index in range(n_plane):
        frames.append(_grid_rotation(normals[:, index], per_alpha, per_beta, data, "ZYX"))
    return np.stack(frames)


def _rcim2020_hand_rotation(data: dict[str, np.ndarray]) -> np.ndarray:
    count = int(scalar(data["m"]))
    normal = vectors(data["nP"])
    normal = normal / np.linalg.norm(normal)
    _, _, _, sensor_r0 = _frame(normal)
    euler = np.column_stack(
        (
            np.linspace(0.0, 2.0 * np.pi, count),
            np.linspace(-np.pi / 4.0, np.pi / 4.0, count),
            np.linspace(-np.pi / 4.0, np.pi / 4.0, count),
        )
    )
    sensor = np.einsum("ij,mjk->mik", sensor_r0, eul_to_rotm(euler, "ZXY"))
    return sensor @ rotations(data["RO2H"]).T


def _line_hand_rotation(data: dict[str, np.ndarray]) -> np.ndarray:
    direction = vectors(data["vL"])
    if direction[0] == 0.0 and direction[1] == 0.0:
        axis_x = np.array([1.0, 0.0, 0.0])
    else:
        axis_x = np.array([-direction[1], direction[0], 0.0])
        axis_x = axis_x / np.linalg.norm(axis_x)
    sensor_r0 = np.column_stack((axis_x, direction, np.cross(axis_x, direction)))
    return _euler_grid(sensor_r0, int(scalar(data["mAlpha"])), int(scalar(data["mBeta"])), data)


def _grid_rotation(normal, m_alpha, m_beta, data, sequence: str) -> np.ndarray:
    _, _, _, sensor_r0 = _frame(normal)
    beta, alpha = np.meshgrid(
        np.linspace(-scalar(data["betaMax"]), scalar(data["betaMax"]), m_beta),
        np.linspace(-scalar(data["alphaMax"]), scalar(data["alphaMax"]), m_alpha),
        indexing="ij",
    )
    euler = np.column_stack(
        (np.zeros(m_alpha * m_beta), alpha.ravel(order="F"), beta.ravel(order="F"))
    )
    sensor = np.einsum("ij,mjk->mik", sensor_r0, eul_to_rotm(euler, sequence))
    return sensor @ rotations(data["RO2H"]).T


def _euler_grid(sensor_r0, m_alpha, m_beta, data) -> np.ndarray:
    return _grid_rotation_from_frame(sensor_r0, m_alpha, m_beta, data)


def _grid_rotation_from_frame(sensor_r0, m_alpha, m_beta, data) -> np.ndarray:
    beta, alpha = np.meshgrid(
        np.linspace(-scalar(data["betaMax"]), scalar(data["betaMax"]), m_beta),
        np.linspace(-scalar(data["alphaMax"]), scalar(data["alphaMax"]), m_alpha),
        indexing="ij",
    )
    count = m_alpha * m_beta
    euler = np.column_stack((np.zeros(count), alpha.ravel(order="F"), beta.ravel(order="F")))
    sensor = np.einsum("ij,mjk->mik", sensor_r0, eul_to_rotm(euler, "ZYX"))
    return sensor @ rotations(data["RO2H"]).T


def _frame(normal: np.ndarray):
    if normal[0] == 0.0 and normal[1] == 0.0:
        axis_x = np.array([1.0, 0.0, 0.0])
    else:
        axis_x = np.array([-normal[1], normal[0], 0.0])
        axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = np.cross(normal, axis_x)
    plane_R = np.column_stack((axis_x, axis_y, normal))
    sensor_r0 = np.column_stack((-axis_x, axis_y, -normal))
    return axis_x, axis_y, plane_R, sensor_r0


def _page_rotations(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=float)
    moved = np.moveaxis(values, (0, 1), (-2, -1))
    return np.swapaxes(moved, 0, 1)

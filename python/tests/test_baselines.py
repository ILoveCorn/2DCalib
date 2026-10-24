"""Baseline solvers against the Octave fixtures, plus the line-index fix."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fixture_io import load_fixture, scalar
from handeye2d.calib import line_rcim2022, plane_iros2015, plane_rcim2020
from handeye2d.calib.line_rcim2022 import _direction_samples
from handeye2d.metrics import calibration_error
from layout import rotations, vectors


def test_rcim2020_matches_the_octave_fixture(fixtures_dir: Path) -> None:
    data = load_fixture(fixtures_dir / "plane_rcim2020.mat")
    translation, rotation, normal, _elapsed = plane_rcim2020(
        vectors(data["tO2H0"]),
        rotations(data["RO2H0"]),
        vectors(data["tH2Wn"]),
        rotations(data["RH2Wn"]),
        vectors(data["pOn"]),
        scalar(data["tol"]),
    )
    _assert_pose(data, translation, rotation)
    _assert_parallel(normal, vectors(data["ne"]))


def test_iros2015_matches_the_octave_fixture(fixtures_dir: Path) -> None:
    data = load_fixture(fixtures_dir / "plane_iros2015.mat")
    translation, rotation, normal, _elapsed = plane_iros2015(
        vectors(data["tO2H0"]),
        rotations(data["RO2H0"]),
        _page_vectors(data["tH2Wn"]),
        _page_rotations(data["RH2Wn"]),
        _page_vectors(data["pOn"]),
        scalar(data["tol"]),
    )
    _assert_pose(data, translation, rotation)
    reference = _page_normals(data["ne"])
    assert normal.shape == reference.shape
    _assert_parallel(normal, reference)


def test_line_rcim2022_matches_the_patched_octave_fixture(fixtures_dir: Path) -> None:
    data = load_fixture(fixtures_dir / "line_rcim2022.mat")
    translation, rotation, _elapsed = line_rcim2022(
        vectors(data["tH2Wn"]), rotations(data["RH2Wn"]), vectors(data["pOn"])
    )
    _assert_pose(data, translation, rotation)


def test_every_pose_keeps_its_own_direction_columns() -> None:
    """MATLAB line 58 writes every pose into columns ``(1:mt-1)+(mt-1)``."""
    point = np.zeros((2, 3, 3))
    translation = np.zeros((2, 3, 3))
    point[0] = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    translation[0] = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
    rotation = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
    samples = _direction_samples(translation, rotation, point, np.eye(3))
    assert samples.shape == (3, 4)
    assert np.linalg.norm(samples[:, :2]) > 0.0
    np.testing.assert_allclose(samples[:, 2:], 0.0)


def _assert_pose(
    data: dict[str, np.ndarray], translation: np.ndarray, rotation: np.ndarray
) -> None:
    np.testing.assert_allclose(translation, vectors(data["tO2He"]), atol=1e-6)
    np.testing.assert_allclose(rotation, rotations(data["RO2He"]), atol=1e-8)
    position, angle = calibration_error(
        vectors(data["tO2H"]), rotations(data["RO2H"]), translation, rotation
    )
    np.testing.assert_allclose(position, scalar(data["err_t"]), atol=1e-6)
    np.testing.assert_allclose(angle, scalar(data["err_R_deg"]), atol=1e-5)


def _assert_parallel(estimate: np.ndarray, reference: np.ndarray) -> None:
    estimate = np.atleast_2d(estimate)
    reference = np.atleast_2d(reference)
    cosine = np.einsum("ij,ij->i", estimate, reference)
    np.testing.assert_allclose(np.abs(cosine), 1.0, atol=1e-6)
    aligned = estimate.copy()
    aligned[cosine < 0.0] *= -1.0
    np.testing.assert_allclose(aligned, reference, atol=1e-6)


def _page_vectors(array: np.ndarray) -> np.ndarray:
    """``(3, 1, m, n_planes)`` to ``(n_planes, m, 3)``."""
    values = np.asarray(array, dtype=float)
    moved = np.moveaxis(values, 0, -1)[0]
    return np.transpose(moved, (1, 0, 2))


def _page_rotations(array: np.ndarray) -> np.ndarray:
    """``(3, 3, m, n_planes)`` to ``(n_planes, m, 3, 3)``."""
    values = np.asarray(array, dtype=float)
    moved = np.moveaxis(values, (0, 1), (-2, -1))
    return np.swapaxes(moved, 0, 1)


def _page_normals(array: np.ndarray) -> np.ndarray:
    values = np.squeeze(np.asarray(array, dtype=float))
    if values.ndim == 1:
        return values.reshape(1, 3)
    if values.shape[0] == 3:
        return values.reshape(3, -1).T
    return values.reshape(-1, 3)


@pytest.mark.parametrize("name", ["plane_rcim2020.mat", "plane_iros2015.mat", "line_rcim2022.mat"])
def test_fixture_names_are_present(fixtures_dir: Path, name: str) -> None:
    assert (fixtures_dir / name).is_file()

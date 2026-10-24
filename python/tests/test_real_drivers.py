"""Drivers for ``real_calibration.m`` and ``real_plane.m``.

The calibration driver is compared with ``result/real_calibration_260219_1146.mat``
using the same pose bounds as ``test_real_data.py``. Wall time is not compared.
``real_plane`` has no saved MATLAB output. Its leave-one-out pattern and the
printed means are checked against ``plane_fit`` on that mat. The paper's
reconstruction table is not in this repository, so it is not asserted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from experiments._real import (
    DIAMETER,
    REFERENCE_MAT,
    load_calibration,
    matlab_cells,
    reference_calibration,
    save_calibration,
    session_names,
)
from experiments.real_calibration import calibrate
from experiments.real_plane import plane_errors, reconstruction_table
from handeye2d.metrics import calibration_error


def _finite_mean(values: np.ndarray) -> float:
    data = values[np.isfinite(values)]
    return float(np.mean(data))


def _available(repo_root: Path) -> bool:
    result = repo_root / "result" / REFERENCE_MAT
    planes = repo_root / "data" / "oct" / "plane"
    spheres = repo_root / "data" / "oct" / "sphere"
    measure = repo_root / "data" / "measure_260219_1144.mat"
    return result.is_file() and measure.is_file() and planes.is_dir() and spheres.is_dir()


@pytest.fixture(scope="module")
def saved(repo_root: Path):
    if not _available(repo_root):
        pytest.skip("real datasets or the saved calibration result are not in this checkout")
    return reference_calibration(repo_root)


@pytest.fixture(scope="module")
def driver(repo_root: Path, saved):
    return calibrate(repo_root)


def test_sorted_sessions_match_the_saved_mat(repo_root: Path, saved) -> None:
    planes = session_names(repo_root / "data" / "oct" / "plane", "oct_plane*.yaml")
    spheres = session_names(repo_root / "data" / "oct" / "sphere", "oct_sphere*.yaml")
    assert saved.plane_files == planes
    assert saved.sphere_files == spheres
    raw = loadmat(repo_root / "result" / REFERENCE_MAT, squeeze_me=False)
    legacy = matlab_cells(raw["planeFiles"])
    assert all(name.startswith("oct_checkerboard_") for name in legacy)
    assert tuple(name.replace("oct_checkerboard", "oct_plane") for name in legacy) == planes
    np.testing.assert_allclose(raw["D"], DIAMETER)


def test_real_calibration_matches_the_saved_poses(driver, saved) -> None:
    assert driver.translation.shape == saved.translation.shape == (3, 10, 5)
    assert driver.rotation.shape == saved.rotation.shape == (3, 3, 10, 5)
    assert driver.time.shape == (10, 5)
    np.testing.assert_allclose(driver.time[:, 4], 0.0)
    for index in range(10):
        np.testing.assert_allclose(
            driver.translation[:, index, 0], saved.translation[:, index, 0], atol=1e-8
        )
        np.testing.assert_allclose(
            driver.rotation[:, :, index, 0], saved.rotation[:, :, index, 0], atol=1e-8
        )
        np.testing.assert_allclose(
            driver.translation[:, index, 2], saved.translation[:, index, 2], atol=1e-8
        )
        _position, angle = calibration_error(
            saved.translation[:, index, 2],
            saved.rotation[:, :, index, 2],
            driver.translation[:, index, 2],
            driver.rotation[:, :, index, 2],
        )
        assert float(angle) <= 1e-4
        position, angle = calibration_error(
            saved.translation[:, index, 1],
            saved.rotation[:, :, index, 1],
            driver.translation[:, index, 1],
            driver.rotation[:, :, index, 1],
        )
        assert float(position) <= 5e-2
        assert float(angle) <= 1e-2
        np.testing.assert_allclose(
            driver.translation[:, index, 3], saved.translation[:, index, 3], atol=1e-8
        )
        np.testing.assert_allclose(
            driver.rotation[:, :, index, 3], saved.rotation[:, :, index, 3], atol=1e-8
        )
        np.testing.assert_allclose(
            driver.translation[:, index, 4], saved.translation[:, index, 4], atol=0.0
        )
        np.testing.assert_allclose(
            driver.rotation[:, :, index, 4], saved.rotation[:, :, index, 4], atol=0.0
        )


def test_calibration_npz_roundtrip(tmp_path: Path, driver) -> None:
    path = tmp_path / "real_calibration.npz"
    save_calibration(path, driver)
    loaded = load_calibration(path)
    np.testing.assert_allclose(loaded.translation, driver.translation)
    np.testing.assert_allclose(loaded.rotation, driver.rotation)
    np.testing.assert_allclose(loaded.time, driver.time)
    assert loaded.plane_files == driver.plane_files
    assert loaded.sphere_files == driver.sphere_files


def test_real_plane_leave_one_out_and_printed_means(repo_root: Path, saved) -> None:
    errors = plane_errors(saved, repo_root)
    assert errors.shape == (10, 10, 5)
    diagonal = np.arange(10)
    off = ~np.eye(10, dtype=bool)
    for method in range(3):
        assert np.isnan(errors[diagonal, diagonal, method]).all()
        assert np.isfinite(errors[:, :, method][off]).all()
        assert np.count_nonzero(np.isnan(errors[:, :, method])) == 10
    for method in (3, 4):
        assert np.isfinite(errors[:, :, method]).all()
    finite = errors[np.isfinite(errors)]
    assert np.all(finite > 0.0)
    assert np.all(finite < 10.0)
    # omitnan means on the saved mat, in millimeters. These are not paper digits.
    expected_overall = (
        0.17041614125112248,
        1.1951347661459304,
        0.24801100463883483,
        0.8673328863039029,
        0.3433101034589676,
    )
    for method, expected in enumerate(expected_overall):
        assert _finite_mean(errors[:, :, method]) == pytest.approx(expected, abs=1e-6)
    table = reconstruction_table(errors)
    assert table.startswith("\n\nPlane reconstruction error:\n")
    body = [line for line in table.splitlines() if line.startswith("\t")]
    assert len(body) == 5
    for line, method in zip(body, range(5), strict=True):
        cells = [part for part in line.split("&")[1:] if part.strip()]
        assert len(cells) == 11
        parsed = [float(cell) for cell in cells]
        for index in range(10):
            expected = _finite_mean(errors[index, :, method])
            assert parsed[index] == pytest.approx(expected, abs=5e-4)
        assert parsed[-1] == pytest.approx(_finite_mean(errors[:, :, method]), abs=5e-4)

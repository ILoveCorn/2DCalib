"""L2 checks against the saved MATLAB real-calibration result.

Method 1 is ``plane_jnt_gh`` on the ten plane sessions. MEAS2023 is the ten
sphere sessions. ``plane_rcim2020`` is compared tightly, and ``plane_tro2016``
loosely, on the same plane sessions. The measured slot is the fiducial
registration in ``data/measure_260219_1144.mat``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from handeye2d.calib import plane_jnt_gh, plane_rcim2020, plane_tro2016, sphere_meas2023
from handeye2d.fitting import plane_fit
from handeye2d.io import load_meas2023_yaml, load_plane_yaml
from handeye2d.metrics import calibration_error

_RESULT = "real_calibration_260219_1146.mat"
_MEASURE = "measure_260219_1144.mat"
_DIAMETER = 14.9981
_MT = 3


def _cells(value: np.ndarray) -> list[str]:
    rows = []
    for item in np.asarray(value, dtype=object).ravel():
        if isinstance(item, np.ndarray):
            rows.append("".join(np.asarray(item).ravel().astype(str)))
        else:
            rows.append(str(item))
    return rows


@pytest.fixture(scope="module")
def real_result(repo_root: Path) -> dict[str, np.ndarray]:
    path = repo_root / "result" / _RESULT
    measure = repo_root / "data" / _MEASURE
    spheres = repo_root / "data" / "oct" / "sphere"
    planes = repo_root / "data" / "oct" / "plane"
    if not path.is_file() or not measure.is_file() or not spheres.is_dir() or not planes.is_dir():
        pytest.skip("real datasets or the saved calibration result are not in this checkout")
    loaded = loadmat(path, squeeze_me=False)
    loaded["__measure"] = loadmat(measure, squeeze_me=False)
    return loaded


def test_measured_slot_matches_the_fiducial_registration(
    real_result: dict[str, np.ndarray],
) -> None:
    methods = _cells(real_result["testedMethods"])
    assert methods[4] == "5 - Measured"
    assert methods[3] == "4 - MEAS-2023"
    measure = real_result["__measure"]
    np.testing.assert_allclose(real_result["alltO2He"][:, :, 4], measure["tO2H"][:, 0, :])
    np.testing.assert_allclose(real_result["allRO2He"][:, :, :, 4], measure["RO2H"])
    np.testing.assert_allclose(real_result["D"], _DIAMETER)
    np.testing.assert_allclose(real_result["sigma0"], [[0.1, 0.01]])
    assert int(np.reshape(real_result["iImax"], ())) == 30
    assert int(np.reshape(real_result["iOmax"], ())) == 50
    np.testing.assert_allclose(real_result["tol"], 1e-5)


def test_meas2023_matches_every_saved_sphere_session(
    real_result: dict[str, np.ndarray], repo_root: Path
) -> None:
    names = _cells(real_result["sphereFiles"])
    assert len(names) == 10
    for index, name in enumerate(names):
        path = repo_root / "data" / "oct" / "sphere" / name
        translation, rotation, centers, _scans = load_meas2023_yaml(path, _MT, _DIAMETER)
        assert translation.shape[1] == _MT
        assert rotation.shape[0] == translation.shape[0] == centers.shape[0]
        estimated_t, estimated_R, _elapsed = sphere_meas2023(translation, rotation, centers)
        np.testing.assert_allclose(estimated_t, real_result["alltO2He"][:, index, 3], atol=1e-8)
        np.testing.assert_allclose(estimated_R, real_result["allRO2He"][:, :, index, 3], atol=1e-8)


def test_plane_jnt_gh_matches_every_saved_plane_session(
    real_result: dict[str, np.ndarray], repo_root: Path
) -> None:
    legacy = _cells(real_result["planeFiles"])
    names = [name.replace("oct_checkerboard", "oct_plane") for name in legacy]
    assert len(names) == 10
    # real_calibration uses the first measured pose for every plane session.
    initial_t = real_result["alltO2He"][:, 0, 4]
    initial_R = real_result["allRO2He"][:, :, 0, 4]
    sigma0 = np.asarray(real_result["sigma0"], dtype=float).reshape(-1)
    inner = int(np.reshape(real_result["iImax"], ()))
    outer = int(np.reshape(real_result["iOmax"], ()))
    tolerance = float(np.reshape(real_result["tol"], ()))
    assert outer == 50
    for index, name in enumerate(names):
        path = repo_root / "data" / "oct" / "plane" / name
        translation, rotation, position, direction, scan = load_plane_yaml(path)
        normal, distance, _residual = plane_fit(initial_t, initial_R, translation, rotation, scan)
        result = plane_jnt_gh(
            initial_t,
            initial_R,
            normal,
            distance,
            sigma0,
            translation,
            rotation,
            position,
            direction,
            inner,
            outer,
            tolerance,
        )
        np.testing.assert_allclose(
            result.tO2H, real_result["alltO2He"][:, index, 0], atol=1e-8, err_msg=name
        )
        np.testing.assert_allclose(
            result.RO2H, real_result["allRO2He"][:, :, index, 0], atol=1e-8, err_msg=name
        )


def test_plane_rcim2020_matches_every_saved_plane_session(
    real_result: dict[str, np.ndarray], repo_root: Path
) -> None:
    legacy = _cells(real_result["planeFiles"])
    names = [name.replace("oct_checkerboard", "oct_plane") for name in legacy]
    initial_t = real_result["alltO2He"][:, 0, 4]
    initial_R = real_result["allRO2He"][:, :, 0, 4]
    tolerance = float(np.reshape(real_result["tol"], ()))
    for index, name in enumerate(names):
        path = repo_root / "data" / "oct" / "plane" / name
        translation, rotation, position, _direction, _scan = load_plane_yaml(path)
        estimated_t, estimated_R, _normal, _elapsed = plane_rcim2020(
            initial_t,
            initial_R,
            translation,
            rotation,
            position,
            tolerance,
        )
        saved_t = real_result["alltO2He"][:, index, 2]
        saved_R = real_result["allRO2He"][:, :, index, 2]
        np.testing.assert_allclose(estimated_t, saved_t, atol=1e-8, err_msg=name)
        _position, angle = calibration_error(saved_t, saved_R, estimated_t, estimated_R)
        assert float(angle) <= 1e-4, name


def test_plane_tro2016_matches_every_saved_plane_session(
    real_result: dict[str, np.ndarray], repo_root: Path
) -> None:
    """Optimizer agreement on the ten saved sessions.

    The largest gap on this result file is 0.033 mm and 0.0054 deg, on
    ``oct_plane_250712_1941.yaml``. The bounds below sit just above that.
    """
    legacy = _cells(real_result["planeFiles"])
    names = [name.replace("oct_checkerboard", "oct_plane") for name in legacy]
    initial_t = real_result["alltO2He"][:, 0, 4]
    initial_R = real_result["allRO2He"][:, :, 0, 4]
    for index, name in enumerate(names):
        path = repo_root / "data" / "oct" / "plane" / name
        translation, rotation, _position, direction, scan = load_plane_yaml(path)
        estimated_t, estimated_R, _normal, _elapsed = plane_tro2016(
            initial_t, initial_R, translation, rotation, direction, scan
        )
        saved_t = real_result["alltO2He"][:, index, 1]
        saved_R = real_result["allRO2He"][:, :, index, 1]
        position_error, angle_error = calibration_error(saved_t, saved_R, estimated_t, estimated_R)
        assert float(position_error) <= 5e-2, name
        assert float(angle_error) <= 1e-2, name


def test_plane_sessions_load_and_fit(real_result: dict[str, np.ndarray], repo_root: Path) -> None:
    legacy = _cells(real_result["planeFiles"])
    names = [name.replace("oct_checkerboard", "oct_plane") for name in legacy]
    assert len(names) == 10
    # real_calibration initializes every plane fit from the first measured pose.
    tO2H = real_result["alltO2He"][:, 0, 4]
    RO2H = real_result["allRO2He"][:, :, 0, 4]
    for name in names:
        path = repo_root / "data" / "oct" / "plane" / name
        translation, rotation, position, direction, scan = load_plane_yaml(path)
        assert translation.shape[0] == rotation.shape[0] == scan.shape[0]
        assert scan.shape[1] <= 5000
        assert scan.shape[2] == 3
        np.testing.assert_allclose(position[:, 1], 0.0, atol=0.0)
        np.testing.assert_allclose(direction[:, 1], 0.0, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(direction, axis=1), 1.0, atol=1e-12)
        assert np.all(direction[:, 0] >= -1e-12)
        # Meters would sit near 0.03; the loader converts to millimeters.
        assert np.max(np.abs(translation)) > 1.0
        normal, _distance, residual = plane_fit(tO2H, RO2H, translation, rotation, scan)
        np.testing.assert_allclose(np.linalg.norm(normal), 1.0, atol=1e-12)
        assert np.isfinite(residual)
        # The first measured pose, which real_calibration uses as the initial
        # hand-eye transform, leaves a sub-millimeter mean absolute residual.
        assert residual < 1.0

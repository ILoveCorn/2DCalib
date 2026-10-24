from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fixture_io import load_fixture, scalar
from handeye2d.metrics import calibration_error
from layout import rotations, vectors

_METHODS = (
    "plane_jnt_gh.mat",
    "plane_alt_gh.mat",
    "plane_jnt_gh_decoupled.mat",
    "plane_rcim2020.mat",
    "plane_iros2015.mat",
    "line_rcim2022.mat",
    "sphere_meas2023.mat",
)


@pytest.mark.parametrize("name", _METHODS)
def test_calibration_error_matches_the_fixture(fixtures_dir: Path, name: str) -> None:
    data = load_fixture(fixtures_dir / name)
    position, angle = calibration_error(
        vectors(data["tO2H"]),
        rotations(data["RO2H"]),
        vectors(data["tO2He"]),
        rotations(data["RO2He"]),
    )
    np.testing.assert_allclose(position, scalar(data["err_t"]), atol=1e-8)
    np.testing.assert_allclose(angle, scalar(data["err_R_deg"]), atol=1e-6)


def test_identical_pose_has_zero_error() -> None:
    translation = np.array([1.0, -2.0, 3.0])
    rotation = np.eye(3)
    position, angle = calibration_error(translation, rotation, translation, rotation)
    np.testing.assert_allclose(position, 0.0, atol=1e-15)
    np.testing.assert_allclose(angle, 0.0, atol=1e-12)

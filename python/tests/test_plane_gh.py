from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fixture_io import load_fixture, scalar
from handeye2d.calib import plane_alt_gh, plane_jnt_gh, plane_jnt_gh_decoupled
from handeye2d.metrics import calibration_error
from layout import rotations, vectors

_SOLVERS = (
    ("plane_jnt_gh.mat", plane_jnt_gh),
    ("plane_alt_gh.mat", plane_alt_gh),
    ("plane_jnt_gh_decoupled.mat", plane_jnt_gh_decoupled),
)


@pytest.mark.parametrize(("name", "solver"), _SOLVERS)
def test_solver_matches_the_octave_fixture(fixtures_dir: Path, name: str, solver) -> None:
    data = load_fixture(fixtures_dir / name)
    result = solver(
        vectors(data["tO2H0"]),
        rotations(data["RO2H0"]),
        vectors(data["n0"]),
        scalar(data["d0"]),
        np.asarray(data["sigma0"], float).reshape(-1),
        vectors(data["tH2Wn"]),
        rotations(data["RH2Wn"]),
        vectors(data["pOn"]),
        vectors(data["vOn"]),
        int(scalar(data["iImax"])),
        int(scalar(data["iOmax"])),
        scalar(data["tol"]),
    )
    np.testing.assert_allclose(result.tO2H, vectors(data["tO2He"]), atol=1e-6)
    np.testing.assert_allclose(result.RO2H, rotations(data["RO2He"]), atol=1e-8)
    np.testing.assert_allclose(result.n, vectors(data["ne"]), atol=1e-6)
    np.testing.assert_allclose(result.d, scalar(data["de"]), atol=1e-6)
    np.testing.assert_allclose(
        result.sigma, np.asarray(data["sigmae"], float).reshape(-1), rtol=1e-5
    )
    np.testing.assert_allclose(result.tH2W, vectors(data["tH2We"]), atol=1e-6)
    np.testing.assert_allclose(result.RH2W, rotations(data["RH2We"]), atol=1e-8)
    position, angle = calibration_error(
        vectors(data["tO2H"]), rotations(data["RO2H"]), result.tO2H, result.RO2H
    )
    np.testing.assert_allclose(position, scalar(data["err_t"]), atol=1e-6)
    # A 1e-15 rotation-matrix gap sits on either side of trace 3. ``arccos``
    # then reports about 1e-6 degrees, and the Octave shim stores 0 below
    # 1e-12 rad. The pose comparisons above are the reference.
    np.testing.assert_allclose(angle, scalar(data["err_R_deg"]), atol=1e-5)

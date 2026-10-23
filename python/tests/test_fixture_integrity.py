from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fixture_io import EXPECTED_MATS, fingerprint, load_fixture, scalar
from oracles import assert_finite, assert_rotations, matlab_round, pose_error

SHIM_NAMES = (
    "axang2rotm.m.in",
    "eul2rotm.m.in",
    "pagemrdivide.m.in",
    "pagemtimes.m.in",
    "quat2rotm.m.in",
    "rotm2axang.m.in",
    "rotm2eul.m.in",
)

# Noiseless (1e-18 covariance) recovery bounds, in mm and degrees.
RECOVERY = {
    "plane_jnt_gh.mat": (1e-6, 1e-6),
    "plane_jnt_gh_decoupled.mat": (1e-4, 1e-4),
    "plane_rcim2020.mat": (1e-3, 1e-2),
    "plane_iros2015.mat": (1e-3, 1e-2),
    "line_rcim2022.mat": (1e-5, 1e-3),
    "sphere_meas2023.mat": (1e-5, 1e-3),
}


def test_only_export_script_is_checked_in_as_matlab(python_root: Path) -> None:
    matlab = sorted(path.relative_to(python_root).as_posix() for path in python_root.rglob("*.m"))
    assert matlab == ["tools/octave/export_fixtures.m"]
    shim_dir = python_root / "tools" / "octave" / "shims"
    shims = sorted(path.name for path in shim_dir.glob("*.m.in"))
    assert tuple(shims) == SHIM_NAMES
    for name in SHIM_NAMES:
        text = (python_root / "tools" / "octave" / "shims" / name).read_text()
        assert text.startswith("function ")


def test_bug_patches_still_match_the_matlab_sources(repo_root: Path, python_root: Path) -> None:
    exporter = (python_root / "tools" / "octave" / "export_fixtures.m").read_text()
    line = (repo_root / "fun" / "calib" / "line_RCIM2022.m").read_text()
    meas = (repo_root / "fun" / "sim" / "simMEAS2023.m").read_text()
    line_bug = "ub(:,(1:(mt-1))+(mt-1)) = RH2Wn(:,:,i)*Rsa*us + ut;"
    line_fix = "ub(:,(1:(mt-1))+(mt-1)*(i-1)) = RH2Wn(:,:,i)*Rsa*us + ut;"
    meas_bug = "xr = min(pcOx(it)+rO(it),zMax) - 1e-4;"
    meas_fix = "xr = min(pcOx(it)+rO(it),xMax) - 1e-4;"
    assert line_bug in line and line_fix not in line
    assert meas_bug in meas and meas_fix not in meas
    assert line_bug in exporter and line_fix in exporter
    assert meas_bug in exporter and meas_fix in exporter
    assert "plane_TRO2016" in exporter


def test_manifest_and_source(fixtures_dir: Path) -> None:
    manifest = json.loads((fixtures_dir / "MANIFEST.json").read_text())
    source = (fixtures_dir / "SOURCE.txt").read_text()
    assert manifest["schema"] == 1
    assert set(manifest["files"]) == set(EXPECTED_MATS)
    assert "bug_line: fun/calib/line_RCIM2022.m:58" in source
    assert "bug_sim: fun/sim/simMEAS2023.m:90" in source
    assert "schema: 1" in source
    assert "plane_TRO2016" in source
    total = 0
    for name, meta in manifest["files"].items():
        path = fixtures_dir / name
        assert path.is_file()
        assert meta["bytes"] == path.stat().st_size
        assert meta["sha256"] == fingerprint(path)
        total += path.stat().st_size
    assert total < 5_000_000
    assert not (fixtures_dir / "plane_tro2016.mat").exists()


def test_sim_plane_shapes(fixtures_dir: Path) -> None:
    data = load_fixture(fixtures_dir / "sim_plane.mat")
    assert_finite(data)
    count = int(scalar(data["mAlpha"]) * scalar(data["mBeta"]))
    assert data["tH2Wn"].shape == (3, 1, count)
    assert data["RH2Wn"].shape == (3, 3, count)
    assert data["pOn"].shape == (3, 1, count)
    assert data["vOn"].shape == (3, 1, count)
    assert data["scanO"].shape == (3, 20, count)
    assert_rotations(data["RH2Wn"])
    assert_rotations(data["RO2H"])


def test_other_simulator_shapes(fixtures_dir: Path) -> None:
    rcim = load_fixture(fixtures_dir / "sim_rcim2022.mat")
    count = int(scalar(rcim["mAlpha"]) * scalar(rcim["mBeta"]))
    mt = int(scalar(rcim["mt"]))
    assert rcim["tH2Wn"].shape == (3, mt, count)
    assert rcim["RH2Wn"].shape == (3, 3, count)
    assert rcim["pOn"].shape == (3, mt, count)
    assert_rotations(rcim["RH2Wn"])

    meas = load_fixture(fixtures_dir / "sim_meas2023.mat")
    count = int(scalar(meas["mAlpha"]) * scalar(meas["mBeta"]))
    mt = int(scalar(meas["mt"]))
    assert meas["tH2Wn"].shape == (3, mt, count)
    assert meas["pcOe"].shape == (3, mt, count)
    assert_rotations(meas["RH2Wn"])

    legacy = load_fixture(fixtures_dir / "sim_rcim2020.mat")
    count = int(scalar(legacy["m"]))
    assert legacy["tH2Wn"].shape == (3, 1, count)
    assert legacy["RH2Wn"].shape == (3, 3, count)
    assert legacy["lOn"].shape == (6, 1, count)
    assert_rotations(legacy["RH2Wn"])

    iros = load_fixture(fixtures_dir / "sim_iros2015.mat")
    n_plane = int(scalar(iros["nPlane"]))
    side = matlab_round(scalar(iros["mAlpha"]) / np.sqrt(n_plane))
    per_plane = side * side
    assert iros["tH2Wn"].shape == (3, 1, per_plane, n_plane)
    assert iros["RH2Wn"].shape == (3, 3, per_plane, n_plane)
    assert iros["pOn"].shape == (3, 1, per_plane, n_plane)
    assert_rotations(iros["RH2Wn"])
    assert_finite(iros)


@pytest.mark.parametrize("name", sorted(RECOVERY))
def test_method_recovers_or_matches_stored_error(fixtures_dir: Path, name: str) -> None:
    data = load_fixture(fixtures_dir / name)
    assert int(scalar(data["schema"])) == 1
    assert_finite(data)
    assert_rotations(data["RO2He"])
    position, angle = pose_error(data["tO2H"], data["RO2H"], data["tO2He"], data["RO2He"])
    assert position == pytest.approx(scalar(data["err_t"]), abs=1e-8)
    assert np.degrees(angle) == pytest.approx(scalar(data["err_R_deg"]), abs=1e-6)
    pos_limit, deg_limit = RECOVERY[name]
    assert position < pos_limit
    assert np.degrees(angle) < deg_limit


def test_alternating_method_improves_on_the_initial_guess(fixtures_dir: Path) -> None:
    """iOmax is 5, so Alt is not required to sit on the noiseless fixed point."""
    data = load_fixture(fixtures_dir / "plane_alt_gh.mat")
    assert_rotations(data["RO2He"])
    initial = pose_error(data["tO2H"], data["RO2H"], data["tO2H0"], data["RO2H0"])
    final = pose_error(data["tO2H"], data["RO2H"], data["tO2He"], data["RO2He"])
    assert final[0] < initial[0]
    assert final[1] < initial[1]
    assert data["dx0"].shape == (6, 1)
    assert data["H0"].shape == (6, 6)
    assert int(scalar(data["iOmax"])) == 5


@pytest.mark.parametrize(
    ("name", "dx_rows"),
    [("plane_jnt_gh.mat", 9), ("plane_jnt_gh_decoupled.mat", 9)],
)
def test_first_gauss_newton_step_was_captured(fixtures_dir: Path, name: str, dx_rows: int) -> None:
    data = load_fixture(fixtures_dir / name)
    assert data["dx0"].shape == (dx_rows, 1)
    assert data["H0"].shape == (dx_rows, dx_rows)
    hessian = np.asarray(data["H0"], float)
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-6)
    sigma = np.reshape(np.asarray(data["sigmae"], float), (-1,))
    assert sigma.shape == (2,)
    assert np.all(sigma > 0)


def test_methods_replay_the_saved_simulators(fixtures_dir: Path) -> None:
    plane = load_fixture(fixtures_dir / "sim_plane.mat")
    for name in ("plane_jnt_gh.mat", "plane_alt_gh.mat", "plane_rcim2020.mat"):
        method = load_fixture(fixtures_dir / name)
        for key in ("tH2Wn", "RH2Wn", "pOn", "vOn", "tO2H", "RO2H"):
            np.testing.assert_array_equal(method[key], plane[key])
    iros = load_fixture(fixtures_dir / "sim_iros2015.mat")
    method = load_fixture(fixtures_dir / "plane_iros2015.mat")
    for key in ("tH2Wn", "RH2Wn", "pOn", "tO2H", "RO2H"):
        np.testing.assert_array_equal(method[key], iros[key])
    line = load_fixture(fixtures_dir / "sim_rcim2022.mat")
    method = load_fixture(fixtures_dir / "line_rcim2022.mat")
    for key in ("tH2Wn", "RH2Wn", "pOn", "tO2H", "RO2H"):
        np.testing.assert_array_equal(method[key], line[key])
    sphere = load_fixture(fixtures_dir / "sim_meas2023.mat")
    method = load_fixture(fixtures_dir / "sphere_meas2023.mat")
    for key in ("tH2Wn", "RH2Wn", "pcOe", "tO2H", "RO2H"):
        np.testing.assert_array_equal(method[key], sphere[key])

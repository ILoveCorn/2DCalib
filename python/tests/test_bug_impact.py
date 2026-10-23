from __future__ import annotations

from pathlib import Path

import numpy as np

from fixture_io import load_fixture, scalar


def test_line_bug_writes_only_the_last_pose(fixtures_dir: Path) -> None:
    bug = load_fixture(fixtures_dir / "bug_impact.mat")
    mt = int(scalar(bug["line_mt"]))
    orientations = int(scalar(bug["line_m"]))
    n_columns = int(scalar(bug["line_ncols"]))
    assert n_columns == orientations * (mt - 1)
    assert np.all(bug["line_nnz_original"] == mt - 1)
    assert np.all(bug["line_nnz_patched"] == n_columns)


def test_line_bug_leaves_orientation_unchanged(fixtures_dir: Path) -> None:
    bug = load_fixture(fixtures_dir / "bug_impact.mat")
    np.testing.assert_allclose(
        bug["line_err_R_patched"],
        bug["line_err_R_original"],
        atol=1e-8,
    )


def test_line_bug_is_dormant_without_noise_and_visible_at_0_1(fixtures_dir: Path) -> None:
    bug = load_fixture(fixtures_dir / "bug_impact.mat")
    levels = np.reshape(np.asarray(bug["line_sigmat"], float), -1)
    quiet = int(np.where(levels == 0)[0][0])
    noisy = int(np.where(np.isclose(levels, 0.1))[0][0])
    assert np.all(bug["line_err_t_patched"][:, quiet] < 1e-5)
    assert np.all(bug["line_err_t_original"][:, quiet] < 1e-5)
    patched = float(np.median(bug["line_err_t_patched"][:, noisy]))
    original = float(np.median(bug["line_err_t_original"][:, noisy]))
    assert original > patched


def test_meas2023_arc_is_clipped_to_zmax_until_patched(fixtures_dir: Path) -> None:
    bug = load_fixture(fixtures_dir / "bug_impact.mat")
    x_max = scalar(bug["meas_xMax"])
    z_max = scalar(bug["meas_zMax"])
    assert x_max > z_max
    np.testing.assert_allclose(bug["meas_xr_original"], z_max - 1e-4, atol=1e-9)
    np.testing.assert_allclose(bug["meas_xr_patched"], x_max - 1e-4, atol=1e-9)
    np.testing.assert_allclose(bug["meas_xl_original"], 1e-4, atol=1e-9)
    np.testing.assert_allclose(bug["meas_xl_patched"], 1e-4, atol=1e-9)


def test_clipped_arc_increases_sphere_calibration_error(fixtures_dir: Path) -> None:
    bug = load_fixture(fixtures_dir / "bug_impact.mat")
    patched = float(np.median(bug["meas_err_t_patched"]))
    original = float(np.median(bug["meas_err_t_original"]))
    assert original > 5.0 * patched
    rotation_original = float(np.median(bug["meas_err_R_original"]))
    rotation_patched = float(np.median(bug["meas_err_R_patched"]))
    assert rotation_original > rotation_patched

"""Smoke tests for the five simulation drivers.

``--n-test 2`` is the check that has to stay fast. The noise script also uses
one plane-fit cloud instead of 50. Statistical agreement with the paper is
not asserted here: the published table digits are not in this repository.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments import sim_accuracy_vs_image_noise as image
from experiments import sim_accuracy_vs_initial_deviation as deviation
from experiments import sim_accuracy_vs_noise_level as noise
from experiments import sim_accuracy_vs_number_of_measurements as counts
from experiments import sim_accuracy_vs_VFE as vfe
from experiments._common import median_iqr
from handeye2d.sim.sim_plane import sim_plane_batch


def test_hazen_iqr_matches_the_matlab_plotting_position() -> None:
    median, width = median_iqr(np.arange(1, 11, dtype=float))
    assert median == 5.5
    assert width == 5.0


def test_sim_plane_batch_draws_independent_sessions() -> None:
    batch = sim_plane_batch(
        4,
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        50.0,
        5.0,
        2.0,
        np.array([1.0, 2.0, 3.0]),
        np.eye(3),
        1e-4 * np.eye(3),
        1e-6 * np.eye(3),
        0.0,
        0.2,
        0.2,
        4,
        4,
        np.random.default_rng(7),
    )
    assert batch.tH2Wn.shape == (4, 16, 3)
    assert batch.RH2Wn.shape == (4, 16, 3, 3)
    assert batch.scanO.shape == (4, 16, 20, 3)
    assert np.isfinite(batch.scanO).all()
    assert not np.allclose(batch.tH2Wn[0], batch.tH2Wn[1])


def test_noise_level_smoke(tmp_path: Path) -> None:
    source = Path(noise.__file__).read_text()
    assert "plane_alt_gh" not in source
    assert "IIout" not in source
    result = noise.main(
        [
            "--n-test",
            "2",
            "--n-jobs",
            "1",
            "--n-plane-fit",
            "1",
            "--no-show",
            "--out",
            str(tmp_path),
            "--seed",
            "0",
        ]
    )
    assert result["e_pos"].shape == (2, 4, 2, 6)
    assert result["plane_fit_res"].shape == (2, 4, 2, 6, 1)
    assert np.isfinite(result["e_pos"]).all()
    assert np.isfinite(result["e_rot"]).all()
    assert (tmp_path / "sim_accuracy_vs_noise_level.npz").is_file()
    assert (tmp_path / "sim_accuracy_vs_noise_level.txt").is_file()
    assert (tmp_path / "sim_accuracy_vs_noise_level.png").stat().st_size > 0
    again = noise.main(
        [
            "--n-test",
            "2",
            "--n-jobs",
            "2",
            "--n-plane-fit",
            "1",
            "--no-show",
            "--out",
            str(tmp_path / "b"),
            "--seed",
            "0",
        ]
    )
    np.testing.assert_allclose(result["e_pos"], again["e_pos"])
    np.testing.assert_allclose(result["e_rot"], again["e_rot"])


def test_measurement_count_smoke(tmp_path: Path) -> None:
    source = Path(counts.__file__).read_text()
    assert "plane_jnt_gh_gpu" not in source
    assert "IIout" not in source
    result = counts.main(
        ["--n-test", "2", "--n-jobs", "1", "--no-show", "--out", str(tmp_path), "--seed", "1"]
    )
    assert result["e_pos"].shape == (2, 7, 6)
    assert result["m"].tolist() == [36, 64, 100, 196, 400, 784, 1024]
    assert np.isfinite(result["time"]).all()
    assert (tmp_path / "sim_accuracy_vs_number_of_measurements.pdf").is_file()


def test_initial_deviation_omits_iteration_columns(tmp_path: Path) -> None:
    result = deviation.main(
        ["--n-test", "2", "--n-jobs", "1", "--no-show", "--out", str(tmp_path), "--seed", "2"]
    )
    assert result["converged"].shape == (2, 5)
    text = (tmp_path / "sim_accuracy_vs_initial_deviation.txt").read_text()
    assert "Inner Iter" not in text
    assert "comp. time (ms)" in text
    assert (tmp_path / "sim_accuracy_vs_initial_deviation_histogram.png").is_file()
    assert np.isfinite(result["e_pos"]).all()


def test_image_noise_and_vfe_smoke(tmp_path: Path) -> None:
    image_result = image.main(
        [
            "--n-test",
            "2",
            "--n-jobs",
            "1",
            "--no-show",
            "--out",
            str(tmp_path / "image"),
            "--seed",
            "3",
        ]
    )
    repeated = image.main(
        [
            "--n-test",
            "2",
            "--n-jobs",
            "2",
            "--no-show",
            "--out",
            str(tmp_path / "image2"),
            "--seed",
            "3",
        ]
    )
    np.testing.assert_allclose(image_result["e_pos"], repeated["e_pos"])
    assert image_result["e_pos"].shape == (2, 3)
    table = (tmp_path / "image" / "sim_accuracy_vs_image_noise.txt").read_text()
    assert "um" in table
    vfe_result = vfe.main(
        [
            "--n-test",
            "2",
            "--n-jobs",
            "1",
            "--no-show",
            "--out",
            str(tmp_path / "vfe"),
            "--seed",
            "4",
        ]
    )
    assert vfe_result["e_pos"].shape == (2, 3)
    assert vfe_result["sigma"].shape == (2, 2)
    assert np.isfinite(vfe_result["sigma"]).all()
    assert (tmp_path / "vfe" / "sim_accuracy_vs_VFE.png").stat().st_size > 0

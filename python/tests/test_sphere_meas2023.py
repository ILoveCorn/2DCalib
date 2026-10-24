from __future__ import annotations

from pathlib import Path

import numpy as np

from fixture_io import load_fixture
from handeye2d.calib import sphere_meas2023
from handeye2d.so3 import exp
from layout import rotations, vectors


def test_sphere_meas2023_matches_the_octave_fixture(fixtures_dir: Path) -> None:
    data = load_fixture(fixtures_dir / "sphere_meas2023.mat")
    translation, rotation, _elapsed = sphere_meas2023(
        vectors(data["tH2Wn"]),
        rotations(data["RH2Wn"]),
        vectors(data["pcOe"]),
    )
    np.testing.assert_allclose(translation, vectors(data["tO2He"]), atol=1e-8)
    np.testing.assert_allclose(rotation, rotations(data["RO2He"]), atol=1e-8)


def test_sphere_meas2023_recovers_a_known_pose() -> None:
    rng = np.random.default_rng(9)
    truth_t = np.array([4.0, -2.0, 11.0])
    truth_R = exp(np.array([0.2, -0.1, 0.3]))
    world_center = np.array([1.0, 2.0, 30.0])
    n_pose, n_position = 5, 3
    hand_R = exp(rng.normal(scale=0.4, size=(n_pose, 3)))
    oct_center = rng.normal(size=(n_pose, n_position, 3))
    # p_w = RH @ (R @ pc + t) + tH  is constant, so tH is determined.
    mapped = np.einsum("ij,mtj->mti", truth_R, oct_center) + truth_t
    hand_t = world_center - np.einsum("mij,mtj->mti", hand_R, mapped)
    estimated_t, estimated_R, _elapsed = sphere_meas2023(hand_t, hand_R, oct_center)
    np.testing.assert_allclose(estimated_t, truth_t, atol=1e-8)
    np.testing.assert_allclose(estimated_R, truth_R, atol=1e-8)

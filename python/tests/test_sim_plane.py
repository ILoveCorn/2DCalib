from __future__ import annotations

from pathlib import Path

import numpy as np

from fixture_io import load_fixture, scalar
from handeye2d.rotations import eul_to_rotm
from handeye2d.sim import sim_plane
from layout import rotations, vectors


def _noise_free_hand_rotation(data: dict[str, np.ndarray]) -> np.ndarray:
    """Hand orientations before the pose-noise multiply in ``simPlane``."""
    normal = vectors(data["nP"])
    axis_x = np.array([-normal[1], normal[0], 0.0])
    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = np.cross(normal, axis_x)
    sensor_r0 = np.column_stack((-axis_x, axis_y, -normal))
    m_alpha = int(scalar(data["mAlpha"]))
    m_beta = int(scalar(data["mBeta"]))
    beta, alpha = np.meshgrid(
        np.linspace(-scalar(data["betaMax"]), scalar(data["betaMax"]), m_beta),
        np.linspace(-scalar(data["alphaMax"]), scalar(data["alphaMax"]), m_alpha),
        indexing="ij",
    )
    euler = np.column_stack(
        (np.zeros(m_alpha * m_beta), alpha.ravel(order="F"), beta.ravel(order="F"))
    )
    sensor_r = np.einsum("ij,mjk->mik", sensor_r0, eul_to_rotm(euler, "ZYX"))
    return sensor_r @ rotations(data["RO2H"]).T


def test_fixture_orientations_follow_the_euler_grid(fixtures_dir: Path) -> None:
    data = load_fixture(fixtures_dir / "sim_plane.mat")
    expected = _noise_free_hand_rotation(data)
    # Pose noise is chol(1e-18 I), so the saved orientations sit on this grid.
    np.testing.assert_allclose(rotations(data["RH2Wn"]), expected, atol=1e-6)


def test_sim_plane_matches_the_grid_and_stays_on_the_plane(fixtures_dir: Path) -> None:
    data = load_fixture(fixtures_dir / "sim_plane.mat")
    drawn = sim_plane(
        vectors(data["pP"]),
        vectors(data["nP"]),
        scalar(data["lP"]),
        scalar(data["xMax"]),
        scalar(data["zMax"]),
        vectors(data["tO2H"]),
        rotations(data["RO2H"]),
        np.asarray(data["Sigmat"], float),
        np.asarray(data["SigmaR"], float),
        scalar(data["sigmaO"]),
        scalar(data["alphaMax"]),
        scalar(data["betaMax"]),
        int(scalar(data["mAlpha"])),
        int(scalar(data["mBeta"])),
        np.random.default_rng(0),
    )
    np.testing.assert_allclose(drawn.RH2Wn, _noise_free_hand_rotation(data), atol=1e-6)
    normal = vectors(data["nP"])
    mapped = drawn.pOn @ rotations(data["RO2H"]).T + vectors(data["tO2H"])
    world = np.einsum("mij,mj->mi", drawn.RH2Wn, mapped)
    world = world + drawn.tH2Wn
    distance = world @ normal - float(vectors(data["pP"]) @ normal)
    np.testing.assert_allclose(distance, 0.0, atol=1e-5)
    assert drawn.scanO.shape == (drawn.tH2Wn.shape[0], 20, 3)
    np.testing.assert_allclose(drawn.pOn[:, 1], 0.0, atol=0.0)
    np.testing.assert_allclose(np.linalg.norm(drawn.vOn, axis=1), 1.0, atol=1e-12)


def test_sim_plane_is_reproducible() -> None:
    kwargs = dict(
        pP=np.zeros(3),
        nP=np.array([0.0, 0.0, 1.0]),
        lP=10.0,
        xMax=5.0,
        zMax=2.0,
        tO2H=np.array([1.0, 2.0, 3.0]),
        RO2H=np.eye(3),
        Sigmat=1e-6 * np.eye(3),
        SigmaR=1e-6 * np.eye(3),
        sigmaO=0.01,
        alphaMax=0.2,
        betaMax=0.2,
        mAlpha=2,
        mBeta=2,
    )
    first = sim_plane(**kwargs, rng=np.random.default_rng(1))
    second = sim_plane(**kwargs, rng=np.random.default_rng(1))
    other = sim_plane(**kwargs, rng=np.random.default_rng(2))
    np.testing.assert_allclose(first.tH2Wn, second.tH2Wn)
    np.testing.assert_allclose(first.scanO, second.scanO)
    assert not np.allclose(first.tH2Wn, other.tH2Wn)

from __future__ import annotations

from pathlib import Path

import numpy as np

from fixture_io import load_fixture
from handeye2d.fitting import circle_fit, plane_fit
from layout import points, rotations, vectors


def test_circle_fit_recovers_a_known_circle() -> None:
    theta = np.linspace(0.2, 2.4, 12)
    center = (1.5, -0.25)
    radius = 2.0
    x = center[0] + radius * np.cos(theta)
    z = center[1] + radius * np.sin(theta)
    xc, zc, fitted = circle_fit(x, z)
    np.testing.assert_allclose((xc, zc, fitted), (*center, radius), atol=1e-10)


def test_plane_fit_recovers_z_equals_constant() -> None:
    grid_x, grid_y = np.meshgrid(np.linspace(-1.0, 1.0, 5), np.linspace(-2.0, 2.0, 4))
    height = 3.5
    scan = np.stack(
        (np.column_stack((grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, height))),),
        axis=0,
    )
    normal, distance, residual = plane_fit(
        np.zeros(3),
        np.eye(3),
        np.zeros((1, 3)),
        np.eye(3)[None, :, :],
        scan,
    )
    if normal[2] < 0:
        normal = -normal
        distance = -distance
    np.testing.assert_allclose(normal, [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(distance, height, atol=1e-12)
    np.testing.assert_allclose(residual, 0.0, atol=1e-12)


def test_plane_fit_matches_the_sim_plane_fixture(fixtures_dir: Path) -> None:
    data = load_fixture(fixtures_dir / "sim_plane.mat")
    normal, distance, residual = plane_fit(
        vectors(data["tO2H"]),
        rotations(data["RO2H"]),
        vectors(data["tH2Wn"]),
        rotations(data["RH2Wn"]),
        points(data["scanO"]),
    )
    truth = vectors(data["nP"])
    truth = truth / np.linalg.norm(truth)
    if normal @ truth < 0:
        normal = -normal
        distance = -distance
    np.testing.assert_allclose(normal, truth, atol=1e-6)
    np.testing.assert_allclose(distance, float(np.reshape(data["pP"], (3,)) @ truth), atol=1e-6)
    assert residual < 1e-6


def test_profile_line_matches_the_sim_plane_fixture(fixtures_dir: Path) -> None:
    """The YAML loader uses this same mean / polyfit construction."""
    data = load_fixture(fixtures_dir / "sim_plane.mat")
    scan = points(data["scanO"])
    x = scan[..., 0]
    z = scan[..., 2]
    position = np.stack((x.mean(axis=1), np.zeros(x.shape[0]), z.mean(axis=1)), axis=-1)
    direction = np.empty_like(position)
    for index in range(x.shape[0]):
        slope = np.polyfit(x[index], z[index], 1)[0]
        theta = np.arctan(slope)
        direction[index] = (np.cos(theta), 0.0, np.sin(theta))
    np.testing.assert_allclose(position, vectors(data["pOn"]), atol=1e-10)
    np.testing.assert_allclose(direction, vectors(data["vOn"]), atol=1e-10)

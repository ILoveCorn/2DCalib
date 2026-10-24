from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from handeye2d import so3


def test_hat_is_skew_and_kills_the_vector() -> None:
    vector = np.array([0.2, -0.4, 0.7])
    skew = so3.hat(vector)
    np.testing.assert_allclose(skew, -skew.T, atol=1e-15)
    np.testing.assert_allclose(skew @ vector, 0.0, atol=1e-15)
    np.testing.assert_allclose(
        skew,
        np.array([[0.0, -0.7, -0.4], [0.7, 0.0, -0.2], [0.4, 0.2, 0.0]]),
    )


def test_hat_batches_match_one_at_a_time() -> None:
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, -1.0], [-0.3, 0.4, 0.5]])
    stacked = so3.hat(vectors)
    for index, vector in enumerate(vectors):
        np.testing.assert_allclose(stacked[index], so3.hat(vector))


def test_exp_of_zero_is_identity_and_matches_rodrigues() -> None:
    np.testing.assert_allclose(so3.exp(np.zeros(3)), np.eye(3), atol=1e-15)
    axis = np.array([1.0, 2.0, 3.0])
    axis = axis / np.linalg.norm(axis)
    angle = 0.4
    np.testing.assert_allclose(
        so3.exp(axis * angle),
        Rotation.from_rotvec(axis * angle).as_matrix(),
        atol=1e-12,
    )


def test_log_exp_roundtrip_away_from_pi() -> None:
    rng = np.random.default_rng(0)
    rotvec = rng.normal(size=(20, 3))
    rotvec *= 2.5 / np.linalg.norm(rotvec, axis=1, keepdims=True)
    recovered = so3.log(so3.exp(rotvec))
    np.testing.assert_allclose(recovered, rotvec, atol=1e-10)
    matrices = Rotation.from_rotvec(rotvec).as_matrix()
    scipy_log = Rotation.from_matrix(matrices).as_rotvec()
    np.testing.assert_allclose(so3.log(matrices), scipy_log, atol=1e-8)


def test_small_angle_log_uses_half_the_vee() -> None:
    angle = 1e-12
    rotation = so3.exp(np.array([angle, 0.0, 0.0]))
    np.testing.assert_allclose(so3.log(rotation), [angle, 0.0, 0.0], atol=1e-18)


def test_right_jacobian_increments_the_exponential() -> None:
    rng = np.random.default_rng(1)
    rotvec = rng.normal(size=3) * 0.3
    step = rng.normal(size=3) * 1e-6
    left = so3.exp(rotvec + step)
    right = so3.exp(rotvec) @ so3.exp(so3.jr(rotvec) @ step)
    np.testing.assert_allclose(left, right, atol=1e-9)


def test_jacobian_fixes_the_rotvec_and_inverts() -> None:
    rng = np.random.default_rng(2)
    rotvecs = np.vstack([np.zeros(3), rng.normal(size=(8, 3))])
    applied = np.einsum("nij,nj->ni", so3.jr(rotvecs), rotvecs)
    np.testing.assert_allclose(applied, rotvecs, atol=1e-12)
    product = so3.jr(rotvecs) @ so3.jr_inv(rotvecs)
    eye = np.broadcast_to(np.eye(3), product.shape)
    np.testing.assert_allclose(product, eye, atol=1e-10)


def test_random_rotvec_stays_inside_the_ball_and_is_reproducible() -> None:
    first = np.stack([so3.random_rotvec(np.random.default_rng(4)) for _ in range(30)])
    second = np.stack([so3.random_rotvec(np.random.default_rng(4)) for _ in range(30)])
    np.testing.assert_allclose(first, second)
    assert np.all(np.linalg.norm(first, axis=1) < np.pi)
    rotations = so3.exp(first)
    np.testing.assert_allclose(np.linalg.det(rotations), 1.0, atol=1e-10)
    other = so3.random_rotvec(np.random.default_rng(5))
    assert not np.allclose(other, first[0])

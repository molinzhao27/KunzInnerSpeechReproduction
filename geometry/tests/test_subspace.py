"""Principal angles, PCA bases and variance explained on synthetic data with known answers.

Run: python -m pytest geometry/tests
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import subspace_angles

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from geometry import subspace  # noqa: E402


def _e(F, *idx):
    E = np.zeros((F, len(idx)))
    E[list(idx), range(len(idx))] = 1.0
    return E


@pytest.mark.parametrize("theta", [1e-7, 1e-3, 0.3, np.pi / 4, 1.2, np.pi / 2])
def test_planted_angle(theta):
    F = 10
    A = _e(F, 0, 1)
    B = np.zeros((F, 2))
    B[0, 0] = 1
    B[1, 1], B[2, 1] = np.cos(theta), np.sin(theta)
    got = subspace.principal_angles(A, B)
    np.testing.assert_allclose(got, np.sort([0.0, theta]), atol=1e-12, rtol=1e-9)


def test_identical_and_orthogonal():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(50, 4))
    np.testing.assert_allclose(subspace.principal_angles(A, A), 0, atol=1e-7)
    np.testing.assert_allclose(subspace.principal_angles(_e(20, 0, 1, 2), _e(20, 3, 4, 5)), np.pi / 2, atol=1e-12)


def test_basis_invariance_and_symmetry():
    rng = np.random.default_rng(1)
    A, B = rng.normal(size=(30, 3)), rng.normal(size=(30, 5))
    ref = subspace.principal_angles(A, B)
    mix = rng.normal(size=(3, 3))  # any invertible re-basing of A's span
    np.testing.assert_allclose(subspace.principal_angles(A @ mix, B), ref, atol=1e-10)
    np.testing.assert_allclose(subspace.principal_angles(B, A), ref, atol=1e-10)
    assert len(ref) == 3 and np.all(np.diff(ref) >= 0)


def test_matches_scipy():
    rng = np.random.default_rng(2)
    for p, q in [(1, 1), (2, 6), (6, 6), (5, 3)]:
        A, B = rng.normal(size=(40, p)), rng.normal(size=(40, q))
        np.testing.assert_allclose(subspace.principal_angles(A, B), np.sort(subspace_angles(A, B)), atol=1e-10)


def test_rank_deficient_input():
    A = np.hstack([_e(10, 0, 1), _e(10, 0)])  # 3 columns, rank 2
    assert len(subspace.orth(A).T) == 2
    np.testing.assert_allclose(subspace.principal_angles(A, _e(10, 0, 1)), 0, atol=1e-12)


def test_random_subspaces_near_orthogonal_in_high_dim():
    rng = np.random.default_rng(3)
    ang = subspace.principal_angles(rng.normal(size=(2000, 2)), rng.normal(size=(2000, 2)))
    assert np.degrees(ang.min()) > 80


def test_pca_basis_recovers_planted_subspace():
    rng = np.random.default_rng(4)
    F, k = 40, 3
    U = np.linalg.qr(rng.normal(size=(F, k)))[0]
    X = rng.normal(size=(500, k)) * [10, 7, 5] @ U.T + 0.1 * rng.normal(size=(500, F)) + 3.0
    ang = subspace.principal_angles(subspace.pca_basis(X, k), U)
    assert np.degrees(ang.max()) < 2


def test_variance_explained_known_fractions():
    rng = np.random.default_rng(5)
    F = 6
    sd = np.array([3.0, 2.0, 1.0, 0.0, 0.0, 0.0])
    X = rng.normal(size=(20000, F)) * sd + 5.0
    np.testing.assert_allclose(subspace.variance_explained(X, _e(F, 0)), 9 / 14, atol=0.01)
    np.testing.assert_allclose(subspace.variance_explained(X, _e(F, 0, 1, 2)), 1.0, atol=1e-12)
    np.testing.assert_allclose(subspace.variance_explained(X, _e(F, 3, 4)), 0.0, atol=1e-12)


def test_stratified_halves():
    rng = np.random.default_rng(6)
    y = np.repeat(np.arange(7), 11)
    h1, h2 = subspace.stratified_halves(y, rng)
    assert len(np.intersect1d(h1, h2)) == 0 and len(h1) + len(h2) == len(y)
    assert set(np.bincount(y[h1])) == {6} and set(np.bincount(y[h2])) == {5}


def test_signal_variance_explained_is_noise_corrected():
    rng = np.random.default_rng(7)
    W, F, n = 7, 20, 10
    U = np.linalg.qr(rng.normal(size=(F, 2)))[0]
    mu = rng.normal(size=(W, 2)) @ U.T * 0.5  # word signal lives entirely in span(U)
    y = np.repeat(np.arange(W), n)
    noise_sd = np.linspace(0.5, 3, F)  # anisotropic noise
    num, den = np.mean([subspace.signal_variance_explained(mu[y] + rng.normal(size=(W * n, F)) * noise_sd, y, U)
                        for _ in range(3000)], axis=0)
    true = ((mu - mu.mean(0)) ** 2).sum()
    assert abs(den - true) / true < 0.08 and abs(num - true) / true < 0.05  # signal VE ~ 1 (den is noisier)
    Xr = mu[y] + rng.normal(size=(W * n, F)) * noise_sd
    raw = subspace.variance_explained(subspace.word_means(Xr, y, W), U)
    assert raw < 0.5  # the raw word-mean VE is dominated by noise here
    Up = np.linalg.qr(np.hstack([U, rng.normal(size=(F, 2))]))[0][:, 2:]  # orthogonal to the signal
    num0 = np.mean([subspace.signal_variance_explained(mu[y] + rng.normal(size=(W * n, F)) * noise_sd, y, Up)[0]
                    for _ in range(400)])
    assert abs(num0) / true < 0.05


def test_noise_whitener_gives_identity_noise():
    rng = np.random.default_rng(8)
    F, W, n = 15, 7, 300
    A = np.eye(F) + 0.3 * rng.normal(size=(F, F))  # correlated, well-conditioned noise
    y = np.repeat(np.arange(W), n)
    X1 = rng.normal(size=(W, F))[y] * 3 + rng.normal(size=(W * n, F)) @ A.T
    X2 = rng.normal(size=(W, F))[y] + rng.normal(size=(W * n, F)) @ A.T
    Wm = subspace.noise_whitener([X1, X2], [y, y])
    resid = X1 - subspace.word_means(X1, y, W)[y]
    raw, white = np.linalg.eigvalsh(np.cov(resid.T)), np.linalg.eigvalsh(np.cov((resid @ Wm).T))
    assert raw.max() / raw.min() > 10  # strongly anisotropic noise ...
    assert np.mean((white > 0.8) & (white < 1.2)) > 0.9  # ... becomes ~isotropic (shrinkage leaves the
    assert (raw.max() / raw.min()) / (white.max() / white.min()) > 10  # weakest direction somewhat below 1)
    np.testing.assert_allclose(Wm, Wm.T, atol=1e-10)

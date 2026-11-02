"""Gain fits, leave-one-word-out map comparison and the noise-corrected signal variance, on synthetic data.

Run: python -m pytest geometry/tests
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from geometry import scaling  # noqa: E402


def _patterns(seed=0, w=7, F=50):
    rng = np.random.default_rng(seed)
    S = rng.normal(size=(w, F))
    return rng, S - S.mean(0)


def test_global_gain_exact():
    _, S = _patterns()
    assert np.isclose(scaling.fit_gain(S, 0.4 * S, per_feature=False), 0.4)


def test_per_feature_gains_exact_and_noisy():
    rng, S = _patterns(F=200)
    g = rng.uniform(-1, 2, size=200)
    np.testing.assert_allclose(scaling.fit_gain(S, S * g), g, atol=1e-12)
    # many samples + noise: unbiased recovery
    X = rng.normal(size=(5000, 200))
    est = scaling.fit_gain(X, X * g + 0.5 * rng.normal(size=X.shape))
    np.testing.assert_allclose(est, g, atol=0.03)


def test_zero_feature_gets_zero_gain():
    S = np.ones((5, 3))
    S[:, 1] = 0
    assert scaling.fit_gain(S, S)[1] == 0


def test_global_gain_model_wins_when_true():
    _, S = _patterns(1)
    T = 0.3 * S
    assert np.isclose(scaling.lowo_r2(S, T, "global"), 1.0)
    assert np.isclose(scaling.lowo_r2(S, T, "electrode"), 1.0)
    assert np.isclose(scaling.insample_r2(S, T, "global"), 1.0)


def test_per_feature_gain_beats_global_when_gains_differ():
    rng, S = _patterns(2, F=300)
    g = np.where(rng.random(300) < 0.5, 1.0, -1.0)  # half the features flip sign: global gain ~ 0
    T = S * g
    assert np.isclose(scaling.lowo_r2(S, T, "electrode"), 1.0)
    assert scaling.lowo_r2(S, T, "global") < 0.2
    assert np.isclose(scaling.insample_r2(S, T, "electrode"), 1.0)


def test_full_map_recovers_linear_map_in_low_dim():
    rng = np.random.default_rng(3)
    S = rng.normal(size=(7, 3))
    Wmap = rng.normal(size=(3, 3))  # a rotation/shear no gain model can express
    T = S @ Wmap
    assert scaling.lowo_r2(S, T, "full", alphas=[1e-10]) > 0.999
    assert scaling.lowo_r2(S, T, "full", alphas=[1e-10, 1.0, 100.0]) > 0.999  # nested choice picks the small alpha
    assert scaling.lowo_r2(S, T, "electrode") < 0.9


def test_unrelated_target_has_no_lowo_r2():
    rng = np.random.default_rng(4)
    r2 = [scaling.lowo_r2(rng.normal(size=(7, 40)), rng.normal(size=(7, 40)), "global") for _ in range(300)]
    assert np.mean(r2) < 0.0  # held-out words: nothing to predict, R^2 at or below zero on average


def test_ridge_predict_matches_primal():
    rng = np.random.default_rng(5)
    X, Y, Xn = rng.normal(size=(6, 20)), rng.normal(size=(6, 20)), rng.normal(size=(2, 20))
    a = 0.7
    lam = a * np.trace(X @ X.T) / 6
    W = np.linalg.solve(X.T @ X + lam * np.eye(20), X.T @ Y)
    np.testing.assert_allclose(scaling.ridge_predict(X, Y, Xn, a), Xn @ W, atol=1e-10)


def test_signal_variance_is_unbiased():
    rng = np.random.default_rng(6)
    W, F, n = 7, 30, 12
    mu = rng.normal(size=(W, F)) * 0.3
    true = ((mu - mu.mean(0)) ** 2).sum()
    y = np.repeat(np.arange(W), n)
    est = [scaling.signal_variance(mu[y] + rng.normal(size=(W * n, F)) * 1.5, y) for _ in range(400)]
    total, corrected = np.mean(est, axis=0)
    assert total > 2 * true  # the raw estimate is inflated by noise ...
    assert abs(corrected - true) / true < 0.05  # ... the corrected one is not


def test_noise_corrected_pattern_correlation():
    rng = np.random.default_rng(7)
    W, F, n = 7, 40, 10
    S = rng.normal(size=(W, F))
    T = 0.2 * S + 0.2 * rng.normal(size=(W, F))  # scaled copy plus an independent component
    Sc, Tc = S - S.mean(0), T - T.mean(0)
    true_r = (Sc * Tc).sum() / np.sqrt((Sc**2).sum() * (Tc**2).sum())
    y = np.repeat(np.arange(W), n)
    est = []
    for _ in range(300):
        Xs, Xt = S[y] + rng.normal(size=(W * n, F)), T[y] + rng.normal(size=(W * n, F))
        est.append([scaling.pattern_cross(Xs, y, Xt, y), scaling.signal_variance(Xs, y)[1],
                    scaling.signal_variance(Xt, y)[1]])
    cr, ss, st = np.mean(est, axis=0)
    assert abs(cr / np.sqrt(ss * st) - true_r) < 0.05
    raw = np.mean([e[0] / np.sqrt(scaling.signal_variance(S[y] + rng.normal(size=(W * n, F)), y)[0]
                                  * scaling.signal_variance(T[y] + rng.normal(size=(W * n, F)), y)[0]) for e in est[:50]])
    assert raw < true_r - 0.1  # without the correction the similarity is attenuated by noise

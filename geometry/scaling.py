"""Is one condition a scaled copy of another? Gain fits vs a full linear map on word-mean patterns.

Conditions share no trials, so the only pairing between them is word identity: every map is fit
from source word means S (w, F) to target word means T (w, F). Models, nested:
  global    T ~ g * S              (1 parameter)
  electrode T[:, f] ~ g_f * S[:, f] (one gain per feature)
  full      T ~ S @ W              (ridge, F x F; alpha by nested leave-one-word-out)
Evaluated by leave-one-word-out (LOWO): fit on w-1 words (centered with their own means), predict
the held-out word's target pattern; R^2 is relative to predicting the training-word mean.
"""

from __future__ import annotations

import numpy as np


def fit_gain(X: np.ndarray, Y: np.ndarray, per_feature: bool = True) -> np.ndarray | float:
    """Least-squares gain(s) for Y ~ g * X (no intercept; center beforehand). X, Y: (n, F)."""
    if per_feature:
        num, den = (X * Y).sum(0), (X * X).sum(0)
        return np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    den = float((X * X).sum())
    return float((X * Y).sum()) / den if den > 0 else 0.0


def ridge_predict(X: np.ndarray, Y: np.ndarray, Xnew: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge map Y ~ X @ W in dual form (n << F): Xnew @ X' (XX' + aI)^-1 Y; alpha is relative to trace(XX')/n."""
    K = X @ X.T
    a = alpha * np.trace(K) / len(X)
    return Xnew @ X.T @ np.linalg.solve(K + a * np.eye(len(X)), Y)


def _predict(model: str, Xs: np.ndarray, Yt: np.ndarray, xh: np.ndarray, alphas, alpha=None) -> np.ndarray:
    if model == "global":
        return xh * fit_gain(Xs, Yt, per_feature=False)
    if model == "electrode":
        return xh * fit_gain(Xs, Yt, per_feature=True)
    if model == "full":
        if alpha is None:  # nested LOWO on the training words
            alpha = alphas[int(np.argmax([lowo_r2(Xs, Yt, "full", alphas, alpha=a) for a in alphas]))]
        return ridge_predict(Xs, Yt, xh[None], alpha)[0]
    raise ValueError(model)


def lowo_r2(S: np.ndarray, T: np.ndarray, model: str, alphas=(1.0,), alpha=None) -> float:
    """Leave-one-word-out R^2 of predicting target word means T from source word means S."""
    res = base = 0.0
    for h in range(len(S)):
        tr = np.arange(len(S)) != h
        ms, mt = S[tr].mean(0), T[tr].mean(0)
        pred = mt + _predict(model, S[tr] - ms, T[tr] - mt, S[h] - ms, alphas, alpha)
        res += ((T[h] - pred) ** 2).sum()
        base += ((T[h] - mt) ** 2).sum()
    return 1.0 - res / base


def insample_r2(S: np.ndarray, T: np.ndarray, model: str) -> float:
    """R^2 of the gain models fit on all words (centered). The full map is trivially 1 for w-1 <= F."""
    Sc, Tc = S - S.mean(0), T - T.mean(0)
    pred = Sc * fit_gain(Sc, Tc, per_feature=(model == "electrode"))
    return 1.0 - ((Tc - pred) ** 2).sum() / (Tc**2).sum()


def pattern_cross(Xs: np.ndarray, ys: np.ndarray, Xt: np.ndarray, yt: np.ndarray) -> float:
    """<S, T> of the centered word means of two conditions. Their trial noise is independent, so this
    is an unbiased estimate of the noise-free cross product. With signal_variance it gives the
    noise-corrected pattern correlation r = <S,T> / sqrt(sig_S sig_T); r^2 is the share of the target's
    signal variance a single global gain can explain, and <S,T> / sig_S the de-attenuated gain."""
    labels = np.unique(ys)
    S = np.stack([Xs[ys == w].mean(0) for w in labels])
    T = np.stack([Xt[yt == w].mean(0) for w in labels])
    return float(((S - S.mean(0)) * (T - T.mean(0))).sum())


def signal_variance(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(total, noise-corrected) sum of squares of the centered word means.

    Each word mean carries noise of variance s^2_f / n_w per feature (s^2 = pooled within-word
    variance); its expected contribution to the centered sum of squares is sum_w (1 - 1/W) sum_f s^2_f / n_w.
    The corrected value is an unbiased estimate of the noise-free signal sum of squares.
    """
    labels = np.unique(y)
    M = np.stack([X[y == w].mean(0) for w in labels])
    n = np.array([(y == w).sum() for w in labels])
    resid = X - M[np.searchsorted(labels, y)]
    s2 = (resid**2).sum(0) / (len(X) - len(labels))
    total = float(((M - M.mean(0)) ** 2).sum())
    noise = float(((1 - 1 / len(labels)) * s2.sum() / n).sum())
    return total, total - noise

"""Subspace geometry: PCA bases, principal angles, cross-condition variance explained."""

from __future__ import annotations

import numpy as np


def orth(A: np.ndarray) -> np.ndarray:
    """Orthonormal basis (F, rank) for the column span of A."""
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    tol = max(A.shape) * np.finfo(float).eps * (s[0] if s.size else 0.0)
    return U[:, s > tol]


def principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Principal angles (radians, ascending) between the column spans of A (F, p) and B (F, q).

    Returns min(p, q) angles. Cosines come from the SVD of Qa'Qb; angles with cos^2 > 1/2 are
    recomputed from sines (the residual of the smaller span after projecting onto the larger),
    which keeps small angles accurate (Knyazev & Argentati 2002, as in scipy).
    """
    Qa, Qb = orth(A), orth(B)
    if Qa.shape[1] > Qb.shape[1]:
        Qa, Qb = Qb, Qa
    cos = np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), 0.0, 1.0)  # descending
    sin = np.clip(np.sort(np.linalg.svd(Qa - Qb @ (Qb.T @ Qa), compute_uv=False)), 0.0, 1.0)  # ascending
    return np.where(cos**2 > 0.5, np.arcsin(sin), np.arccos(cos))


def pca_basis(M: np.ndarray, k: int) -> np.ndarray:
    """Top-k principal axes (F, k) of the rows of M, after centering."""
    Mc = M - M.mean(0)
    _, _, Vt = np.linalg.svd(Mc, full_matrices=False)
    return Vt[:k].T


def variance_explained(M: np.ndarray, U: np.ndarray) -> float:
    """Fraction of the (centered) variance of the rows of M inside span(U); U orthonormal (F, k)."""
    Mc = M - M.mean(0)
    tot = (Mc**2).sum()
    return float(((Mc @ U) ** 2).sum() / tot) if tot > 0 else np.nan


def signal_variance_explained(X: np.ndarray, y: np.ndarray, U: np.ndarray) -> tuple[float, float]:
    """Noise-corrected (signal variance inside span(U), total signal variance) of the word means of X.

    Word means carry trial noise; its expected share of the centered sum of squares inside span(U) is
    sum_w (1 - 1/W) / n_w * tr(U' S U), S = pooled within-word covariance. Subtracting it from numerator
    and denominator gives unbiased estimates when U does not depend on X's trials. Ratio = signal VE.
    """
    labels = np.unique(y)
    M = np.stack([X[y == w].mean(0) for w in labels])
    n = np.array([(y == w).sum() for w in labels])
    R = X - M[np.searchsorted(labels, y)]
    c = ((1 - 1 / len(labels)) / n).sum() / (len(X) - len(labels))
    Mc = M - M.mean(0)
    num = ((Mc @ U) ** 2).sum() - c * ((R @ U) ** 2).sum()
    den = (Mc**2).sum() - c * (R**2).sum()
    return float(num), float(den)


def noise_whitener(Xs: list[np.ndarray], ys: list[np.ndarray]) -> np.ndarray:
    """(F, F) matrix W such that X @ W has (approximately) identity trial-noise covariance.

    Multivariate noise normalization (Walther et al. 2016): the noise covariance is the Ledoit-Wolf
    shrinkage estimate of the within-word residuals, pooled over all given conditions, so every
    condition gets the same linear map and cross-condition comparisons stay fair.
    """
    from sklearn.covariance import ledoit_wolf

    R = np.vstack([X - np.stack([X[y == w].mean(0) for w in np.unique(y)])[np.searchsorted(np.unique(y), y)]
                   for X, y in zip(Xs, ys)])
    C = ledoit_wolf(R, assume_centered=True)[0]
    vals, vecs = np.linalg.eigh(C)
    return vecs / np.sqrt(vals) @ vecs.T


def word_means(X: np.ndarray, y: np.ndarray, n_words: int) -> np.ndarray:
    return np.stack([X[y == w].mean(0) for w in range(n_words)])


def stratified_halves(y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random split into two halves with each label split evenly (odd counts: extra goes to half 1)."""
    h1, h2 = [], []
    for w in np.unique(y):
        i = rng.permutation(np.flatnonzero(y == w))
        h1.append(i[: (len(i) + 1) // 2])
        h2.append(i[(len(i) + 1) // 2:])
    return np.sort(np.concatenate(h1)), np.sort(np.concatenate(h2))

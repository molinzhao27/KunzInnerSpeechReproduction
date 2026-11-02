"""Within- and cross-condition word decoding with shared fold decoders.

Each condition A gets a stratified k-fold CV. The decoder of every fold is tested on A's held-out
fold (within-condition accuracy) and on all trials of every other condition B (cross-condition
accuracy = mean over folds). Within and cross accuracies therefore come from the very same
decoders, trained on the same number of trials.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .data import Cond, permute_within


def make_clf(name: str, cfg: dict):
    if name == "lda":  # Ledoit-Wolf shrinkage (best linear model in Phase 3)
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if name == "logreg":
        return LogisticRegression(C=cfg["decoding"]["logreg_C"], max_iter=5000)
    raise ValueError(name)


def center(conds: dict[str, Cond], how: str) -> dict[str, Cond]:
    """'condition': subtract each condition's own mean over its trials (label-free)."""
    if how == "none":
        return conds
    return {k: Cond(c.X - c.X.mean(0), c.y, c.block) for k, c in conds.items()}


def cv_cross(conds: dict[str, Cond], pairs: list, clf: str, cfg: dict, seed: int) -> dict:
    """Returns {"within": {A: acc}, "cross": {(A, B): acc}, "preds": {(A, B): (folds, n_B) predictions}}."""
    trains = {a for a, _ in pairs} | set(conds)
    within, cross, preds = {}, {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for a in sorted(trains):
            A = conds[a]
            skf = StratifiedKFold(cfg["decoding"]["folds"], shuffle=True, random_state=seed)
            correct = 0
            targets = [b for x, b in pairs if x == a]
            fold_preds = {b: [] for b in targets}
            for tr, te in skf.split(A.X, A.y):
                m = make_clf(clf, cfg).fit(A.X[tr], A.y[tr])
                correct += (m.predict(A.X[te]) == A.y[te]).sum()
                for b in targets:
                    fold_preds[b].append(m.predict(conds[b].X))
            within[a] = correct / len(A.y)
            for b in targets:
                P = np.stack(fold_preds[b])
                preds[(a, b)] = P
                cross[(a, b)] = float((P == conds[b].y).mean())
    return {"within": within, "cross": cross, "preds": preds}


def cross_null(P: np.ndarray, B: Cond, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """Null accuracies for fixed decoders' predictions P (folds, n_B): B's labels permuted within blocks."""
    return np.array([(P == permute_within(B.y, B.block, rng)).mean() for _ in range(n_perm)])


def within_null(A: Cond, clf: str, cfg: dict, n_perm: int, rng: np.random.Generator, seed: int) -> np.ndarray:
    """Null within-condition CV accuracies: labels permuted within blocks, CV re-run (folds re-stratified)."""
    out = []
    for _ in range(n_perm):
        Ap = Cond(A.X, permute_within(A.y, A.block, rng), A.block)
        out.append(cv_cross({"a": Ap}, [], clf, cfg, seed)["within"]["a"])
    return np.array(out)


def p_value(obs: float, null: np.ndarray) -> float:
    return float((1 + (null >= obs).sum()) / (1 + len(null)))

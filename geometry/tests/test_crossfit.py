"""Noise-normalized (whitened) statistics must stay unbiased when there are more features than residual
degrees of freedom: on pure noise, the cross-fitted estimates show no word signal, while whitening with
all trials (the naive way) manufactures one.

Run: python -m pytest geometry/tests
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from geometry import analyses, scaling, subspace  # noqa: E402
from geometry.data import Cond  # noqa: E402

CFG = {"words": list("abcdefg"), "conditions": ["attempted", "inner", "perceived"],
       "scaling": {"sources": ["attempted"], "target": "inner", "ridge_alphas": [1.0]},
       "controls": {"match_trials": True, "n_match_reps": 4, "n_perm": 0}}


def _noise_conds(rng, F=120, n=10, W=7):
    y = np.repeat(np.arange(W), n)  # 3 x 70 trials - 21 means = 189 residual dof > F/2; halves: 84 < F
    A = rng.normal(size=(F, F)) / np.sqrt(F) + np.eye(F)
    return {c: Cond(rng.normal(size=(W * n, F)) @ A.T, y, np.zeros(W * n, int)) for c in CFG["conditions"]}


def test_naive_whitening_inflates_signal():
    rng = np.random.default_rng(0)
    fr = []
    for _ in range(10):
        conds = _noise_conds(rng)
        Wm = subspace.noise_whitener([c.X for c in conds.values()], [c.y for c in conds.values()])
        X, y = conds["inner"].X @ Wm, conds["inner"].y
        total, corrected = scaling.signal_variance(X, y)
        fr.append(corrected / total)
    assert np.mean(fr) > 0.1  # pure noise, yet a sizeable "reliable" fraction


def test_crossfit_whitening_shows_no_signal_on_noise():
    rng = np.random.default_rng(1)
    fr = []
    for i in range(10):
        rows, _ = analyses.gains(_noise_conds(rng), CFG, i, {}, nulls=False, models=("global",), whiten=True)
        fr.append(rows[0]["target_reliable_frac"])
    assert np.mean(fr) < 0.03  # reliable fraction is clipped at 0, so its mean sits just above 0

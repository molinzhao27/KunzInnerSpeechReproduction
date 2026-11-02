"""One (run, region) job: decoding, subspace alignment and gain fits, each with its controls.

Every analysis uses matched trial counts (each condition subsampled to the smallest per-word
count, repeated and averaged) unless noted, and shuffled-label nulls.
"""

from __future__ import annotations

import numpy as np
from threadpoolctl import threadpool_limits

from . import decode, scaling, subspace
from .data import Cond, matched_subsample, permute_within


def _subsamples(conds: dict[str, Cond], cfg: dict, seed: int) -> list[dict[str, Cond]]:
    c = cfg["controls"]
    if not c["match_trials"]:
        return [conds]
    return [matched_subsample(conds, np.random.default_rng([seed, r])) for r in range(c["n_match_reps"])]


def decoding(conds: dict[str, Cond], cfg: dict, seed: int, base: dict, nulls: bool = True) -> list[dict]:
    rng = np.random.default_rng([seed, 1])
    pairs = [tuple(p) for p in cfg["pairs"]]
    rows = []
    variants = [(True, clf, cen) for clf in cfg["decoding"]["classifiers"] for cen in cfg["decoding"]["centering"]]
    variants += [(False, "lda", cen) for cen in cfg["decoding"]["centering"]]  # unmatched-count control
    for matched, clf, cen in variants:
        subs = _subsamples(conds, cfg, seed) if matched else [conds]
        res = [decode.cv_cross(decode.center(s, cen), pairs, clf, cfg, seed + r) for r, s in enumerate(subs)]
        do_null = nulls and matched and clf == "lda"
        for a in cfg["conditions"]:
            acc = [r["within"][a] for r in res]
            row = base | dict(classifier=clf, center=cen, matched=matched, kind="within", train=a, test=a,
                              acc=np.mean(acc), acc_sd=np.std(acc), n_per_word=int(np.bincount(subs[0][a].y).min()))
            if do_null and cen == "none":  # LDA within-condition accuracy is invariant to centering
                null = decode.within_null(subs[0][a], clf, cfg, cfg["controls"]["n_perm"], rng, seed)
                row |= dict(null_mean=null.mean(), null_q95=np.quantile(null, 0.95), p=decode.p_value(row["acc"], null))
            rows.append(row)
        for a, b in pairs:
            acc = [r["cross"][(a, b)] for r in res]
            row = base | dict(classifier=clf, center=cen, matched=matched, kind="cross", train=a, test=b,
                              acc=np.mean(acc), acc_sd=np.std(acc), n_per_word=int(np.bincount(subs[0][a].y).min()))
            if do_null:  # fixed decoders; test labels permuted within blocks, independently per subsample
                null = np.mean([decode.cross_null(r["preds"][(a, b)], s[b], cfg["controls"]["n_perm_cross"], rng)
                                for r, s in zip(res, subs)], axis=0)
                row |= dict(null_mean=null.mean(), null_q95=np.quantile(null, 0.95), p=decode.p_value(row["acc"], null))
            rows.append(row)
    return rows


def _whitener(sub: dict[str, Cond], idx: dict[str, np.ndarray]) -> np.ndarray:
    """Noise whitener from the given trials of every condition (true labels)."""
    return subspace.noise_whitener([d.X[idx[c]] for c, d in sub.items()], [d.y[idx[c]] for c, d in sub.items()])


def _split_means(sub: dict[str, Cond], W: int, rng, shuffle: str | None = None, whiten: bool = False):
    """Per condition: word means, condition-centered single trials and the half-2 trials (X, y).

    whiten: noise-normalize with a whitener fit on half 1 only (cross-fitting). A whitener fit on all
    trials would amplify directions its residuals don't span (fewer residual dof than features), where
    the half-2 noise correction then sees no noise: noise-corrected signal estimates would be inflated.
    """
    halves = {c: subspace.stratified_halves(d.y, rng) for c, d in sub.items()}
    Wm = _whitener(sub, {c: h[0] for c, h in halves.items()}) if whiten else None
    means, trials, half2 = {}, {}, {}
    for c, d in sub.items():
        h1, h2 = halves[c]
        X = d.X @ Wm if whiten else d.X
        y1 = permute_within(d.y[h1], d.block[h1], rng) if c == shuffle else d.y[h1]
        means[c] = (subspace.word_means(X[h1], y1, W), subspace.word_means(X[h2], d.y[h2], W))
        trials[c] = (X[h1] - X[h1].mean(0), X[h2] - X[h2].mean(0))
        half2[c] = (X[h2], d.y[h2])
    return means, trials, half2


def _crossfit_views(s: dict[str, Cond], rng) -> list[dict[str, Cond]]:
    """Two noise-normalized views of a subsample: each half whitened with the other half's whitener."""
    halves = {c: subspace.stratified_halves(d.y, rng) for c, d in s.items()}
    views = []
    for fit, ev in ((0, 1), (1, 0)):
        Wm = _whitener(s, {c: h[fit] for c, h in halves.items()})
        views.append({c: Cond(d.X[halves[c][ev]] @ Wm, d.y[halves[c][ev]], d.block[halves[c][ev]])
                      for c, d in s.items()})
    return views


def subspaces(conds: dict[str, Cond], cfg: dict, seed: int, base: dict, nulls: bool = True,
              whiten: bool = False) -> tuple[list, list]:
    """Cross-validated alignment: PCs from half 1 of condition a, evaluated on half 2 of b (a == b: ceiling).

    Null: a's half-1 word labels shuffled within blocks (its 'word subspace' is then noise).
    Returns (angle rows, variance-explained rows).
    """
    sc, W = cfg["subspace"], len(cfg["words"])
    rng = np.random.default_rng([seed, 2])
    pairs = [tuple(p) for p in cfg["pairs"]] + [(c, c) for c in cfg["conditions"]]
    kA, kT = sc["k_max_avg"], sc["k_max_trial"]
    n_perm = cfg["controls"]["n_perm"] if nulls else 0

    def one(sub, shuffle=None):
        ang, ve = {}, {}
        for a in sorted({p[0] for p in pairs}):
            means, trials, half2 = _split_means(sub, W, rng, shuffle=a if shuffle else None, whiten=whiten)
            Ua = subspace.pca_basis(means[a][0], kA)
            Ta = subspace.pca_basis(trials[a][0], kT)
            for b in [q for p, q in pairs if p == a]:
                Ub = subspace.pca_basis(means[b][1], kA)
                for k in range(1, kA + 1):
                    ang[(a, b, k)] = np.degrees(subspace.principal_angles(Ua[:, :k], Ub[:, :k]))
                    ve[(a, b, "avg", k)] = subspace.variance_explained(means[b][1], Ua[:, :k])
                    ve[(a, b, "signal", k)] = subspace.signal_variance_explained(*half2[b], Ua[:, :k])
                if not shuffle:
                    for k in range(1, kT + 1):
                        ve[(a, b, "trial", k)] = subspace.variance_explained(trials[b][1], Ta[:, :k])
        return ang, ve

    obs = [one(matched_subsample(conds, rng)) for _ in range(sc["n_splits"])]
    null = [one(matched_subsample(conds, rng), shuffle=True) for _ in range(n_perm)]
    ang_rows, ve_rows = [], []
    for key in obs[0][0]:
        a, b, k = key
        A = np.stack([o[0][key] for o in obs])  # (splits, k)
        row = base | dict(source=a, target=b, kind="within" if a == b else "cross", k=k,
                          mean_angle=A.mean(), mean_cos2=(np.cos(np.radians(A)) ** 2).mean(),
                          **{f"angle{i + 1}": A[:, i].mean() for i in range(k)})
        if null:
            N = np.stack([n[0][key] for n in null])
            ncos = (np.cos(np.radians(N)) ** 2).mean(1)
            row |= dict(null_mean_angle=N.mean(), null_mean_cos2=ncos.mean(), null_cos2_q95=np.quantile(ncos, 0.95),
                        p=decode.p_value(row["mean_cos2"], ncos))
        ang_rows.append(row)
    def ratio(vals):  # signal VE: ratio of the summed (num, den) over splits, stabler than a mean of ratios
        v = np.asarray(vals, dtype=float)
        return v[:, 0].sum() / v[:, 1].sum()

    for key in obs[0][1]:
        a, b, pca_on, k = key
        if pca_on == "signal":  # (num, den) pairs; signal denominators <= 0 mean no measurable word signal
            v, vw = [o[1][key] for o in obs], [o[1][(b, b, pca_on, k)] for o in obs]
            den = np.mean([x[1] for x in v])
            ve_ = ratio(v) if den > 0 else np.nan
            within = ratio(vw) if den > 0 else np.nan
            row = base | dict(source=a, target=b, kind="within" if a == b else "cross", pca_on=pca_on, k=k,
                              ve=ve_, ve_sd=np.nan, ve_within_target=within, alignment=ve_ / within,
                              target_signal_ss=den)
            if null:
                # the shuffle only touches the source's half 1, so the target's signal SS (den) is unaffected:
                # each null split's numerator is scaled by the observed mean den (per-split null: conservative)
                nv = np.array([n[1][key][0] for n in null]) / den
                row |= dict(null_ve=np.mean(nv), null_q95=np.quantile(nv, 0.95), p=decode.p_value(ve_, nv))
        else:
            v = np.array([o[1][key] for o in obs])
            within = np.array([o[1][(b, b, pca_on, k)] for o in obs])
            row = base | dict(source=a, target=b, kind="within" if a == b else "cross", pca_on=pca_on, k=k,
                              ve=v.mean(), ve_sd=v.std(), ve_within_target=within.mean(),
                              alignment=v.mean() / within.mean())
            if null and pca_on == "avg":
                nv = np.array([n[1][key] for n in null])
                row |= dict(null_ve=nv.mean(), null_q95=np.quantile(nv, 0.95), p=decode.p_value(row["ve"], nv))
        ve_rows.append(row)
    return ang_rows, ve_rows


def gains(conds: dict[str, Cond], cfg: dict, seed: int, base: dict, nulls: bool = True,
          models: tuple = ("global", "electrode", "full"), whiten: bool = False) -> tuple[list, np.ndarray]:
    """Target (inner) word means predicted from each source's word means: global gain, per-electrode gain,
    full ridge map, all leave-one-word-out. Null: the target's word labels permuted (no identity).

    whiten: noise-normalized space, cross-fitted (each half whitened with the other half's whitener;
    statistics are averaged over both halves), so the noise corrections stay unbiased."""
    sc, W = cfg["scaling"], len(cfg["words"])
    rng = np.random.default_rng([seed, 3])
    subs = _subsamples(conds, cfg, seed)
    if whiten:
        subs = [v for s in subs for v in _crossfit_views(s, rng)]
    models = list(models)
    rows, electrode_gains = [], []
    for src in sc["sources"]:
        vals = {m: [] for m in models} | {f"{m}_insample": [] for m in models if m != "full"}
        sig, gg, cross = {c: [] for c in (src, sc["target"])}, [], []
        for s in subs:
            cross.append(scaling.pattern_cross(s[src].X, s[src].y, s[sc["target"]].X, s[sc["target"]].y))
            S = subspace.word_means(s[src].X, s[src].y, W)
            T = subspace.word_means(s[sc["target"]].X, s[sc["target"]].y, W)
            for m in models:
                vals[m].append(scaling.lowo_r2(S, T, m, sc["ridge_alphas"]))
            for m in models:
                if m == "full":
                    continue
                vals[f"{m}_insample"].append(scaling.insample_r2(S, T, m))
            for c in sig:
                sig[c].append(scaling.signal_variance(s[c].X, s[c].y))
            gg.append(scaling.fit_gain(S - S.mean(0), T - T.mean(0), per_feature=False))
            if src == "attempted" and "electrode" in models:
                electrode_gains.append(scaling.fit_gain(S - S.mean(0), T - T.mean(0), per_feature=True))
        tot_t, cor_t = np.mean(sig[sc["target"]], axis=0)
        tot_s, cor_s = np.mean(sig[src], axis=0)
        cr = np.mean(cross)
        ok = cor_s > 0 and cor_t > 0
        row = base | dict(source=src, target=sc["target"], global_gain=np.mean(gg),
                          gain_deattenuated=cr / cor_s if cor_s > 0 else np.nan,
                          signal_ratio=np.sqrt(cor_t / cor_s) if ok else np.nan,
                          signal_corr=cr / np.sqrt(cor_s * cor_t) if ok else np.nan,
                          target_reliable_frac=max(cor_t, 0) / tot_t, source_reliable_frac=max(cor_s, 0) / tot_s)
        row |= {f"r2_{k}": np.mean(v) for k, v in vals.items()}
        if nulls:
            null = {m: [] for m in models}
            for i in range(cfg["controls"]["n_perm"]):
                s = subs[i % len(subs)]
                perm = rng.permutation(W)
                while np.all(perm == np.arange(W)):
                    perm = rng.permutation(W)
                S = subspace.word_means(s[src].X, s[src].y, W)
                T = subspace.word_means(s[sc["target"]].X, s[sc["target"]].y, W)[perm]
                for m in models:
                    null[m].append(scaling.lowo_r2(S, T, m, sc["ridge_alphas"]))
                null.setdefault("corr", []).append(((S - S.mean(0)) * (T - T.mean(0))).sum() / np.sqrt(cor_s * cor_t)
                                                   if ok else np.nan)
            for m in models:
                n = np.array(null[m])
                row |= {f"null_{m}_q95": np.quantile(n, 0.95), f"p_{m}": decode.p_value(row[f"r2_{m}"], n)}
            if ok:
                n = np.array(null["corr"])
                row |= dict(null_signal_corr_q95=np.quantile(n, 0.95), p_signal_corr=decode.p_value(row["signal_corr"], n))
        rows.append(row)
    return rows, np.mean(electrode_gains, axis=0) if electrode_gains else np.array([])


def cross_half(early: dict[str, Cond], late: dict[str, Cond], cfg: dict, seed: int, base: dict) -> list[dict]:
    """Train on all trials of one block half, test on the other half (every condition pair, both directions)."""
    rng = np.random.default_rng([seed, 4])
    n = min(np.bincount(c.y).min() for d in (early, late) for c in d.values())
    reps = cfg["controls"]["n_match_reps"]
    rows = []
    for cen in cfg["decoding"]["centering"]:
        for (h_tr, tr), (h_te, te) in [(("early", early), ("late", late)), (("late", late), ("early", early))]:
            accs, nulls = {}, {}
            for r in range(reps):
                str_ = decode.center(matched_subsample(tr, np.random.default_rng([seed, r, 0]), n), cen)
                ste = decode.center(matched_subsample(te, np.random.default_rng([seed, r, 1]), n), cen)
                with threadpool_limits(1):
                    for a in cfg["conditions"]:
                        m = decode.make_clf("lda", cfg).fit(str_[a].X, str_[a].y)
                        for b in cfg["conditions"]:
                            P = m.predict(ste[b].X)[None]
                            accs.setdefault((a, b), []).append((P == ste[b].y).mean())
                            nulls.setdefault((a, b), []).append(
                                decode.cross_null(P, ste[b], cfg["controls"]["n_perm_cross"], rng))
            for (a, b), v in accs.items():
                null = np.mean(nulls[(a, b)], axis=0)
                rows.append(base | dict(center=cen, train=a, train_half=h_tr, test=b, test_half=h_te,
                                        acc=np.mean(v), n_per_word=n, null_q95=np.quantile(null, 0.95),
                                        p=decode.p_value(np.mean(v), null)))
    return rows

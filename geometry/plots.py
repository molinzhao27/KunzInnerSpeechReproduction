"""Figures for geometry/results/ (PNG). Colors: reference palette slots 1-3 (validated all-pairs)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COND_COLOR = {"attempted": "#2a78d6", "inner": "#eb6834", "perceived": "#1baf7a"}
INK, MUTED, GRID, NULL = "#0b0b0b", "#52514e", "#e4e3df", "#b5b4ae"
SHORT = {"attempted": "A", "inner": "I", "perceived": "P"}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
    "figure.dpi": 150, "savefig.bbox": "tight", "figure.facecolor": "white",
})


def _save(fig, out: Path, name: str, tight: bool = True):
    if tight:
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / f"{name}.png")
    plt.close(fig)


def decoding_by_region(dec: pd.DataFrame, cfg: dict, out: Path, center: str):
    d = dec[(dec.run == "main") & (dec.classifier == "lda") & dec.matched & (dec.center == center)].copy()
    # LDA within-condition accuracy does not depend on centering; its null is computed once (center = none)
    wn = dec[(dec.run == "main") & (dec.classifier == "lda") & dec.matched & (dec.center == "none") & (dec.kind == "within")]
    wn = wn.set_index(["region", "train"]).null_q95
    w_ = d.kind == "within"
    d.loc[w_, "null_q95"] = [wn.get((r, t), np.nan) for r, t in zip(d[w_].region, d[w_].train)]
    regions = list(cfg["regions"])
    conds = cfg["conditions"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    w = 0.26
    for ax, test in zip(axes, conds):
        for i, train in enumerate(conds):
            sub = d[(d.test == test) & (d.train == train)].set_index("region").reindex(regions)
            x = np.arange(len(regions)) + (i - 1) * w
            ax.bar(x, sub.acc, w * 0.9, color=COND_COLOR[train], label=f"train on {train}")
            ax.scatter(x, sub.null_q95, marker="_", s=60, color=INK, linewidths=1.5, zorder=3,
                       label="null 95th pct." if i == 0 else None)
        ax.axhline(cfg["chance"], color=MUTED, ls="--", lw=1)
        ax.set_xticks(np.arange(len(regions)), regions, rotation=45, ha="right")
        ax.set_title(f"test on {test}", color=INK)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("word accuracy (7 words; dashed = chance)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02),
               title="bar matching the panel's test condition = within-condition 10-fold CV; others = cross-condition",
               title_fontsize=8)
    fig.suptitle(f"Within- vs cross-condition decoding, shrinkage LDA, matched trials, centering = {center}", color=INK)
    _save(fig, out, f"fig_decoding_regions_center-{center}")


def subspace_ve(ve: pd.DataFrame, cfg: dict, out: Path, run="main", region="all", pca_on="avg", space="mnn"):
    d = ve[(ve.run == run) & (ve.region == region) & (ve.pca_on == pca_on) & (ve.space == space)]
    pairs = [tuple(p) for p in cfg["pairs"]]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharey=True)
    for ax, (a, b) in zip(axes.T.ravel(), pairs):
        c = d[(d.source == a) & (d.target == b)].sort_values("k")
        wt = d[(d.source == b) & (d.target == b)].sort_values("k")
        ax.plot(c.k, c.ve, color=COND_COLOR[a], marker="o", ms=4, label=f"{a} PCs (cross)")
        ax.plot(wt.k, wt.ve, color=COND_COLOR[b], ls="--", marker="o", ms=4, label=f"{b} PCs (own, split-half)")
        if "null_q95" in c and c.null_q95.notna().any():
            ax.fill_between(c.k.to_numpy(float), 0, c.null_q95.to_numpy(float), color=NULL, alpha=0.5, lw=0, label=f"null 95% ({a} labels shuffled)")
        ax.set_title(f"{b} variance, {a} PCs", color=INK, fontsize=9)
        ax.set_xlabel("k (top PCs)")
        ax.legend(fontsize=7, loc="upper left")
    for ax in axes[:, 0]:
        ax.set_ylabel("fraction of held-out variance")
    what = {"avg": "word-mean PCs, raw held-out variance", "signal": "word-mean PCs, noise-corrected held-out signal variance",
            "trial": "single-trial PCs"}[pca_on]
    fig.suptitle(f"Cross-validated subspace overlap: {what} ({space} space; {run}, {region})", color=INK)
    _save(fig, out, f"fig_subspace_ve_{pca_on}_{space}_{run}_{region}")


def principal_angles(ang: pd.DataFrame, cfg: dict, out: Path, run="main", region="all", space="mnn"):
    k = cfg["subspace"]["k_max_avg"]
    d = ang[(ang.run == run) & (ang.region == region) & (ang.k == k) & (ang.space == space)]
    cols = [f"angle{i + 1}" for i in range(k)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, (a, b) in zip(axes, [("attempted", "inner"), ("perceived", "inner"), ("attempted", "perceived")]):
        r = d[(d.source == a) & (d.target == b)].iloc[0]
        x = np.arange(1, k + 1)
        ax.plot(x, r[cols].astype(float), color=COND_COLOR[a], marker="o", ms=5, label=f"{a} vs {b}")
        for c, ls in ((a, "--"), (b, ":")):
            w = d[(d.source == c) & (d.target == c)].iloc[0]
            ax.plot(x, w[cols].astype(float), color=COND_COLOR[c], ls=ls, marker="o", ms=4, label=f"{c} half 1 vs half 2")
        ax.axhline(r.null_mean_angle, color=NULL, lw=4, alpha=0.8, label=f"null mean ({a} labels shuffled)")
        ax.set_title(f"{a} vs {b}", color=INK)
        ax.set_xlabel("principal angle index")
        ax.set_ylim(0, 92)
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("angle (deg)")
    fig.suptitle(f"Principal angles between {k}-dim word-mean subspaces (split halves; {space} space; {run}, {region})", color=INK)
    _save(fig, out, f"fig_principal_angles_{space}_{run}_{region}")


def alignment_by_region(ve: pd.DataFrame, cfg: dict, out: Path, space="mnn", pca_on="avg"):
    k = cfg["subspace"]["k_report"]
    d = ve[(ve.run == "main") & (ve.pca_on == pca_on) & (ve.k == k) & (ve.kind == "cross") & (ve.space == space)]
    regions = list(cfg["regions"])
    pairs = [("attempted", "inner"), ("inner", "attempted"), ("perceived", "inner"), ("inner", "perceived")]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    w = 0.2
    hatch = {0: None, 1: "///"}
    for i, (a, b) in enumerate(pairs):
        s = d[(d.source == a) & (d.target == b)].set_index("region").reindex(regions)
        x = np.arange(len(regions)) + (i - 1.5) * w
        ax.bar(x, s.alignment, w * 0.9, color=COND_COLOR[a if a != "inner" else b], hatch=hatch[a == "inner"],
               edgecolor="white", lw=0, label=f"{a} PCs -> {b} variance")
    ax.axhline(1, color=MUTED, ls="--", lw=1)
    ax.set_xticks(np.arange(len(regions)), regions)
    ax.set_ylabel(f"alignment index (k = {k})\ncross VE / target's own VE")
    ax.set_title(f"Word-subspace alignment by array / electrode group ({space} space, {pca_on} VE; 1 = as aligned as "
                 "the condition with itself)", color=INK, fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    _save(fig, out, f"fig_alignment_regions_{space}_{pca_on}")


def scaling(sc: pd.DataFrame, eg: pd.DataFrame, cfg: dict, out: Path):
    regions = list(cfg["regions"])
    labels = {"global": "one global gain", "electrode": "gain per feature", "full": "full linear map (ridge)"}
    shades = {"global": 1.0, "electrode": 0.65, "full": 0.35}
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.2))
    for ax, space in zip(axes[0], ["zscore", "mnn"]):
        d = sc[(sc.run == "main") & (sc.source == "attempted") & (sc.space == space)].set_index("region").reindex(regions)
        models = [m for m in labels if f"r2_{m}" in d and d[f"r2_{m}"].notna().any()]
        w = 0.8 / len(models)
        for i, m in enumerate(models):
            x = np.arange(len(regions)) + (i - (len(models) - 1) / 2) * w
            ax.bar(x, d[f"r2_{m}"], w * 0.9, color=COND_COLOR["attempted"], alpha=shades[m], label=labels[m])
            ax.scatter(x, d[f"null_{m}_q95"], marker="_", s=50, color=INK, lw=1.5, zorder=3,
                       label="null 95th pct. (words permuted)" if i == 0 else None)
        ax.scatter(np.arange(len(regions)), d.target_reliable_frac, marker="D", s=18, color=COND_COLOR["inner"],
                   zorder=4, label="noise ceiling (reliable fraction of inner)")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(np.arange(len(regions)), regions, rotation=45, ha="right")
        ax.set_ylim(-1, 1)
        ax.set_ylabel("leave-one-word-out R²")
        ax.set_title(f"inner word means predicted from attempted ({space} space)", color=INK)
        ax.legend(fontsize=6.5, loc="upper right")
    ax = axes[1, 0]
    d = sc[(sc.run == "main") & (sc.space == "mnn")]
    for i, src in enumerate(cfg["scaling"]["sources"]):
        s = d[d.source == src].set_index("region").reindex(regions)
        x = np.arange(len(regions)) + (i - 0.5) * 0.38
        ax.bar(x, s.signal_corr.clip(-1.5, 1.5), 0.34, color=COND_COLOR[src], label=f"{src} vs inner")
        ax.scatter(x, s.null_signal_corr_q95, marker="_", s=50, color=INK, lw=1.5, zorder=3,
                   label="null 95th pct." if i == 0 else None)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axhline(1, color=MUTED, lw=0.8, ls="--")
    ax.set_xticks(np.arange(len(regions)), regions, rotation=45, ha="right")
    ax.set_ylabel("noise-corrected pattern correlation\n(r² = inner signal explained by one gain)")
    ax.set_title("Similarity of word-mean patterns (mnn space; missing = no measurable signal)", color=INK)
    ax.legend(fontsize=7, loc="upper right", ncol=3)
    ax.set_ylim(top=1.3)
    ax = axes[1, 1]
    g = eg[eg.region == "all"]
    glob = sc[(sc.run == "main") & (sc.region == "all") & (sc.source == "attempted") & (sc.space == "zscore")]
    glob = glob.global_gain.iloc[0]
    ax.hist(g.gain.clip(-3, 3), bins=60, color=COND_COLOR["attempted"])
    ax.axvline(glob, color=INK, lw=1.5, label=f"global gain {glob:.2f}")
    ax.axvline(0, color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("in-sample per-feature gain, inner / attempted (zscore space, clipped ±3)")
    ax.set_ylabel("features")
    ax.set_title("per-feature gains, all arrays", color=INK)
    ax.legend(fontsize=7)
    fig.suptitle("Is inner speech a scaled copy of attempted speech? (main run, matched trials)", color=INK)
    _save(fig, out, "fig_scaling")


def robustness(dec: pd.DataFrame, ve: pd.DataFrame, sc: pd.DataFrame, cfg: dict, out: Path):
    runs = list(cfg["runs"])
    k = cfg["subspace"]["k_report"]
    d = dec[(dec.region == "all") & (dec.classifier == "lda") & dec.matched & (dec.center == "condition")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    series = [("attempted", "attempted"), ("inner", "inner"), ("perceived", "perceived"),
              ("attempted", "inner"), ("perceived", "inner")]
    w = 0.16
    for i, (a, b) in enumerate(series):
        s = d[(d.train == a) & (d.test == b)].set_index("run").reindex(runs)
        x = np.arange(len(runs)) + (i - 2) * w
        ax = axes[0]
        ax.bar(x, s.acc, w * 0.9, color=COND_COLOR[a], hatch=None if a == b else "///", edgecolor="white", lw=0,
               label=f"{SHORT[a]}->{SHORT[b]}" + (" (within)" if a == b else ""))
    axes[0].axhline(cfg["chance"], color=MUTED, ls="--", lw=1)
    axes[0].set_title("decoding, all arrays (cond.-centered)", color=INK, fontsize=9)
    v = ve[(ve.region == "all") & (ve.pca_on == "signal") & (ve.k == k) & (ve.space == "mnn")]
    for i, (a, b) in enumerate([("attempted", "inner"), ("inner", "attempted"), ("perceived", "inner")]):
        s = v[(v.source == a) & (v.target == b)].set_index("run").reindex(runs)
        axes[1].bar(np.arange(len(runs)) + (i - 1) * 0.26, s.alignment, 0.24,
                    color=COND_COLOR[a if a != "inner" else b], hatch="///" if a == "inner" else None,
                    edgecolor="white", lw=0, label=f"{SHORT[a]} PCs -> {SHORT[b]}")
    axes[1].axhline(1, color=MUTED, ls="--", lw=1)
    axes[1].set_title(f"alignment index (signal VE, k = {k}, mnn)", color=INK, fontsize=9)
    s = sc[(sc.region == "all") & (sc.space == "mnn")]
    for i, src in enumerate(cfg["scaling"]["sources"]):
        r = s[s.source == src].set_index("run").reindex(runs)
        x = np.arange(len(runs)) + (i - 0.5) * 0.38
        axes[2].bar(x, r.signal_corr.clip(-1.5, 1.5), 0.34, color=COND_COLOR[src], label=f"{SHORT[src]} vs I")
        axes[2].scatter(x, r.null_signal_corr_q95, marker="_", s=50, color=INK, lw=1.5, zorder=3,
                        label="null 95th pct." if i == 0 else None)
    axes[2].axhline(0, color=MUTED, lw=0.8)
    axes[2].axhline(1, color=MUTED, lw=0.8, ls="--")
    axes[2].set_title("pattern correlation with inner (mnn)", color=INK, fontsize=9)
    for ax in axes:
        ax.set_xticks(np.arange(len(runs)), runs, rotation=30, ha="right")
        ax.legend(fontsize=7)
    fig.suptitle("Robustness across inner-speech variants, day-1 block halves and day 2 (interleaved)", color=INK)
    _save(fig, out, "fig_robustness_runs")


def cross_half(ch: pd.DataFrame, cfg: dict, out: Path, region="all"):
    conds = cfg["conditions"]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), constrained_layout=True)
    for ax, (h_tr, h_te) in zip(axes, [("early", "late"), ("late", "early")]):
        d = ch[(ch.region == region) & (ch.center == "condition") & (ch.train_half == h_tr)]
        M = d.pivot(index="train", columns="test", values="acc").reindex(index=conds, columns=conds)
        P = d.pivot(index="train", columns="test", values="p").reindex(index=conds, columns=conds)
        im = ax.imshow(M.values, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                v = M.values[i, j]
                ax.text(j, i, f"{v:.2f}" + ("*" if P.values[i, j] < 0.05 else ""), ha="center", va="center",
                        color="white" if v > 0.6 else INK, fontsize=9)
        ax.set_xticks(range(3), conds)
        ax.set_yticks(range(3), conds)
        ax.set_xlabel(f"test ({h_te} blocks)")
        ax.set_ylabel(f"train ({h_tr} blocks)")
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label="accuracy")
    fig.suptitle(f"Train on one half of the session, test on the other ({region}; * p < 0.05 vs permuted test labels)",
                 color=INK, fontsize=9)
    _save(fig, out, f"fig_session_split_cross_half_{region}", tight=False)


def window_sweep(ws: pd.DataFrame, cfg: dict, out: Path):
    d = ws[ws.center == "condition"].copy()
    d["window"] = d.window_start_ms.astype(str) + "+" + d.window_length_ms.astype(str)
    wins = list(dict.fromkeys(d.sort_values(["window_start_ms", "window_length_ms"]).window))
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for a, b in [("attempted", "attempted"), ("inner", "inner"), ("perceived", "perceived"),
                 ("attempted", "inner"), ("perceived", "inner"), ("inner", "attempted")]:
        s = d[(d.train == a) & (d.test == b)].set_index("window").reindex(wins)
        ax.plot(wins, s.acc, color=COND_COLOR[a], ls="-" if a == b else ("--" if b == "inner" else ":"),
                marker="o", ms=4, label=f"{SHORT[a]}->{SHORT[b]}")
    ax.axhline(cfg["chance"], color=MUTED, ls="--", lw=1)
    ax.set_xlabel("window (start + length, ms after each condition's event)")
    ax.set_ylabel("accuracy")
    ax.set_title("Window sensitivity (main run, all arrays, LDA, condition-centered)", color=INK)
    ax.legend(fontsize=7, ncol=2)
    _save(fig, out, "fig_window_sweep")


def make_all(t: dict, cfg: dict, out: Path):
    t = {k: v.infer_objects() for k, v in t.items()}  # older pandas leaves NaN-padded columns as object
    for cen in cfg["decoding"]["centering"]:
        decoding_by_region(t["decoding"], cfg, out, cen)
    for space in cfg["spaces"]:
        for pca_on in ("avg", "signal", "trial"):
            subspace_ve(t["ve"], cfg, out, pca_on=pca_on, space=space)
        principal_angles(t["angles"], cfg, out, space=space)
        alignment_by_region(t["ve"], cfg, out, space=space)
    alignment_by_region(t["ve"], cfg, out, space="mnn", pca_on="signal")
    scaling(t["scaling"], t["electrode_gains"], cfg, out)
    robustness(t["decoding"], t["ve"], t["scaling"], cfg, out)
    cross_half(t["cross_half"], cfg, out)
    window_sweep(t["window_sweep"], cfg, out)

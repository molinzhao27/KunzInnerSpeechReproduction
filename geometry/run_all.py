#!/usr/bin/env python
"""Regenerate every geometry result (CSV + PNG) in geometry/results/.

    python geometry/run_all.py            # full run (config: geometry/config.yaml)
    python geometry/run_all.py --quick    # few permutations/repeats, writes geometry/results/quick/
    python geometry/run_all.py --plots-only   # redraw the figures from the saved CSVs
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geometry import analyses, data, plots  # noqa: E402


def _seed(cfg: dict, *names: str) -> int:
    return cfg["seed"] + zlib.crc32("/".join(names).encode()) % 100_000


def _base(cfg, run_name, region, t, cols):
    return dict(run=run_name, session=t.session, inner_behavior=cfg["runs"][run_name]["inner"], region=region,
                n_features=len(cols))


def job(cfg: dict, run_name: str, region: str, t: data.Trials) -> dict:
    run, reg = cfg["runs"][run_name], cfg["regions"][region]
    cols = t.columns(reg["arrays"], reg["kinds"])
    conds = data.conditions(t, cfg, run, cols, run.get("blocks"))
    base, seed = _base(cfg, run_name, region, t, cols), _seed(cfg, run_name, region)
    with threadpool_limits(1):
        dec = analyses.decoding(conds, cfg, seed, base)
        ang, ve, sc = [], [], []
        for space in cfg["spaces"]:
            # mnn: one noise-whitening map shared by the three conditions, cross-fitted inside the analyses;
            # whitening mixes electrodes, so there is no per-electrode gain model there
            whiten = space == "mnn"
            models = ("global", "full") if whiten else ("global", "electrode", "full")
            a, v = analyses.subspaces(conds, cfg, seed, base | dict(space=space), whiten=whiten)
            s, g = analyses.gains(conds, cfg, seed, base | dict(space=space), models=models, whiten=whiten)
            ang, ve, sc = ang + a, ve + v, sc + s
            if space == "zscore":
                eg = g
    return dict(decoding=dec, angles=ang, ve=ve, scaling=sc, electrode_gains=(run_name, region, eg, cols))


def half_job(cfg: dict, region: str, t: data.Trials) -> list[dict]:
    run, reg = cfg["runs"]["main"], cfg["regions"][region]
    cols = t.columns(reg["arrays"], reg["kinds"])
    early, late = (data.conditions(t, cfg, run, cols, h) for h in ("early", "late"))
    with threadpool_limits(1):
        return analyses.cross_half(early, late, cfg, _seed(cfg, "half", region), _base(cfg, "main", region, t, cols))


def sweep_job(cfg: dict, window: tuple, t: data.Trials) -> list[dict]:
    c = copy.deepcopy(cfg)
    c["decoding"]["classifiers"] = ["lda"]
    reg = cfg["regions"]["all"]
    cols = t.columns(reg["arrays"], reg["kinds"])
    conds = data.conditions(t, c, cfg["runs"]["main"], cols)
    base = _base(cfg, "main", "all", t, cols) | dict(window_start_ms=window[0], window_length_ms=window[1])
    with threadpool_limits(1):
        return [r for r in analyses.decoding(conds, c, _seed(cfg, "sweep"), base, nulls=False) if r["matched"]]


def data_summary(cfg: dict, trials: dict) -> pd.DataFrame:
    rows = []
    for sess, t in trials.items():
        split = cfg["sessions"][sess].get("split_block")
        for b in np.unique(t.behavior):
            m = t.behavior == b
            counts = np.bincount(t.word[m], minlength=len(cfg["words"]))
            row = dict(session_key=sess, session=t.session, behavior=b, bin_ms=t.bin_ms, n_trials=int(m.sum()),
                       per_word_min=int(counts.min()), per_word_max=int(counts.max()),
                       blocks=" ".join(map(str, np.unique(t.block[m]))))
            if split is not None:
                row |= dict(n_early=int((m & (t.block <= split)).sum()), n_late=int((m & (t.block > split)).sum()))
            rows.append(row)
    return pd.DataFrame(rows)


NAMES = dict(decoding="decoding", angles="subspace_angles", ve="subspace_variance_explained",
             scaling="scaling_gain_fit", cross_half="session_split_cross_half", window_sweep="window_sweep",
             electrode_gains="scaling_electrode_gains")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "geometry" / "config.yaml"))
    ap.add_argument("--quick", action="store_true", help="smoke test: few permutations / repeats")
    ap.add_argument("--n-jobs", type=int, help="override config n_jobs (e.g. the cores a gq CPU slot has)")
    ap.add_argument("--plots-only", action="store_true", help="redraw every figure from the CSVs in the results folder")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = ROOT / cfg["paths"]["results"]
    if args.n_jobs:
        cfg["n_jobs"] = args.n_jobs
    if args.quick:
        cfg["controls"] |= dict(n_perm=10, n_perm_cross=50, n_match_reps=2)
        cfg["subspace"]["n_splits"] = 5
        out = out / "quick"
    out.mkdir(parents=True, exist_ok=True)
    if args.plots_only:
        plots.make_all({k: pd.read_csv(out / f"{v}.csv") for k, v in NAMES.items()}, cfg, out)
        print(f"figures redrawn in {out}")
        return 0
    t0 = time.time()

    win = (cfg["window"]["start_ms"], cfg["window"]["length_ms"])
    sweep = [tuple(w) for w in cfg["window_sweep"]]
    iso = data.load_trials(cfg, "isolated", [win] + sweep)
    trials = {"isolated": iso[win], "interleaved": data.load_trials(cfg, "interleaved", [win])[win]}
    summary = data_summary(cfg, trials)
    summary.to_csv(out / "data_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)

    tasks = [delayed(job)(cfg, r, g, trials[cfg["runs"][r]["session"]]) for r in cfg["runs"] for g in cfg["regions"]]
    tasks += [delayed(half_job)(cfg, g, trials["isolated"]) for g in cfg["regions"]]
    tasks += [delayed(sweep_job)(cfg, w, iso[w]) for w in sweep]
    res = Parallel(n_jobs=cfg["n_jobs"], verbose=5)(tasks)
    n_main = len(cfg["runs"]) * len(cfg["regions"])
    main_res, half_res, sweep_res = res[:n_main], res[n_main:n_main + len(cfg["regions"])], res[n_main + len(cfg["regions"]):]

    tables = {k: pd.DataFrame([row for r in main_res for row in r[k]]) for k in ("decoding", "angles", "ve", "scaling")}
    tables["cross_half"] = pd.DataFrame([row for r in half_res for row in r])
    tables["window_sweep"] = pd.DataFrame([row for r in sweep_res for row in r])
    eg = [r["electrode_gains"] for r in main_res if r["electrode_gains"][0] == "main"]
    tables["electrode_gains"] = pd.DataFrame([
        dict(region=g, array=trials["isolated"].feat_array[c], kind=trials["isolated"].feat_kind[c], column=int(c), gain=v)
        for _, g, gains, cols in eg for c, v in zip(cols, gains)])
    for k, df in tables.items():
        df.to_csv(out / f"{NAMES[k]}.csv", index=False, float_format="%.6g")

    plots.make_all(tables, cfg, out)
    info = dict(seconds=round(time.time() - t0, 1), quick=args.quick, python=platform.python_version(),
                numpy=np.__version__, sklearn=sklearn.__version__, machine=platform.machine(), config=cfg)
    (out / "run_info.json").write_text(json.dumps(info, indent=1, default=str))
    print(f"done in {info['seconds']} s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

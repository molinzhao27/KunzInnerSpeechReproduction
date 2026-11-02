"""Window-mean trial features for the geometry analysis, built on the reproduction's loaders.

Session file -> kunzdecoder.features.preprocess (optional smoothing / block centering, then
z-scoring over all bins of the file = per-session z-scoring) -> one mean per trial over a
fixed window relative to that behavior's event (go onset, or stimulus onset for passive
behaviors whose stimulus plays in the delay epoch). Results are cached in geometry/cache/.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kunzdecoder.features import canonical_arrays, preprocess, window_means  # noqa: E402
from kunzdecoder.io import load_session, parse_cue  # noqa: E402
from kunzdecoder.windows import session_file  # noqa: E402

CACHE_VERSION = 1


@dataclass
class Trials:
    X: np.ndarray  # (n, F) window means, z-scored per session
    word: np.ndarray  # (n,) 0-based index into cfg["words"]
    behavior: np.ndarray  # (n,) behavior tag
    block: np.ndarray  # (n,) recording block
    feat_array: np.ndarray  # (F,) array name per feature
    feat_kind: np.ndarray  # (F,) "tx" | "sbp"
    session: str  # "T15 2024.04.07"
    bin_ms: int

    def columns(self, arrays: list[str], kinds: list[str]) -> np.ndarray:
        return np.flatnonzero(np.isin(self.feat_array, arrays) & np.isin(self.feat_kind, kinds))


def session_path(cfg: dict, session: str) -> Path:
    sc = cfg["sessions"][session]
    return session_file(ROOT / cfg["paths"]["raw"] / sc["task"], cfg["participant"], sc["file_tag"])


def _cache_file(cfg: dict, session: str, window: tuple[int, int]) -> tuple[Path, dict]:
    path = session_path(cfg, session)
    params = {"v": CACHE_VERSION, "src": path.name, "size": path.stat().st_size,
              "sigma_ms": cfg["preprocess"]["smoothing_sigma_ms"],
              "block_center": cfg["preprocess"]["block_center"],
              "align": cfg["sessions"][session]["align"], "window": list(window), "words": cfg["words"]}
    key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]
    return ROOT / cfg["paths"]["cache"] / f"{session}_{window[0]}_{window[1]}_{key}.npz", params


def load_trials(cfg: dict, session: str, windows: list[tuple[int, int]]) -> dict[tuple[int, int], Trials]:
    """Trials for each (start_ms, length_ms) window; the session file is read only if a window isn't cached."""
    windows = [tuple(int(v) for v in w) for w in windows]
    out, missing = {}, []
    for w in windows:
        f, _ = _cache_file(cfg, session, w)
        if f.exists():
            z = np.load(f, allow_pickle=False)
            out[w] = Trials(X=z["X"], word=z["word"], behavior=z["behavior"].astype(str), block=z["block"],
                            feat_array=z["feat_array"].astype(str), feat_kind=z["feat_kind"].astype(str),
                            session=str(z["session"]), bin_ms=int(z["bin_ms"]))
        else:
            missing.append(w)
    if not missing:
        return out

    s = canonical_arrays(load_session(session_path(cfg, session)))
    pc = cfg["preprocess"]
    x = preprocess(s, pc["smoothing_sigma_ms"], center_blocks=pc["block_center"], zscore=True)
    parsed = [parse_cue(s.participant, s.cue_names[c - 1]) for c in s.cues]
    words = {w: i for i, w in enumerate(cfg["words"])}
    keep = np.array([w in words for w, _ in parsed])
    word = np.array([words.get(w, -1) for w, _ in parsed])[keep]
    behavior = np.array([b for _, b in parsed])[keep]
    align = cfg["sessions"][session]["align"]
    delay_aligned = np.array([align.get(b, align["default"]) == "delay" for b in behavior])
    anchors = np.where(delay_aligned, s.delay[keep, 0], s.go[keep, 0])
    block = s.trial_block()[keep]

    n_ch = s.tx.shape[1]
    feat_array = np.empty(2 * n_ch, dtype=object)
    for name, ch in s.arrays.items():
        feat_array[ch] = feat_array[ch + n_ch] = name
    feat_kind = np.array(["tx"] * n_ch + ["sbp"] * n_ch)

    for w in missing:
        start, length = (round(v / s.bin_ms) for v in w)
        X = window_means(x, anchors, s.block, np.array([start]), length)[:, 0]
        t = Trials(X=X, word=word, behavior=behavior, block=block, feat_array=feat_array.astype(str),
                   feat_kind=feat_kind, session=f"{s.participant} {s.date}", bin_ms=s.bin_ms)
        f, params = _cache_file(cfg, session, w)
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez(f, X=t.X, word=t.word, behavior=t.behavior, block=t.block, feat_array=t.feat_array,
                 feat_kind=t.feat_kind, session=t.session, bin_ms=t.bin_ms, params=json.dumps(params))
        out[w] = t
    return out


@dataclass
class Cond:
    """One condition's trials restricted to a region: X (n, F), y (n,), block (n,)."""

    X: np.ndarray
    y: np.ndarray
    block: np.ndarray

    def take(self, idx: np.ndarray) -> "Cond":
        return Cond(self.X[idx], self.y[idx], self.block[idx])


def conditions(t: Trials, cfg: dict, run: dict, cols: np.ndarray, blocks: str | None = None) -> dict[str, Cond]:
    """{attempted, inner, perceived} -> Cond for one run (condition -> behavior map) and region columns."""
    split = cfg["sessions"][run["session"]].get("split_block")
    out = {}
    for c in cfg["conditions"]:
        m = t.behavior == run[c]
        if blocks == "early":
            m &= t.block <= split
        elif blocks == "late":
            m &= t.block > split
        idx = np.flatnonzero(m)
        out[c] = Cond(t.X[np.ix_(idx, cols)].astype(np.float64), t.word[idx], t.block[idx])
    return out


def matched_subsample(conds: dict[str, Cond], rng: np.random.Generator, n: int | None = None) -> dict[str, Cond]:
    """Subsample every condition to n trials per word (default: the smallest count of any word/condition)."""
    if n is None:
        n = min(np.bincount(c.y).min() for c in conds.values())
    out = {}
    for k, c in conds.items():
        idx = np.concatenate([rng.choice(np.flatnonzero(c.y == w), n, replace=False) for w in np.unique(c.y)])
        out[k] = c.take(np.sort(idx))
    return out


def permute_within(y: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle labels within each group (recording block); every block holds every word."""
    out = y.copy()
    for g in np.unique(groups):
        i = np.flatnonzero(groups == g)
        out[i] = y[rng.permutation(i)]
    return out

# Geometry of inner vs attempted speech (T15)

How does the neural code for inner speech relate to attempted speech, and to perceived speech, in T15's
motor-cortex arrays? This folder runs the full analysis on CPU (numpy, scipy, scikit-learn). It reuses
the reproduction's loaders (`src/kunzdecoder/io.py`, `features.py`, `windows.py`) and changes none of
them. Key numbers and caveats: [`results.md`](results.md).

```bash
python geometry/run_all.py              # everything -> geometry/results/ (CSV + PNG)
python geometry/run_all.py --quick      # smoke test (few permutations) -> geometry/results/quick/
python -m pytest geometry/tests         # principal angles, gain fits, noise corrections, whitening
```

All settings are in [`config.yaml`](config.yaml). The full run takes about 20 min on 12 cores
(seconds and versions are in `results/run_info.json`). It was run through gq on vast-3's CPU queue:
`gq submit --on vast3 --project bit --cpu --cwd /workspace/KunzInnerSpeechReproduction -- /workspace/envs/bit/bin/python geometry/run_all.py --n-jobs 12`.
That checkout holds only `src/kunzdecoder`, `geometry/` and the two T15 files below.

## What the data offers for T15

The data comes from the public Dryad release; see [`docs/DATA_NOTES.md`](../docs/DATA_NOTES.md) for the
file format. Nothing was re-downloaded. T15 has **two recording days** with these word tasks:

| Session | File | Behaviors | Layout |
|---|---|---|---|
| **2024.04.07** (isolated) | `isolatedVerbalBehaviors/t15.2024.04.07_allBehaviors_raw.mat` (union of the 7 per-behavior files; 100 min, blocks 3–19) | all 7 | one behavior per block |
| **2024.06.14** (interleaved, 2 months later) | `interleavedVerbalBehaviors/t15.2024.06.14_interleavedVerbalBehaviors_raw.mat` (53 min, blocks 3, 4, 5, 7, 8) | attempted, imagined listening, listening | three behaviors share every block |

**Conditions used here.** Each condition maps to one of the session's behaviors:

| Condition | Behavior tag | Notes |
|---|---|---|
| attempted | `attempted` (attempted vocalized) | `mouthed` (attempted mimed) also exists but is not used |
| inner | `imagined1stMotor` (motoric inner speech), primary | Alternatives run as robustness checks: `imagined1stAuditory`, `imaginedListening` (3rd-person auditory). Day 2 has only `imaginedListening` |
| perceived | `listening` (passive listening) | `reading` (silent reading) also exists but is not used |

**Words:** 7 (*ban, choice, day, feel, kite, though, were*), plus a DO_NOTHING cue that is excluded.

**Trials per word** (`results/data_summary.csv`):

| Behavior | Day 1 (isolated) | Day 1 blocks (early ≤ 12 / late ≥ 13) | Day 2 (interleaved) |
|---|---|---|---|
| attempted | 20 (140 trials) | 12 / 19 → 10 / 10 | 22 |
| mouthed | 20 | 9 / 17 | — |
| imagined1stMotor | 30 (210) | 10, 11 / 18 → 20 / 10 | — |
| imagined1stAuditory | 30 | 4, 5 / 14 → 20 / 10 | — |
| imaginedListening | 30 | 7, 8 / 16 → 20 / 10 | 44 |
| listening | 20 | 6 / 15 → 10 / 10 | 22 |
| reading | 20 | 3 / 13 → 10 / 10 | — |

**Arrays and electrodes.** There are four 64-electrode Utah arrays, so 256 electrodes:

| Array | Electrodes | Region |
|---|---|---|
| `i6v` | 0–63 | inferior 6v |
| `M1` | 64–127 | area 4; Fig 1E calls it "T15-4" |
| `55b` | 128–191 | area 55b |
| `s6v` | 192–255 | superior 6v |

**Bins and features.**
- **Bin size:** 10 ms on both days.
- **Features:** two per electrode, threshold crossings (`binnedTX`, LRR-denoised counts) and spike-band
  power (`spikePow`, >250 Hz). That gives 512 features in all, 128 per array.
- **Timing:** the delay is 1.5 s and go is 3 s for active behaviors. In day-1 listening and reading,
  the stimulus plays in the 1.5 s *delay* epoch and "go" is only 0.5 s of inter-trial interval. On
  day 2, the listening audio plays at go.

**Not available for T15:**
- **More word-task days.** The other Dryad archives (sentences, copy task, …) are different tasks.
  So a "session split" can only mean (a) early vs late blocks of day 1, or (b) day 1 vs day 2 with
  `imaginedListening` as the inner condition.
- **Within-day separation of behavior from time.** Day 1 confounds behavior with block: each block
  holds a single behavior.

## Pipeline

| Step | What | Code |
|---|---|---|
| Preprocessing | `[TX, SBP]` per bin → optional smoothing (default none) → **z-score per session** over all bins of the session file. Every condition shares one scale, so gains are comparable across conditions. Then one mean per trial over a fixed window relative to each behavior's event: go onset, or stimulus onset for day-1 listening. The default window, 0–1500 ms, was set a priori; a sweep is reported. Optional block-mean subtraction (`block_center`) is off: on day 1 it would remove each behavior's mean | `data.py` |
| Controls on every analysis | **Matched trial counts:** each condition is subsampled to the smallest per-word count (20 on day 1), 10 subsamples averaged. **Shuffled-label nulls:** word labels are permuted within recording blocks | `analyses.py` |
| Subspaces | PCA on each condition's **word means** (k ≤ 6) and on its **single trials** (k ≤ 20). Everything is **cross-validated** over 50 random split-halves: PCs come from half 1 of condition *a* and are evaluated on half 2 of *b*; *a = b* gives the reliability ceiling. Reported: principal angles, the variance of *b* explained by *a*'s top-k PCs (raw, and **noise-corrected** so it counts only reliable word signal), and the alignment index = cross ÷ own | `subspace.py` |
| Scaling | Inner word means are predicted from attempted (or perceived) word means. Three nested models: one global gain ⊂ one gain per feature ⊂ full linear map (ridge). Each is scored by **leave-one-word-out** R² against a noise ceiling and a word-permutation null. Also reported: the noise-corrected pattern correlation *r*, where *r*² is the share of inner's word signal that a single gain explains | `scaling.py` |
| Decoding | Shrinkage LDA (and L2 logistic regression, C = 0.1) with 10-fold CV within each condition. Each fold's decoder is also tested on all trials of the other conditions, so within- and cross-condition accuracies come from the same decoders. Nulls: within, CV re-run on shuffled labels; cross, the decoders stay fixed and the test labels are permuted. Centering: none, or each condition's own mean subtracted (label-free) | `decode.py` |
| Where it lives | Every analysis is repeated for the 4 arrays, for 6v (i6v + s6v), and for TX-only and SBP-only features | `config.yaml: regions` |
| Session split | (a) Day-1 early vs late blocks: the full analysis within each half, plus train on one half and test on the other. (b) Day 2 (interleaved) vs day 1, both with `imaginedListening` as inner | `config.yaml: runs`, `analyses.cross_half` |

**Two coordinate systems** for the subspace and scaling analyses (`config.yaml: spaces`):
- `zscore`: electrode coordinates, as specified.
- `mnn`: multivariate noise normalization (Walther et al. 2016). The features are whitened by the
  Ledoit-Wolf covariance of the within-word residuals, pooled over the three conditions, so one map is
  shared by all of them.
- **The whitening is cross-fitted.** The map is fit on one half of the trials, and every statistic is
  computed on the other half. With 512 features and fewer residual degrees of freedom, a whitener fit
  on all trials amplifies directions the residuals don't span. There the noise correction sees no noise,
  so it invents signal: on pure noise, naive whitening reports a reliable fraction of 0.24, against 0.005
  cross-fitted (`tests/test_crossfit.py`).

Why both: with 20 trials per word, inner speech's word structure is so small compared with the
high-variance noise directions that it is not measurable in `zscore` space. Its noise-corrected signal
estimate is ≈ 0, although inner speech decodes at about 45%. `mnn` is the space LDA decodes in, and
there the inner signal is reliable. The per-feature gain model exists only in `zscore` space, because
whitening mixes electrodes.

## Outputs (`results/`)

| File | Content |
|---|---|
| `data_summary.csv` | trials per behavior, session and block half |
| `decoding.csv` | within/cross accuracy for every run × region × classifier × centering × matched, with null 95th percentile and p |
| `subspace_angles.csv`, `subspace_variance_explained.csv` | principal angles and VE for every k, pair, space, run and region, with nulls |
| `scaling_gain_fit.csv`, `scaling_electrode_gains.csv` | LOWO R² per model, gains, noise ceilings, pattern correlation, nulls; per-feature gains |
| `session_split_cross_half.csv`, `window_sweep.csv` | the early↔late transfer matrix; accuracy vs window |
| `fig_*.png` | one figure per analysis (listed in `results.md`) |
| `run_info.json` | runtime, library versions, the exact config used |

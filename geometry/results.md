# Results: geometry of inner vs attempted speech (T15)

**Run.** Full run on 2026-11-02 on vast-3's CPU queue (gq job `vast3:368`, 977 s on 12 cores, Python 3.10,
scikit-learn 1.7.2). Details: `results/run_info.json`.

**Main analysis settings** (anything else is noted where it appears):
- **Data:** day 1 (2024.04.07). Inner = motoric inner speech; perceived = passive listening.
- **Features:** all 512 (4 arrays × 64 electrodes × {TX, SBP}).
- **Window:** 0–1500 ms after each condition's event.
- **Trials:** matched to 20 per word, 10 subsamples averaged.
- **Chance:** 14.3%.
- **p-values:** the smallest possible p is 0.005 for 200 permutations and 0.001 for 1000.

**Short answer.**
- **Attempted → inner:** inner speech carries a weak, partly scaled-down copy of attempted speech's word
  code.
  - About half of inner's reliable word structure lines up with the attempted pattern (noise-corrected
    pattern correlation r = 0.74).
  - Its size is about ¼ of attempted's.
  - A decoder trained on attempted speech recovers 73% of inner speech's own above-chance accuracy.
- **Perceived → inner:** the overlap is clearly weaker (r = 0.32; transfer index 0.40).
- **Where:** strongest on i6v, 55b and 6v; absent on s6v, which carries no inner-speech information.
- **Day 2:** the pattern holds on a second recording day where conditions are interleaved within blocks.

## 1. Cross-condition decoding (shrinkage LDA, all arrays)

| Train → test | No centering | Condition-centered | Null 95th pct. |
|---|---|---|---|
| attempted → attempted (within, 10-fold) | 98.3 | 98.3 | 18.6 |
| inner → inner (within) | 43.9 | 43.9 | 17.9 |
| perceived → perceived (within) | 79.4 | 79.4 | 19.3 |
| **attempted → inner** | **31.5** | **36.0** | 15–16 |
| inner → attempted | 64.4 | 82.4 | 15–16 |
| **perceived → inner** | **24.2** | **26.1** | 15–16 |
| inner → perceived | 27.4 | 34.4 | 15–16 |
| attempted → perceived / perceived → attempted | 23.8 / 59.4 | 29.8 / 71.6 | 15–16 |

**Transfer index** = (cross − chance) / (inner within − chance):

| Direction | Condition-centered | No centering |
|---|---|---|
| attempted → inner | **0.73** | 0.58 |
| perceived → inner | **0.40** | 0.33 |

**Significance and robustness.**
- **Significance:** every cross cell is p = 0.001 (fixed decoders, 1000 within-block label permutations),
  except on s6v.
- **Classifier:** logistic regression gives the same picture (attempted → inner 37.0, perceived → inner 30.0,
  condition-centered).
- **Matched trials:** matching barely matters. Attempted → inner is 36.0 matched and 35.3 unmatched.

## 2. Is inner speech a scaled-down copy of attempted speech?

All arrays, word-mean patterns in the noise-normalized (`mnn`) space, cross-fitted:

| Measure | attempted → inner | perceived → inner |
|---|---|---|
| Noise-corrected pattern correlation *r* (null 95th) | **0.74** (0.35), p = 0.005 | 0.32 (0.27), p = 0.045 |
| *r*² = share of inner's reliable word signal one gain explains | **≈ 0.55** | ≈ 0.10 |
| Size of inner's word signal relative to the source | **0.24×** | 0.58× |
| Global gain (de-attenuated) | 0.15 (0.18) | 0.09 (0.19) |
| Leave-one-word-out R², one global gain (noise ceiling 0.198) | **0.088**, p = 0.005 (≈ 44% of the ceiling) | 0.008, p = 0.01 |
| Leave-one-word-out R², full linear map (ridge) | −0.001, p = 0.60 | −0.003 |

**One gain vs per-electrode gains vs a full map** (z-scored electrode space):

| Model | In-sample R² | Held-out-word R² |
|---|---|---|
| One gain for all features | 0.07 | 0.07 |
| One gain per feature | 0.22 | **−0.58** |

- **Per-feature gains don't generalize.** They fit better in-sample but fail on held-out words.
- **The full map is unidentifiable.** It can't be learned from 6 training words; the nested CV always
  picks maximal shrinkage.
- **Bottom line:** a single global gain is the only map that generalizes to unseen words. It explains
  about half of inner's reliable word structure; the rest is not predictable from attempted speech with
  7 words.
- **Per array** (*r*, attempted vs inner): i6v 0.73, M1 0.85, 55b 0.89, 6v 0.74, TX-only 0.79,
  SBP-only 0.77. All p ≤ 0.01.

## 3. Subspaces (`mnn` space, held-out noise-corrected signal variance, k = 3, all arrays)

**Variance explained**, compared with each target's own split-half PCs:

| PCs from → target | Captured | Target's own PCs | p |
|---|---|---|---|
| attempted → inner | **33%** | 16% | 0.085 |
| inner → attempted | 11% | 58% | 0.015 |
| perceived → inner | 6% | 16% | 0.11 |
| attempted → perceived | 16% | 45% | 0.07 |

- **attempted → inner:** p = 0.085 is not significant on all arrays together; it is p = 0.02 on i6v and
  on 6v.
- **Inner's own PCs are a poor reference.** They are estimated from 10 trials per word, which is why
  attempted's PCs beat them.

**Principal angles** (k = 3):

| Subspaces compared | Angles | Notes |
|---|---|---|
| attempted vs inner | 65° / 73° / 83° | null mean 78°; mean cos² 0.10 vs null 95th 0.09, p = 0.02 |
| inner half vs inner half | 71° / 78° / 86° | inner's own reliability bound |
| perceived vs inner | 77° / 83° / 88° | ≈ null |

Attempted–inner angles are as small as inner's own reliability allows, so the angles can't separate
"same subspace" from "partly rotated". Perceived and inner are close to orthogonal.

**In raw z-scored space, inner's word structure isn't measurable.** Its reliable fraction is 0.007. The
largest noise directions dominate, so raw PCA and angles there sit at the null level. That is why the
`mnn` space is the primary one for the geometry.

## 4. Where it lives (condition-centered accuracy %, *r* from section 2)

| Region | A within | I within | P within | A→I | P→I | r(A, I) |
|---|---|---|---|---|---|---|
| all | 98.3 | 43.9 | 79.4 | 36.0 | 26.1 | 0.74 |
| i6v | 98.3 | 39.6 | 81.4 | 35.5 | 25.0 | 0.73 |
| M1 (area 4) | 94.6 | 22.6 | 28.9 | 22.2 | 21.0 | 0.85 |
| 55b | 96.4 | 34.9 | 64.7 | 32.0 | 19.2 | 0.89 |
| s6v | 73.0 | 13.9 (p = 0.34) | 14.6 (n.s.) | 17.9 | 12.2 | (0.96, unreliable) |
| 6v (i6v + s6v) | 97.9 | 36.9 | 72.9 | 32.2 | 24.9 | 0.74 |
| TX only | 97.6 | 41.0 | 75.6 | 34.3 | 23.8 | 0.79 |
| SBP only | 98.5 | 43.6 | 80.6 | 34.2 | 26.2 | 0.77 |

- **Inner speech and transfer:** i6v and 55b carry both. M1 has weak inner speech (22.6%) but a high
  pattern correlation. s6v has no inner or perceived word information.
- **Feature type:** TX and SBP behave the same.

## 5. Controls

**Shuffled-label nulls.** The null 95th percentile is 17–20% for within-condition decoding and 15–16% for
cross-condition decoding. Scaling and subspace nulls are in their CSVs.

**Matched trial counts.** These are the default. Unmatched gives the same results: attempted → inner 35.3
vs 36.0, inner → attempted 87.4 vs 82.4.

**Day-1 block halves** (blocks ≤ 12 vs ≥ 13; 10 trials per word):

| Half | attempted → inner | inner within | r(A, I) |
|---|---|---|---|
| early | 31.6 | 28.7 | 0.68 |
| late | 44.8 | 58.0 | 0.70 |

Training on one half and testing on the other gives attempted early → inner late 50.0 and attempted
late → inner early 22.1. Inner → inner across halves is 42.4 / 32.3. All of these are p = 0.001. The
asymmetry between directions is large, so single-block halves are noisy.

**Second day** (2024.06.14, interleaved blocks; the only inner condition is imagined listening):

| Session | attempted → inner | perceived → inner | inner within | r(A, I) | r(P, I) |
|---|---|---|---|---|---|
| day 2 | 29.5 | 34.1 | 37.3 | 0.62 (null 0.27) | 0.54 (null 0.28) |
| day 1, same mapping | 41.5 | 37.9 | 55.1 | 0.68 | 0.50 |

The attempted–inner overlap replicates on a different day, with conditions sharing blocks, so it is
not a single-day or block artifact. Perceived → inner is relatively stronger with imagined listening as
the inner condition, on both days.

**Inner-speech variants** (day 1):
- **Auditory inner speech:** attempted → inner 26.1, within 39.6. Its pattern correlation (1.15) is not
  interpretable, because its reliable fraction is only 0.03.
- **Imagined listening:** shown in the day-2 table above (day 1, same mapping).

**Window sweep** (`fig_window_sweep.png`):

| Window after the event | attempted → inner | inner within |
|---|---|---|
| 0–500 ms | 38.5 | 34.6 |
| 500–1000 ms | 34.9 | 31.9 |
| 1000–1500 ms | 16.0 | 25.1 |

Inner-speech information and transfer sit in the first second. The 0–1500 ms default is not the best
window: 0–1000 ms gives attempted → inner 41.8.

## Leakage and too-good-to-be-true checks

1. **Found and fixed: inflated signal estimates from in-sample whitening.**
   - **Symptom:** the first full run whitened with all trials. Its noise-corrected "reliable signal"
     *rose* when the trials were halved (0.56 → 0.65–0.68), which is impossible.
   - **Cause:** 512 features against 189–399 residual degrees of freedom. The whitener amplifies
     directions the residuals don't span, and the noise correction sees no noise there.
   - **Size:** on pure noise it reports a reliable fraction of 0.24, against 0.005 cross-fitted
     (`tests/test_crossfit.py`).
   - **Fix:** whitening is now cross-fitted. All numbers above are from the corrected re-run; the first
     run was discarded.
2. **Inner → attempted (82%) beats inner → inner (44%).**
   - **Not leakage:** the trials are disjoint and in different blocks, the decoders are fixed, and the
     null permutes only the test labels.
   - **Explanation:** the inner-trained decoder's axes also carry attempted speech's word signal, which
     is about 4× stronger, so its accuracy rises when applied there. That fits the scaled-copy picture;
     it is not an anomaly.
3. **Attempted within-condition accuracy of 98%** with 512 features is plausible. Phase 3's shrinkage LDA
   reached 92–94% on single T15 arrays.
4. **Ratios with tiny denominators are not real effects.** Treat a metric as unmeasurable when the
   target's reliable fraction is below about 0.05. Examples:
   - noise-corrected *r* > 1: z-scored space *r* = 3.5; auditory inner speech 1.15; s6v 0.96;
   - alignment index > 1 (up to 11 on M1, and 5.6 in `fig_robustness_runs.png`).

   These come from denominators near zero (the target's own signal, or its noisy split-half PCs).
5. **Label-free steps that do use test-condition data.**
   - **Per-session z-scoring** uses every bin, including test trials.
   - **Condition centering** subtracts the test condition's own mean, which is transductive. It raises
     cross-condition accuracy by 2–18 points, so both versions are reported. Note that the "no centering"
     column on day 1 still includes behavior × block offsets.
6. **On day 1, condition is confounded with block,** so cross-condition differences there partly
   reflect drift. Day 2 interleaves the conditions and replicates the main effect.
7. **The window is fixed, not tuned here,** but choosing it drew on the Phase 1 timing from the same data
   (a mild dependence). The sweep shows the conclusions hold for every early window.
8. **The whitener uses word labels** (within-word residuals). It is shared by all conditions and fit on
   held-out halves, and the shuffled-label nulls use the same true-label whitener. This is acceptable,
   but note it.
9. **Possible cue-reading activity.** The text cue is shown in the delay before go. If it stays on
   screen during go, part of the inner and attempted word information could be visual or reading
   activity; the data files don't say.

## Figures

| File | What it shows |
|---|---|
| `fig_decoding_regions_center-{none,condition}.png` | within- and cross-condition decoding per region |
| `fig_scaling.png` | gain vs full-map R², pattern correlation, per-feature gains |
| `fig_subspace_ve_{avg,signal,trial}_{mnn,zscore}_main_all.png` | variance explained vs k |
| `fig_principal_angles_{mnn,zscore}_main_all.png` | principal angles |
| `fig_alignment_regions_*.png` | alignment index per region |
| `fig_robustness_runs.png` | inner-speech variants, block halves, day 2 |
| `fig_session_split_cross_half_all.png` | train on one block half, test on the other |
| `fig_window_sweep.png` | accuracy vs window |

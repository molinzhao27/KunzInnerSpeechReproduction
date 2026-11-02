# KunzInnerSpeechReproduction

Reproduction of Kunz et al. 2025 (Phases 0–3; see README.md) plus a CPU-only geometry analysis in `geometry/`.

## Current status

- **2026-11-02: added `geometry/`**, a T15 analysis of inner vs attempted vs perceived speech:
  - subspaces, gain fits and cross-decoding, with matched trials, shuffled nulls and session splits;
  - one script, `python geometry/run_all.py`, with settings in `geometry/config.yaml`;
  - 28 tests in `geometry/tests`.
  - Reproduction code is unchanged; only the root README and `.gitignore` were edited.
- **Key results** (all arrays, `geometry/results.md`):
  - Attempted → inner decoding is 36% vs 44% within inner (transfer index 0.73); perceived → inner is
    26% (0.40).
  - Inner's word pattern correlates with attempted's at r = 0.74 (noise-corrected) and is about ¼ its
    size.
  - One global gain generalizes to held-out words (R² 0.088 of a 0.198 ceiling). Per-electrode gains and
    a full map do not.
  - Inner speech is absent on s6v. The result replicates on day 2 (interleaved).
- **Decision:** geometry uses a cross-fitted noise-normalized (`mnn`) space. In raw z-scored space the
  inner signal is unmeasurable, and in-sample whitening inflated the signal estimates (a bug found and
  fixed; the first full run was discarded).
- **Where it runs:** on vast-3's CPU queue via gq, in the `kunzdecoder` container at
  `/workspace/KunzInnerSpeechReproduction` (code plus the two T15 files only). After pulling the CSVs,
  run `--plots-only` locally to redraw the figures.
- **Open:**
  - nothing is committed yet;
  - possible follow-ups: a 0–1000 ms window (better transfer), mouthed and reading as extra conditions,
    and more words or days for identifying a full map.

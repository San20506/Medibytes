# Proposal: English-Only Denoising WER Pilot

## Why

`docs/ARCHITECTURE.md:50` names DeepFilterNet4 the Stage-1 audio-enhancement
"WINNER quality" and `:56` turns that into the decision "`DeepFilterNet4` when
GPU, `RNNoise` fallback CPU". `gpu_voice_denoising_strategy.md:189-241` likewise
recommends an "Adaptive Layered Pipeline" as the winner for medical voice
quality. Neither document contains a single measured number, and both are
contradicted by the two studies this repository has actually run.

The only English-valid word-error-rate evidence is thin, and thin in two
different ways:

1. `evidence/pilot-b1/results/accuracy-probe-english.csv` (20 rows) covers 4
   English bases over 8 conditions. Its own `note` column says "4 English bases
   only; 11 of 15 bases are hi/ta and are not measurable here", and
   `evidence/README.md:75-80` records that the probe was a throwaway run
   outside the frozen evaluator.
2. `evidence/matrix-v2/results/matrix-results.csv` (1,440 rows) does cover the
   same 4 English bases across real DEMAND noise, and reports every denoiser as
   net-harmful to word accuracy. Pooling the 0 dB and 5 dB targets and reading
   the shipped file directly: `none` 0.1589, `dpdfnet2-onnx` 0.1685 (+6.1%
   relative), `sherpa-gtcrn-simple` 0.1917 (+20.7%), `deepfilternet3` 2.1667
   (+1264%).

Four base clusters cannot support a confidence interval. That is the whole
blocker: `CHANGELOG.md:129-141` records that a production `pipeline/` Stage-1
`AudioEnhancer` default is explicitly deferred until this evidence gap is closed,
and no amount of re-reading a 4-cluster number closes it. Meanwhile
`evidence/matrix-v2/results/matrix-results.csv` has a second, separate defect —
its `si_sdr_db` column is populated on 23 of 96 English `none` rows and on
**zero** rows for `dpdfnet2-onnx` and `sherpa-gtcrn-simple`, because
`evidence/matrix-v2/matrix_eval.py:281` returns `None` whenever a negative
SI-SDR residual makes the projection bound degenerate.

This change builds the study that can actually decide the question: 40 English
bases, real DEMAND noise, the exact arm set `none` / `dpdfnet2-onnx` /
`sherpa-gtcrn-simple`, two decoder tiers, and a **pre-declared** acceptance rule
stated in `design.md` and implemented in code before any number is produced. The
deliverable is a decision-grade `decision.csv` that either names a Stage-1
default or states that `BackendId.NONE` remains it.

## What Changes

- Add a self-contained `en_pilot/` package at the repository root that builds a
  40-base English-only corpus from the already-downloaded FLEURS `en_us`
  validation parquet (394 rows; the frozen corpus consumed only the first 4
  eligible), mixing it against the existing 18-file DEMAND bank at 0 dB and
  5 dB for 1,440 conditions.
- Reuse the frozen corpus primitives — `select_fleurs_rows`'s privacy and
  duration gate, `canonicalize_clean`'s −3 dBFS normalisation, `active_mask` and
  `mix_at_target_snr`'s masked-SNR mixing, `float_to_pcm16` — read-only, rather
  than extending the frozen harness, whose `corpus_id` and 15-group assumptions
  are pinned by `eval/corpus/build_corpus.py` and both JSON Schemas.
- Run one enhancement process per backend, each under that backend's own worktree
  venv, recording `actual_backend` and `variant_id` read off the returned
  `EnhancementOutput` on every row — the column whose absence in
  `evidence/matrix-v2/results/enhance-all.csv` made that study's dispatch
  unauditable.
- Score every condition with `eval/metrics.py`'s own `levenshtein_counts`,
  `evaluation_tokens`, `frozen_normalize`, `snr_db` and `si_sdr_db`, replacing
  the hand-rolled dynamic-programming WER loop and inline SI-SDR bound in
  `evidence/matrix-v2/matrix_eval.py:262-290`.
- Transcribe with two Whisper tiers — `small` as the declared primary, `base` as
  the continuity tier comparable to `matrix-v2` — recording the resolved device
  and compute type on every row, and scoring a clean-baseline block per base so
  the decoder's WER floor is measured rather than assumed.
- Compute a paired cluster bootstrap over the 40 base groups (10,000 resamples,
  seed 1729) with Holm correction across the two candidates, and apply a
  pre-declared rule: `adopt` iff `ci95_high <= 0.0` and `ci95_low >= -0.02`.
- Emit `evidence/en-pilot/README.md` stating the rule, whether it was met, the
  measured RTF, and — in the same terms `evidence/README.md:79-80` uses — that no
  clinical metric was measured and nothing here supports a clinical or safety
  claim.

## Capabilities

### Modified Capabilities

- `denoise-evaluation`: adds a decision-grade English-only WER study with a
  40-cluster paired bootstrap, a pre-declared acceptance rule, auditable
  per-backend dispatch provenance, and a clean-floor WER measurement. The
  single-backend-per-run, fail-closed, hash-verified guarantees of the original
  capability are unchanged, and the frozen `denoise-pilot-b1` harness is not
  modified by this change.

## Breaking Changes

None.

## Scope Boundaries

- `denoise-pilot-b2-gap-closure` is in flight in parallel. This change is
  English-only and does **not** depend on its pending Task Group 5.0 decoder
  re-pin: it declares `small` as its own primary tier and states the reasoning
  locally, and it touches none of the files that change owns
  (`eval/contracts/report.schema.json`, `eval/contracts/run.schema.json`,
  `eval/metrics.py`, `eval/reporting.py`, `eval/tests/`). All `en_pilot/` code
  imports those modules read-only.
- `deepfilternet3` is excluded from the arm set: it is already refuted on real
  noise (+1264% relative WER, output SNR pinned near −3 dB regardless of input),
  and `denoise-pilot-b2-gap-closure/tasks.md` item 5.6 states it "already failed
  real noise and is not owed a re-run".
- Hindi, Tamil and Hindi-English code-mix buckets are out of scope. The pinned
  `base` decoder tier produces unusable output for them
  (`denoise-pilot-b2-gap-closure/tasks.md` Task Group 4), so a WER number there
  would measure the decoder, not the enhancer.
- This change does **not** select a Stage-1 `AudioEnhancer` production default in
  `pipeline/` code. It produces the evidence a later change acts on.
- This change does not rewrite `docs/ARCHITECTURE.md` or
  `gpu_voice_denoising_strategy.md`. Its finding corrects them; acting on that
  correction belongs to whatever change consumes the result.

## Rollback

Delete `en_pilot/`, `openspec/changes/denoise-pilot-en-only/`, and
`evidence/en-pilot/`. Nothing else in the repository is modified: `eval/`,
`demo/` and `evidence/matrix-v2/` are untouched by this change, and every large
artefact lands under the gitignored data root
(`/home/sandy/.local/share/medibytes-eval/`), which can be deleted independently.

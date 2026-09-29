# Proposal: Denoise Pilot Gap Closure (DeepFilterNet4, Bucket-Aware WER, Real-Noise Coverage)

## Why

`docs/ARCHITECTURE.md` names DeepFilterNet4 the Stage-1 audio-enhancement "WINNER
quality" and RNNoise the "WINNER CPU-constrained" fallback. The frozen
`denoise-pilot-b1` evaluation (`evidence/pilot-b1/`, `evidence/README.md`)
directly contradicts both claims on the evidence actually gathered: RNNoise is
measurably harmful (SI-SDR −7 to −27 dB depending on summary), and
DeepFilterNet**3** — the only DeepFilterNet architecture actually tested, since
`eval/config/methods/deepfilternet3.json` pins `deepfilternet==0.5.6`,
architecture `DeepFilterNet3` — ranked 3rd-best on the frozen synthetic
`pilot-15-v1` corpus but collapsed catastrophically on real DEMAND noise in
`evidence/matrix-v2/` (WER +1264% vs `none`, output SNR pinned near −3 dB
regardless of input). No production default was selected by `denoise-pilot-b1`,
deliberately.

Before a production `pipeline/` implementation can pick a Stage-1
`AudioEnhancer` default, three concrete gaps in `denoise-pilot-b1` need to be
closed rather than carried forward as caveats:

1. **DeepFilterNet4 was never tested.** Only v3 was pinned and evaluated. It is
   not yet confirmed that a distinct, separately-releasable "DeepFilterNet4"
   architecture exists upstream at all — ARCHITECTURE.md's citation has not
   been verified against the actual `Rikorose/DeepFilterNet` release history.
2. **Word-error-rate impact is unmeasured for 11 of 15 corpus bases.**
   `evidence/pilot-b1/test-data/reference-transcripts.csv` already carries
   native-script Hindi, Tamil, and Hindi-English code-mix reference
   transcripts for every base (this was verified directly, not assumed), and
   `eval/metrics.py::evaluation_tokens()` is already tested as Unicode-safe
   across English/Hindi/Tamil/code-mix (`eval/tests/test_text_metrics.py`). The
   gap is not missing ground truth or an English-only tokenizer — it is that
   (a) `eval/metrics.py::frozen_normalize()` hard-calls
   `demo.stt_extract.normalize_text()`, an English/Hinglish-specific
   normalizer, so only the *normalized* WER half is English-bound; (b) the
   official `denoise-pilot-b1` deliverable never ran or surfaced the
   already-script-safe *raw*-token WER path per bucket — the only WER numbers
   published (`results/accuracy-probe-english.csv`) came from a throwaway
   script "run outside the frozen evaluator" and explicitly excluded the
   non-English buckets; and (c) `report.schema.json`'s `aggregate.text` is a
   single pooled WER object with no per-bucket breakdown, even though
   `eval/reporting.py`'s bootstrap comparison already stratifies by `bucket`
   internally (`_bootstrap`/strata code, ~line 1013).
3. **Real-noise coverage exists for only 4 of 13 catalogued backends.**
   `evidence/matrix-v2/` tested `none`, `dpdfnet2-onnx`, `sherpa-gtcrn-simple`,
   and `deepfilternet3` — the three arms with a favorable synthetic-corpus CI,
   plus baseline. `deepfilternet3`'s synthetic ranking (3rd, favorable CI)
   completely inverted on real noise. No other backend showed a favorable
   synthetic CI, so the only clearly missing real-noise candidate is
   DeepFilterNet4, if it exists as a distinct, evaluable arm (see gap 1).

This change closes gaps 1–3 so that a later, separate change can pick a
defensible Stage-1 `AudioEnhancer` production default on evidence gathered by
this project, not on ARCHITECTURE.md's unverified literature-review claims.

## What Changes

- Verify whether DeepFilterNet4 is a real, separately-pinnable upstream
  release distinct from the already-tested DeepFilterNet3 architecture. If it
  is, add it to `eval/config/method-catalog.json` and
  `eval/config/methods/deepfilternet4.json` following the exact schema
  `deepfilternet3.json` already uses, run it through the frozen `pilot-15-v1`
  synthetic corpus via the existing per-method-worktree governance, and run it
  through the `evidence/matrix-v2/` real-DEMAND-noise methodology. If it is
  not a real, distinct release, record that finding as the closure of gap 1
  (no new arm to add) rather than inventing a fabricated pin.
- Add a bucket-aware WER path: compute and surface the raw-token WER
  (`eval/metrics.py::_wer` over `evaluation_tokens()`, already Unicode-safe)
  per `bucket` (`en-US-proxy`, `hi`, `ta`, `hi-en-code-mix`) inside the frozen
  harness itself, for `none` and every arm with a favorable synthetic-corpus
  CI, instead of only in a disconnected, unpinned "accuracy probe" script.
  Leave `frozen_normalize()`-based *normalized* WER explicitly
  `not_applicable` outside `en-US-proxy` — building per-language (Devanagari,
  Tamil) text normalizers is out of scope for this change.
- Extend `run.schema.json`/`report.schema.json` (version bump from `1.0.0`) so
  an individual arm report can carry a per-bucket WER breakdown alongside the
  existing pooled `aggregate.text` object, without weakening
  `additionalProperties: false` strictness.
- Extend `evidence/matrix-v2/`'s real-noise methodology
  (`matrix_eval.py`, `build_noise_matrix.py`) to include DeepFilterNet4, if
  gap 1 confirms it exists as an evaluable arm.
- Re-run `eval/reporting.py`'s gated comparison (`cluster_bootstrap`,
  `holm_adjust`, `compare_runs`) across the resulting arm set and bucket
  breakdown, and produce an `evidence/pilot-b2/` deliverable mirroring
  `evidence/pilot-b1/`'s README/results/timing/xlsx structure, stating plainly
  whether a Stage-1 default can now be justified or still cannot.

## Capabilities

### Modified Capabilities

- `denoise-evaluation`: adds bucket-aware WER scoring (raw-token path,
  schema-versioned per-bucket report field), an optional DeepFilterNet4
  method arm gated on a real upstream-release verification step, and extended
  real-noise matrix coverage. Single-backend-per-run, fail-closed, and
  hash-verified provenance guarantees from the original capability are
  unchanged.

## Breaking Changes

- `eval/contracts/report.schema.json` and `eval/contracts/run.schema.json`
  version bump from `schema_version: "1.0.0"`; any external consumer pinned to
  `1.0.0`'s exact pooled-only `aggregate.text` shape must be updated. No
  compatibility alias is kept, consistent with `denoise-pilot-worktrees`'
  "clean cutover" precedent.
- If DeepFilterNet4 is confirmed real, `eval/config/method-catalog.json` gains
  a 13th row (`denoise-pilot-12-v1`'s `records` currently has exactly 12) and
  `demo/denoise.py`'s `BackendId` enum gains a 15th member (currently exactly
  14: `NONE` and `NOISEREDUCE` as the two non-catalogued controls, plus the
  12 catalogued backends). Any code that assumes those counts must be
  re-verified, not assumed compatible.

## Scope Boundaries

- This change does **not** select a Stage-1 `AudioEnhancer` production
  default. That is a separate, later change that consumes this change's
  `evidence/pilot-b2/` deliverable.
- This change does **not** build per-language text normalizers for Devanagari
  or Tamil script. Normalized WER stays `en-US-proxy`-only; non-English
  buckets report raw-token WER only, explicitly labeled as such.
- This change does **not** create the `pipeline/` production folder or any
  `AudioEnhancer`/`STTEngine`/etc. interface code from `docs/ARCHITECTURE.md`
  section 3. That work is blocked on this change's results per the user's
  explicit decision.
- This change does not unblock `percepnet`, `speecht5`, or `cmsis-dsp` (still
  `blocked_prerequisite`/`not_comparable`/`blocked_target` per
  `eval/config/method-catalog.json`) — none showed a favorable synthetic CI,
  so real-noise coverage is not owed to them by this change's own logic.

## Rollback

Remove this OpenSpec change and revert the catalog/schema/reporting additions
as one unit, exactly as `denoise-pilot-worktrees`' rollback section specifies.
`evidence/pilot-b2/` (gitignored, like `evidence/pilot-b1/`) can be deleted
independently without affecting the frozen `denoise-pilot-b1` baseline or any
`pilot-15-v1` corpus artifact, since this change adds arms and report fields
rather than modifying existing ones in place.

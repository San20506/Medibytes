# Design: Denoise Pilot Gap Closure

## Evidence This Design Is Built On

Verified directly against the repository (not assumed) before writing this
design:

- `evidence/pilot-b1/test-data/reference-transcripts.csv` carries a native-
  script reference transcript for all 15 bases: `sample-001`–`004` English,
  `005`–`008` Devanagari Hindi, `009`–`012` Tamil script, `013`–`015` Hindi-
  English code-mix (Devanagari + Latin in one line). Ground truth is not
  missing for any bucket.
- `eval/tests/test_text_metrics.py::test_unicode_tokenization_handles_english_hindi_tamil_and_code_mix`
  and `::test_code_mix_punctuation_and_whitespace_do_not_create_tokens` already
  cover `eval/metrics.py`'s tokenizer across all four buckets. The raw-token
  WER path (`_wer` over `evaluation_tokens()`) is script-agnostic today.
- `eval/metrics.py::frozen_normalize()` is hard-wired to
  `demo.stt_extract.normalize_text()` — the demo's English/Hinglish
  normalizer (`HINGLISH_MAP`, BP/SpO2 phonetic fixes, English number-word
  conversion). Applying it to Devanagari or Tamil script text is out of its
  design envelope; this is the actual English-only boundary, not the
  tokenizer.
- `eval/reporting.py` already computes bootstrap strata keyed by `bucket`
  (search `strata: dict[str, dict[...]] = {"bucket": {}, ...}` and the
  `("bucket", condition_status["bucket"])` grouping key, ~line 1013–1019).
  Bucket-level statistical infrastructure partially exists; what is missing is
  (a) a real per-bucket WER number to feed it for non-English buckets and (b)
  a schema-legal place to publish that number in an individual arm's
  `report.json`.
- `eval/contracts/report.schema.json`'s `aggregate.text` is one pooled object
  (`additionalProperties: false`, no bucket key) — confirmed by reading the
  schema directly.
- `eval/config/methods/deepfilternet3.json` pins `package: "deepfilternet"`,
  `package_version: "0.5.6"`, `parameters.architecture: "DeepFilterNet3"`. The
  installed `deepfilternet` package version already determines which
  architecture ships; there is no separately-versioned "DeepFilterNet4"
  package known to this design with confidence — see Open Questions.
- `evidence/matrix-v2/results/matrix-results.csv` (per `evidence/README.md`)
  scored exactly 4 backends: `none`, `dpdfnet2-onnx`, `sherpa-gtcrn-simple`,
  `deepfilternet3`. These are exactly the arms with a favorable paired SI-SDR
  delta on the synthetic corpus (`dpdfnet2-onnx` +5.85 dB,
  `sherpa-gtcrn-simple` +5.30 dB, `deepfilternet3` +3.94 dB); every other arm
  was flat-to-harmful, so no other existing arm is owed real-noise testing by
  this design's own "test what looked promising" logic.

## Gap 1: DeepFilterNet4 Verification and Method Arm

**Verification first, implementation second.** Before writing a
`deepfilternet4.json` method config, Task Group 1 requires confirming against
the actual `Rikorose/DeepFilterNet` (or successor) release history whether a
distinct "DeepFilterNet4" model architecture has ever shipped, under what
package/version, with what license and checkpoint provenance. This project's
existing configs (`percepnet.json`'s `blocked_prerequisite`,
`speecht5.json`'s `not_comparable`) already establish the precedent that "the
literature/doc says X exists" is not sufficient evidence to pin X — a config
row must cite a concrete, checkable upstream source and revision, the same
standard `deepfilternet3.json` meets today.

Two outcomes, both acceptable closures of gap 1:

- **DeepFilterNet4 exists and is pinnable.** Add
  `eval/config/methods/deepfilternet4.json` mirroring `deepfilternet3.json`'s
  shape exactly (`schema_version`, `backend: "deepfilternet4"`, `variant_id`,
  `source`, `source_revision`, `license`, `package`, `package_version`,
  `model_path`, `model_sha256`, `parameters.architecture`, `timeout_s`). Add a
  new `BackendId.DEEPFILTERNET4` value to `demo/denoise.py`'s enum, which
  currently has exactly 14 members (`NONE` and `NOISEREDUCE` as the two
  non-catalogued controls, plus the 12 catalogued backends) — adding
  DeepFilterNet4 makes 15, a 13th catalogued backend; confirm this count at
  implementation time rather than assuming it has drifted. Add a
  corresponding `demo/denoise_backends/deepfilternet4.py`
  adapter (or confirm the existing `deepfilternet3.py`-equivalent adapter, if
  any, can be parameterized by architecture rather than duplicated — check
  whether a `demo/denoise_backends/deepfilternet3.py` adapter file actually
  exists yet, since `demo/denoise_backends/` was confirmed earlier in this
  project's history to have real adapters only for `none` and `noisereduce`;
  `deepfilternet3` may itself still be eval-only, not demo-wired). Add a new
  row to `eval/config/method-catalog.json` (row 14, `default_disposition:
  "run_required"`). Create the isolated method worktree from the frozen B0 SHA
  per the original `denoise-pilot-worktrees` governance (design.md "B0
  Worktree Freeze and Sequence"), since B0 itself has not changed and that
  constraint still holds. Run it through the frozen `pilot-15-v1` harness
  (`python -m eval.harness run --method deepfilternet4 ...`) for synthetic-
  corpus comparability, then through `evidence/matrix-v2/`'s
  `matrix_eval.py` for real-noise comparability.
- **DeepFilterNet4 does not exist as a distinct, checkable release.** Record
  this explicitly in `evidence/pilot-b2/README.md` (or equivalent) as the
  closure of gap 1: `docs/ARCHITECTURE.md`'s "DeepFilterNet4" citation could
  not be verified against upstream, and the only real DeepFilterNet
  architecture evaluated to date (v3) already collapsed on real noise. No new
  method config is added in this branch of the outcome; `tasks.md` accounts
  for both branches explicitly rather than assuming the first.

## Gap 2: Bucket-Aware WER

**Two-tier WER, matching what is actually script-safe today:**

- **Raw-token WER** (`eval/metrics.py::_wer` over `evaluation_tokens()`):
  already Unicode-safe, already tested across all four buckets. This is
  scored for every bucket, for `none` and every arm carried into gap 3's
  real-noise-tested set (i.e. the arms that matter for a production decision:
  `none`, `dpdfnet2-onnx`, `sherpa-gtcrn-simple`, `deepfilternet3`, and
  `deepfilternet4` if it exists).
- **Normalized WER** (`frozen_normalize()` → `demo.stt_extract.normalize_text`):
  stays `en-US-proxy`-only. For `hi`, `ta`, and `hi-en-code-mix` buckets, the
  normalized-WER fields are populated with the schema's existing
  `not_applicable` shape (`eval/contracts/report.schema.json`'s
  `definitions.notApplicable` pattern, already used for clinical metrics) with
  reason `"language_normalizer_not_implemented"` — a new but structurally
  identical reason string, not a new mechanism.

**Real transcription first.** Before any schema or reporting change, Task
Group 2 requires actually running the frozen `downstream-v1.json` decoder
(`faster-whisper base-int8`) against the existing Hindi/Tamil/code-mix
conditions and inspecting the raw hypothesis output. `demo/stt_extract.py`'s
`transcribe()` already captures Whisper's auto-detected `info.language` per
segment, so multilingual decoding is expected to work in principle — but
whether `base-int8` produces usable Devanagari/Tamil script (vs. mis-detected
Latin transliteration or silent failures) on this specific corpus is an
empirical unknown, not a known code gap. If quality is too poor to produce a
meaningful WER signal (e.g. near-100% WER from language mis-detection rather
than genuine transcription error), that finding is recorded as-is — it would
mean the *downstream decoder*, not the *enhancement stage*, is the blocker for
a Hindi/Tamil production decision, which is itself a useful, honest result.

**Schema change.** Bump `eval/contracts/report.schema.json` and
`eval/contracts/run.schema.json` `schema_version` to `"1.1.0"`. Add
`aggregate.text_by_bucket`: an object keyed by the four bucket strings, each
value shaped like the existing `aggregate.text` object (raw WER fields) plus
the `notApplicable`-shaped normalized-WER sub-object. Keep the existing pooled
`aggregate.text` field unchanged in shape (still English-only in practice,
since it was always computed from whatever conditions had a scored reference)
for backward-legibility of the `1.0.0` pilot-b1 reports, but require every
`1.1.0` report to also populate `text_by_bucket`. `additionalProperties:
false` is preserved throughout — this is an additive, versioned field, not a
relaxation.

**Reporting change.** `eval/reporting.py::build_arm_report` gains the bucket-
stratified WER aggregation, reusing the bootstrap strata grouping that already
exists for the comparison step (~line 1013) rather than adding a second,
divergent bucket-grouping implementation.

## Gap 3: Real-Noise Matrix Extension

Only DeepFilterNet4 (if confirmed real per gap 1) is added to
`evidence/matrix-v2/`'s methodology. `matrix_eval.py` and
`build_noise_matrix.py` are config-driven per backend (confirm the exact
config surface at implementation time — Task Group 3 includes reading both
files in full before assuming a config-only change is sufficient); the
expectation is that adding a fourth-to-fifth backend is a config addition, not
new code, since the same four backends already share one evaluator. If reading
the files reveals backend-specific code branches rather than pure config, that
finding gets recorded and the task re-scoped, not silently worked around.

## Statistics Re-Run

`eval/reporting.py`'s `cluster_bootstrap` (10,000 resamples, seed `1729`) and
`holm_adjust` are re-run across the full arm set (old 13 + any new
DeepFilterNet4 arm) and the new bucket-stratified WER numbers, producing an
updated `comparison.json`/`comparison.md`. The existing SI-SDR/SNR paired-
delta ranking from `denoise-pilot-b1` is not recomputed from scratch — those
numbers are unaffected by this change (audio metrics were already
bucket-blind-safe) — only the WER and any new-arm rows are added.

## Deliverable

`evidence/pilot-b2/` mirrors `evidence/pilot-b1/`'s structure exactly
(`README.md`, `hypotheses-tested.csv`, `test-data/`, `results/`, `timing/`,
`xlsx/`), so the two pilots remain directly comparable file-for-file. The new
`README.md` states plainly, for each of the three gaps, what was found and
whether it changes the Stage-1 production-default picture — including the
honest possibility that all three gaps close without producing a clear
winner, matching `denoise-pilot-b1`'s own precedent of not forcing a verdict
the evidence does not support.

## Verification and Concurrency Constraints

- `python -m eval.harness verify-run --run-root <path>` must pass for every
  new run exactly as it does for `denoise-pilot-b1`'s existing runs.
- `python -m eval.contracts` schema validation must accept every `1.1.0`
  report and reject any `1.0.0`-shaped report missing `text_by_bucket`.
- New/changed code is covered by extending `eval/tests/test_reporting.py`,
  `eval/tests/test_text_metrics.py`, and `eval/tests/test_contracts.py` — not
  a new, separate test module — consistent with how `denoise-pilot-worktrees`
  extended the same suite for its own additions.
- This change does not touch the frozen `denoise-pilot-b0` tag, the
  `pilot-15-v1` corpus, or any already-landed arm's existing `run.json` —
  those stay byte-identical; only new runs and new schema/reporting code are
  added.

## Open Questions

- **Does DeepFilterNet4 exist as a distinct, separately-pinnable upstream
  release?** Not resolved by this design — resolving it is the first task in
  `tasks.md`. If it does not exist, `docs/ARCHITECTURE.md`'s citation is
  itself in question and that should be flagged back to whoever maintains that
  doc, separately from this change.
- **Is `faster-whisper base-int8` an adequate Hindi/Tamil/code-mix decoder at
  all**, independent of enhancement backend? If the empirical check in gap 2
  shows the decoder itself performs too poorly to produce a meaningful signal,
  a follow-up change (out of scope here) may need to evaluate a stronger or
  language-specific decoder tier before a Stage-1 decision can lean on
  non-English WER at all.
- **Is raw-token WER alone sufficient signal for a production decision on
  non-English buckets**, or does the missing normalized-WER half (numerals,
  phonetic fixes) matter enough that a later change must build per-language
  normalizers before this data can be trusted for Hindi/Tamil? Left to
  whoever makes the Stage-1 default decision, informed by `evidence/pilot-b2/`
  reporting the gap explicitly rather than silently.

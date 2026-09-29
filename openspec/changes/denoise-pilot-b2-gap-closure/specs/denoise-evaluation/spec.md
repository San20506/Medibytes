# Delta Specification: Denoise Pilot Gap Closure

## MODIFIED Requirements

### Requirement: Paired audio and prediction metrics

The evaluator MUST score each valid enhanced output against its time-aligned reference, MUST report raw and normalized WER, MUST report raw-token WER per language bucket for every scored condition, and MUST distinguish unavailable metrics from measured zero.

#### Scenario: Compute paired audio metrics

- **GIVEN** finite aligned reference and enhanced audio, with the reverberant clean signal used as the reference for `echo`
- **WHEN** audio metrics are computed
- **THEN** SNR and delta SNR use `10*log10(sum(ref^2)/sum((enhanced-ref)^2))`, SI-SDR uses zero-mean target projection, and LSD uses 20 ms Hann windows, 50% overlap, 512 FFT, and epsilon `1e-10`
- **AND** clipping ratio/count, DC offset, maximum and p99.99 adjacent discontinuity, RMS dBFS, exact sample-count drift, and duration drift are recorded
- **AND** wideband PESQ MOS-LQO and STOI record package versions or return `not_applicable` for an incompatible/missing reference

#### Scenario: Compute deterministic text metrics

- **GIVEN** the immutable source reference and a real prediction
- **WHEN** WER is computed
- **THEN** Unicode NFC, casefold, punctuation/whitespace tokenization, and Levenshtein distance produce numerator, denominator, substitutions, deletions, insertions, and raw WER
- **AND** the exact frozen `demo.stt_extract.normalize_text` is then applied to both texts to produce normalized WER and exact match, for the `en-US-proxy` bucket only

#### Scenario: Score WER per language bucket

- **GIVEN** a condition's `bucket` is `en-US-proxy`, `hi`, `ta`, or `hi-en-code-mix`, each with a native-script reference transcript
- **WHEN** an arm's report is built
- **THEN** `aggregate.text_by_bucket` carries one raw-token WER object per bucket present in the run, computed identically to the pooled raw WER regardless of script
- **AND** the normalized-WER sub-object for `hi`, `ta`, and `hi-en-code-mix` is `not_applicable` with reason `language_normalizer_not_implemented`, never a fabricated or English-normalizer-derived number
- **AND** a bucket is omitted from `aggregate.text_by_bucket` only if its decoder output was recorded as not usable for WER scoring (language mis-detection or empty transcription), never silently coerced into a misleading zero or `not_applicable` masking a real failure

#### Scenario: Keep clinical claims inapplicable

- **GIVEN** the selected public non-clinical corpus
- **WHEN** medical WER, medication/dose accuracy, clinical NER F1, field accuracy, and critical-error rate are requested
- **THEN** each is `not_applicable`, never zero
- **AND** empty clinical gold is used only to report extraction false-positive fields plus crash/schema status

#### Scenario: Measure resources and paired uncertainty

- **GIVEN** a runnable backend
- **WHEN** resources are measured
- **THEN** Stage 1 accuracy runs once, followed by one warm-up and three measured repeats in fresh processes
- **AND** cold load/inference, warm inference, p50/p95/p99, real-time factor, child CPU time, peak RSS, and NVML peak VRAM are recorded, using `unavailable` rather than zero when VRAM cannot be measured
- **AND** candidate results use paired `candidate − none` deltas with a 10,000-resample cluster bootstrap over the 15 `group_id` values, seed `1729`, and Holm correction across all row families present in the run, including any row added by this change

## ADDED Requirements

### Requirement: Conditional DeepFilterNet4 arm

A DeepFilterNet4 method arm MUST NOT be added to the catalog or evaluated unless its existence as a distinct, separately-pinnable upstream release is verified against a concrete, checkable source; an unverifiable claim MUST be recorded as a negative finding, never a fabricated pin.

#### Scenario: DeepFilterNet4 confirmed real

- **GIVEN** a concrete upstream source and revision distinct from the already-pinned `deepfilternet3` architecture (`deepfilternet==0.5.6`, architecture `DeepFilterNet3`)
- **WHEN** the verification task completes
- **THEN** a new catalog row and `eval/config/methods/deepfilternet4.json` are added with the same required fields as `deepfilternet3.json` (`source`, `source_revision`, `license`, `package`, `package_version`, `model_path`, `model_sha256` policy, `parameters`, `timeout_s`)
- **AND** it is run through both the frozen `pilot-15-v1` synthetic corpus and the real-DEMAND-noise matrix methodology before any comparison includes it

#### Scenario: DeepFilterNet4 not verifiable

- **GIVEN** no concrete, checkable upstream source distinguishes a "DeepFilterNet4" architecture from the already-tested `DeepFilterNet3`
- **WHEN** the verification task completes
- **THEN** no new catalog row, method config, or worktree is created
- **AND** the negative finding is recorded in the pilot-b2 deliverable, including that `docs/ARCHITECTURE.md`'s "DeepFilterNet4" citation could not be verified against upstream

### Requirement: Bucket-stratified report schema

An individual arm's `report.json` MUST expose raw-token WER separately per language bucket, versioned so pooled-only `1.0.0` reports remain distinguishable from bucket-aware `1.1.0` reports.

#### Scenario: Validate a 1.1.0 report

- **GIVEN** a report declares `schema_version: "1.1.0"`
- **WHEN** it is validated against `eval/contracts/report.schema.json`
- **THEN** validation fails if `aggregate.text_by_bucket` is absent
- **AND** validation fails if any bucket entry adds a property `additionalProperties: false` does not declare

#### Scenario: Reject a stale pooled-only claim

- **GIVEN** a report declares `schema_version: "1.1.0"`
- **WHEN** `aggregate.text` is populated but `aggregate.text_by_bucket` is missing or empty
- **THEN** validation fails, so a `1.1.0` report can never silently regress to `1.0.0`'s pooled-only behavior

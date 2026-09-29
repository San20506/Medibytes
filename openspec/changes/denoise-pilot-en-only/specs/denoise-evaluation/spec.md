# Delta Specification: English-Only Denoising WER Pilot

## ADDED Requirements

### Requirement: English-only WER pilot corpus

The study MUST admit its bases through the frozen corpus privacy gate and
duration window, MUST canonicalise them with the frozen peak normalisation, and
MUST fail closed rather than writing a partial corpus; it MUST NOT modify the
frozen `eval/corpus/build_corpus.py` path, whose `corpus_id` and group count are
pinned by `eval/corpus/manifest.py` and both JSON Schemas.

#### Scenario: Admit forty English bases

- **GIVEN** the FLEURS `en_us` validation parquet with 394 rows
- **WHEN** the corpus builder runs
- **THEN** exactly 40 bases are admitted, ordered by ascending numeric FLEURS `id`, each between 3 s and 12 s after decoding
- **AND** every rejected row is rejected by the frozen privacy reasons (digits, URLs, emails, phone numbers, privacy tokens), not by a rule restated in the study
- **AND** each canonical WAV peaks within −3.01..−2.99 dBFS, measured after PCM16 quantisation

#### Scenario: Keep the frozen English bases byte-identical

- **GIVEN** the frozen `pilot-15-v1` corpus already on disk
- **WHEN** the English pilot corpus is built
- **THEN** `sample-001` through `sample-004` are byte-identical to the frozen corpus's first four bases, verified by SHA-256
- **AND** a digest difference halts the study rather than producing comparable-looking numbers over a different corpus

#### Scenario: Under-populated source split

- **GIVEN** fewer than 40 eligible rows
- **WHEN** the corpus builder runs
- **THEN** it raises and leaves no clean directory, manifest or sources file behind

### Requirement: Namespaced real-noise mixture matrix

Every mixture MUST be built with the frozen masked-SNR primitives, MUST be
derived from a seed namespace that cannot collide with any other study, and MUST
be checked against the licensed noise manifest before use.

#### Scenario: Build the mixture matrix

- **GIVEN** 40 canonical clean bases, 18 licensed DEMAND recordings and SNR targets 0.0 and 5.0 dB
- **WHEN** the matrix builder runs
- **THEN** 1,440 mixtures are written, each with `condition_id` of the form `{sample_id}_{noise_id}_{int(snr)}db`
- **AND** each noise recording's SHA-256 matches `evidence/noise-bank/demand-bank-manifest.jsonl` before it is used, and a mismatch is a hard failure
- **AND** `abs(measured_snr_db - snr_db) <= 0.5` for every row, and no quantised mixture clips
- **AND** the mixture record carries `bucket`, which the predecessor study left permanently null

#### Scenario: Originals are never modified

- **GIVEN** the clean base WAVs are hashed before the sweep
- **WHEN** the sweep completes
- **THEN** every clean source WAV is re-hashed and the run fails if any digest changed

### Requirement: Auditable per-backend dispatch

Each enhancement backend MUST run in its own process under its own repository
and virtual environment, and every recorded row MUST carry the backend and
variant the adapter actually reported.

#### Scenario: One repository per process

- **GIVEN** three backends, each owning a distinct `demo.denoise_backends` adapter module
- **WHEN** the enhancement sweep runs
- **THEN** each backend is launched as a separate process that prepends exactly one repository root to `sys.path` before importing `demo`
- **AND** a process that already has a `demo` package in `sys.modules` refuses to run rather than mixing repositories
- **AND** concurrent threads are used only within a single backend

#### Scenario: Report the adapter's own identity

- **GIVEN** an `EnhancementOutput` returned by `demo.denoise.run_backend`
- **WHEN** the row is written to `enhance-<backend>.csv`
- **THEN** `requested_backend`, `actual_backend` and `variant_id` are read off that output
- **AND** a row whose `actual_backend` differs from its `requested_backend` is a hard failure at the scoring step, not a warning
- **AND** the pinned `model_path` is resolved under the data root, and its absence is a hard failure naming the file, never a silent substitution

### Requirement: Decision-grade scoring on frozen metrics

Scoring MUST use the frozen metric functions, MUST treat an incomplete sweep as
a failure rather than a smaller number, and MUST keep the decoder's clean floor
out of every noisy aggregate.

#### Scenario: Score one condition

- **GIVEN** a matrix condition, an enhancement backend and a decoder tier
- **WHEN** it is scored
- **THEN** WER is the frozen `levenshtein_counts` over `evaluation_tokens()` text, with the reference passed through the same frozen normalisation applied to the hypothesis
- **AND** SNR and SI-SDR are `eval.metrics.snr_db` and `eval.metrics.si_sdr_db`, with the reference reconstructed as the clean base scaled by the mixture's recorded `pre_mix_gain`
- **AND** every noisy row carries an SI-SDR value; a missing one means the run is not finished

#### Scenario: Refuse to score an incomplete sweep

- **GIVEN** a missing `(condition_id, backend, decoder)` triple, an errored enhance row, or an errored transcript
- **WHEN** scoring runs
- **THEN** it prints the offending keys and exits non-zero
- **AND** it never drops the row and reports a smaller corpus as if it were complete

#### Scenario: Measure the clean floor separately

- **GIVEN** one clean baseline per base, transcribed with `backend = "none"` and `condition_id = sample_id`
- **WHEN** results are written
- **THEN** those rows appear in `per-condition.csv` with an empty `snr_target_db`
- **AND** they are excluded from every per-base aggregate and from every decision

### Requirement: Pre-declared paired decision rule

The Stage-1 decision MUST be made by a rule fixed in code and stated in
`design.md` before any run, applied to a paired cluster bootstrap over the base
groups, with a multiple-comparison correction across the candidate family.

#### Scenario: Bootstrap the paired WER delta

- **GIVEN** one mean WER per base per backend at a decoder tier
- **WHEN** the delta against `none` is bootstrapped
- **THEN** the delta is paired within base, the estimator resamples whole base groups, and the interval is the 2.5/97.5 percentile of 10,000 resamples with seed 1729
- **AND** p-values are Holm-adjusted across the candidate backends with the family's declared total count
- **AND** a group count other than the study's own base count is rejected rather than silently accepted

#### Scenario: Apply the stated rule

- **GIVEN** a candidate backend's bootstrap interval at a decoder tier
- **WHEN** the decision is computed
- **THEN** the decision is `adopt` if and only if `ci95_high <= 0.0` and `ci95_low >= -0.02`, and `reject` otherwise
- **AND** the `none` row is `baseline`
- **AND** `mean_delta_snr_db` and `mean_rtf` are reported but never gate
- **AND** no threshold is widened, no family is re-picked and no backend is added in order to obtain an `adopt`

#### Scenario: Disagreement between decoder tiers

- **GIVEN** the `small` and `base` tiers disagree on the direction of a delta
- **WHEN** the deliverable is written
- **THEN** both tiers' decisions are reported and neither is adopted on the strength of the other

# Delta Specification: Denoise Pilot Evaluation

## ADDED Requirements

### Requirement: Deterministic source admission

The system MUST resolve the pilot corpus only from the pinned source revisions and the deterministic selection policy, and MUST record the resolved literal identities used by the build.

#### Scenario: Admit the three FLEURS buckets

- **GIVEN** only the FLEURS validation parquets at revision `168de341b3db6859a9bac1c50a2ef5e3b47647e0` are eligible
- **WHEN** each of `en_us`, `hi_in`, and `ta_in` is selected in ascending numeric `id` order
- **THEN** exactly four records per locale are admitted with 3–12 second successful mono decodes, nonempty transcripts, and no digits, URLs, email addresses, phone numbers, or privacy-sensitive tokens
- **AND** the English records are labeled `en-US-proxy`, never `en-IN`

#### Scenario: Admit the Hindi-English code-mix bucket

- **GIVEN** `Hindi-English_test.tar.gz` from OpenSLR SLR104
- **WHEN** eligible segments are ordered by `recording_id` and then segment start sample
- **THEN** exactly three segments from the first three distinct eligible recordings are admitted
- **AND** each segment is 3–12 seconds, successfully decodable to mono, and passes the same transcript/privacy filter
- **AND** all three are labeled `hi-en-code-mix`

#### Scenario: Reject a changed source

- **GIVEN** a required FLEURS parquet, the MUCS archive, or downloaded model/source artifact does not match its recorded SHA-256
- **WHEN** verification runs
- **THEN** the build or method arm fails with the changed digest visible
- **AND** no unverified mirror, mutable revision, or substitute artifact is admitted

#### Scenario: Preserve literal provenance

- **GIVEN** deterministic source selection has completed
- **WHEN** the corpus manifest is written
- **THEN** it records the source dataset, release/revision or archive URL, digest, split, literal source ID and member path, client/recording ID where published, source row, transcript and transcript digest, SPDX license, and attribution
- **AND** later builds use those literals instead of rerunning selection

### Requirement: Exact paired-corpus admission and rebuild

The system MUST construct and verify exactly 15 unique base groups and 30 canonical WAV files, and MUST reject any corpus that does not match the approved distribution and audio contract.

#### Scenario: Build canonical source audio

- **GIVEN** an admitted clean source
- **WHEN** it is canonicalized
- **THEN** the clean recording is 16 kHz mono signed PCM16 WAV with a canonical 44-byte RIFF header
- **AND** source metadata is dropped and empty, clipped, all-zero, nonfinite, or duration-mismatched output is rejected
- **AND** the clean peak is `-3.00 ± 0.01 dBFS`

#### Scenario: Build the exact noise distribution

- **GIVEN** 15 admitted bases
- **WHEN** paired noisy conditions are assigned in ascending `sample_id` order
- **THEN** each base has exactly one noisy file
- **AND** `white`, `pink_voss16`, `hum50`, `am_babble_proxy`, and `echo` each occur exactly three times
- **AND** the result contains exactly 15 clean files, 15 noisy files, equal clean/noisy sample counts per group, and zero clipping

#### Scenario: Generate deterministic active-mask noise

- **GIVEN** a `sample_id` and one of the five approved noise types
- **WHEN** its random seed is derived from the first eight little-endian bytes of `SHA-256("medibytes-corpus-v1|{sample_id}|{noise_type}")`
- **THEN** `numpy.random.Generator(numpy.random.PCG64(seed))` produces the samples and the manifest records the 16-character seed hex
- **AND** a 20 ms RMS window with 10 ms hop defines activity as RMS above `max_frame_rms * 10^(-40/20)`
- **AND** additive noise is scaled to `5.0 ± 0.1 dB` active-mask SNR
- **AND** `echo` uses direct delay `0 ms`, `RT60=0.4 s`, unit-leading-energy normalization, and SNR against the reverberant clean reference with one global headroom gain if needed to remain below `-1 dBFS`

#### Scenario: Encode the fixed synthetic conditions

- **GIVEN** a noisy condition other than the fixed echo definition
- **WHEN** its samples are generated
- **THEN** `white` uses independent `N(0,1)` samples
- **AND** `pink_voss16` uses the pinned 16-row Voss-McCartney implementation
- **AND** `hum50` sums 50/100/150 Hz components at amplitudes `0.60/0.25/0.10` plus white noise at `0.05`
- **AND** `am_babble_proxy` amplitude-modulates four independently seeded Voss streams at `3.1/4.7/6.3/8.1` Hz

#### Scenario: Complete a corpus record

- **GIVEN** an admitted base and its paired noisy file
- **WHEN** its manifest record is emitted
- **THEN** it includes schema/corpus/group identity, `split_role=smoke`, `tuning_allowed=false`, bucket/language/code-mix/locale fields, all source and license provenance, and for both clean and noisy audio the path, SHA-256, bytes, sample rate, channels, codec, sample count, duration, and peak
- **AND** it includes noise type/seed/target/measured SNR/active RMS/post-mix gain/clipping count, build-tool and decoder hashes, privacy decision, and `reference_status=source_reference`
- **AND** neither source nor prediction is relabeled as clinical gold

#### Scenario: Accept the exact corpus

- **GIVEN** a completed build
- **WHEN** corpus validation runs
- **THEN** it accepts only 15 unique `group_id` values, 30 WAVs, the `4/4/4/3` English/Hindi/Tamil/code-mix split, three cases of every noise type, complete hashes/provenance, equal paired sample counts, `5.0 ± 0.1 dB` measured noisy SNR, zero clipping, and a valid canonical audio contract
- **AND** it returns a nonzero terminal result for every violation without omitting the bad record

#### Scenario: Reproduce byte-identical audio

- **GIVEN** the accepted manifest and the same pinned sources and tools
- **WHEN** the corpus is rebuilt into a separate output root
- **THEN** every canonical clean and noisy file hash matches the original manifest
- **AND** sample counts, durations, SNR values, and generated checksums match byte-for-byte

### Requirement: Strict single-backend identity and no fallback

The system MUST execute exactly one explicitly requested enhancement backend, MUST prove requested and actual identity, and MUST fail rather than mock, substitute, cascade, or silently degrade.

#### Scenario: Expose the exact backend catalog

- **GIVEN** the shared cleaner is imported
- **WHEN** its backend enum is inspected
- **THEN** it contains exactly `none`, `noisereduce`, `rnnoise`, `deepfilternet3`, `dpdfnet2-onnx`, `sherpa-gtcrn-simple`, `percepnet`, `nemo-se-den-sb-16k`, `speecht5`, `speex-denoise`, `speechbrain-metricgan-plus`, `voicefixer-mode0`, `cmsis-dsp`, and `spectral-subtraction`

#### Scenario: Run exactly one adapter

- **GIVEN** one `EnhancementConfig` requesting one catalog backend
- **WHEN** `run_backend` executes
- **THEN** it probes and invokes only `demo.denoise_backends.<backend-with-underscores>`
- **AND** requested and actual backend IDs and the concrete variant match
- **AND** no second backend, cached synthetic result, mock output, or fallback adapter participates

#### Scenario: Reject an invalid enhancement result

- **GIVEN** an adapter reports a backend/variant mismatch, fallback use, multiple backends, empty or nonfinite samples, the wrong channel count, or uncompensated length
- **WHEN** output validation runs
- **THEN** enhancement fails closed
- **AND** the actual adapter error or validation reason is retained

#### Scenario: Use the none control honestly

- **GIVEN** backend `none`
- **WHEN** enhancement runs twice with equivalent standardized input
- **THEN** each run returns the input unchanged with `actual_backend=none`
- **AND** the two runs have identical samples and SHA-256

#### Scenario: Use the noisereduce control honestly

- **GIVEN** backend `noisereduce` and the pinned `noisereduce==3.0.2` dependency or execution is unavailable
- **WHEN** enhancement runs
- **THEN** the run fails with `BACKEND_UNAVAILABLE` and never returns gain-only or other substituted audio
- **AND** `noisereduce` remains a control rather than one of the 12 catalog rows

#### Scenario: Return measured cleaner provenance

- **GIVEN** arbitrary input audio and one explicit backend
- **WHEN** the shared cleaner decodes, resamples to 16 kHz mono, applies the existing RMS target `0.1` and peak ceiling `0.98`, runs diagnostic RMS VAD without trimming, compensates declared delay, and saves PCM16
- **THEN** it returns measured pre/post RMS dBFS, sample counts, duration drift, requested/actual backend, variant, model hash, runtime version, `fallback_used=false`, input/output hashes, and processing time
- **AND** it does not return nominal `snr_before_db`, `snr_after_db`, or a hard-coded `+6.0` gain

### Requirement: Strict real STT and deterministic downstream state

Evaluation MUST use the real deterministic `faster-whisper` path and MUST reject mock, cached, corrected, fallback, or compositionally contaminated results.

#### Scenario: Propagate a real-model failure

- **GIVEN** `transcribe(..., strict=True, use_llm=False)`
- **WHEN** `faster-whisper` raises an exception
- **THEN** the exception propagates and no `MOCK_TEXTS`, cached transcript, or alternate provider is returned

#### Scenario: Keep explicit demo fixtures labeled

- **GIVEN** an explicit product-demo request for `model="mock"`
- **WHEN** it runs outside strict evaluation
- **THEN** the transcript is labeled `is_mock=true` with the other provider/model provenance
- **AND** any attempt to use that result as evaluation evidence is rejected

#### Scenario: Freeze downstream execution

- **GIVEN** an evaluation arm
- **WHEN** transcription, normalization, and entity extraction run
- **THEN** they use `faster-whisper` `base-int8`, CPU, INT8, `beam_size=1`, word timestamps, `temperature=0`, strict mode, regex-only extraction, no LLM, no corrected/human artifacts, and no cache/idempotency reuse
- **AND** the result records actual provider/model/device/compute type/runtime/model hash or snapshot and rejects an unexpected downstream hash

#### Scenario: Report deterministic entity spans

- **GIVEN** a real transcript and the exact `normalized_en` string
- **WHEN** entities are emitted
- **THEN** each entity has deterministic `start_char`, `end_char`, and `span_status` of `exact`, `ambiguous`, or `not_found`
- **AND** ambiguous or absent spans count as incorrect in strict scoring without changing extraction matching semantics

### Requirement: Isolated condition results and retained failures

The evaluator MUST execute every condition in a fresh process, MUST keep artifacts isolated and immutable after finalization, and MUST retain every terminal failure.

#### Scenario: Produce the exact run layout

- **GIVEN** an accepted corpus and a unique run ID
- **WHEN** the run starts under `$MEDIBYTES_EVAL_DATA_ROOT/runs/<run_id>`
- **THEN** it writes `run.json`, `environment.json`, and `arms/<backend>/<variant>/report.json`, `report.md`, and `conditions/<condition_id>/{status.json,enhanced.wav,enhancement.json,transcript.json,entities.json,fields.json,resources.json,metrics.json}`

#### Scenario: Isolate process state

- **GIVEN** multiple noisy conditions for one method
- **WHEN** they execute
- **THEN** each condition starts in a fresh child process with model/adapter state reset or freshly constructed
- **AND** no adapter, model, cache, transcript, entity, field, or output from one condition can initialize another

#### Scenario: Retain every terminal outcome

- **GIVEN** a crash, OOM, timeout, empty output, invalid audio, rejected input, unavailable prerequisite, or other execution failure
- **WHEN** the condition terminates
- **THEN** a terminal `status.json` and the available diagnostic/artifact files are retained under that condition
- **AND** the row/run still accounts for the condition and never omits it or reports nominal success

#### Scenario: Enforce strict records

- **GIVEN** a corpus, run, condition, or report record
- **WHEN** it is validated against its Draft-07 schema
- **THEN** unknown properties are rejected with `additionalProperties: false`
- **AND** missing required provenance, status, hashes, geometry, or metric fields are rejected

#### Scenario: Protect a finalized run

- **GIVEN** a finalized run ID
- **WHEN** anyone attempts to overwrite it
- **THEN** the write is rejected
- **AND** resume is allowed only when the corpus, manifest, config, and downstream hashes are identical; otherwise a new run ID is required

### Requirement: Paired audio and prediction metrics

The evaluator MUST score each valid enhanced output against its time-aligned reference, MUST report raw and normalized WER, and MUST distinguish unavailable metrics from measured zero.

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
- **AND** the exact frozen `demo.stt_extract.normalize_text` is then applied to both texts to produce normalized WER and exact match

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
- **AND** candidate results use paired `candidate − none` deltas with a 10,000-resample cluster bootstrap over the 15 `group_id` values, seed `1729`, and Holm correction across all 12 row families

### Requirement: Twelve terminal row dispositions

Every authoritative CSV row MUST have one terminal individual disposition before comparison, and non-comparable or blocked evidence MUST never be assigned fabricated quality results.

#### Scenario: Execute viable rows individually

- **GIVEN** the row has a reproducible pinned enhancer artifact and its prerequisites pass
- **WHEN** its isolated arm runs
- **THEN** RNNoise, DeepFilterNet3, DPDFNet2 ONNX, Sherpa GTCRN, NeMo SE denoising, Speex DSP, SpeechBrain MetricGAN+, VoiceFixer mode 0, and spectral subtraction each account for all 30 conditions and publish one terminal individual report
- **AND** failures remain visible and no other method/model is substituted

#### Scenario: Disposition PercepNet without a fake adapter

- **GIVEN** PercepNet has only the evidence pin at commit `8ffae4337d23f920176ac2a7426e84610fa338ab` and no authoritative pretrained checkpoint or reproducible training recipe
- **WHEN** its worktree preflight completes
- **THEN** the terminal result is `blocked_prerequisite` with reason `no_authoritative_pretrained_checkpoint_or_reproducible_training_recipe`
- **AND** no executable adapter, trained unreported model, or placeholder is committed

#### Scenario: Disposition SpeechT5 without ASR/TTS substitution

- **GIVEN** the official SpeechT5 source pin exposes no released standalone audio-to-audio enhancement checkpoint or adapter
- **WHEN** its worktree preflight completes
- **THEN** the terminal result is `not_comparable` with reason `no_released_speech_enhancement_checkpoint_or_audio_to_audio_adapter`
- **AND** no `speecht5_asr`, `speecht5_tts`, generic codec, or placeholder enhancement output is used

#### Scenario: Disposition CMSIS-DSP without a host accuracy claim

- **GIVEN** CMSIS-DSP provides kernels but no selected end-to-end enhancement algorithm or specified Cortex-M/Cortex-A target/audio transport
- **WHEN** its worktree preflight completes
- **THEN** the terminal result is `blocked_target` with reason `no_selected_enhancement_algorithm_or_target_hardware`
- **AND** no host wrapper is presented as CMSIS-DSP denoising accuracy

#### Scenario: Retain a NeMo prerequisite or runtime failure

- **GIVEN** the selected non-commercial NeMo checkpoint is pinned
- **WHEN** this workstation cannot satisfy its prerequisites or execute it
- **THEN** the row retains `blocked_prerequisite` or `failed_runtime` with the actual error
- **AND** no other NeMo model, backend, or device label is substituted

### Requirement: Frozen B0 and immutable isolated method execution

The shared baseline/evaluator MUST be frozen before method worktrees are created, and each method worktree MUST remain a sibling of that exact baseline until its terminal report exists.

#### Scenario: Freeze B0

- **GIVEN** corpus tooling/manifest, shared cleaner/backends, strict STT, evaluator, configs, tests, and OpenSpec artifacts pass acceptance
- **WHEN** the integration commit is created
- **THEN** its SHA is recorded as `BASE_SHA`, tagged `denoise-pilot-b0`, and started only from a clean worktree with `python -m pytest eval/tests demo/tests -q` passing
- **AND** no method worktree exists before this point

#### Scenario: Create worktrees sequentially

- **GIVEN** the current row's implementation, targeted test, preflight, full run or terminal disposition, and individual report are complete
- **WHEN** the next CSV row begins
- **THEN** a new sibling worktree is created from exactly `BASE_SHA`, never from another method branch
- **AND** the prior worktree/branch remains available for inspection
- **AND** method branches are not merged during this phase

#### Scenario: Restrict method changes

- **GIVEN** a method worktree exists
- **WHEN** its diff from B0 is inspected
- **THEN** changes are limited to `demo/denoise_backends/<id>.py`, a co-located method config, method-only dependency/native-wrapper files, and one targeted `demo/tests/test_denoise_<id>.py`

#### Scenario: Reject shared-file drift

- **GIVEN** `demo/denoise.py`, `demo/audio_clean.py`, `demo/stt_extract.py`, `demo/coords.py`, templates/assets, corpus tooling/manifest, evaluator, or downstream config differs from B0 in a method worktree
- **WHEN** that run is verified
- **THEN** the run is invalid and the method worktree must be reset to B0 before evaluation
- **AND** an invalid run cannot contribute to ranking or comparison

### Requirement: Final single-backend-only comparison gate

Comparison MUST be enabled only after all 12 row dispositions exist and MUST remain a comparison of isolated single-backend arms, never a compatibility or layering decision.

#### Scenario: Block early or incomplete comparison

- **GIVEN** fewer than 12 terminal row reports exist, or any candidate contains fallback, multiple/layered backends, mock/cached/corrected output, absolute result paths, or a changed downstream hash
- **WHEN** comparison is requested
- **THEN** the command rejects the input and writes no quality ranking

#### Scenario: Compare eligible individual arms

- **GIVEN** all 12 terminal row reports exist
- **WHEN** comparison runs
- **THEN** it reports completion/failure, paired audio deltas, raw/normalized WER, clean-speech safety, latency/resources, license status, and language/noise strata
- **AND** it ranks only valid `ok` single-backend rows
- **AND** blocked, not-comparable, target-blocked, offline/generative, research/GPU, and realtime evidence remain in distinct execution-class sections
- **AND** no row is assigned `production_supported` or selected as a production default

#### Scenario: Exclude compatibility and layering

- **GIVEN** a user requests cascades, fallback routing, sequential composition, or layered processing
- **WHEN** the pilot comparison is considered
- **THEN** those behaviors are rejected as outside this change and require a later separately approved plan

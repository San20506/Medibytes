# Tasks: MediBytes Denoising Pilot and B0 Worktrees

## 1. OpenSpec Artifacts

- [ ] 1.1 Review `proposal.md`, `specs/denoise-evaluation/spec.md`, and `design.md` against the approved plan; resolve any contradiction before changing runtime code.
- [ ] 1.2 Commit the four OpenSpec artifacts as the first change, with no corpus data, dependencies, implementation stubs, or method worktrees.

## 2. Reproducible 15-Base Corpus

- [ ] 2.1 Add `eval/corpus/selection.json` with the immutable policy: FLEURS revision `168de341b3db6859a9bac1c50a2ef5e3b47647e0`; four eligible validation rows each for `en_us`, `hi_in`, and `ta_in`; three distinct-recording MUCS segments; English label `en-US-proxy`; common 3–12 second, mono, transcript, and privacy filters.
- [ ] 2.2 Add the absolute/empty-root guard and default workstation export `MEDIBYTES_EVAL_DATA_ROOT=/home/sandy/.local/share/medibytes-eval`; add `eval/.cache/` to `.gitignore` while keeping canonical audio/models outside git.
- [ ] 2.3 Implement `python -m eval.corpus.build_corpus download` with the exact FLEURS `hf_hub_download` call/revision and `curl --fail --location --retry 3` for the size-`443929204` MUCS archive; compute/persist parquet and archive SHA-256 values and fail on a changed digest.
- [ ] 2.4 Implement deterministic selection in numeric-FLEURS-ID and MUCS recording/start-sample order, retaining the exact privacy decisions and the first four/first three distinct eligible records.
- [ ] 2.5 Implement 16 kHz mono PCM16 canonicalization with a 44-byte RIFF header, pinned `imageio-ffmpeg==0.6.0` or hash-recorded system FFmpeg, metadata removal, validation, and clean peak `-3.00 ± 0.01 dBFS`.
- [ ] 2.6 Implement the five exact noise generators, PCG64 seed derivation, 16-character seed manifest field, 20 ms/10 ms active mask, `5.0 ± 0.1 dB` scaling, echo reference/headroom rule, and zero-clipping validation.
- [ ] 2.7 Implement `build`, `verify`, and separate-output `rebuild`; require 15 groups, 30 WAVs, `4/4/4/3` buckets, `3/3/3/3/3` noises, equal pair lengths, complete provenance, and byte-identical rebuild hashes.
- [ ] 2.8 Write `manifest.jsonl`, `source-checksums.sha256`, and `THIRD_PARTY_NOTICES.md` with every required source, license, clean/noisy hash/geometry, noise, tool, decoder, privacy, and `reference_status=source_reference` field.
- [ ] 2.9 Add corpus contract tests for selection order, privacy rejection, exact distributions, seed/SNR determinism, header/geometry, changed-digest rejection, schema rejection, and rebuild identity.
- [ ] 2.10 Run and retain evidence for:

```bash
export MEDIBYTES_EVAL_DATA_ROOT=/home/sandy/.local/share/medibytes-eval
uv run --python 3.12 python -m eval.corpus.build_corpus download
uv run --python 3.12 python -m eval.corpus.build_corpus build --spec eval/corpus/selection.json --out "$MEDIBYTES_EVAL_DATA_ROOT/corpus/pilot-15-v1"
uv run --python 3.12 python -m eval.corpus.build_corpus verify --manifest eval/corpus/manifest.jsonl --data-root "$MEDIBYTES_EVAL_DATA_ROOT"
uv run --python 3.12 python -m eval.corpus.build_corpus rebuild --out "$MEDIBYTES_EVAL_DATA_ROOT/corpus/pilot-15-v1-rebuild"
```

## 3. Shared Cleaner, Strict STT, and Evaluator

### 3.1 Backend and cleaner contract

- [ ] 3.1.1 Add the exact `BackendId` values and frozen `EnhancementConfig`, `BackendProbe`, and `EnhancementOutput` dataclasses in `demo/denoise.py`.
- [ ] 3.1.2 Implement `run_backend` to import/probe/invoke exactly one `demo.denoise_backends.<backend-with-underscores>` adapter and reject mismatch, fallback, multiple backends, empty/nonfinite/wrong-channel/rate/delay-length output without catching into another backend.
- [ ] 3.1.3 Add `none` as unchanged input and `noisereduce==3.0.2` as a control that raises `BACKEND_UNAVAILABLE` when absent; keep it outside the 12-row catalog.
- [ ] 3.1.4 Cut over `clean_audio` to `(in_path, out_path, backend=BackendId.NONE, config=None)`, use the 16 kHz mono/RMS `0.1`/peak `0.98`/untrimmed-VAD/delay-validation/PCM16 boundary, remove `apply_denoise` and nominal SNR fields, and return all measured hashes/provenance/timing.
- [ ] 3.1.5 Update every `demo/pipeline.py`, `demo/app.py`, test, and future caller to the new signature; replace `--no-denoise` with `--denoiser` plus optional `--denoiser-config`, defaulting to `none`, and display measured provenance.
- [ ] 3.1.6 Split `eval/requirements-eval.txt` from product requirements, pin/remove duplicate NumPy entries, and document method worktrees as the sole owners of method-specific dependencies.

### 3.2 Strict real STT and downstream freeze

- [ ] 3.2.1 Add `strict: bool = False` to `transcribe`; propagate every real-model failure in strict mode and retain explicitly labeled `model="mock"` only for investor fixtures.
- [ ] 3.2.2 Add provider/requested/actual model/device/compute type/beam/runtime/model-hash provenance and deterministic entity `start_char`, `end_char`, `span_status` against exact `normalized_en`.
- [ ] 3.2.3 Add `eval/config/downstream-v1.json` with `base-int8`, CPU INT8, beam `1`, word timestamps, temperature `0`, strict mode, regex extraction, `use_llm=false`, no corrected/human/cache reuse, and its frozen hash.
- [ ] 3.2.4 Reject evaluation evidence containing mock/fallback engines, cached/idempotent output, corrected entities, changed downstream settings/hash, absolute result paths, or multiple/layered backends.

### 3.3 Harness, schemas, metrics, and resources

- [ ] 3.3.1 Add the `validate-corpus`, `preflight`, `run`, `verify-run`, and gated `compare` CLIs under `eval/harness.py`; keep comparison unavailable until all 12 terminal row dispositions exist.
- [ ] 3.3.2 Execute every condition in a fresh child process, reset method state, serialize measured runs, enforce per-condition timeouts, and retain terminal records for crash/OOM/timeout/empty/invalid/rejected outcomes.
- [ ] 3.3.3 Implement the exact `run.json`, `environment.json`, arm report, and per-condition result layout under `$MEDIBYTES_EVAL_DATA_ROOT/runs/<run_id>`; reject absolute result paths and writes to product demo state.
- [ ] 3.3.4 Add strict Draft-07 corpus/run/condition/report schemas with `additionalProperties: false`; make finalized run IDs immutable and permit resume only for identical manifest/config/downstream/corpus hashes.
- [ ] 3.3.5 Implement aligned SNR/delta SNR, SI-SDR, 20 ms/50%-overlap/512-FFT LSD, wideband PESQ/STOI, clipping/DC/discontinuity/RMS/count/duration drift, and explicit `not_applicable` handling.
- [ ] 3.3.6 Implement NFC/casefold/tokenized Levenshtein WER with S/D/I components plus frozen-normalizer WER/exact match; keep all clinical metrics `not_applicable` and report empty-gold false-positive fields plus crash/schema status.
- [ ] 3.3.7 Implement one accuracy pass, one warm-up, three measured fresh-process repeats, cold/warm latency, p50/p95/p99, real-time factor, child CPU/RSS, and NVML VRAM or `unavailable`.
- [ ] 3.3.8 Implement paired candidate-minus-`none` deltas, 10,000 cluster bootstraps over 15 bases with seed `1729`, Holm correction across 12 row families, and labels that cannot produce `production_supported`.
- [ ] 3.3.9 Add `eval/config/method-catalog.json` for CSV rows 2–13 with the exact backend/variant, source/revision, license/notice, model/checkpoint hash policy, execution class, prerequisites, and terminal disposition defined in the design.

### 3.4 Permanent shared verification

- [ ] 3.4.1 Add `demo/tests/test_audio_clean.py` for deterministic `none`, measured provenance, unavailable/failing `noisereduce`, exact identity, and rejection of nonfinite/empty/all-zero/wrong-channel/rate/uncompensated-length output.
- [ ] 3.4.2 Add `demo/tests/test_stt_contract.py` for propagated real failures, labeled explicit mock, evaluator mock rejection, deterministic spans/status, and frozen downstream provenance.
- [ ] 3.4.3 Add the exact `eval/tests/test_contracts.py`, `test_corpus.py`, `test_runner.py`, `test_audio_metrics.py`, `test_text_metrics.py`, `test_clinical_metrics.py`, and `test_reporting.py` coverage for strict schemas, 15/30 distributions, cache/static-output rejection, failure retention, per-condition reset, hand-calculated metrics, multilingual/code-mix WER, 15-cluster bootstrap, and no production verdict.
- [ ] 3.4.4 Run existing numeral tests plus `python -m pytest eval/tests demo/tests -q`; do not use current tone files, `demo scripts/*.mp3`, or cached outputs as evidence.
- [ ] 3.4.5 Validate the corpus, run two distinct-ID `none` evaluations, and require 30/30 terminal `ok`, identical enhanced hashes, `actual_backend=none`, `fallback_used=false`, real `faster-whisper:base-cpu-int8`, and no demo-state writes.

```bash
uv run --python 3.12 python -m eval.harness validate-corpus --manifest eval/corpus/manifest.jsonl --data-root "$MEDIBYTES_EVAL_DATA_ROOT"
```

- [ ] 3.4.6 Inject an unavailable `noisereduce` dependency and a `faster-whisper` exception; require terminal failure artifacts and no nominal, mock, cached, gain-only, or fallback success.

## 4. Freeze B0

- [ ] 4.1 Confirm corpus/tooling/manifest, shared cleaner/backend/STT, evaluator, configs, tests, reports, and OpenSpec artifacts are complete and reviewed.
- [ ] 4.2 From a clean integration worktree with the passing shared suite, commit the accepted state and set `BASE_SHA=$(git rev-parse HEAD)`; tag the exact commit `denoise-pilot-b0`.
- [ ] 4.3 Verify no method worktree or branch was created from a pre-B0 commit; preserve the clean B0 state for every sibling.
- [ ] 4.4 Launch Streamlit from the frozen B0/demo state with one clean and one noisy corpus sample and explicit `none`; observe upload, enhanced-audio playback, a real transcript, entity/field display, and a unique export path without cached sample fixtures.

## 5. Sequential B0 Method Worktrees

For each viable row below, replace `<backend>` and `<run-id>` with that row's exact ID, run its targeted test and `preflight`, execute one real `clean_audio()` fixture, run all 30 conditions, verify the run root, exercise the actual CLI/UI-equivalent pipeline with a unique output root, and persist the individual terminal report before starting the next row:

```bash
uv run --python 3.12 python -m pytest demo/tests/test_denoise_<id>.py -q
uv run --python 3.12 python -m eval.harness preflight --method <backend> --config eval/config/downstream-v1.json
uv run --python 3.12 python -m eval.harness run --method <backend> --run-id <run-id> --data-root "$MEDIBYTES_EVAL_DATA_ROOT" --results-root "$MEDIBYTES_EVAL_DATA_ROOT/runs/<run-id>"
uv run --python 3.12 python -m eval.harness verify-run --run-root "$MEDIBYTES_EVAL_DATA_ROOT/runs/<run-id>"
```

Every viable row must prove exact requested/actual backend and variant, complete source/model/config/runtime provenance, `fallback_used=false`, finite mono 16 kHz output with correct delay/length, deterministic state reset, real downstream output or visible retained failure, and all 30 conditions accounted for.

### 5.1 CSV Row 2 — RNNoise

- [ ] 5.1.1 Run `git worktree add -b eval/rnnoise-v0.2 ../Medibytes-eval-rnnoise "$BASE_SHA"`.
- [ ] 5.1.2 Pin Xiph `v0.2`/model `0b50c45`, hash archive and generated `src/rnn_data.c`, build upstream, and add only the narrow state/frame C wrapper, co-located config, method dependencies, and `test_denoise_rnnoise.py`.
- [ ] 5.1.3 Implement 16↔48 kHz signed-PCM16 processing with one `DenoiseState` per sample, `rnnoise_get_frame_size()`, and reset/flush; complete the common viable-row gate before row 3.

### 5.2 CSV Row 3 — DeepFilterNet3

- [ ] 5.2.1 Run `git worktree add -b eval/deepfilternet-v0.5.6-dfn3 ../Medibytes-eval-deepfilternet "$BASE_SHA"`.
- [ ] 5.2.2 Pin `Rikorose/DeepFilterNet` `v0.5.6` and the actual `DeepFilterNet3.zip`; hash archive/config/checkpoint and implement the CPU PyTorch `init_df()`/`enhance()` adapter with 16↔48 kHz, one state/sample, and explicit delay/padding; never label it DFN4.
- [ ] 5.2.3 Complete the common viable-row gate, recording an explicitly selected GPU separately if used, before row 4.

### 5.3 CSV Row 4 — DPDFNet2 ONNX

- [ ] 5.3.1 Run `git worktree add -b eval/dpdfnet-v0.6.0 ../Medibytes-eval-dpdfnet "$BASE_SHA"`.
- [ ] 5.3.2 Pin `dpdfnet==0.6.0`, commit `de503b39ddfb16023b9d599b05ca872877506047`, snapshot `dd6818d00f50c836fed43a6243ebe49116de5964`, and `dpdfnet2.onnx` SHA-256 `4f0ee28935b4a32abecc717d745416976565834d839601acf43031094b4dc94c`; use the local 16 kHz ONNX CPU path, package-default `attn_limit_db` recorded as such, and reset/flush streaming state.
- [ ] 5.3.3 Complete the common viable-row gate before row 5.

### 5.4 CSV Row 5 — Sherpa-ONNX GTCRN

- [ ] 5.4.1 Run `git worktree add -b eval/sherpa-onnx-1.13.8-gtcrn ../Medibytes-eval-sherpa-onnx "$BASE_SHA"`.
- [ ] 5.4.2 Pin `sherpa-onnx==1.13.8`, `sherpa-onnx-bin`, toolkit commit `362ddf2c077e3c04759c93b7a47212a5c9108ca7`, and the hashed public `gtcrn_simple.onnx`; retain Apache-2.0/GTCRN MIT notices and implement the 16 kHz CPU `OfflineSpeechDenoiser` with provider/thread provenance.
- [ ] 5.4.3 Complete the common viable-row gate before PercepNet; keep this result distinct from DPDFNet.

### 5.5 CSV Row 6 — PercepNet Evidence Only

- [ ] 5.5.1 Run `git worktree add -b eval/percepnet ../Medibytes-eval-percepnet "$BASE_SHA"`.
- [ ] 5.5.2 Verify and retain evidence for unofficial source commit `8ffae4337d23f920176ac2a7426e84610fa338ab`, missing generated `nnet_data.cpp`/weights, and the README's absent pretrained model.
- [ ] 5.5.3 Run preflight and persist exactly one `blocked_prerequisite` disposition with reason `no_authoritative_pretrained_checkpoint_or_reproducible_training_recipe`; do not add an adapter, train an unreported model, or commit a placeholder, and leave the branch at B0.

### 5.6 CSV Row 7 — NVIDIA NeMo SE Denoising

- [ ] 5.6.1 Run `git worktree add -b eval/nemo-2.7.3-se-den ../Medibytes-eval-nemo "$BASE_SHA"`.
- [ ] 5.6.2 Pin `nemo-toolkit==2.7.3`, NVIDIA-NeMo/Speech commit `1d4ee423806d461f9146ae982f9da8eb32495ae7`, snapshot `fe9725b1ae90f7f6b11d9b95c5f9ccec736a211f`, checkpoint SHA-256 `9de7f36d08df4109402e68e41d78e781c8da6a3d7691a674c5231b840151ff09`, and CC-BY-NC-SA-4.0 research-only notice.
- [ ] 5.6.3 If preflight and runtime prerequisites pass, implement 16 kHz deterministic ODE denoising with `num_steps=10`, seed `1729`, actual device and all NeMo/PyTorch/CUDA versions, then complete the common viable-row gate; otherwise persist the actual `blocked_prerequisite` or `failed_runtime` without substituting a model before row 8.

### 5.7 CSV Row 8 — SpeechT5 Evidence Only

- [ ] 5.7.1 Run `git worktree add -b eval/speecht5 ../Medibytes-eval-speecht5 "$BASE_SHA"`.
- [ ] 5.7.2 Verify and retain evidence for official source commit `5d66cf5f37e97f4a1999ad519537decc16d852af` and the absence of a standalone enhancement checkpoint/audio-to-audio API.
- [ ] 5.7.3 Run preflight and persist exactly one `not_comparable` disposition with reason `no_released_speech_enhancement_checkpoint_or_audio_to_audio_adapter`; do not use `speecht5_asr`, `speecht5_tts`, a generic codec, or a placeholder, and leave the branch at B0.

### 5.8 CSV Row 9 — Speex DSP Denoise Only

- [ ] 5.8.1 Run `git worktree add -b eval/speexdsp-1.2.1 ../Medibytes-eval-speexdsp "$BASE_SHA"`.
- [ ] 5.8.2 Pin SpeexDSP `1.2.1`, commit `7a158783df74efe7c2d1c6ee8363c1e695c71226`, BSD-style license, and a narrow wrapper around the documented preprocess init/control/run/destroy calls.
- [ ] 5.8.3 Use 160 samples at 16 kHz, enable denoise, explicitly disable AGC/VAD/dereverb/residual echo/echo state, record upstream default suppression, reset one state/sample, and complete the common viable-row gate before row 10.

### 5.9 CSV Row 10 — SpeechBrain MetricGAN+

- [ ] 5.9.1 Run `git worktree add -b eval/speechbrain-1.1.1-metricgan-plus ../Medibytes-eval-speechbrain "$BASE_SHA"`.
- [ ] 5.9.2 Pin SpeechBrain `1.1.1`, commit `89ead74d163463d30c62329a09cfdb4c54f5abc1`, model `speechbrain/metricgan-plus-voicebank` revision `a196ce26b3bdace6fa1d819017584bdbcce462a8`, `enhance_model.ckpt`, and Apache-2.0 notices.
- [ ] 5.9.3 Implement only 16 kHz `SpectralMaskEnhancement` MetricGAN+ with one fresh model/state per run; do not substitute or pool SepFormer/SEGAN, and complete the common viable-row gate before row 11.

### 5.10 CSV Row 11 — VoiceFixer Mode 0

- [ ] 5.10.1 Run `git worktree add -b eval/voicefixer-0.1.3-mode0 ../Medibytes-eval-voicefixer "$BASE_SHA"`.
- [ ] 5.10.2 Pin `voicefixer==0.1.3`, Zenodo `5600188`, mode `0`, `vf.ckpt` MD5 `1f479dabb9fa5724de5e20f2164cbc7a`, `model.ckpt-1490000_trimed.pt` MD5 `ac6de16561dd6042daf8bb3a806dc818`, and post-download SHA-256 values.
- [ ] 5.10.3 Implement CPU `VoiceFixer.restore(..., cuda=False, mode=0)`, label `generative_restoration`, reconcile documented 44.1 kHz algorithmic padding, resample to 16 kHz, and report waveform/speaker/term preservation separately before completing the common viable-row gate.

### 5.11 CSV Row 12 — CMSIS-DSP Evidence Only

- [ ] 5.11.1 Run `git worktree add -b eval/cmsis-dsp-v1.18.0 ../Medibytes-eval-cmsis-dsp "$BASE_SHA"`.
- [ ] 5.11.2 Verify and retain evidence for release `1.18.0`, commit `4b6f26593e9d3b24a9bc93b3646a3a859238eae2`, Apache-2.0, kernel-only scope, and absence of a selected enhancer/target/audio transport.
- [ ] 5.11.3 Run preflight and persist exactly one `blocked_target` disposition with reason `no_selected_enhancement_algorithm_or_target_hardware`; do not add a host wrapper or accuracy claim, and leave the branch at B0.

### 5.12 CSV Row 13 — Spectral Subtraction

- [ ] 5.12.1 Run `git worktree add -b eval/pyroomacoustics-0.10.1-spectral-subtraction ../Medibytes-eval-spectral-subtraction "$BASE_SHA"`.
- [ ] 5.12.2 Pin `pyroomacoustics==0.10.1`, commit `f02b01dd6609709e2089aefa5d1e59c91d3a0601`, and its license; implement the 16 kHz `apply_spectral_sub`/online path with `nfft=512`, `db_reduc=10`, `lookback=12`, `beta=3`, `alpha=1.2`, fixed per-sample state, and recorded config hash.
- [ ] 5.12.3 Use spectral subtraction only, never pooled iterative Wiener filtering or `noisereduce`, and complete the common viable-row gate plus the twelfth terminal individual report before comparison.

## 6. Final Single-Backend Comparison Gate

- [ ] 6.1 Verify the disposition file contains all 12 CSV rows exactly once: every viable attempted row has an individual evaluated report with any per-condition failures retained; PercepNet is `blocked_prerequisite`; SpeechT5 is `not_comparable`; CMSIS-DSP is `blocked_target`; and a machine-wide NeMo prerequisite/runtime failure uses its actual `blocked_prerequisite` or `failed_runtime` disposition.
- [ ] 6.2 Verify every viable attempted row accounts for 30 conditions and every failure is retained; verify no blocked/not-comparable/target-blocked row has an accuracy score and no method used a fallback, mock, cache, correction, or another model/backend.
- [ ] 6.3 Enable and run comparison only after the 12-report gate, using the accepted `none` run and all valid individual candidate run roots:

```bash
uv run --python 3.12 python -m eval.harness compare --baseline-run <none-run> --candidate-run <method-run>... --output eval/reports/<comparison-id>
```

- [ ] 6.4 Verify per-row completion/failure, paired audio deltas/CIs, raw/normalized WER, clean-speech safety, latency/resources, license status, and language/noise strata; rank only valid `ok` single-backend rows and keep execution classes separate.
- [ ] 6.5 Confirm the report has 12 dispositions, no `production_supported` verdict, no selected production default, and no cascade, fallback routing, sequential composition, or layered processing.
- [ ] 6.6 Remove temporary build caches/scaffolds, preserve manifests/source and checkpoint hashes/notices/terminal reports, update project documentation/changelog with the pilot limitation, and mark the OpenSpec tasks complete only after the actual UI and CLI evidence is observed.

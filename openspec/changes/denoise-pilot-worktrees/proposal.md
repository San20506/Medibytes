# Proposal: MediBytes 15-Sample Denoising Pilot and Isolated Method Worktrees

## Why

MediBytes needs a reproducible, auditable first denoising comparison before product claims, compatibility work, or layered processing. The current shared cleaner/STT boundaries can hide backend or mock failures, and the 12-row method inventory is not yet executable. This change defines a bounded 15-base paired-audio smoke corpus, a fail-closed single-backend contract, a deterministic real-model evaluator, and isolated method worktrees that can be inspected without contaminating the shared baseline.

The corpus is intentionally non-clinical: it can screen paired audio quality, real `faster-whisper` WER, crashes, latency, and resource use, but it cannot support production medical WER, medication/dose, clinical NER, field-accuracy, safety, or `<0.1%` critical-error claims.

## What Changes

- Add a reproducible `pilot-15-v1` corpus with exactly 15 clean bases and 15 paired noisy conditions: English `en_us` (`en-US-proxy`, never `en-IN`) 4, Hindi `hi_in` 4, Tamil `ta_in` 4, and Hindi-English code-mix 3. Selection uses FLEURS revision `168de341b3db6859a9bac1c50a2ef5e3b47647e0` and OpenSLR SLR104, with literal post-download provenance and byte-identical rebuilds.
- Assign exactly three conditions to each of `white`, `pink_voss16`, `hum50`, `am_babble_proxy`, and `echo`, at a measured active-mask target of `5.0 ± 0.1 dB` SNR.
- Replace the nominal cleaner/STT behavior with explicit provenance and failure contracts. The exact backend catalog is `none`, `noisereduce`, `rnnoise`, `deepfilternet3`, `dpdfnet2-onnx`, `sherpa-gtcrn-simple`, `percepnet`, `nemo-se-den-sb-16k`, `speecht5`, `speex-denoise`, `speechbrain-metricgan-plus`, `voicefixer-mode0`, `cmsis-dsp`, and `spectral-subtraction`. Evaluation invokes exactly one requested adapter; it never mocks enhancement, substitutes a backend, or turns a crash into success.
- Freeze the shared baseline/evaluator as B0, then create each sibling worktree from the exact B0 SHA in CSV order. Only the current method adapter, method-local configuration/dependencies, and one targeted test may differ from B0.
- Execute the 12 inventory rows with these terminal dispositions:

| CSV row | Backend ID | Concrete execution | Expected disposition |
|---:|---|---|---|
| 2 | `rnnoise` | Xiph RNNoise `v0.2`, model `0b50c45` | implemented/evaluated or retained runtime failure |
| 3 | `deepfilternet3` | DeepFilterNet `v0.5.6`, DeepFilterNet3 artifact | implemented/evaluated or retained runtime failure |
| 4 | `dpdfnet2-onnx` | DPDFNet `0.6.0`, `dpdfnet2.onnx` | implemented/evaluated or retained runtime failure |
| 5 | `sherpa-gtcrn-simple` | sherpa-onnx `1.13.8`, `gtcrn_simple.onnx` | implemented/evaluated or retained runtime failure |
| 6 | `percepnet` | evidence-only source pin; no executable artifact | `blocked_prerequisite`: `no_authoritative_pretrained_checkpoint_or_reproducible_training_recipe` |
| 7 | `nemo-se-den-sb-16k` | NeMo `2.7.3`, `se_den_sb_16k_small.nemo` | implemented/evaluated when prerequisites pass; otherwise `blocked_prerequisite` or `failed_runtime` |
| 8 | `speecht5` | evidence-only source pin; no enhancement checkpoint | `not_comparable`: `no_released_speech_enhancement_checkpoint_or_audio_to_audio_adapter` |
| 9 | `speex-denoise` | SpeexDSP `1.2.1`, denoise-only | implemented/evaluated or retained runtime failure |
| 10 | `speechbrain-metricgan-plus` | MetricGAN+ only | implemented/evaluated or retained runtime failure |
| 11 | `voicefixer-mode0` | VoiceFixer `0.1.3`, mode `0` | implemented/evaluated as `generative_restoration`, or retained runtime failure |
| 12 | `cmsis-dsp` | evidence-only release pin; no target selected | `blocked_target`: `no_selected_enhancement_algorithm_or_target_hardware` |
| 13 | `spectral-subtraction` | `pyroomacoustics==0.10.1`, spectral subtraction only | implemented/evaluated or retained runtime failure |

- Require one terminal individual report for every row before enabling comparison. Compare only valid `ok` single-backend rows; keep realtime, offline/generative, research/GPU, blocked, and target-only evidence separate. Rank only evaluated rows and do not select a production default.

## Capabilities

### New Capabilities

- `denoise-evaluation`: deterministic corpus admission, provenance, fail-closed enhancement and STT, isolated condition execution, paired metrics, terminal failures, row dispositions, and a gated single-backend comparison.

### Modified Capabilities

- The shared cleaner accepts an explicit backend/configuration and returns measured, auditable provenance instead of nominal SNR metadata.
- The evaluation STT path is strict and real; explicit demo mocks remain possible only outside evaluation and are labeled and rejected by the evaluator.

## Breaking Changes

- `demo.audio_clean.clean_audio` changes from `apply_denoise: bool` to `backend: BackendId = BackendId.NONE` plus optional `EnhancementConfig`; no compatibility alias remains.
- The product CLI drops `--no-denoise`, defaults `--denoiser` to `none`, and accepts an optional backend configuration.
- Evaluation is a clean cutover: callers, configs, schemas, and reports use the new backend and provenance contracts.

## Scope Boundaries

- This phase does not implement cascades, fallback routing, sequential/layered processing, production selection, or claims beyond the smoke pilot.
- The shared baseline and evaluator finish before any method worktree exists. Method branches are not merged during this phase.
- Audio, downloaded archives, and model weights remain under `/home/sandy/.local/share/medibytes-eval`; only policy, manifests, checksums, notices, configs, code, and reports are versioned.

## Rollback

Remove the OpenSpec change and revert the corpus/backend/evaluator cutover and B0 tag/worktrees as one unit. Delete external data only after preserving manifests, checksums, notices, reports, and failure evidence. No method branch is a dependency of another method branch, so an individual method arm can be discarded without changing the frozen shared baseline.

# Design: MediBytes Denoising Pilot and B0 Worktrees

## Technical Approach

The change creates four frozen layers in dependency order:

1. a source-pinned, deterministic 15-base paired corpus;
2. a shared cleaner/backend and strict real-STT contract;
3. a schema-validating, failure-retaining evaluator and scorer;
4. twelve sibling worktrees that may change only one method adapter each, followed by a gated single-backend comparison.

No method worktree is created until the shared corpus, baseline, evaluator, tests, and these OpenSpec artifacts are accepted and committed as B0. B0 is the integration boundary: all method worktrees branch from its exact SHA, never from one another.

## Affected Files

The implementation cutover adds or changes these shared surfaces:

- `eval/corpus/selection.json`, `eval/corpus/build_corpus.py`, `eval/corpus/manifest.jsonl`, `eval/corpus/source-checksums.sha256`, and `eval/corpus/THIRD_PARTY_NOTICES.md` — source policy, collector, provenance, and notices.
- `demo/denoise.py` and `demo/denoise_backends/*.py` — backend identity, configuration, probe, and one-adapter execution.
- `demo/audio_clean.py` — shared decode/preprocess/enhance/validate/save boundary and measured provenance.
- `demo/pipeline.py`, `demo/app.py`, and product requirements — clean call-site cutover.
- `demo/stt_extract.py` and `demo/coords.py` — strict STT, normalization, and deterministic entity spans.
- `eval/harness.py`, supporting `eval/` modules, Draft-07 schemas, evaluator/downstream/method configs, reports, and tests — isolated execution, validation, metrics, and terminal dispositions.

A method worktree may add only `demo/denoise_backends/<id>.py`, a co-located method config, method-only dependency/native-wrapper files, and one targeted `demo/tests/test_denoise_<id>.py`.

## Shared Audio Contract

### Common format and data boundary

- The evaluator data root is the absolute, nonempty environment value `MEDIBYTES_EVAL_DATA_ROOT=/home/sandy/.local/share/medibytes-eval`. Corpus, downloaded archives, model weights, and run outputs live under that root; repository files contain only policy, manifests, hashes, notices, configs, code, and reports.
- Corpus and evaluator audio is 16 kHz mono signed PCM16 WAV. A canonical RIFF header is exactly 44 bytes. Source metadata is not propagated.
- `clean_audio` accepts source input at any decodable rate/channel layout, decodes it, and resamples to 16 kHz mono before the common RMS target `0.1`, peak ceiling `0.98`, and diagnostic RMS VAD with trimming disabled.
- Enhancement output is delay-compensated, finite, nonempty, mono, and exactly the common input sample count before save. The saved output is 16 kHz PCM16.

### Public shared API

`demo/denoise.py` exposes:

- `BackendId(StrEnum)` with exactly `none`, `noisereduce`, `rnnoise`, `deepfilternet3`, `dpdfnet2-onnx`, `sherpa-gtcrn-simple`, `percepnet`, `nemo-se-den-sb-16k`, `speecht5`, `speex-denoise`, `speechbrain-metricgan-plus`, `voicefixer-mode0`, `cmsis-dsp`, and `spectral-subtraction`.
- Frozen `EnhancementConfig(backend, variant_id, model_path, model_sha256, parameters, timeout_s)`, where parameters are JSON values and the model fields are optional only when the selected method has no model.
- Frozen `BackendProbe(available, reason, source_revision, runtime_version, model_sha256, native_sample_rate, delay_samples, stateful)`.
- Frozen `EnhancementOutput(audio, actual_backend, variant_id, native_sample_rate, delay_samples, metadata, provenance)`.
- `run_backend(audio, sample_rate, config) -> EnhancementOutput`, which imports exactly `demo.denoise_backends.<backend-with-underscores>`, calls that module's `probe(config)` once and `enhance(audio, sample_rate, config)` once, and validates identity/geometry/provenance. It never catches an adapter failure to select another implementation.

`clean_audio(in_path: Path, out_path: Path, backend: BackendId = BackendId.NONE, config: EnhancementConfig | None = None) -> dict[str, JSONValue]` returns measured pre/post RMS dBFS, sample counts, duration drift, requested/actual backend, variant, model SHA-256, runtime version, `fallback_used=false`, input/output hashes, and processing time. The old `apply_denoise` argument and nominal SNR fields are removed.

### Native-rate adapter boundary

Only a method adapter crosses the 16 kHz/native boundary:

| Backend ID | Native rate | Adapter contract |
|---|---:|---|
| `none` | 16 kHz | Return standardized input unchanged. |
| `noisereduce` | 16 kHz | Control only; pin `noisereduce==3.0.2`; unavailable dependency is `BACKEND_UNAVAILABLE`. |
| `rnnoise` | 48 kHz | One state/sample through `rnnoise_get_frame_size()`; reset/flush between samples. |
| `deepfilternet3` | 48 kHz | CPU PyTorch `init_df()`/`enhance()` against the staged DeepFilterNet3 directory; one state/sample. |
| `dpdfnet2-onnx` | 16 kHz | Local pinned ONNX model; reset streaming state for every sample/condition. |
| `sherpa-gtcrn-simple` | 16 kHz | `OfflineSpeechDenoiser` CPU path; record provider and thread count. |
| `percepnet` | n/a | Evidence-only disposition; no adapter. |
| `nemo-se-den-sb-16k` | 16 kHz | `se_den_sb_16k_small.nemo`; ODE sampling `num_steps=10`, seed `1729`. |
| `speecht5` | n/a | Evidence-only disposition; no ASR/TTS/codec adapter. |
| `speex-denoise` | 16 kHz | 160-sample `speex_preprocess_run` frames; one state/sample. |
| `speechbrain-metricgan-plus` | 16 kHz | MetricGAN+ `SpectralMaskEnhancement`; one fresh model/state per run. |
| `voicefixer-mode0` | 44.1 kHz output | `VoiceFixer.restore(..., cuda=False, mode=0)` by default, then resample to 16 kHz. |
| `cmsis-dsp` | n/a | Evidence-only target disposition; no host accuracy adapter. |
| `spectral-subtraction` | 16 kHz | Fixed spectral-subtraction configuration; one state/sample. |

A native input resampler uses a duration-preserving endpoint mapping to `round(common_samples * native_rate / 16000)`. The adapter declares its algorithm delay in `BackendProbe` and `EnhancementOutput`; the shared boundary removes exactly that many compensated samples. Rate conversion to 16 kHz uses the inverse duration-preserving mapping. After the declared delay and any method-documented algorithmic padding are reconciled, the result must equal the original common sample count exactly. VoiceFixer may crop only the padding identified by its restoration path and records the crop count. Any other remaining length mismatch, wrong rate/channel count, empty/all-zero/nonfinite audio, backend/variant mismatch, or fallback flag fails closed. There is no generic tail trim or silent zero-padding fallback.

## Strict Downstream Contract

`demo.stt_extract.transcribe` adds `strict: bool = False`. Strict mode propagates every `faster-whisper` exception and never returns `MOCK_TEXTS`. Explicit `model="mock"` remains available only for investor fixtures and is labeled `is_mock=true` with provider, requested/actual model, device, compute type, beam size, runtime version, and model hash/snapshot when available.

`eval/config/downstream-v1.json` freezes `faster-whisper` `base-int8` on CPU with INT8, `beam_size=1`, word timestamps, `temperature=0`, strict mode, regex-only extraction, `use_llm=false`, no corrected/human artifact, and no cache/idempotency reuse. Entity offsets are calculated against the exact frozen `normalized_en` string and carry `exact|ambiguous|not_found`. Evaluation rejects mock/fallback engines, cached output, corrected entities, a changed downstream hash, absolute result paths, and a non-single-backend composition.

## Reproducible Corpus

### Source selection and downloads

`eval/corpus/selection.json` is the immutable pre-download policy:

- FLEURS locales `en_us`, `hi_in`, and `ta_in` use `hf_hub_download(repo_id="google/fleurs", repo_type="dataset", revision="168de341b3db6859a9bac1c50a2ef5e3b47647e0", filename="<locale>/validation/0000.parquet")`.
- Each locale sorts eligible validation rows by numeric `id` and admits the first four with decoded 3–12 second mono audio, nonempty transcript, and no digits, URLs, email addresses, phone numbers, or privacy-sensitive tokens.
- MUCS uses `curl --fail --location --retry 3` for `https://openslr.elda.org/resources/104/Hindi-English_test.tar.gz`, requires size `443929204`, and orders by `recording_id` then segment start sample. It admits the first eligible 3–12 second segment from each of the first three distinct recordings under the same privacy filter.
- The archive digest is computed on the first verified download and persisted in `source-checksums.sha256`; every parquet digest is also persisted. Later builds consume the literal IDs/paths in `manifest.jsonl` and reject changed digests.
- Canonicalization uses the `imageio-ffmpeg==0.6.0` bundled FFmpeg. A system FFmpeg is eligible only if its binary SHA-256 is recorded in the manifest. Empty, clipped, all-zero, nonfinite, or duration-mismatched conversion fails. Clean audio is peak-normalized to `-3.00 ± 0.01 dBFS`.

The collector provides `download`, `build --spec ... --out ...`, `verify --manifest ... --data-root ...`, and `rebuild --out ...` subcommands. It rejects a relative or empty data root. `eval/.cache/` is temporary and ignored; canonical external data is never staged in git.

### Paired generation

Each `sample_id`/noise pair uses the first eight little-endian bytes of `SHA-256("medibytes-corpus-v1|{sample_id}|{noise_type}")` in `numpy.random.Generator(numpy.random.PCG64(seed))`; the manifest records all 16 seed hex characters. A 20 ms RMS window and 10 ms hop define active frames above `max_frame_rms * 10^(-40/20)`. Additive conditions are scaled to `5.0 ± 0.1 dB` active-mask SNR. `echo` convolves with a deterministic exponentially decaying noise RIR at direct delay `0 ms` and `RT60=0.4 s`, normalizes leading energy to one, uses the reverberant clean signal as its reference, and applies one global headroom gain only if the mixture would exceed `-1 dBFS`.

In ascending `sample_id` order, the five fixed conditions are repeated three times each: `white` is independent `N(0,1)`; `pink_voss16` is the pinned 16-row Voss-McCartney stream; `hum50` is 50/100/150 Hz at `0.60/0.25/0.10` plus white noise `0.05`; `am_babble_proxy` modulates four independently seeded Voss streams at `3.1/4.7/6.3/8.1` Hz.

### Manifest and admission

Each JSONL base record contains schema/corpus/sample/group identity, `split_role=smoke`, `tuning_allowed=false`, bucket/language/code-mix/locale fields, dataset/release/revision/archive/digest/split/ID/path/speaker/row/transcript fields, transcript digest, and SPDX license/attribution. Both clean and noisy records contain path, SHA-256, bytes, sample rate, channels, codec, sample count, duration, and peak. Noisy records additionally contain noise type, seed, target/measured SNR, active RMS, post-mix gain, and clipping count. The base record also contains build-tool/decoder hashes, privacy decision, and `reference_status=source_reference`.

Admission requires 15 unique groups, 30 WAVs, `4/4/4/3` English/Hindi/Tamil/code-mix, `3/3/3/3/3` noise cases, equal paired sample counts, zero clipping, complete hashes/provenance, and all noisy SNRs in `5.0 ± 0.1 dB`. Rebuild uses a separate output root and requires byte-identical hashes and metadata.

## Evaluation Data Model and Isolation

### Result layout

A run ID maps one-to-one to:

```text
$MEDIBYTES_EVAL_DATA_ROOT/runs/<run_id>/
  run.json
  environment.json
  arms/<backend>/<variant>/
    report.json
    report.md
    conditions/<condition_id>/
      status.json
      enhanced.wav
      enhancement.json
      transcript.json
      entities.json
      fields.json
      resources.json
      metrics.json
```

`run.json` binds corpus/manifest/config/downstream/environment hashes, backend/variant, lifecycle, and condition inventory. `status.json` is always present and terminal. Enhancement records requested/actual IDs, variant, source/model/config hashes, runtime/device, delay/length/peak validity, and `fallback_used=false`. Transcript/entity/field records identify strict real STT, normalized text/hash, spans/statuses, false positives, and mock/cache/correction rejection. Resource and metric records preserve unavailable values explicitly.

Corpus, run, condition, and report Draft-07 schemas set `additionalProperties: false` and require identity, status, hashes, geometry, provenance, and metric-state fields appropriate to the record. A finalized run ID cannot be overwritten. Resume requires identical manifest, corpus, config, and downstream hashes; otherwise the caller must use a new ID.

### Process and failure model

- Every condition executes in a fresh child process and constructs/resets method state before work.
- Stage 1 executes once for accuracy. Resource/latency measurement then performs one warm-up and three measured repeats in fresh processes, serially, so measurements do not compete for CPU/RAM/GPU.
- Crash, OOM, timeout, empty output, invalid audio, and rejected input all produce a terminal condition status plus every artifact that can safely be written. Missing enhancement-dependent downstream artifacts are explained by the status; the condition is never omitted.
- A report accounts for all 30 expected conditions for an attempted viable row. A blocked/not-comparable/target-blocked row has one explicit terminal disposition and no quality score.
- No runner writes under `demo/cleaned`, `demo/transcripts`, `demo/entities`, `demo/exports`, or `demo/_state`, and no condition reuses another condition's cache/state.

## Metrics and Statistics

For aligned finite `ref` and `enhanced` arrays:

- `SNR = 10*log10(sum(ref^2) / sum((enhanced-ref)^2))`; delta SNR is paired candidate minus `none`.
- SI-SDR subtracts the reference mean, projects the zero-mean enhanced signal onto the zero-mean target, and reports the scale-invariant target-to-error ratio.
- LSD uses 20 ms Hann windows, 50% overlap, 512 FFT, and epsilon `1e-10`.
- PESQ MOS-LQO is wideband only; STOI and PESQ record package versions. Missing or incompatible references return `not_applicable`.
- Clipping ratio/count, DC offset, maximum and p99.99 adjacent discontinuity, RMS dBFS, exact sample-count drift, and duration drift are always structurally represented.

Text scoring first applies Unicode NFC, casefold, punctuation/whitespace tokenization, and Levenshtein WER, retaining numerator, denominator, substitutions, deletions, and insertions. It then applies the exact frozen `demo.stt_extract.normalize_text` to reference and prediction for normalized WER and exact match. Medical WER, medication/dose accuracy, clinical NER F1, field accuracy, and critical-error rate are `not_applicable`; empty clinical gold is used only for false-positive field counts and crash/schema status.

Latency/resource records include cold model-load/inference, warm inference, p50/p95/p99, real-time factor, child CPU time, peak RSS, and NVML peak VRAM. An unavailable VRAM counter is `unavailable`, never zero.

Paired candidate-minus-`none` uncertainty uses 10,000 cluster-bootstrap resamples of the 15 base `group_id` values with seed `1729`, followed by Holm correction across all 12 row families. Statistical outcome labels are `observed`, `inconclusive`, `invalid`, `blocked_prerequisite`, or `not_comparable`; they are distinct from terminal row dispositions such as `failed_runtime` or `blocked_target`. `production_supported` is unavailable in this pilot.

## Method Catalog and Concrete Pins

`eval/config/method-catalog.json` preserves the authoritative CSV row and normalizes these records. Each record includes the exact backend/variant, upstream source/revision, authoritative license/notice, model/checkpoint hash policy, execution class, prerequisites, and default terminal disposition. Dynamic archive/extracted hashes are computed and pinned when downloaded; they may not be replaced by branch names.

| CSV row/order | Backend and execution class | Pin and concrete behavior | Default disposition |
|---:|---|---|---|
| 2 | `rnnoise` / `realtime_laptop` | Xiph tag `v0.2`, model `0b50c45`, BSD-style license; hash `rnnoise_data-0b50c45.tar.gz` and generated `src/rnn_data.c`; build via upstream `autogen.sh`, `configure`, `make`; narrow C wrapper; 48 kHz signed PCM16; full state reset/flush. | Implement/evaluate or retain failure. |
| 3 | `deepfilternet3` / `realtime_laptop` | `Rikorose/DeepFilterNet` tag `v0.5.6`, pinned `models/DeepFilterNet3.zip`; hash archive plus extracted config/checkpoint; Python `init_df()`/`enhance()`, CPU PyTorch, one state/sample, 48 kHz around model. GPU is a separately labeled actual device, never inferred. | Implement/evaluate or retain failure. |
| 4 | `dpdfnet2-onnx` / `realtime_laptop` | `dpdfnet==0.6.0`, commit `de503b39ddfb16023b9d599b05ca872877506047`, Apache-2.0; snapshot `dd6818d00f50c836fed43a6243ebe49116de5964`; `dpdfnet2.onnx` SHA-256 `4f0ee28935b4a32abecc717d745416976565834d839601acf43031094b4dc94c`; local 16 kHz ONNX CPU path, reset streaming state. | Implement/evaluate or retain failure. |
| 5 | `sherpa-gtcrn-simple` / `realtime_laptop` | `sherpa-onnx==1.13.8`, `sherpa-onnx-bin`, toolkit commit `362ddf2c077e3c04759c93b7a47212a5c9108ca7`, Apache-2.0 toolkit; public `gtcrn_simple.onnx` asset hash recorded with GTCRN MIT notice; 16 kHz CPU `OfflineSpeechDenoiser`; record provider/threads. | Implement/evaluate or retain failure. |
| 6 | `percepnet` / `realtime_laptop` evidence only | Unofficial source commit `8ffae4337d23f920176ac2a7426e84610fa338ab`; generated `nnet_data.cpp`/weights and pretrained checkpoint are absent. | `blocked_prerequisite`, exact spec reason; no adapter. |
| 7 | `nemo-se-den-sb-16k` / `research_gpu` | `nemo-toolkit==2.7.3`; NVIDIA-NeMo/Speech commit `1d4ee423806d461f9146ae982f9da8eb32495ae7`; snapshot `fe9725b1ae90f7f6b11d9b95c5f9ccec736a211f`; `se_den_sb_16k_small.nemo` SHA-256 `9de7f36d08df4109402e68e41d78e781c8da6a3d7691a674c5231b840151ff09`; model card CC-BY-NC-SA-4.0, research-only; 16 kHz, `num_steps=10`, seed `1729`, actual device/versions recorded. | Implement/evaluate only if runnable; else `blocked_prerequisite` or `failed_runtime`. Never substitute a model. |
| 8 | `speecht5` / `research_gpu` evidence only | Official source commit `5d66cf5f37e97f4a1999ad519537decc16d852af`; no standalone enhancement checkpoint/audio-to-audio API. | `not_comparable`, exact spec reason; no ASR/TTS/codec adapter. |
| 9 | `speex-denoise` / `realtime_laptop` | SpeexDSP `1.2.1`, commit `7a158783df74efe7c2d1c6ee8363c1e695c71226`, BSD-style; 160 samples/10 ms; `SPEEX_PREPROCESS_SET_DENOISE=1`; AGC, VAD, dereverb, residual echo suppression, and echo state explicitly off; upstream default suppression recorded. | Implement/evaluate or retain failure. |
| 10 | `speechbrain-metricgan-plus` / `offline_laptop` | SpeechBrain `1.1.1`, commit `89ead74d163463d30c62329a09cfdb4c54f5abc1`; model `speechbrain/metricgan-plus-voicebank`, revision `a196ce26b3bdace6fa1d819017584bdbcce462a8`, `enhance_model.ckpt`, Apache-2.0; 16 kHz `SpectralMaskEnhancement`; MetricGAN+ only, not pooled SepFormer/SEGAN. | Implement/evaluate or retain failure. |
| 11 | `voicefixer-mode0` / `offline_laptop`, `generative_restoration` | `voicefixer==0.1.3`, Zenodo `5600188`, mode `0`; `vf.ckpt` MD5 `1f479dabb9fa5724de5e20f2164cbc7a`, `model.ckpt-1490000_trimed.pt` MD5 `ac6de16561dd6042daf8bb3a806dc818`; record SHA-256 too; CPU `restore` by default; 44.1 kHz output resampled to 16 kHz; waveform/speaker/term preservation reported separately. | Implement/evaluate or retain failure. |
| 12 | `cmsis-dsp` / `target_only` evidence only | Release `1.18.0`, commit `4b6f26593e9d3b24a9bc93b3646a3a859238eae2`, Apache-2.0; kernels but no selected end-to-end algorithm or Cortex-M/Cortex-A target/audio transport. | `blocked_target`, exact spec reason; no host adapter. |
| 13 | `spectral-subtraction` / `realtime_laptop` | `pyroomacoustics==0.10.1`, commit `f02b01dd6609709e2089aefa5d1e59c91d3a0601`; license recorded from pin; 16 kHz `apply_spectral_sub`/online path with `nfft=512`, `db_reduc=10`, `lookback=12`, `beta=3`, `alpha=1.2`; spectral subtraction only, no pooled Wiener filtering. | Implement/evaluate or retain failure. |

`noisereduce` is intentionally absent from this 12-row catalog and exists only as a shared control.

## B0 Worktree Freeze and Sequence

After corpus, shared code, tests, configs, reports, and artifacts are committed, record `BASE_SHA=$(git rev-parse HEAD)`, tag `denoise-pilot-b0`, and require a clean checkout plus `python -m pytest eval/tests demo/tests -q`. Worktrees are created from that SHA in this exact order:

| Order | Branch | Sibling path |
|---:|---|---|
| 1 | `eval/rnnoise-v0.2` | `../Medibytes-eval-rnnoise` |
| 2 | `eval/deepfilternet-v0.5.6-dfn3` | `../Medibytes-eval-deepfilternet` |
| 3 | `eval/dpdfnet-v0.6.0` | `../Medibytes-eval-dpdfnet` |
| 4 | `eval/sherpa-onnx-1.13.8-gtcrn` | `../Medibytes-eval-sherpa-onnx` |
| 5 | `eval/percepnet` | `../Medibytes-eval-percepnet` |
| 6 | `eval/nemo-2.7.3-se-den` | `../Medibytes-eval-nemo` |
| 7 | `eval/speecht5` | `../Medibytes-eval-speecht5` |
| 8 | `eval/speexdsp-1.2.1` | `../Medibytes-eval-speexdsp` |
| 9 | `eval/speechbrain-1.1.1-metricgan-plus` | `../Medibytes-eval-speechbrain` |
| 10 | `eval/voicefixer-0.1.3-mode0` | `../Medibytes-eval-voicefixer` |
| 11 | `eval/cmsis-dsp-v1.18.0` | `../Medibytes-eval-cmsis-dsp` |
| 12 | `eval/pyroomacoustics-0.10.1-spectral-subtraction` | `../Medibytes-eval-spectral-subtraction` |

The current row must finish its targeted test, real `clean_audio` smoke, preflight, product-equivalent smoke, 30-condition run or explicit terminal disposition, and individual report before the next `git worktree add -b <branch> <path> "$BASE_SHA"` command. Completed siblings remain available. Branches never branch from one another and are not merged in this phase.

Before every method run, compare shared files to B0. `demo/denoise.py`, `demo/audio_clean.py`, `demo/stt_extract.py`, `demo/coords.py`, templates/assets, corpus tooling/manifest, evaluator, and downstream config must be byte-identical. A shared diff invalidates the run and requires resetting that method worktree to B0; fixing the shared contract requires a separately accepted new baseline and regenerated results, not an in-place method patch.

## Environment, Licenses, and Data Safety

- All evaluation/method environments use Python 3.12. The workstation's Python 3.14 is not used. With installed `uv 0.12.13`, create environments using `uv python install 3.12`, `uv venv --python 3.12 .venv`, and `uv pip install -r <lock>` in the relevant checkout.
- A method package that officially rejects Python 3.12 uses an isolated officially supported environment; the deviation and versions are recorded. Published package metadata is not modified to force support.
- No credentials or secrets are embedded in source/config/manifests. Gated, authenticated, or license-unaccepted artifacts are prerequisites. A failed official download preserves the error and does not authorize a mirror.
- Method archives, generated wrappers, model weights, and runtime outputs stay in the data root or method checkout's ignored environment. Versioned artifacts contain source/license/attribution, exact revisions, cryptographic digests, and notices. NeMo remains research-only because its selected model card is CC-BY-NC-SA-4.0.

## Verification and Concurrency Constraints

Verification advances only at phase boundaries:

1. **Corpus gate** — run download, build, verify, and separate rebuild commands; require exact counts/distributions, `5.0 ± 0.1 dB`, no clipping, complete provenance, and identical rebuild hashes.
2. **Shared/B0 gate** — run `python -m pytest eval/tests demo/tests -q`; validate corpus; run `none` twice under distinct run IDs and require 30/30 terminal `ok`, identical enhanced hashes, exact backend identity, and no demo output/cache writes. Inject unavailable `noisereduce` and a real-STT exception and require retained terminal failures.
3. **Per-method gate** — in the current sibling only, run its targeted test, `preflight`, one real `clean_audio()` fixture, full 30-condition `run`, and actual CLI/UI-equivalent product smoke. Require finite mono 16 kHz audio, exact requested/actual backend/variant, no fallback, complete hashes/runtime/config, valid delay/length, and visible real STT/entity artifacts or retained failure.
4. **Final gate** — after all 12 terminal reports exist, run `compare`; require every disposition, all attempted conditions accounted for, paired deltas/CIs, no contaminated/layered input, and no `production_supported` verdict.
5. **Actual UI gate** — launch Streamlit from the frozen B0/demo state with one clean and one noisy corpus file and explicit `none`; observe upload, playback, real transcript, entities/fields, and export without cached demo fixtures.

Corpus rebuilds, finalized runs, and resource measurements are serialized. Method worktrees are sequential, and no two resource-heavy methods run concurrently. Independent read-only inspection of completed worktrees/reports may occur concurrently but cannot alter a frozen worktree. Finalized run directories are immutable; correction requires a new run ID.

## Security and Failure Considerations

- Corpus privacy filtering occurs before literal IDs are committed; an ambiguous privacy token rejects admission rather than being redacted into a different sample.
- Schema validation, hash binding, path-root checks, timeout/resource capture, and backend identity checks fail before a result can be ranked.
- Missing or unverifiable source/model artifacts, runtime crashes, and unsupported targets remain visible terminal evidence. They never cause silent substitution.
- Absolute result paths, cache/idempotency reuse, and writes into product demo state are rejected to keep evaluation reproducible and auditable.

## Migration and Rollback

The cleaner/STT signatures cut over together: update all exact callers, remove `apply_denoise` and `--no-denoise`, and do not retain compatibility aliases. B0 is the first fully migrated integration commit. Rolling back the pilot removes the OpenSpec change and the shared cutover as one unit; completed method branches and external data can be discarded independently after reports/manifests/notices are preserved.

## Open Questions

None. The backend identities, corpus distribution, fail-closed behavior, method pins/dispositions, B0 sequencing, metric formulas, and comparison boundary are fixed by this design.

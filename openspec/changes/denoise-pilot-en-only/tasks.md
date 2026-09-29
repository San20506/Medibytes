# Tasks: English-Only Denoising WER Pilot

## 1. OpenSpec Artifacts

- [ ] 1.1 Review `proposal.md`, `design.md`, and `specs/denoise-evaluation/spec.md` against this task list; resolve any contradiction before changing runtime code.
- [ ] 1.2 Commit the four OpenSpec artifacts as the first change, with no schema/config/code edits bundled in.

## 2. Pinned Study Constants

- [ ] 2.1 Add `en_pilot/config.py` holding literals only: `DATA_ROOT_DEFAULT`, `CORPUS_ID = "en-40-v1"`, `CLEAN_DIR`, `MATRIX_DIR = "matrix-en-40"`, `OUT_DIR = "en-pilot-run"`, `BASE_COUNT = 40`, `NOISE_BANK_DIR`, `SNR_TARGETS_DB = (0.0, 5.0)`, `SAMPLE_RATE`, `TARGET_PEAK_DBFS`, `MIN_DURATION_S`, `MAX_DURATION_S`, `SEED_NAMESPACE = "medibytes-en-pilot-v1"`, `SNR_TOLERANCE_DB = 0.5`, `CLIPPING_CEILING`, `BOOTSTRAP_ITERATIONS = 10_000`, `BOOTSTRAP_SEED = 1729`, `HOLM_FAMILY_COUNT = 2`, `WER_DELTA_CI_LOW_MIN = -0.02`, `WER_DELTA_CI_HIGH_MAX = 0.0`, `PRIMARY_DECODER = "small"`, `CONTINUITY_DECODERS = ("base",)`, `BACKENDS`.
- [ ] 2.2 Pin `BACKEND_REPOS` to the per-backend worktree roots already used by `evidence/matrix-v2/matrix_eval.py:45-50`, with the `none` control mapping to the compare worktree; expose a `--backend-repo-root` override on the enhance CLI rather than editing the constant when a worktree moves.
- [ ] 2.3 Confirm the arm set excludes `deepfilternet3`: it is refuted on real noise (`matrix-v2` WER +1264% relative, output SNR pinned near −3 dB regardless of input) and `denoise-pilot-b2-gap-closure/tasks.md` item 5.6 states it is not owed a re-run.

## 3. 40-Base English Corpus

- [ ] 3.1 Add `en_pilot/build_corpus.py` calling the frozen primitives rather than extending `eval/corpus/build_corpus.py`, whose `corpus_id` and 15-group assumptions are pinned by `eval/corpus/manifest.py` and both JSON Schemas.
- [ ] 3.2 Admit bases through `select_fleurs_rows(reader.rows, lambda row: reader.probe(row, decoder), limit=40)` so the frozen privacy gate and 3–12 s window are reused, and decode/canonicalise through `decode_bytes` + `canonicalize_clean` so the −3 dBFS normalisation and its hard-fail band are reused.
- [ ] 3.3 Write `DATA_ROOT/corpus/en-40-v1/clean/<sample_id>.wav`, `manifest.jsonl` (including the exact FLEURS `path`, row group and row index, because `fleurs_id` repeats across speakers) and `sources.json` with the driver Python version, the pinned `imageio-ffmpeg` digest and the parquet digest.
- [ ] 3.4 Fail closed and write nothing if fewer than `BASE_COUNT` rows pass the privacy/duration gate.

```bash
.venv/bin/python -m en_pilot.build_corpus
.venv/bin/python -c "import json,pathlib;rows=[json.loads(l) for l in pathlib.Path('/home/sandy/.local/share/medibytes-eval/corpus/en-40-v1/manifest.jsonl').read_text().splitlines()];print(len(rows),[r['sample_id'] for r in rows[:4]],rows[0]['bucket'],rows[0]['language_profile'])"
# expected: 40 ['sample-001', 'sample-002', 'sample-003', 'sample-004'] en-US-proxy en-US-proxy
```

- [ ] 3.5 Assert the superset claim by digest before anything downstream runs: if `sample-001` differs from the frozen corpus, stop — the canonicalisation path has drifted and the study is not comparable to `matrix-v2`.

```bash
sha256sum /home/sandy/.local/share/medibytes-eval/corpus/en-40-v1/clean/sample-001.wav \
          /home/sandy/.local/share/medibytes-eval/corpus/pilot-15-v1/clean/sample-001.wav
# expected: two identical digests
```

## 4. Real-Noise Matrix

- [ ] 4.1 Add `en_pilot/build_matrix.py` modelled on `evidence/matrix-v2/build_noise_matrix.py`, importing the frozen audio primitives from this repository instead of that file's hardcoded `_REPO`.
- [ ] 4.2 Derive each seed as the first 8 bytes little-endian of `sha256(f"{SEED_NAMESPACE}|{sample_id}|{noise_id}|{snr_db}")` so no mixture is ever shared with `matrix-v2`.
- [ ] 4.3 Verify every noise recording against `evidence/noise-bank/demand-bank-manifest.jsonl` before use; a digest mismatch is a hard failure, not a recorded observation.
- [ ] 4.4 Mix with the frozen `active_mask` and `mix_at_target_snr`, quantise with `float_to_pcm16`, and hard-fail on clipping or on `abs(measured_snr_db - snr_db) > SNR_TOLERANCE_DB`.
- [ ] 4.5 Write the 27-field `matrix-v2` provenance record plus the `bucket` column `matrix-v2` left permanently `null`, so 40 bases × 18 noises × 2 SNRs = 1,440 mixtures are each traceable to a licensed noise file, a seed and an achieved SNR.
- [ ] 4.6 Re-hash every clean source WAV after the sweep and fail if any changed.

```bash
.venv/bin/python -m en_pilot.build_matrix
# expected: {"mixtures": 1440, "bases": 40, "noises": 18, "originals_unchanged": true, ...}
```

## 5. Enhancement, One Process Per Backend

- [ ] 5.1 Add `en_pilot/enhance.py`, invoked once per backend, that refuses to run for a backend outside `config.BACKENDS` and refuses to run when a backend's worktree or its `.venv/bin/python` is absent.
- [ ] 5.2 Prepend exactly one repository root to `sys.path` before any `demo` import, then import `run_backend`, `BackendId` and `EnhancementConfig` from it; never import a second repository, and refuse to continue if a `demo` package is already in `sys.modules`. Threads are for work *within* one backend only.
- [ ] 5.3 Rebuild each `EnhancementConfig` by reading `eval/config/methods/<backend>.json`, resolving `model_path` under the data root and failing with a message naming the file when it is absent — never silently substituting a model.
- [ ] 5.4 Write `enhance-<backend>.csv` with `requested_backend`, `actual_backend` and `variant_id` read off the returned `EnhancementOutput`, so dispatch is auditable from the artefact rather than inferred from a comment.
- [ ] 5.5 Also transcribe one clean baseline per base with `backend="none"` and `condition_id = sample_id`, so the decoder's WER floor is measured; `matrix-v2` transcribed 15 clean clips and never scored them.
- [ ] 5.6 Drop `matrix-v2`'s dead `--batch-size` argument.

```bash
PYTHONPATH="$PWD" /home/sandy/Projects/Medibytes-worktrees/eval-dpdfnet/.venv/bin/python \
  -m en_pilot.enhance --backend dpdfnet2-onnx --data-root "$MEDIBYTES_EVAL_DATA_ROOT" --workers 4
# every row of enhance-dpdfnet2-onnx.csv must read actual_backend=dpdfnet2-onnx, variant_id=v0.6.0
```

## 6. Transcription, Both Decoder Tiers

- [ ] 6.1 Add `en_pilot/transcribe.py`, invoked once per decoder tier, using `ctranslate2.get_cuda_device_count() > 0 → cuda/float16` else `cpu/int8` — CTranslate2 carries its own CUDA runtime, so the driver venv's torch is irrelevant — and recording the resolved device and compute type on every row.
- [ ] 6.2 Transcribe with `beam_size=1, word_timestamps=True, temperature=0`, matching `eval/config/downstream-v1.json`'s `stt` block exactly, over every `(condition_id, backend)` plus every clean base.
- [ ] 6.3 Store `demo.stt_extract.normalize_text(text)["normalized_en"]` as the hypothesis. Do not re-implement normalisation.
- [ ] 6.4 Append to `transcripts-<decoder>.jsonl`, resume-safe on `condition_id + "|" + backend + "|" + decoder"`.
- [ ] 6.5 Record a per-clip exception as an `error` field with empty text, and let `score.py` turn it into a hard failure rather than scoring an empty hypothesis.

```bash
.venv/bin/python -m en_pilot.transcribe --decoder small --data-root "$MEDIBYTES_EVAL_DATA_ROOT" --streams 4
.venv/bin/python -m en_pilot.transcribe --decoder base  --data-root "$MEDIBYTES_EVAL_DATA_ROOT" --streams 4
```

## 7. Scoring, Statistics and the Decision Rule

- [ ] 7.1 Add `en_pilot/score.py` writing `per-condition.csv` and `per-base.csv`, scoring WER with the frozen `levenshtein_counts` over `evaluation_tokens()` and the same `frozen_normalize` applied to the reference, and SNR/SI-SDR with the frozen `snr_db`/`si_sdr_db` — not `matrix-v2`'s hand-rolled DP or its inline projection bound.
- [ ] 7.2 Treat an `actual_backend != requested_backend` row, a missing `(condition_id, backend, decoder)` triple and an errored transcript as hard failures that print the offending keys and exit non-zero.
- [ ] 7.3 Score the clean baselines into their own `snr_target_db == ""` block and keep them out of every aggregate.
- [ ] 7.4 Add `en_pilot/stats.py` with a 40-group paired cluster bootstrap (10,000 resamples, seed 1729, percentile interval, centred-exceedance p), reusing `eval.reporting.holm_adjust` with `family_count=2`; do not call `eval.reporting.cluster_bootstrap`, which raises for any group count other than 15.
- [ ] 7.5 Implement the pre-declared rule in one function: `adopt` iff `ci95_high <= 0.0 and ci95_low >= -0.02`, else `reject`; the `none` row is always `baseline`; `mean_delta_snr_db` and `mean_rtf` are reported but never gate.
- [ ] 7.6 Emit `decision.csv` and `comparison.json`, the latter carrying the full bootstrap objects, the Holm map, the rule constants and the sha256 of `demo/stt_extract.py` and `demo/denoise_backends/*.py`.

## 8. Tests

- [ ] 8.1 Add `en_pilot/tests/test_stats.py`: 40 identical deltas give `ci95_low == ci95_high == observed` and `p_value == 1.0`; all-negative deltas give `ci95_high < 0`; `decide` returns `adopt` for `[-0.01, 0.0]` and `reject` for `[-0.03, -0.01]`; and a 39- or 41-group input is rejected by `bootstrap_paired`, proving the group count is 40 and not the frozen 15.
- [ ] 8.2 Add `en_pilot/tests/test_build_matrix.py`: mixing at 0.0 and 5.0 dB reproduces `measured_snr_db` within `SNR_TOLERANCE_DB` using the frozen `active_mask`; a deliberately clipping mixture raises instead of writing; `derive_seed` is stable across calls and differs across `(sample_id, noise_id, snr_db)`.
- [ ] 8.3 Do not add tests under `eval/tests/`; that directory belongs to the frozen harness and to `denoise-pilot-b2-gap-closure` items 5.4/5.5.

```bash
.venv/bin/python -m pytest en_pilot/tests eval/tests demo/tests -q
```

## 9. Driver, Run and Write-Up

- [ ] 9.1 Add `en_pilot/run_study.sh` (`set -euo pipefail`) running from the repository root: `build_corpus` → `build_matrix` → one `enhance` invocation per backend, each launched by that backend's own `.venv/bin/python` → `transcribe --decoder small` → `transcribe --decoder base` → `score`, printing each phase's JSON summary.
- [ ] 9.2 Confirm `per-condition.csv` carries an SI-SDR value on every noisy row before the run is called finished; `matrix-v2` has zero for both candidates on English.
- [ ] 9.3 Confirm the clean-floor block shows materially lower WER than the 0 dB and 5 dB blocks for every backend; if it does not, the decoder or the reference is wrong and every noisy number is void.
- [ ] 9.4 Write `evidence/en-pilot/README.md` with the per-backend per-decoder decision table, the confidence interval, the measured RTF, the rule verbatim, and whether it was met — stating plainly whether a Stage-1 default was selected, and stating in `evidence/README.md:79-80`'s terms that no clinical metric was measured over public non-clinical speech.
- [ ] 9.5 Update `evidence/README.md` to describe `en-pilot` alongside `pilot-b1` and `matrix-v2`, and change its "two studies" heading to three.
- [ ] 9.6 Add a `CHANGELOG.md` entry following the repository's own convention, stating plainly whether a Stage-1 default was selected and, if `none` wins, saying so without hedging.

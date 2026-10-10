# Denoising pilot — changelog and result summary

Date: 2026-09-27. Baseline: tag `denoise-pilot-b0`, commit `8e10cb7`.
Corpus: `pilot-15-v1`, 15 public non-clinical samples (FLEURS read speech in
`en_us`/`hi_in`/`ta_in` plus 3 MUCS Hindi-English conversational code-mix),
each with a clean reference and one noisy copy at 5.0 ± 0.1 dB SNR across five
noise types. 15 bases × clean + noisy = 30 conditions per method.

## Scope limit, stated up front

This pilot cannot support a production, clinical or safety claim. Medical WER,
medication and dose accuracy, clinical NER F1, field accuracy and
critical-error rate are `not_applicable` by design, because the corpus is
non-clinical. The product demo still defaults to the `none` backend; no
adapter from this pilot is wired into product code, and no cascade, fallback
routing or layered processing was evaluated.

WER is additionally near-meaningless in these results: the recogniser already
scores 1.20 against the `none` baseline, so WER differences mostly reflect
recogniser failure on this corpus rather than denoiser quality.

## Terminal dispositions, all 12 catalog rows

| row | backend | terminal | conditions | commit |
|---:|---|---|---|---|
| 2 | rnnoise | observed | 30/30 | `bbc85fa` |
| 3 | deepfilternet3 | observed | 30/30 | `254f36a` |
| 4 | dpdfnet2-onnx | observed | 30/30 | `8407443` |
| 5 | sherpa-gtcrn-simple | observed | 30/30 | `52baa48` |
| 6 | percepnet | blocked_prerequisite | 0/0 | at B0 |
| 7 | nemo-se-den-sb-16k | invalid | 29/30 | `2f148fe` |
| 8 | speecht5 | not_comparable | 0/0 | at B0 |
| 9 | speex-denoise | observed | 30/30 | `28c9ed1` |
| 10 | speechbrain-metricgan-plus | observed | 30/30 | `c4f1330` |
| 11 | voicefixer-mode0 | observed | 30/30 | `3bd3a0e` |
| 12 | cmsis-dsp | blocked_target | 0/0 | at B0 |
| 13 | spectral-subtraction | invalid | 22/30 | `7a57785` |

## Aggregate results

| row | backend | SNR | SI-SDR | LSD | PESQ | STOI | WER |
|---:|---|---:|---:|---:|---:|---:|---:|
| — | none (baseline) | 4.01 | −0.12 | 14.21 | 2.92 | 0.84 | 1.20 |
| 5 | sherpa-gtcrn-simple | 16.55 | 15.65 | 15.42 | 2.45 | 0.84 | 1.51 |
| 4 | dpdfnet2-onnx | 15.24 | 14.41 | 21.09 | 2.29 | 0.83 | 1.55 |
| 7 | nemo-se-den-sb-16k | 15.07 | 13.09 | 14.03 | 2.31 | 0.84 | 0.89 |
| 3 | deepfilternet3 | 11.78 | 11.33 | 18.49 | 2.44 | 0.85 | 1.15 |
| 10 | speechbrain-metricgan-plus | 3.63 | 3.45 | 17.55 | 2.46 | 0.81 | 0.86 |
| 9 | speex-denoise | −2.44 | −22.95 | 16.79 | 2.73 | 0.67 | 1.11 |
| 13 | spectral-subtraction | −2.56 | −26.78 | 21.56 | 1.95 | 0.54 | 1.35 |
| 11 | voicefixer-mode0 | −2.24 | −29.35 | 16.87 | 1.69 | 0.75 | 0.94 |
| 2 | rnnoise | −3.01 | −27.05 | 18.85 | 2.83 | 0.52 | 1.80 |
| 6 | percepnet | n/a | n/a | n/a | n/a | n/a | n/a |
| 8 | speecht5 | n/a | n/a | n/a | n/a | n/a | n/a |
| 12 | cmsis-dsp | n/a | n/a | n/a | n/a | n/a | n/a |

Only four arms beat the no-denoising baseline. Six are at or below it, four of
them substantially worse than doing nothing. LSD rises for the strongest
denoisers, which means they alter the spectral envelope rather than only
removing noise. PESQ has no true wideband mode, so that column is weak
evidence and is reported rather than leaned on.

## Retained failures, not suppressed

- **nemo-se-den-sb-16k**, `sample-004-noisy`: output exceeded PCM16 full scale
  and the shared cleaner refused rather than silently clipping. Real
  integration risk for that model.
- **spectral-subtraction**, 8 conditions, all `*-clean`: the pinned aggressive
  parameters (`db_reduc=10`, `beta=3`, `alpha=1.2`) drive already-clean speech
  to silence. 22/30 ok.
- **percepnet** `blocked_prerequisite`, **speecht5** `not_comparable**, **cmsis-dsp**
  `blocked_target`: no executable artifact, no adapter written, branch left at B0.

## Artifact hazards found and contained

1. **DPDFNet2.** The staged `dpdfnet2.onnx` (3,883,008 bytes, SHA-256
   `6ecabbde…`) was not the pinned Ceva-IP artifact. Preserved as
   `dpdfnet2.unverified-6ecabbde.onnx`; the official revision `dd6818d0…` was
   downloaded and verified as `4f0ee289…`.
2. **VoiceFixer.** The 0.1.3 import-time downloader left a 55,205,888-byte
   `vf.ckpt` (MD5 `8d6f2d80…`) that does not match the pinned
   `1f479dab…`; the true file is 489,307,071 bytes. VoiceFixer loads the
   analysis module with `strict=False`, so the wrong file loads silently. The
   adapter pins size, MD5 and SHA-256 and re-verifies the exact paths
   VoiceFixer reads before constructing any model.

## Documented deviations

- **NeMo.** The pinned checkpoint's embedded config targets its two training
  losses at `nemo.collections.audio.losses.{MSE,MAE}Loss`, which moved to
  `nemo.collections.audio.losses.audio` in nemo-toolkit 2.7.3, so an unmodified
  restore aborts before any weight is read. The adapter rewrites exactly those
  two strings in memory; the archive is unmodified, the embedded config digest
  is pinned, weights load with `strict=True`, and the remap count is asserted
  to equal 2.
- **SpeexDSP.** Built from the pinned Xiph commit 1.2.1 in place, no source
  patch; noise suppression left at the upstream default of −15 dB and queried
  with `GET_NOISE_SUPPRESS` rather than overridden.

## Comparison gate: not run

The single-backend comparison could not be executed because of a defect in
frozen shared evaluator code. `eval/reporting.py:893` initialises
`per_metric_p_values` with bare metric names while line 999 indexes it with
composite `f"{group}.{name}"` keys, raising `KeyError: 'audio.snr_db'` for the
first candidate with 15 complete paired groups.

Because `reporting.py` is inside `IMPLEMENTATION_FILES`, changing it changes
`implementation_sha256`, which `verify_run` requires to match every finalized
run. Fixing it therefore invalidates all 12 existing reports and requires a new
baseline plus a full regeneration of every arm, not a one-line patch on top of
the current artifacts. This was left unpatched deliberately.

The adapters were assembled for verification in this worktree so
that `verify_run`'s adapter-hash check could pass across arms. No method branch
was merged.

## Path note, 2026-09-28

Worktrees were consolidated under `~/Projects/Medibytes-worktrees/` after this
document was written; see `~/Projects/Medibytes-worktrees/PATH-MIGRATION.md` for
the old-to-new mapping. Results were re-verified from the new locations.
`~/Projects/Medibytes/evidence/` now holds the pilot exports, the tagged test
data and the DEMAND noise-matrix study.

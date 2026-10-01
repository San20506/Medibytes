# MedASR with and without the denoising layer (2026-10-01)

Two variants of the demo pipeline, differing in one setting, measured on the
40-base English corpus and the full 1,440-condition DEMAND noise matrix the n=40
denoising pilot already built. Both commands run in **this** repository:

```bash
# Variant 1 - denoising ON + MedASR
python demo/pipeline.py --in <wav> --model medasr \
    --denoiser sherpa-gtcrn-simple \
    --denoiser-config demo/denoise_backends/sherpa-gtcrn-simple.config.json

# Variant 2 - denoising OFF + MedASR
python demo/pipeline.py --in <wav> --model medasr --denoiser none
```

Raw artefacts are in `evidence/medasr-denoise-20261001/`, which is gitignored
like the rest of `evidence/`.

## Bottom line

**Denoising helps MedASR — and MedASR is still the wrong decoder for this
corpus.** Both things are true and the second one is the bigger number:

| | Mean WER, 1,440 noisy conditions |
|---|---|
| Whisper `base`, no denoising | **0.1560** |
| Whisper `base` + GTCRN | 0.2066 |
| **MedASR + GTCRN (variant 1)** | **0.3790** |
| **MedASR, no denoising (variant 2)** | **0.4007** |

The best MedASR variant is 2.4× worse than the plain incumbent. Nothing here
argues for changing the pipeline default. What it does establish is that the
denoising question has a **decoder-specific answer**, which no previous
measurement in this repo had tested.

## The denoising result

Paired per-base deltas, 40 bases, 10,000-iteration bootstrap (seed 1729), scored
by `en_pilot.score.word_error_rate` — the same frozen normalisation and tokeniser
the n=40 study used:

| Decoder | Denoiser | Δ WER vs no denoising | 95% CI | p |
|---|---|---|---|---|
| **MedASR** | `sherpa-gtcrn-simple` | **−0.0217** | [−0.0396, −0.0051] | 0.014 |
| **MedASR** | `dpdfnet2-onnx` | **−0.0707** | [−0.0960, −0.0491] | <0.001 |
| Whisper `base` | `sherpa-gtcrn-simple` | +0.0505 | [+0.0289, +0.0790] | 0.001 |
| Whisper `base` | `dpdfnet2-onnx` | +0.0018 | [−0.0082, +0.0108] | 0.707 |

Both denoisers improve MedASR; neither improves Whisper. The sign flip belongs to
the decoder, not to one enhancement backend. `dpdfnet2-onnx` was run as a
robustness arm, not as one of the two requested variants.

### The printed decision says `reject` — read why

`en_pilot/stats.py` states its rule in code: `adopt iff ci95_high <= 0.0 and
ci95_low >= -0.02`, documented as accepting "a default which is provably no worse
than no denoising, and costs at most 2 absolute WER points." MedASR + GTCRN has
`ci95_high = -0.0051` (passes) and `ci95_low = -0.0396` (fails the −0.02 bound).
The entire interval is below zero, so what fired is the lower bound, not evidence
of harm. The rule was left exactly as written; the scorer now emits a
`decision_note` recording this rather than letting the word `reject` stand alone.

### How much of the gain is "the model said nothing at all"

MedASR emits an empty hypothesis on some noisy clips, which scores WER 1.0. That
happens 61 times undenoised, 14 times with GTCRN, 4 times with dpdfnet. Dropping
the 64 conditions where any arm went empty:

| Denoiser | Δ WER, all 1,440 | Δ WER, 1,376 with empties excluded |
|---|---|---|
| `sherpa-gtcrn-simple` | −0.0217 | −0.0147 |
| `dpdfnet2-onnx` | −0.0707 | −0.0614 |

Roughly a third of the GTCRN gain and an eighth of the dpdfnet gain is
empty-output rescue; the majority is ordinary accuracy on clips both arms
transcribed.

## This is not a test of MedASR's medical accuracy

Clean-speech floor WER: **MedASR 0.2620** vs Whisper `base` **0.1051**, on the
same 40 clean bases.

`en-40-v1` is FLEURS — read Wikipedia sentences about global warming and
mortality rates. MedASR is fine-tuned on physician dictation and emits report
structure (`[EXAM TYPE]`, `[FINDINGS]`, `{period}`, `{colon}`); on its own bundled
radiology sample it produces a clean, correctly punctuated report. **Nothing here
measures MedASR in the domain it was built for**, which is also the domain this
product records. That test needs clinical audio with reference transcripts;
`demo/audio_in/` currently holds only synthetic tones.

## Pipeline/harness parity

The sweep decoded the pilot's pre-enhanced files; the demo pipeline was then run
on 5 of those same conditions per variant, end to end, to check the measurement
describes the shipped path:

| Variant | Demo mean WER (n=5) | Harness mean WER (same 5) |
|---|---|---|
| GTCRN + MedASR | 0.4679 | 0.4679 |
| none + MedASR | 0.4734 | 0.5142 |

Close, and the same direction, but **not identical**, for two reasons worth
knowing: `demo/audio_clean.py` RMS-normalises to 0.1 before the backend, which
`en_pilot/enhance.py` does not (it calls `run_backend` directly); and the demo
hands MedASR a file path with `chunk_length_s=20`, while the sweep passes a raw
array. The harness therefore measures the backend in isolation and the demo adds
a gain stage around it. n=5 is far too small to compare the magnitudes.

## Incidental fix: the pilot was scoring half its noise matrix

`en_pilot/stats.py` filtered noisy rows with `not row["snr_target_db"]`. That is
True for the float `0.0` as well as for `""`. Scoring runs in memory where the
field is a float, so **every 0 dB condition — the harder half, 720 of 1,440 — was
silently dropped from every decision**, while the CSV written beside it kept them
(strings, so `"0.0"` is truthy). Fixed to test emptiness, with a regression test.

Conclusions are unchanged; the evidence behind them is now complete:

| Candidate (Whisper `base`) | Before, 5 dB only | After, all 1,440 |
|---|---|---|
| `sherpa-gtcrn-simple` | +0.0443, p=0.032 | +0.0505, p=0.0006 |
| `dpdfnet2-onnx` | +0.0011, p=0.872 | +0.0018, p=0.707 |

`en-pilot-run/results/decision.csv` has been regenerated; the pre-fix table is at
`evidence/medasr-denoise-20261001/en-pilot-decision-base-refixed.csv`.

## What changed in the project

- `demo/stt_extract.py`
  - `transcribe(..., model="medasr")` routes to `transcribe_medasr()`, running
    `google/medasr` through the `transformers` ASR pipeline on CUDA when
    available. Same result shape and `stt_provenance` keys as the faster-whisper
    path; `strict=True` still propagates failures.
  - `medasr_detokenize()` resolves MedASR's dictation markup: `{period}` → `.`,
    `{colon}` → `:`, `{new paragraph}` → blank line, unrecognised `{directive}`
    dropped, `[SECTION]` headers and `</s>` removed. Without it every `{period}`
    tokenises to the word "period" and inflates WER.
  - The mock fallback was factored into `_mock_result()` so both real backends
    share one labelled fallback.
- `demo/denoise_backends/sherpa_gtcrn_simple.py` — ported from the
  `eval-sherpa-onnx` worktree with its `sources.json` and requirements, so
  variant 1 runs here instead of only in that worktree. Needs
  `sherpa-onnx==1.13.8` and `sherpa-onnx-bin==1.13.8`, both now in `.venv`. The
  adapter fails closed on variant id, model path, digest and parameters, so
  `sherpa-gtcrn-simple.config.json` ships the exact values it demands.
- `demo/pipeline.py` — `--model medasr` documented.
- `en_pilot/transcribe_medasr.py` — decodes the already-enhanced pilot audio
  under `en-pilot-run/enhance-<backend>/`; nothing is re-enhanced, so a MedASR row
  and a Whisper row for the same `(condition, backend)` are the same bytes decoded
  twice. Resume-safe; keeps `raw_text` beside the detokenised text.
- `en_pilot/score_medasr.py` — scores both variants with the frozen WER rule and
  `en_pilot.stats.bootstrap_paired`, plus the sentence-stratified robustness
  bootstrap. `--context-decoder` picks the Whisper tier to report alongside.
- `en_pilot/stats.py` — the 0 dB fix above.

## Reproduce

```bash
.venv/bin/python -m en_pilot.transcribe_medasr --device cuda \
    --arms none,sherpa-gtcrn-simple,dpdfnet2-onnx
.venv/bin/python -m en_pilot.score_medasr --denoiser sherpa-gtcrn-simple
```

2,920 clips (both requested variants plus clean floors) in **62 s** on one GPU,
RTF **0.0027**. Decoding is greedy CTC — the `transformers` pipeline default, and
the mode the model card documents for its own evaluation. The shipped
`lm_6.kenlm` shallow-fusion LM was **not** used.

## Limits

1. **Not medical audio.** The denoising conclusion is solid; the absolute WER
   says nothing about MedASR in its own domain.
2. **English only.** MedASR is an English model, so the Hindi/Tamil/code-mix case
   this product cares about is untested here.
3. **Whisper `small` was not the reference.** `transcripts-small.jsonl` in that
   run directory is an interrupted sweep (806 of 1,440 conditions, 23 of 40
   bases; ~100 rows of it were appended during this session before the job was
   stopped). The complete tier is `base`, which is what every Whisper number
   above uses. Completing `small` is one resume-safe command — `python -m
   en_pilot.transcribe --decoder small --streams 4` — but CTranslate2 cannot load
   CUDA on this host, so it is roughly a 6-hour CPU job.
4. **The adoption rule was not re-derived.** `reject` on MedASR + GTCRN is the
   plausibility floor firing on an improvement, not a quality verdict.

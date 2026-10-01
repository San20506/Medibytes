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
corpus.** Both are true and the second is the bigger number:

| | Mean WER, 1,440 noisy conditions |
|---|---|
| Whisper `base`, no denoising | **0.1560** |
| Whisper `base` + GTCRN | 0.2066 |
| **Variant 1 — MedASR + GTCRN** | **0.3816** |
| **Variant 2 — MedASR, no denoising** | **0.4046** |

The better MedASR variant is 2.4× worse than the plain incumbent. Nothing here
argues for changing the pipeline default. What it does establish is that the
denoising question has a **decoder-specific answer**, which no previous
measurement in this repo had tested.

## The denoising result

Measured twice, on two different chains, over all 1,440 conditions. Paired
per-base deltas across 40 bases, 10,000-iteration bootstrap (seed 1729), scored
with `en_pilot.score.word_error_rate` — the same frozen normalisation and
tokeniser the n=40 study used.

**The shipped chain** — `demo.audio_clean.clean_audio` (RMS normalisation,
backend, resample, validation) then `demo.stt_extract.transcribe(model="medasr")`,
i.e. exactly what `demo/pipeline.py` runs, including the 3 clips GTCRN enhancement
refused outright (`422 ENHANCE_FAIL`, scored as the empty transcript the product
would have produced):

| Variant | Mean WER | Δ vs no denoising | 95% CI | p |
|---|---|---|---|---|
| 1. MedASR + GTCRN | 0.3816 | **−0.0229** | [−0.0423, −0.0052] | 0.015 |
| 2. MedASR, none | 0.4046 | — | — | — |

**The harness chain** — the pilot's pre-enhanced files decoded directly, which
isolates the backend from `clean_audio`'s gain stage:

| Decoder | Denoiser | Δ WER | 95% CI | p |
|---|---|---|---|---|
| MedASR | `sherpa-gtcrn-simple` | **−0.0217** | [−0.0396, −0.0051] | 0.014 |
| MedASR | `dpdfnet2-onnx` | **−0.0707** | [−0.0960, −0.0491] | <0.001 |
| Whisper `base` | `sherpa-gtcrn-simple` | +0.0505 | [+0.0289, +0.0790] | 0.001 |
| Whisper `base` | `dpdfnet2-onnx` | +0.0018 | [−0.0082, +0.0108] | 0.707 |

The two chains agree to within 0.0012 WER, so `clean_audio`'s normalisation is
not what produces the gain. (An early 5-clip spot check suggested otherwise; at
n=5 that was noise.) Both denoisers improve MedASR and neither improves Whisper,
so the sign flip belongs to the decoder, not to one enhancement backend.
`dpdfnet2-onnx` was a robustness arm, not one of the two requested variants.

### The printed decision says `reject` — read why

`en_pilot/stats.py` states its rule in code: `adopt iff ci95_high <= 0.0 and
ci95_low >= -0.02`, documented as accepting "a default which is provably no worse
than no denoising, and costs at most 2 absolute WER points." MedASR + GTCRN has
`ci95_high = -0.0051` (passes) and `ci95_low = -0.0396` (fails the −0.02 bound).
The entire interval is below zero, so what fired is the lower bound, not evidence
of harm. The rule was left exactly as written; the scorer now emits a
`decision_note` recording this rather than letting the word `reject` stand alone.

### How much of the gain is "the model said nothing at all"

MedASR emits an empty hypothesis on some noisy clips, which scores WER 1.0. On
the shipped chain that is 64 times undenoised and 14 times with GTCRN. Dropping
the conditions where any arm went empty:

| Chain / denoiser | Δ WER, all 1,440 | Δ WER, empties excluded |
|---|---|---|
| shipped, `sherpa-gtcrn-simple` | −0.0229 | −0.0175 (n=1,372) |
| harness, `sherpa-gtcrn-simple` | −0.0217 | −0.0147 (n=1,376) |
| harness, `dpdfnet2-onnx` | −0.0707 | −0.0614 (n=1,376) |

So roughly a quarter of the GTCRN gain and an eighth of the dpdfnet gain is
empty-output rescue; the majority is ordinary accuracy on clips both arms
transcribed.

### Variant 1 fails outright on 3 of 1,440 conditions

GTCRN enhancement produced out-of-range audio on `sample-019_demand_street_02_5db`
and two `sample-029_demand_street_*` conditions, and `clean_audio` refused them
with `422 ENHANCE_FAIL: backend output exceeds PCM16 full scale`. Variant 2 has no
such failures. They are counted as failed transcripts in the numbers above rather
than dropped, and are listed in `failed_cells` in
`comparison-pipeline-gtcrn.json`. 3 in 1,440 is small, but it is a failure mode
variant 2 does not have.

## This is not a test of MedASR's medical accuracy

Clean-speech floor WER: **MedASR 0.2571** on the shipped chain (0.2620 on the
harness) vs Whisper `base` **0.1051**, on the same 40 clean bases.

`en-40-v1` is FLEURS — read Wikipedia sentences about global warming and
mortality rates. MedASR is fine-tuned on physician dictation and emits report
structure (`[EXAM TYPE]`, `[FINDINGS]`, `{period}`, `{colon}`); on its own bundled
radiology sample it produces a clean, correctly punctuated report. **Nothing here
measures MedASR in the domain it was built for**, which is also the domain this
product records. That test needs clinical audio with reference transcripts;
`demo/audio_in/` currently holds only synthetic tones.

## Pipeline/harness parity

The two chains differ in real ways — `demo/audio_clean.py` RMS-normalises to 0.1
before the backend, which `en_pilot/enhance.py` does not (it calls `run_backend`
directly); the demo hands MedASR a file path with `chunk_length_s=20` where the
harness passes a raw array; and `clean_audio` can refuse a clip. An early 5-clip
spot check made it look as though normalisation supplied most of the undenoised
arm's accuracy, so both chains were run over all 1,440 conditions:

| | Shipped chain | Harness chain |
|---|---|---|
| MedASR, no denoising | 0.4046 | 0.4007 |
| MedASR + GTCRN | 0.3816 | 0.3790 |
| **Paired Δ** | **−0.0229** [−0.0423, −0.0052] | **−0.0217** [−0.0396, −0.0051] |
| Clean floor | 0.2571 | 0.2620 |
| Empty hypotheses, none / GTCRN | 64 / 14 | 61 / 14 |

They agree. The n=5 reading was noise. The harness remains the right instrument
for isolating a backend; the shipped-chain numbers are the ones to quote for the
variants' accuracy, and are what the Bottom line uses.

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

`en-pilot-run/results/decision.csv` has been regenerated. Both tables are kept:
`en-pilot-decision-base-BEFORE-0db-fix.csv` and `...-AFTER-0db-fix.csv` in
`evidence/medasr-denoise-20261001/`.

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
  bootstrap. `--decoder` picks which MedASR sweep to score and
  `--context-decoder` which Whisper tier to report alongside.
- `en_pilot/transcribe_pipeline.py` — runs the real `clean_audio` →
  `stt_extract.transcribe(model="medasr")` chain over the whole matrix, so the
  headline numbers describe the shipped path and not a harness approximation
  of it. A clip `clean_audio` refuses is recorded as a failed transcript, not
  dropped.
- `en_pilot/stats.py` — the 0 dB fix above.

## Reproduce

```bash
# the two variants through demo/pipeline.py's own chain (the headline numbers)
.venv/bin/python -m en_pilot.transcribe_pipeline --workers 8
.venv/bin/python -m en_pilot.score_medasr --denoiser sherpa-gtcrn-simple \
    --decoder medasr-pipeline

# the backend in isolation, plus the dpdfnet robustness arm
.venv/bin/python -m en_pilot.transcribe_medasr --device cuda \
    --arms none,sherpa-gtcrn-simple,dpdfnet2-onnx
.venv/bin/python -m en_pilot.score_medasr --denoiser sherpa-gtcrn-simple
```

The shipped chain is 2,960 (clip, arm) pairs: 108 s of CPU enhancement on 8
workers, then 224 s of GPU decoding. The harness chain decodes 2,920 clips in
62 s, RTF **0.0027**. Decoding is greedy CTC — the `transformers` pipeline
default, and the mode the model card documents for its own evaluation. The
shipped `lm_6.kenlm` shallow-fusion LM was **not** used.

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
4. **The adoption rule was not re-derived.** `reject` on MedASR + GTCRN is
   `ci95_low = -0.0396 < -0.02` firing with the whole interval below zero, not a
   finding of harm.

# MedASR with and without the denoising layer (2026-10-01)

Two variants of the demo pipeline, differing in one setting, measured on the
same 40-base English corpus and the same 1,440-condition DEMAND noise matrix the
n=40 denoising pilot used:

| Variant | Command |
|---|---|
| 1. denoising **on** + MedASR | `python demo/pipeline.py --in <wav> --denoiser sherpa-gtcrn-simple --model medasr` |
| 2. denoising **off** + MedASR | `python demo/pipeline.py --in <wav> --denoiser none --model medasr` |

Raw artefacts (`comparison-gtcrn.json`, `comparison-dpdfnet.json`,
`per-base-gtcrn.csv`) are in `evidence/medasr-denoise-20261001/`, which is
gitignored like the rest of `evidence/`.

## Headline

**For MedASR, the denoising layer helps — the opposite of what it does to
Whisper.** Measured on the identical audio files, scored by the identical frozen
WER rule, paired and bootstrapped over whole bases:

| Decoder | Denoiser | Mean WER, 1,440 noisy conditions | Paired Δ vs no denoising | 95% CI | p |
|---|---|---|---|---|---|
| **MedASR** | none | **0.4007** | — | — | — |
| **MedASR** | `sherpa-gtcrn-simple` | **0.3790** | **−0.0217** | [−0.0396, −0.0051] | 0.014 |
| **MedASR** | `dpdfnet2-onnx` | **0.3390**¹ | **−0.0707** | [−0.0960, −0.0491] | <0.001 |
| Whisper `small` | none | 0.0788² | — | — | — |
| Whisper `small` | `sherpa-gtcrn-simple` | 0.1089² | **+0.0301** | [+0.0177, +0.0443] | <0.001 |

¹ dpdfnet was run as a robustness arm, not as one of the two requested variants.
² Whisper figures come from the pre-existing `transcripts-small.jsonl`, which is
an **interrupted** sweep covering 806 of 1,440 conditions and 23 of 40 bases.
They are context, not a like-for-like headline. See *Limits*.

Both denoisers improve MedASR and both degrade Whisper, so the sign flip is a
property of the decoder, not of one enhancement backend.

### The printed decision says `reject` — read the note

`en_pilot/stats.py`'s pre-declared rule is `adopt iff ci95_high <= 0 and
ci95_low >= -0.02`. The lower bound was written as a plausibility guard for
candidates expected to be neutral or harmful. MedASR + GTCRN improves by *more*
than that floor allows, so the rule fires `reject` even though the entire
confidence interval sits below zero. `decision_note` in `comparison-*.json`
states this; the rule itself was left untouched rather than retro-fitted to a
result it did not anticipate.

### How much of the gain is "the model said nothing at all"

MedASR emits an empty hypothesis on some noisy clips, which scores WER 1.0.
That happens 61 times on the undenoised arm, 14 times with GTCRN and 4 times
with dpdfnet. Dropping the 64 conditions where any arm went empty:

| Denoiser | Δ WER, all 1,440 | Δ WER, 1,376 with empties excluded |
|---|---|---|
| `sherpa-gtcrn-simple` | −0.0217 | −0.0147 |
| `dpdfnet2-onnx` | −0.0707 | −0.0614 |

So roughly a third of the GTCRN gain and an eighth of the dpdfnet gain is
empty-output rescue; the majority is ordinary accuracy on clips both arms
transcribed.

## MedASR is far worse than Whisper on this corpus — and that is expected

Clean-speech floor WER: **MedASR 0.2620** vs Whisper-`small` **0.095** (the
latter from the n=120 calibration run). On the shared noisy subset MedASR is
+0.33 WER behind Whisper-`small`.

`en-40-v1` is FLEURS — read Wikipedia sentences about global warming and
mortality rates. MedASR is fine-tuned on physician dictation (radiology,
internal medicine, family medicine) and emits report structure: `[EXAM TYPE]`,
`[FINDINGS]`, `{period}`, `{colon}`. On its own bundled radiology sample it
produces a clean, correctly punctuated report. **This measurement says nothing
about MedASR's accuracy on medical dictation**, which is what it was built for
and what this product records. It establishes the *denoising* answer on a real,
pre-registered corpus; the domain question needs clinical audio with reference
transcripts, and `demo/audio_in/` currently holds only synthetic tones.

## What changed in the project

- `demo/stt_extract.py`
  - `transcribe(..., model="medasr")` routes to `transcribe_medasr()`, which runs
    `google/medasr` through the `transformers` ASR pipeline on CUDA when
    available. Same result shape, same `stt_provenance` keys, same explicit mock
    fallback as the faster-whisper path (`strict=True` still propagates).
  - `medasr_detokenize()` resolves MedASR's dictation markup: `{period}` → `.`,
    `{colon}` → `:`, `{new paragraph}` → blank line, unrecognised `{directive}`
    dropped, `[SECTION]` headers and `</s>` removed. Without this every
    `{period}` tokenises to the word "period" and inflates WER.
  - The mock fallback was factored into `_mock_result()` so both real backends
    share one labelled fallback instead of duplicating it.
- `demo/pipeline.py` — `--model medasr` documented; `--denoiser` already
  selected the enhancement backend, so no change was needed there.
- `en_pilot/transcribe_medasr.py` — transcribes the **already-enhanced** pilot
  audio under `en-pilot-run/enhance-<backend>/`. Nothing is re-enhanced, so a
  MedASR row and a Whisper row for the same `(condition, backend)` are the same
  bytes decoded twice. Resume-safe; keeps `raw_text` beside the detokenised text.
- `en_pilot/score_medasr.py` — scores both variants with
  `en_pilot.score.word_error_rate` (the frozen normalisation and tokeniser) and
  `en_pilot.stats.bootstrap_paired` (10,000 iterations, seed 1729), plus the
  sentence-stratified robustness bootstrap. Cross-decoder numbers are restricted
  to conditions both decoders actually scored.

## Reproduce

```bash
.venv/bin/python -m en_pilot.transcribe_medasr --device cuda \
    --arms none,sherpa-gtcrn-simple,dpdfnet2-onnx
.venv/bin/python -m en_pilot.score_medasr --denoiser sherpa-gtcrn-simple
```

2,920 clips (both requested variants + clean floors) in **62 s** on one GPU,
RTF **0.0027**. Decoding is greedy CTC, the `transformers` pipeline default and
the mode the model card documents for its own evaluation; the shipped
`lm_6.kenlm` shallow-fusion LM was **not** used.

## Limits

1. **Not medical audio.** See above. The denoising conclusion is solid; the
   absolute WER is not a statement about MedASR in its own domain.
2. **The Whisper reference arm is partial.** `transcripts-small.jsonl` covers
   806 of 1,440 conditions across 23 of 40 bases, unevenly. Every cross-decoder
   number is computed on that shared subset so neither side gets an easier
   slice, but the n=23 base bootstrap is weaker than the MedASR n=40. Completing
   it is a single resume-safe command — `python -m en_pilot.transcribe --decoder
   small --streams 4` — but CTranslate2 cannot load CUDA on this host, so it
   runs CPU-only at roughly 6 hours.
3. **One corpus, one language.** English only; MedASR is an English model.
4. **The adoption rule was not re-run against its own threshold.** `reject` here
   is the plausibility floor firing on an improvement, not a quality verdict.

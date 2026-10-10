# Proposal: Raw ASR Decoding Presets

## Why

The transcript is the ceiling for everything downstream: a drug name the decoder never
writes cannot be repaired by fuzzy matching or an LLM (`CHANGELOG.md`: "the loss is the
decoder not emitting the word"). Today the Whisper decode in `stt_extract.transcribe` is
hard-wired to one setting - greedy (`beam_size=1`), `temperature=0.0` (no fallback), language
auto-detected - and the product is English-only. Two observed failures trace to that setting:

- A runaway decode on United-Syn-Med `SYN_024` (held-out sample): 136 hypothesis words for a
  19-word sentence, because `temperature=0.0` disables faster-whisper's own repetition fallback.
- Drug and brand names garbled by a greedy decode ("Illumatebarone", "ipamidal", "Glymperid").

Whether beam search, a pinned language, a temperature fallback or a short medical prompt
improves the *raw transcript* has not been measured. This change makes each setting a named,
reproducible preset and defines how they are compared, so the answer is evidence, not opinion.

## What Changes

- Named decoding presets for the faster-whisper path, selectable per run (`--decode`).
- The default preset reproduces today's behaviour exactly.
- The Whisper model is loaded once per process instead of once per clip.
- A reusable raw-transcript scorer (word error rate, exact sentences, drug/brand heard rate,
  runaway count) and a paired comparison between presets.
- A fixed evaluation protocol: choose on tuning sets, confirm once on a held-back test set.

## Out of Scope

- Fuzzy repair, the LLM validator, context repair, the second decoder pass, discharge notes.
  "Raw" means none of them: the score reads `transcript.json` only.
- MedASR decoding (CTC, no text prompt, no beam options here). `medasr-lm` stays opt-in.
- Changing the default preset. That is a decision taken from this change's results, separately.

## Impact

`base_pipeline/stt_extract.py`, `base_pipeline/run.py`, new `base_pipeline/decode_presets.py`,
new `clinical_eval/raw_asr_score.py`, new gold file for the held-back test set.

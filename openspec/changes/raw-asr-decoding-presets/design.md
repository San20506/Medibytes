# Design: Raw ASR Decoding Presets

## Presets

A preset is a dict of `faster_whisper.WhisperModel.transcribe` keyword arguments, keyed by name.
Presets are data in `base_pipeline/decode_presets.py`; no preset logic lives in `transcribe`.

| id | name | settings (on top of the shared guard `no_speech_threshold=0.6`, `compression_ratio_threshold=2.4`) |
|---|---|---|
| P1 | `greedy` | `beam_size=1`, `temperature=0.0` - byte-identical to the pre-change call |
| P2 | `beam5` | `beam_size=5`, `temperature=0.0` |
| P3 | `beam5-en` | P2 + `language="en"` |
| P4 | `beam5-en-fallback` | P3 + `temperature=[0.0,0.2,0.4,0.6,0.8,1.0]` |
| P5 | `beam5-en-fallback-prompt` | P4 + `initial_prompt="Medical dictation. Medication names, brand names and dosages."` |

Each preset adds exactly one change over its predecessor, so a difference between neighbours
is attributable to that one change.

## Raw means raw

Scoring reads `transcript.json["text"]` only. Runs for this change pass `--no-second-pass`,
`--no-validate` and no `--llm`. The extractor still runs (it always does) but plays no part.

## Metrics

- **WER** over the clip set: `sum(edit distance) / sum(reference words)`, with tokens
  `re.findall(r"[a-z0-9]+")` after lower-casing and removing `'` and `-` (hyphen -> space).
- **Exact sentences**: clips with zero edits.
- **Heard rate**: brand and substance gold names found in the transcript by
  `clinical_eval.syn_med_score.heard >= HEARD_RATIO`, the scorer the project already uses.
- **Runaway clips**: hypothesis words > 3x reference words. A count, reported beside WER, because
  one runaway clip can dominate a pooled WER (it was 120 of 139 errors in one half-sample).
- **Paired comparison** between two presets: per-clip error counts, clip-level bootstrap interval
  on the WER difference, and the number of clips better/worse. Pooled percentages alone are not
  used to call a winner.

## Evaluation protocol (anti-overfitting)

- Tuning sets: United-Syn-Med dev sample (40 clips) and held-out sample (40 clips), which earlier
  work has already looked at.
- Test set: 30 clips drawn at random without replacement from the 200-clip pool, none overlapping
  the tuning samples; gold written before any run. It is scored **once per preset, after** the
  tuning scores are recorded, and no setting is changed in response to it.

## Concurrency

The GPU has 8 GB. Large-v3 in float16 needs about 3.5 GB, so at most two decodes may be resident.
Every GPU run takes `flock /tmp/medibytes-gpu.lock`.

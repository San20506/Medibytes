# Tasks: Raw ASR Decoding Presets

## 1. Spec
- [x] 1.1 Proposal, design and delta spec written before any code.

## 2. Mechanism
- [x] 2.1 `base_pipeline/decode_presets.py`: the five presets as data, `get(name)` raising on unknown names.
- [x] 2.2 `stt_extract.transcribe(decode=...)`; default reproduces the pre-change call exactly; provenance records the preset and kwargs.
- [x] 2.3 Module-level `WhisperModel` cache keyed by (size, device, compute type).
- [x] 2.4 `run_stt_extract(decode=...)` and `run.py --decode`; reject non-default presets with `--model medasr`.
- [x] 2.5 Tests: default call kwargs, preset call kwargs, unknown preset, medasr rejection, model reuse.

## 3. Scoring
- [x] 3.1 `clinical_eval/raw_asr_score.py`: WER, exact sentences, heard rate, runaway clips, paired bootstrap comparison.
- [x] 3.2 Gold + sample list for the held-back test set saved in the repo.
- [x] 3.3 Tests for WER tokenisation, runaway detection, paired comparison.

## 4. Evaluation (one subagent per preset)
- [x] 4.1 P1 `greedy`   - [x] 4.2 P2 `beam5`   - [x] 4.3 P3 `beam5-en`   - [x] 4.4 P4 `beam5-en-fallback`   - [x] 4.5 P5 `beam5-en-fallback-prompt`
- [x] 4.6 Each runs the pipeline with its preset on both tuning sets, then once on the test set, and writes `runs/decode/<preset>/results.json`.
- [x] 4.7 Cross-check against the independent scratch experiment: initially disagreed on dev40; explained (raw mp3 without word timestamps vs cleaned audio with them) and reproduced exactly with matching settings. See results.md finding 6.
- [x] 4.8 Comparison report and a recommendation against the adoption criterion: `results.md`. Default preset NOT changed in this change.

# Delta Specification: Raw ASR Decoding Presets

## ADDED Requirements

### Requirement: Named decoding presets

The Whisper decode path MUST accept a named preset that fully determines the decoding keyword arguments, and the default preset MUST reproduce the pre-change call exactly.

#### Scenario: Default preserves behaviour

- **GIVEN** no preset is named
- **WHEN** a clip is transcribed with a faster-whisper model
- **THEN** `transcribe` is called with `beam_size=1`, `word_timestamps=True`, `temperature=0.0`, `no_speech_threshold=0.6`, `compression_ratio_threshold=2.4` and no `language` or `initial_prompt`

#### Scenario: Select a preset

- **GIVEN** `--decode beam5-en` (or `decode="beam5-en"`)
- **WHEN** a clip is transcribed
- **THEN** the call uses exactly the keyword arguments of that preset plus `word_timestamps=True`
- **AND** `transcript.json["stt_provenance"]["decode_preset"]` is `beam5-en` and records the keyword arguments used

#### Scenario: Reject an unknown preset

- **GIVEN** a preset name that is not defined
- **WHEN** a run starts
- **THEN** it fails before any audio is decoded, naming the valid presets

#### Scenario: Presets do not apply to MedASR

- **GIVEN** `--model medasr` and a non-default preset
- **WHEN** a run starts
- **THEN** it fails with a message that presets apply to faster-whisper models only, rather than silently ignoring the flag

### Requirement: One model load per process

The system MUST load a given Whisper model (size, device, compute type) once per process and reuse it.

#### Scenario: Reuse

- **GIVEN** two clips transcribed in one process with the same model and device
- **WHEN** the second clip is transcribed
- **THEN** `WhisperModel` is not constructed again

### Requirement: Raw transcript scoring

The scorer MUST compute word error rate, exact-sentence count, drug/brand heard rate and runaway-clip count from `transcript.json` text alone, and MUST NOT read extracted entities.

#### Scenario: Word error rate

- **GIVEN** a reference sentence and a transcript
- **WHEN** WER is computed
- **THEN** both are lower-cased, apostrophes removed, hyphens treated as spaces, and tokenised on `[a-z0-9]+`
- **AND** the edit distance is word-level Levenshtein, and set WER is total edits over total reference words

#### Scenario: Runaway decode is visible

- **GIVEN** a clip whose transcript has more than 3x the reference word count
- **WHEN** a run is scored
- **THEN** the clip is listed as a runaway and the set also reports WER with runaway clips counted, so a single clip cannot hide in the pooled figure

#### Scenario: Paired comparison

- **GIVEN** two scored runs over the same clips
- **WHEN** they are compared
- **THEN** the report gives per-clip better/worse/equal counts and a clip-level bootstrap interval for the WER difference
- **AND** a preset is called better only when that interval excludes zero

### Requirement: Evaluation protocol

Presets MUST be compared under a protocol that keeps the test set out of the selection.

#### Scenario: Test set is read once

- **GIVEN** the tuning scores for every preset are recorded
- **WHEN** the held-back test set is scored
- **THEN** each preset is scored on it once, and no preset definition changes afterwards in response

#### Scenario: Adoption criterion

- **GIVEN** the paired comparison against `greedy` on the test set
- **WHEN** deciding whether a preset should become the default
- **THEN** it MUST lower pooled WER with an interval excluding zero, MUST NOT increase the runaway-clip count, and MUST NOT lower the substance heard rate

# base_pipeline

The voice pipeline, on its own: **denoise -> ASR -> extraction with fuzzy drug
repair against a national drug reference**, with an optional local LLM.
No UI, no templates, no server.

```
audio (wav/mp3/m4a/mp4)
  | audio_clean.py   16 kHz mono, anti-aliased resample, RMS normalise, VAD ratio
  |   denoise.py + denoise_backends/   none (default) | dpdfnet2-onnx | sherpa-gtcrn-simple | noisereduce
  v
  | stt_extract.py   transcribe(): medasr -> small-int8 -> mock (never straight to mock)
  |   medasr_lm.py   opt-in kenlm beam search (measured worse; off by default)
  |   second decoder pass unions drug rows (and vitals, keyed on the reading) the primary missed
  v
  | stt_extract.py   normalize_text(), extract_entities(): vitals, drugs, allergies, negation
  |   drug_reference.py + assets/   2,542 substances / 71,872 brands (Common Drug Codes for India)
  |   fuzzy repair of misheard names, dose-aware, brands never create rows
  v
  | term_validate.py   LLM check of each fuzzy match (optional)
  | llm_extract.py     LLM-primary extraction / tidy (optional)
  | llm_client.py      any local OpenAI-compatible server
  | context_repair.py  --context-repair: LLM places misheard drug names from sentence context (RED rows only)
```

## Try it on new audio

```bash
python -m base_pipeline.run clip.wav folder_of_clips/ --out runs/try1
python -m base_pipeline.run noisy.wav --denoiser none --denoiser dpdfnet2-onnx   # compare arms
```

Per clip and denoiser arm you get `cleaned.wav`, `transcript.json`,
`entities.json` and `report.md` (heard -> chosen drug, candidates, reference
hit, colour, confidence). `summary.md` indexes the run; `review.csv` has a blank
`correct (y/n)` column per extracted item, so a manual pass ends with a tally.

Defaults are for repeatable manual testing: regex extraction, MedASR with the
automatic `small-int8` second pass (`--no-second-pass` to drop it), and the fuzzy
validator only if an LLM server is actually serving the model.

## Local LLM

One client, any server speaking the OpenAI-compatible API, held resident (no
per-call process start, unlike the old `ollama run` subprocess):

```bash
OLLAMA_KEEP_ALIVE=1h ollama serve && ollama pull llama3.2:3b      # default URL
llama-server -m model.gguf --port 8080                            # llama.cpp
python -m base_pipeline.run clip.wav --llm-url http://127.0.0.1:8080/v1 --llm   # extraction too
```

Env: `MEDIBYTES_LLM_URL`, `MEDIBYTES_LLM_MODEL`, `MEDIBYTES_LLM_KEY`. With no server
every LLM step is skipped and says so on the row; nothing fails.

## Tests

```bash
python -m pytest base_pipeline clinical_eval
```

## Known limits (from CHANGELOG.md; read before trusting a number)

- **English only.** MedASR destroys code-mixed audio (`BP 130 by 80` -> "BP aches or
  teaeth by a se"); use `--model small-int8` for Hindi/Tamil/Hinglish.
- The extractor is at 100% on *perfect* text; the loss is the decoder not emitting the word.
- Drug names are the weak axis; vitals are strong. Gold is agent-authored, not clinician-reviewed.
- Denoising has never beaten `none` on clean clinical audio. The default stays `none`.

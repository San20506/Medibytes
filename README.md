# MediBytes — voice to structured clinical notes

A research pipeline and its evaluation harnesses: **denoise → speech-to-text → clinical entity
extraction with fuzzy drug-name repair against a national drug reference**, plus an optional local
LLM. It is not a validated clinical tool. Every number in this repository comes from a command you
can rerun, and `docs/PIPELINE_EVAL_REPORT.md` says what each one does and does not establish.

The earlier investor-demo shell (`demo/`: templates, A4 exports, the Verify gate, `run_demo.py`) was
retired; its history is in git and `CHANGELOG.md`.

## Layout

```
base_pipeline/   the pipeline: one module per stage, one registry per choice
  presets.py       named approaches (a preset = a full stage selection)
  denoise_backends/  one adapter per denoiser arm, selected by BackendId
  asr.py extract.py stt_extract.py   speech-to-text, extraction, and the glue between them
eval/            denoiser/ASR evaluation harness (frozen contracts, paired statistics)
  en_pilot/        the English-only denoising WER pilot
clinical_eval/   clinical fact accuracy: component eval on the 7-clip set, gold files, scorers
app.py           Streamlit UI over base_pipeline
whisper_demo/    self-contained Whisper large-v3 demo service (imports nothing above; see below)
tests/           cross-package tests (import graph, the UI)
fixtures/audio/  small audio samples and the script they were recorded from
docs/            reports and plans; docs/archive/ is history only
openspec/        change proposals and specs
```

`evidence/` and `runs/` hold outputs and are gitignored; models and corpora live outside the
repository, under the data root (below).

## Pipeline

```
audio (wav/mp3/m4a/mp4)
  | audio_clean.py   16 kHz mono, anti-aliased resample, RMS normalise, VAD ratio
  |   denoise.py + denoise_backends/   none (default) | one of the arms below
  v
  | asr.py           transcribe(): medasr -> small-int8 -> mock (never straight to mock)
  |   medasr_lm.py   opt-in kenlm beam search (measured worse; off by default)
  |   second decoder pass unions drug rows (and vitals, keyed on the reading) the primary missed
  v
  | extract.py       normalize_text(), extract_entities(): vitals, drugs, allergies, negation
  |   drug_reference.py + assets/   2,542 substances / 71,872 brands (Common Drug Codes for India)
  |   fuzzy repair of misheard names, dose-aware, brands never create rows
  v
  | term_validate.py   LLM check of each fuzzy match (optional)
  | llm_extract.py     LLM-primary extraction / tidy (optional)
  | llm_client.py      any local OpenAI-compatible server
  | context_repair.py  LLM places misheard drug names from sentence context (RED rows only)
```

Fallbacks are never silent: a mock transcript says `is_mock`, an unavailable denoiser raises
`BACKEND_UNAVAILABLE`, and a skipped LLM step says why on the row.

## Setup

Python 3.12.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r base_pipeline/requirements.txt
pip install pytest
pip install -r requirements-ui.txt                     # only for app.py
pip install -r base_pipeline/requirements-server.txt   # only for the HTTP backend
```

`rnnoise` and `deepfilternet3` also need `pip install scipy` in this environment. The pinned
evaluation environment for `eval/` is `eval/requirements-eval.txt`; the test suite below does not need it.

Machine-local data (models, corpora, built native libraries) is found through
`$MEDIBYTES_EVAL_DATA_ROOT`, default `~/.local/share/medibytes-eval`. Nothing in the repository
hard-codes a home directory.

## Run

```bash
python -m base_pipeline.run clip.wav folder_of_clips/ --out runs/try1
python -m base_pipeline.run noisy.wav --denoiser none --denoiser dpdfnet2-onnx   # compare arms
python -m base_pipeline.run clip.wav --preset regex-only --model small-int8
streamlit run app.py
```

Per clip and denoiser arm you get `cleaned.wav`, `transcript.json`, `entities.json`, `verify.json`,
`discharge.json` and `report.md` (heard → chosen drug, candidates, reference hit, colour,
confidence). `summary.md` indexes the run and `review.csv` has a blank `correct (y/n)` column per
extracted item, so a manual pass ends with a tally. Every run also writes `manifest.json`: preset,
resolved config, the flags you overrode, git commit and dirty flag, python and package versions.

### Presets

An approach is a name: `--preset NAME` picks a full stage selection from `presets.py` (default
`baseline`, what you get with no flags); any flag you type overrides it.

| preset | changes from `baseline` | evidence |
|---|---|---|
| `baseline` | nothing; the validator runs only if an LLM server is up | the shipped chain |
| `regex-only` | term validator off, so no LLM is ever called | report arm A/A2 |
| `no-second-pass` | primary decoder only | `--no-second-pass` |
| `validator` | LLM term validator forced on (failures noted per row) | report arm B2 |
| `context-repair` | validator + LLM context repair of RED rows | report arm C3 |
| `llm` | LLM-primary extraction and the discharge note | not measured |
| `whisper-medasr` | SNR-gated denoise → Whisper → MedASR variant | CLI only |

A preset fixes flags, not code: the report's arms were run on code that has since changed (the
"guard" fix, the first context-repair version), so the names map to them approximately.

### HTTP backend

A thin FastAPI service over `run.run_clip`, so a run through the API cannot drift from the CLI.

```bash
python -m base_pipeline.server                 # http://127.0.0.1:8000/docs
uvicorn clinical_eval.server:app               # the same, plus POST /api/eval/run
```

| route | purpose |
|---|---|
| `GET /api/health` | which denoiser backends actually load, via each adapter's own `probe()` |
| `POST /api/audio/jobs` | multipart upload + `denoiser`, `model`, `use_llm`; returns `202` and a `job_id` |
| `GET /api/audio/jobs/{id}` | status, clean meta, transcript, entities, timings, and `observed` |
| `DELETE /api/audio/jobs/{id}` | drop a job record |
| `POST /api/eval/run` | (`clinical_eval.server` only) the 7-clip component eval and its score |

`observed` reports `actual_denoiser`, `stt_engine`, `stt_provenance` (including `is_mock`) and a
`mismatches` list when what ran differs from what was requested: without it a scored run can
quietly be a fallback measuring nothing. Job outputs go to `runs/_server_jobs/`
(`MEDIBYTES_JOB_ROOT` to change it).

### Local LLM

One client, any server speaking the OpenAI-compatible API, held resident:

```bash
OLLAMA_KEEP_ALIVE=1h ollama serve && ollama pull llama3.2:3b      # default URL
llama-server -m model.gguf --port 8080                            # llama.cpp
python -m base_pipeline.run clip.wav --llm-url http://127.0.0.1:8080/v1 --llm   # extraction too
```

Env: `MEDIBYTES_LLM_URL`, `MEDIBYTES_LLM_MODEL`, `MEDIBYTES_LLM_KEY`. With no server every LLM step
is skipped and says so on the row; nothing fails.

## Denoiser arms

Each arm is `denoise_backends/<id>.py`, an entry in `denoise.BackendId`, and a frozen method config
in `eval/config/methods/<id>.json`. The default is `none`: denoising has never beaten it on clean
clinical audio. An arm whose runtime or model is missing fails with `BACKEND_UNAVAILABLE`; it never
falls back to another backend.

| id | runtime | notes |
|---|---|---|
| `none`, `noisereduce` | none / `noisereduce` | identity control; spectral-gating control |
| `dpdfnet2-onnx`, `sherpa-gtcrn-simple` | `onnxruntime`, `sherpa-onnx` | model under the data root |
| `rnnoise` | native library, `scipy` | `denoise_backends/build_rnnoise.sh` builds it with pinned `-march=x86-64-v2`; the adapter pins its SHA-256 |
| `deepfilternet3` | CPython 3.11 venv at `./venv`, plus `scipy` here | `build_deepfilternet3.sh` builds the venv from the hashed lock |
| `speex-denoise` | built SpeexDSP | under `methods/speexdsp` in the data root |
| `spectral-subtraction`, `speechbrain-metricgan-plus`, `voicefixer-mode0`, `nemo-se-den-sb-16k` | one package each | `denoise_backends/requirements-<id>.txt`; install into a venv dedicated to the arm |
| `percepnet`, `speecht5`, `cmsis-dsp` | none | catalog identities with no adapter |

Run a heavy arm under its own interpreter, e.g.
`<arm venv>/bin/python -m base_pipeline.run clip.wav --denoiser nemo-se-den-sb-16k`.

## Evaluation

- `python -m eval.harness {validate-corpus,preflight,run,verify-run,compare}`: the frozen
  denoiser comparison: paired deltas against the `none` baseline with bootstrap intervals and Holm
  correction. Needs `$MEDIBYTES_EVAL_DATA_ROOT`.
- `eval/en_pilot/run_study.sh`: the English-only denoising WER pilot, one process per backend.
- `python -m clinical_eval.{prepare,run,score}`: clinical fact accuracy on the 7-clip set;
  `clinical_eval.syn_med*` and `clinical_eval.raw_asr_score` cover the synthetic drug-name set and
  raw decoding presets. Gold is agent-authored, not clinician-reviewed.
- Reports: `docs/PIPELINE_EVAL_REPORT.md` (the 30-clip set and the LLM stages),
  `docs/DENOISING_PILOT_20260927.md` and `eval/reports/` (the 12-row denoiser comparison).

## Whisper large-v3 demo

`whisper_demo/` is a separate, self-contained service: audio -> Whisper large-v3 -> extraction ->
optional LLM discharge note. No denoiser, MedASR, second pass or mock. It has its own
requirements and imports nothing from the packages above (its extraction code is a fork; see the
header of each file).

```bash
pip install -r whisper_demo/requirements.txt -r whisper_demo/requirements-gpu.txt   # gpu file: Linux CUDA only
python -m whisper_demo                    # http://127.0.0.1:8001/docs
curl -F file=@clip.mp3 -F language=auto http://127.0.0.1:8001/api/jobs
curl http://127.0.0.1:8001/api/jobs/<job_id>
```

`GET /api/health` reports the device the model settled on (`cuda float16`, or `cpu int8` with the
reason). Env: `WHISPER_DEMO_CORS_ORIGINS` (comma list; none by default), `WHISPER_DEMO_JOB_ROOT`,
`WHISPER_DEMO_PRELOAD=0`, `WHISPER_DEMO_PORT`. LLM steps use `MEDIBYTES_LLM_URL/MODEL/KEY`.

## Tests

```bash
python -m pytest tests base_pipeline clinical_eval eval whisper_demo -q -rs
```

An arm's tests skip, with the probe's reason shown by `-rs`, where that arm cannot run (its package,
model or built library is missing), so a fresh clone passes. Run them under the arm's own interpreter
to exercise it, and set `MEDIBYTES_REQUIRE_ARMS=<ids|all>` on a machine that has the arm so an
unavailable one fails instead of skipping.

## Known limits

Read these before trusting a number (details in `CHANGELOG.md` and the reports).

- **English only for the default decoder.** MedASR destroys code-mixed audio (`BP 130 by 80` →
  "BP aches or teaeth by a se"); use `--model small-int8` for Hindi/Tamil/Hinglish.
- The extractor is at 100% on *perfect* text; the loss is the decoder not emitting the word.
- Drug names are the weak axis; vitals are strong.
- Denoising has never beaten `none` on clean clinical audio.
- Clips 001–007 of the 30-clip set were used to develop the extractor: quote the held-out clips (008–030) only.

## Troubleshooting

| symptom | fix |
|---|---|
| mp3 fails to decode | `pip install imageio-ffmpeg soundfile` (bundled ffmpeg, no admin) |
| `ModuleNotFoundError: faster_whisper` | install it, or use `--model mock` (rows are labelled `is_mock`) |
| `BACKEND_UNAVAILABLE` | the arm's runtime or model is missing or does not match its pin; see the message |
| LLM steps say "skipped" | no server answering, or the model is not served; see Local LLM |

## Docs

`docs/IDEA.md` product definition · `docs/ARCHITECTURE.md` selected methods ·
`docs/PIPELINE_ARCHITECTURE_REPORT.md` the production architecture · `docs/PRODUCTION_PLAN_L7.md`
the production plan · `docs/EDGE_CASES.md` · `docs/HMS_INTEGRATION_REPORT.md` · `docs/archive/` history.

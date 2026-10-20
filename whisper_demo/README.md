# whisper_demo

Self-contained demo service: audio -> **Whisper large-v3** -> clinical extraction -> optional LLM
discharge note. Nothing here imports `base_pipeline`, `clinical_eval` or `eval`; copy the folder
anywhere and run it.

## Run

```bash
pip install -r requirements.txt            # CPU is enough to start; add requirements-gpu.txt for CUDA
python -m whisper_demo                     # http://127.0.0.1:8001/docs
```

## API

| route | purpose |
|---|---|
| `GET /api/health` | model, device, compute type, fallback reason, LLM reachability |
| `POST /api/jobs` | multipart `file`, `language` (`auto`/`en`/`hi`/`ta`), `use_llm`, `validate_terms`, `context_repair`; returns `202` and `job_id` |
| `GET /api/jobs/{id}` | status, transcript (with word times), entities, verify-backed discharge note, timings, `observed` |
| `DELETE /api/jobs/{id}` | drop a job record |

Outputs are also written to `jobs/<job_id>/` (`transcript.json`, `entities.json`, `verify.json`,
`discharge.json`); `jobs/` is gitignored.

## Behaviour worth knowing

- **No mock, no fallback decoder.** If audio cannot be decoded or large-v3 fails, the job is `failed`
  with the reason. Silence, under 0.6 s, and under 5 % voiced frames are rejected before the model
  runs, because large-v3 answers silence with "Thank you." / "Thank you for watching."
- **Device.** CUDA float16 only if a real warm-up decode succeeds (model construction alone does not
  prove cuBLAS loads); otherwise CPU int8, which is far too slow for a live demo. `/api/health` says
  which one you got.
- **Language.** `auto` lets Whisper detect per clip; forcing `en`/`hi`/`ta` is available.
- **LLM steps** (`use_llm`, term validation, context repair) need an OpenAI-compatible server:
  `MEDIBYTES_LLM_URL`, `MEDIBYTES_LLM_MODEL`, `MEDIBYTES_LLM_KEY` (or `OPENAI_API_KEY`). Without one
  each step is skipped and says so on the output.
- **Known limitation (inherited).** The LLM discharge note can list a *denied* symptom under
  `present_illness` with no negation flag ("Patient denies chest pain" -> `chest pain`). The
  entities and the allergy list do carry `negated`. `base_pipeline` behaves identically, and the
  forked code is deliberately left unpatched; do not present the note's symptom list as
  negation-aware.
- **Forked code.** `extract.py`, `drug_reference.py`, `context_repair.py`, `vitals_kind.py`,
  `llm_client.py`, `llm_extract.py`, `term_validate.py`, `verify.py`, `assets/` and
  `discharge_schema.json` are copies of `base_pipeline/` at `c68d7a9`; each `.py` says so in its first
  line. They are not kept in sync.

## Env

`WHISPER_DEMO_CORS_ORIGINS` (comma-separated; default none), `WHISPER_DEMO_JOB_ROOT`,
`WHISPER_DEMO_PRELOAD` (`0` skips loading at startup), `WHISPER_DEMO_MAX_UPLOAD_BYTES`,
`WHISPER_DEMO_HOST`, `WHISPER_DEMO_PORT`.

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
| `GET /api/health` | model, device, compute type, fallback reason, local-LLM reachability, `form_llm` (provider, model, `key_present`; never the key) |
| `POST /api/jobs` | multipart `file`, `language` (`auto`/`en`/`hi`/`ta`), `template` (`discharge` default, or `admission`), `use_llm`, `validate_terms`, `context_repair`; returns `202` and `job_id` |
| `GET /api/jobs/{id}` | status, transcript (with word times), entities, verify-backed discharge note, timings, `observed` |
| `DELETE /api/jobs/{id}` | drop a job record |

Outputs are also written to `jobs/<job_id>/` (`transcript.json`, `entities.json`, `verify.json`,
`discharge.json`); `jobs/` is gitignored.

## Admission form (`template=admission`)

Fills a structured admission note from the transcript and returns it as `form` (also written to
`jobs/<job_id>/admission.json`; the discharge note is not produced for this template). Fields are in
`admission_schema.json` (patient, chief complaints with duration, HPI, **pertinent negatives**, past
medical/surgical history, drug history, allergies, family/personal history, vitals, examination,
provisional diagnosis, investigations, admission plan with orders).

- **The transcript is sent to OpenAI** (`WHISPER_DEMO_FORM_URL`, default `https://api.openai.com/v1`,
  model `WHISPER_DEMO_FORM_MODEL`, default `gpt-4.1-mini`, key from `OPENAI_API_KEY`). Every result records
  `provenance.sent_to`. Use synthetic audio for demos; do not send real patient audio until data-handling
  (DPDP Act) is cleared.
- **Heard-only.** The model proposes values; a deterministic gate keeps a value only if its
  `source_sentence` is quoted verbatim from the transcript, every number in it was spoken, a drug name
  is a real drug that appears in the cited sentence, and a symptom's own clause is not negated. Anything
  else is dropped and counted in `provenance.dropped_unproven`. `status: "verified"` means *quoted from the
  transcript*, not clinician-reviewed. A field not dictated is `null`.
- **Denials.** A denied symptom ("denies cough") always ends in `pertinent_negatives`, even if the model
  filed it as a complaint; a denied history ("denies diabetes") is dropped.
- **What the gate also does.** A value the model gave without a source gets one from the first transcript
  sentence that holds its numbers and words (marked `needs_review`); polarity and drug role (already taking
  vs newly ordered) are read from the whole sentence; sex, mode and no-known-allergies need cue words; a
  verified drug the model left out is added to the orders or drug history by its sentence cue (`needs_review`,
  dose and unit only); an explicit "no known allergies" sets that field. Every drop is listed with its reason
  in `provenance.drops`.
- **Using a local model.** `WHISPER_DEMO_FORM_URL=http://localhost:11434/v1 WHISPER_DEMO_FORM_MODEL=llama3.2:3b
  OPENAI_API_KEY=anything` keeps the transcript on the machine. A 3B model is grounded but misses fields (see the
  CHANGELOG); the gate protects precision, not recall.
- **No silent fallback.** With no key, an HTTP error, or unusable JSON after 3 attempts the job still
  succeeds with an all-null form and `provenance.llm_engine = "form-skipped (reason)"`.
- Not measured: there is no scorer, so recall on real dictation is unknown.

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
- **Known limitation (inherited, discharge template only).** The LLM discharge note can list a *denied* symptom under
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
`WHISPER_DEMO_HOST`, `WHISPER_DEMO_PORT`, `WHISPER_DEMO_FORM_MODEL`, `WHISPER_DEMO_FORM_URL`.

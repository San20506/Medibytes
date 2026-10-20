# MediBytes: voice to structured clinical notes

A research pipeline, not a validated clinical tool: **audio -> Whisper large-v3 -> clinical entity
extraction with fuzzy drug-name repair against a national drug reference -> optional local-LLM
discharge note**, served over HTTP. Nothing is mocked: a failed decode or transcription marks the job
`failed` with the reason.

The earlier MedASR + denoiser pipeline, its evaluation harness and the `clinical_eval` scorers were
retired in the Whisper rebuild. Their code is at git tag `archive/pre-whisper-rebuild`, and the reports
written for them are in `docs/archive/previous-pipeline/`. Numbers in those reports describe that
pipeline, not this one.

## Layout

```
whisper_demo/    the pipeline and its FastAPI service (self-contained; see its README)
tests/           repo-level tests (every import in whisper_demo resolves)
fixtures/audio/  audio samples; only the luvvoice.com-*.mp3 clips are speech
docs/archive/    history only
```

`runs/` and `evidence/` are gitignored outputs of the retired pipeline.

## Setup and run

Python 3.12.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r whisper_demo/requirements.txt
pip install -r whisper_demo/requirements-gpu.txt     # Linux + CUDA only
python -m whisper_demo                               # http://127.0.0.1:8001/docs
curl -F file=@clip.mp3 http://127.0.0.1:8001/api/jobs
curl -F file=@clip.mp3 -F template=admission http://127.0.0.1:8001/api/jobs   # admission form (sends the transcript to OpenAI)
```

`GET /api/health` reports the device the model settled on: `cuda float16`, or `cpu int8` with the
reason (far too slow for a live demo). Environment variables and API details are in
`whisper_demo/README.md`.

## Tests

```bash
python -m pytest tests whisper_demo -q -rs
```

The real-model test runs only where `Systran/faster-whisper-large-v3` is in the Hugging Face cache.

## Known limits

- `template=admission` sends the transcript to OpenAI; see `whisper_demo/README.md` before using real audio.
- The LLM discharge note can list a denied symptom under `present_illness` with no negation flag;
  entities and allergies do carry `negated`.
- Large-v3 answers silence and tones with "Thank you." / "Thank you for watching."; silent and
  sub-0.6 s clips are rejected before the model runs, but quiet non-speech noise is not.
- No accuracy figure for this pipeline is recorded here: the scorers and gold files went with the
  retired pipeline.

# Archive

Kept for history. Nothing here describes the current layout; start from the repository `README.md`.

`previous-pipeline/` holds the reports and plans written for the MedASR + denoiser pipeline (`base_pipeline/`,
`eval/`, `clinical_eval/`), retired in the Whisper rebuild. The code is at tag `archive/pre-whisper-rebuild`.

| file | what it is |
|---|---|
| `SESSION-SUMMARY.md` | a session summary on turning the denoising evidence into a production audio pipeline, with a "Path to ~95%" section on which metrics to target |
| `2026-10-05-best-configuration-spec.md` | the highest-scoring configuration measured on the 7-clip dataset, specified for reproduction; written against the old `demo/` layout (`demo/pipeline.py`, `demo/stt_extract.py`), now retired |
| `new.md` | an earlier draft of `../PIPELINE_ARCHITECTURE_REPORT.md`: 99% of its lines appear in the report, which has since grown; only its stage headings are numbered differently |

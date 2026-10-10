# MediBytes denoising comparison pilot-15-v1-b1

Single-backend rows only. Candidate metrics are paired deltas against the verified `none` baseline over 15 base groups (10,000 resamples, seed 1729, Holm family count 12).

| CSV row | Backend | Class | Disposition | Outcome | Normalized WER delta (95% CI) |
|---:|---|---|---|---|---:|
| 2 | rnnoise | realtime_laptop | run_required | inconclusive | not_comparable |
| 3 | deepfilternet3 | realtime_laptop | run_required | observed | not_comparable |
| 4 | dpdfnet2-onnx | realtime_laptop | run_required | observed | not_comparable |
| 5 | sherpa-gtcrn-simple | realtime_laptop | run_required | observed | not_comparable |
| 6 | percepnet | realtime_laptop | blocked_prerequisite | blocked_prerequisite | not_comparable |
| 7 | nemo-se-den-sb-16k | research_gpu | run_required | invalid | not_comparable |
| 8 | speecht5 | research_gpu | not_comparable | not_comparable | not_comparable |
| 9 | speex-denoise | realtime_laptop | run_required | inconclusive | not_comparable |
| 10 | speechbrain-metricgan-plus | offline_laptop | run_required | inconclusive | not_comparable |
| 11 | voicefixer-mode0 | offline_laptop | run_required | inconclusive | not_comparable |
| 12 | cmsis-dsp | target_only | blocked_target | blocked_target | not_comparable |
| 13 | spectral-subtraction | realtime_laptop | run_required | invalid | not_comparable |

Blocked/not-comparable/target-blocked rows are not assigned quality scores. This 15-base non-clinical smoke corpus cannot establish production medical performance, safety, or a default backend.

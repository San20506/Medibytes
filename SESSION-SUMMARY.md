# Session Summary: Denoising Evidence → Production Audio Pipeline

**Scope note on "95% quality" below**: no single number in this session
means "95%." Treated it as shorthand for *transcription + routing
correctness*, and propose two concrete metrics to actually target (WER and
false-discard rate) — see "Path to ~95%" for why neither can be reported
with confidence yet, and what closes that gap.

## 1. What was done

### A. Closed evidence gaps in the existing denoising research (Medibytes repo)

- **DeepFilterNet4 does not exist.** Checked `Rikorose/DeepFilterNet`'s
  actual release history: latest tag is `v0.5.6` (2023), same version
  already tested as `deepfilternet3`. `docs/ARCHITECTURE.md`'s "WINNER
  quality" citation was unverifiable.
- **LL-SDR (new 2026 candidate) tested harmful on this corpus.** 5-sample
  smoke test: mean ΔSI-SDR ≈ **-2.43 dB**, worst on the echo/code-mix
  condition (reconstruction correlation -0.29). Caught and fixed a real
  measurement trap along the way (the model's declared `get_delay()`
  looked like a needed compensation but wasn't — true alignment is lag≈0).
- **IndicConformer/IndicWhisper: MIT-licensed, ready, blocked by
  environment, not by anything about the models.** Confirmed license
  clean for proprietary use. 4 install attempts across 2 Python versions
  and 2 filesystems all failed identically (`Disk quota exceeded`) —
  narrowed to a sandbox-level write cap, not the workstation's actual
  disk (116GB genuinely free).
- **Found (not built) a decision-grade English WER study** already sitting
  in the repo from an earlier session (`en_pilot/`, n=40 pre-registered,
  bootstrap CI, Holm correction): **rejects both remaining denoising
  candidates** — `dpdfnet2-onnx` (Δwer +0.001, CI crosses zero, p=0.87) and
  `sherpa-gtcrn-simple` (Δwer +0.044, entirely worse, p=0.03). Verified it
  was real, tested (23/23 passing), and legitimate before including it.

### B. Integration decision (Medibytes repo, `feature/denoise-stt-evidence-integration`, pushed)

- **No denoiser integrated.** `none` (pass-through) confirmed correct —
  now backed by two independent lines of evidence (LL-SDR smoke test +
  the n=40 en_pilot study), not just the original n=4/n=8 pilot.
- **STT default bumped `tiny-int8` → `small-int8`.** The only change with
  clear positive evidence: `tiny`/`base` produce wrong-script or
  hallucinated output on Hindi/Tamil/code-mix, independent of denoising.

### C. Full production pipeline build (separate `Product` repo, `feature/audio-diagnostic-pipeline-task1`, pushed)

Implemented all 7 tasks of `docs/audio-diagnostic-routing-plan.md` — a
FastAPI audio-diagnostic-and-routing service (job API → bounded FFmpeg
ingestion → VAD/clipping/SNR diagnostics → quality-based routing →
optional GTCRN enhancement → faster-whisper ASR → supervised, crash/hang-
recoverable execution). **118/118 tests passing.**

**The one finding worth remembering**: a real end-to-end HTTP test against
real speech (not a unit test) caught a genuine bug in Task 3's VAD adapter
— missing a 64-sample context prefix that Silero's own reference wrapper
requires, which made it score real English speech at **0.001 mean
probability**, indistinguishable from silence. Fixed; verified the fix
(0.001 → 0.83 mean probability); re-ran the same real end-to-end check —
completed job, transcript exactly matching the known reference text.

DPCRN intentionally not implemented (plan's own "keep disabled until
parity passes" guidance). `AUDIO_PIPELINE_ENABLED` stays `false` by
default — see `docs/audio-pipeline-operations.md` in the Product repo for
the full "what has and hasn't been verified" accounting.

## 2. What we actually know about quality today

| Measurement | Value | Confidence |
|---|---|---|
| Denoising WER impact (English, n=40) | No candidate helps; both rejected | **High** — pre-registered, bootstrapped |
| Denoising WER impact (real noise, matrix-v2) | Every backend made WER worse | **High** — 1,440 scored conditions |
| Hindi/Tamil decoder: `base` vs `small` | `base` = wrong script/garbage; `small` = usable | **High** — direct comparison, real audio |
| GTCRN enhancement latency | 420–750ms p50/p95 on 2–8s clips | **Medium** — n=7, but consistent |
| Pipeline WER (smoke corpus) | ~10–15% on clean English | **Low** — n=4, unnormalized metric (see below) |
| False-discard rate | 0/7 smoke conditions | **Very low** — no near-boundary case tested |
| Routing thresholds (20dB/5dB/5%) | Untouched "engineering defaults" | **None** — never calibrated |

## 3. Path to ~95% — ranked by evidence-to-effort ratio

1. **Fix WER measurement before trusting any quality number.** The
   pipeline's own eval script uses unnormalized word matching (no
   punctuation/number-word normalization) — the reported 10–15% WER on
   clean speech is inflated by measurement artifact, not necessarily real
   error. *You cannot know today whether you're at 80% or 95% quality
   because the ruler itself is wrong.* Effort: low (reuse Medibytes'
   `eval/metrics.py::frozen_normalize`-style normalization). This is the
   single highest-leverage next step — everything else is calibrated
   against a number we don't trust yet.
2. **Build a real calibration corpus and run Task 7's evaluation for
   real.** Everything measured so far is smoke-scale (n=1–5 per
   condition). The plan's own release gates require bootstrap-uncertainty
   WER on a real held-out set across languages/noise types, plus a
   reviewed false-discard set — none of that exists yet. This is the
   actual gate between "looks plausible" and "95%, defensibly."
3. **Upgrade the ASR decoder tier, not the denoiser.** This session's own
   evidence rules out denoising as a lever (rejected at n=40, p=0.87/0.03)
   but shows decoder tier is real and large (Hindi/Tamil: garbage → usable
   from one size bump). Testing `small`/`medium` on English, and
   IndicConformer/IndicWhisper on Hindi/Tamil, is the evidence-backed next
   move — not further denoiser tuning.
4. **Unblock IndicConformer/IndicWhisper.** License-clear, ready to test,
   blocked only by this session's sandbox disk-write cap. Running the
   install outside that sandbox (or with the cap raised) is a low-effort
   unlock specifically for Hindi/Tamil quality, where generic Whisper is
   currently the weak point.
5. **Stop investing in denoising further** unless the calibration corpus
   (item 2) surfaces a specific condition where it helps. Two adequately-
   powered studies now agree it doesn't, and one candidate made things
   catastrophically worse (+1264% WER on real noise). Redirecting that
   effort to items 1–4 has a better payoff.
6. **Calibrate the routing thresholds and close the false-discard blind
   spot.** The 20dB/5dB SNR bands and 5% speech-fraction discard rule are
   still the plan's original untouched defaults. The specific edge case
   the plan itself warns about — a brief valid sentence inside a long
   silent recording — was never actually tested near the 5% boundary.
7. **GTCRN latency** (420–750ms vs. upstream's 0.07 RTF claim) is real but
   lower priority than the above, since current evidence says enhancement
   isn't earning its cost anyway.
8. **Operational hardening** (retention sweep not yet implemented, no
   accelerator path exercised, no competing-workload measurement) — needed
   for production readiness but orthogonal to transcription accuracy.

## References

- Medibytes: `CHANGELOG.md`, `openspec/changes/denoise-pilot-b2-gap-closure/`,
  `openspec/changes/denoise-pilot-en-only/`, `evidence/README.md`
- Product: `docs/audio-pipeline-operations.md`, `docs/audio-diagnostic-routing-plan.md`,
  commit history on `feature/audio-diagnostic-pipeline-task1`

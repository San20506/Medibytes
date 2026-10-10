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

**Updated after actually measuring** (2026-09-29, second pass): found real
test data already sitting in the eval data root from an earlier session (40
real English bases with reference transcripts + a real DEMAND-noise matrix,
`~/.local/share/medibytes-eval/corpus/en-40-v1` + `matrix-en-40`) — no
need to build a smoke corpus, a real one already existed. Fixed the WER
metric's normalization bug, then ran 120 real files × 3 arms = 360
evaluations against the actual Product pipeline. Full detail in the Product
repo's `evidence/en40-calibration-2026-09-29/`.

| Measurement | Value | Confidence |
|---|---|---|
| Denoising WER impact (English, n=40, en_pilot) | No candidate helps; both rejected | **High** — pre-registered, bootstrapped |
| Denoising WER impact (real noise, matrix-v2) | Every backend made WER worse | **High** — 1,440 scored conditions |
| **Pipeline routing WER impact (n=120, real corpus)** | **`dynamic_routing` 35% relatively worse than always-direct-ASR** (0.201 vs 0.149 weighted-mean WER on 80 noisy conditions) | **High** — real corpus, real routing code, not a fake |
| Clean-speech pipeline WER (n=40, normalized) | **0.095** | **High** — real corpus, fixed metric |
| Hindi/Tamil decoder: `base` vs `small` | `base` = wrong script/garbage; `small` = usable | **High** — direct comparison, real audio |
| GTCRN enhancement latency | 500–820ms p50/p95 on 2–8s clips (n=120) | **Medium-high** |
| False-discard rate | 0/120 real conditions | **Medium** — still no near-5%-boundary case tested |
| Routing thresholds (20dB/5dB/5%) | Untouched "engineering defaults" — **and now shown actively harmful**, not just uncalibrated | **High** |

## 3. Path to ~95% — ranked by evidence-to-effort ratio

**Done, this pass**: WER measurement fixed (was counting "U.N." vs "un" as
an error); a real, properly-sized corpus was found and used, not built
from scratch — 4x larger than the plan's own held-out-set ambition needs
to start being useful. Both items that were "#1 highest leverage" in the
previous version of this list are now closed.

1. **Fix the routing thresholds — this is now the single highest-priority
   item, upgraded from #6 by the measurement above.** The pipeline's own
   `dynamic_routing` is *choosing* to enhance in exactly the conditions
   where enhancement hurts, making its real-world WER measurably worse
   than the trivial "always use direct ASR" baseline. This isn't a
   hypothetical gap anymore — it's quantified, real-corpus harm. Fix:
   raise the SNR bands in `routing.py` (or disable `enhance_gtcrn` as a
   selectable route until re-calibrated) so `direct_asr` becomes the
   default near-everywhere GTCRN is currently chosen. Not applied yet —
   flagged for confirmation since it changes already-committed default
   behavior.
2. **Upgrade the ASR decoder tier, not the denoiser.** Now confirmed
   twice at real scale: denoising doesn't help (en_pilot n=40, and now
   this n=120 pipeline-level run) but decoder tier does (Hindi/Tamil:
   garbage → usable from one size bump). Testing `small`/`medium` on
   English against this same 40-base corpus, and IndicConformer/
   IndicWhisper on Hindi/Tamil, is the evidence-backed next move.
3. **Unblock IndicConformer/IndicWhisper.** License-clear, ready to test,
   blocked only by a sandbox disk-write cap, not the models. Still
   unresolved.
4. **Extend the n=120 real-corpus run to a genuine held-out split with
   bootstrap CIs**, and add a Hindi/Tamil arm using the pilot-15-v1
   corpus's own native-script references (already sitting in
   `evidence/pilot-b1/test-data/reference-transcripts.csv` — likely
   another "check the folders before building" case). Needed to move from
   "real and large enough to trust a decision" to "meets the plan's own
   section-11 release-gate bar."
5. **Test a boundary case for the false-discard risk.** 0/120 real
   conditions were false-discarded, but none was built to sit near the 5%
   speech-fraction threshold — the plan's own named risk (a brief valid
   sentence inside a long silent recording) still hasn't been tested at
   the boundary.
6. **Stop investing in denoising further.** Now three independent studies
   agree (en_pilot n=40, matrix-v2, and this session's n=120 pipeline
   run) — redirect effort to items 1–4.
7. **GTCRN latency** (500–820ms vs. upstream's 0.07 RTF claim) — real, but
   lower priority than fixing the routing that's choosing to pay it for a
   quality loss.
8. **Operational hardening** (retention sweep, accelerator path,
   competing-workload measurement) — orthogonal to transcription accuracy.

## References

- Medibytes: `CHANGELOG.md`, `openspec/changes/denoise-pilot-b2-gap-closure/`,
  `openspec/changes/denoise-pilot-en-only/`, `evidence/README.md`
- Product: `docs/audio-pipeline-operations.md`, `docs/audio-diagnostic-routing-plan.md`,
  commit history on `feature/audio-diagnostic-pipeline-task1`

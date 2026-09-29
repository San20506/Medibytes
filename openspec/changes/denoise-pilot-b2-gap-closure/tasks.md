# Tasks: Denoise Pilot Gap Closure

## 1. OpenSpec Artifacts

- [ ] 1.1 Review `proposal.md`, `design.md`, and `specs/denoise-evaluation/spec.md` against this task list; resolve any contradiction before changing runtime code.
- [ ] 1.2 Commit the four OpenSpec artifacts as the first change, with no schema/config/code edits bundled in.

## 2. DeepFilterNet4 Verification (blocking, do first) — RESOLVED 2026-09-28

- [x] 2.1 Checked `Rikorose/DeepFilterNet`'s release history directly via `gh api repos/Rikorose/DeepFilterNet/releases`: latest tag is `v0.5.6` (published 2023-08-31), the exact version already pinned in `eval/config/methods/deepfilternet3.json`. Last commit to the repo was 2024-09-25 (`gh api repos/Rikorose/DeepFilterNet/commits?since=...`). `gh api search/code?q=DeepFilterNet4+repo:Rikorose/DeepFilterNet` returns `total_count: 0`. README has no mention of a "DeepFilterNet4"/"DF4"/v4 architecture.
- [x] 2.2 N/A — no such release exists.
- [x] 2.3 **Negative finding, confirmed**: DeepFilterNet4 does not exist as a distinct, releasable upstream artifact. `docs/ARCHITECTURE.md`'s citation of "DeepFilterNet4" as the Stage-1 audio-enhancement "WINNER quality" is unverifiable against upstream and should be treated as inaccurate — the only real DeepFilterNet architecture is v3, already tested in `denoise-pilot-b1`, already shown to collapse on real noise (WER +1264%). Task Groups 3 and 6 (real-noise matrix extension for DeepFilterNet4) do not apply and are skipped. This closes gap 1 with a documented negative result rather than a fabricated pin.

## 3. DeepFilterNet4 Method Arm — SKIPPED (Task Group 2 found no such release exists)

## 3A. New Candidate Backends Found During Research (2026-09-28, not yet in scope — see Task Group 9)

While researching Task Groups 2 and 4, two candidates surfaced that are more
promising than anything in the current 12-row catalog and are worth a
follow-up change (not silently folded into this one's scope):

- **LL-SDR** (arXiv 2603.20242, code at `github.com/jingyi49/llsdr`, weights on
  Hugging Face): RTF 0.0108 — ~5x faster than the current catalog's fastest
  arm (`sherpa-gtcrn-simple` at RTF 0.053) — and explicitly evaluated on both
  reverberant and non-reverberant conditions on DNS-Challenge 2020, i.e. it
  targets exactly the real-noise robustness property `deepfilternet3` failed
  on in `matrix-v2`. Token-based (Variance-Ordered Residual Vector Quantizer),
  not a drop-in spectral/waveform model like the current catalog — adapter
  effort is unknown until read directly.
- **AI4Bharat IndicConformer** (`huggingface.co/ai4bharat/indic-conformer-600m-multilingual`,
  MIT license, `github.com/AI4Bharat/IndicConformerASR`) and **IndicWhisper**
  (fine-tuned Whisper on the Vistaar dataset, lowest WER on 39/59 Vistaar
  benchmarks): purpose-built for all 22 official Indian languages, directly
  relevant to Task Group 4's finding that generic `faster-whisper base` is
  inadequate for Hindi/Tamil/code-mix — these are downstream-decoder
  candidates, not enhancement backends, and belong in a Stage-2 (STT)
  evaluation, not this Stage-1 (denoising) change.
- Two directly relevant papers for Stage 2 STT selection: "ASR Under the
  Stethoscope: Evaluating Biases in Clinical Speech Recognition across Indian
  Languages" (arXiv 2512.10967, Nov 2025 — benchmarks IndicWhisper/Whisper/
  Sarvam/Google STT/Gemma3n/Omnilingual/Vaani/Gemini on **real clinical
  interview data** in Hindi/Kannada/Indian English) and "SamaVaani: Auditing
  and Debiasing Multilingual Clinical ASR for Indian Languages" (arXiv
  2606.26901).

## 4. Bucket-Aware WER — Empirical Check — RESOLVED 2026-09-28

- [x] 4.1 Confirmed: `demo/stt_extract.py::transcribe` captures Whisper's auto-detected `info.language` per call; the frozen `downstream-v1.json` pins `model: "base-int8"` (`actual_model: "base"`), CPU, no forced language.
- [x] 4.2 Ran `faster-whisper base` (CPU, int8 — the exact pinned tier) against all 11 non-English `clean` + `noisy` conditions (`sample-005`..`015`) directly (script: ad hoc, not yet folded into the harness — see 4.4) and compared to the native-script references in `evidence/pilot-b1/test-data/reference-transcripts.csv`.
- [x] 4.3 **Finding: the pinned `base` decoder tier is not usable for Hindi/Tamil/code-mix, independent of any enhancement backend.** On *clean* audio alone: `hi` samples produced Urdu-script output (sample-007, language mis-detected as `ur`), repeated-character hallucination (sample-008: `"4 サン มี ھا ھا ھا ھا..."`), or Latin transliteration instead of Devanagari (sample-005, -006). `ta` samples were better but still degraded (sample-010 clean mis-detected as `ml`/Malayalam). `hi-en-code-mix` collapsed to English-transliteration or empty output. On *noisy* audio it got categorically worse: empty transcriptions (sample-006, -013), language mis-detection cascading across `te`/`ml`/`fr` (sample-010, -012, -015 noisy), and on sample-015 noisy a complete hallucination in **French**, unrelated to the audio. This is a decoder-tier failure, not an enhancement-stage question — no denoising backend can fix wrong-script/wrong-language output.
- [x] **Follow-up empirical check**: re-ran the same 3 representative clean samples (`005` hi, `009` ta, `013` code-mix) on `faster-whisper small` (CPU, int8 — one tier up). Result: **all three produced correct-script, topically correct, normal-magnitude-error transcription** (e.g. sample-005: `"जल लिए दंगा सम्मन्दी उखरनो से लेस अदिकारियों ने आहते में परवेश किया..."` vs ref `"जल्द ही दंगा संबंधी उपकरणों से लैस अधिकारियों ने अहाते में प्रवेश किया..."` — recognizable Devanagari, right content, ordinary ASR substitution errors, not garbage). **Model size, not denoising, is the dominant lever for Hindi/Tamil/code-mix WER measurability.** This does not yet prove `small` is sufficient for a production WER comparison (only 3 samples, no noisy-condition retest, no code-mix depth) — it proves `base` is categorically inadequate and a one-tier bump already changes the failure mode from "unusable" to "measurable."
- [x] 4.4 Gate decision, all three non-English buckets: **decoder-blocked at the `base` tier.** Task Groups 5 (schema/reporting) can proceed once the downstream config is re-pinned to at least `small-int8` (or, better, evaluated against AI4Bharat's IndicConformer/IndicWhisper — see Task Group 3A — before committing to a vanilla-Whisper tier bump as the permanent fix). Re-running 4.2's full 11-sample check against `small` and against IndicConformer, on both clean and noisy conditions, is required before Task Group 5 can trust any Hindi/Tamil/code-mix WER number this change produces.

## 5. Decoder Re-Pin and Bucket-Aware WER — Schema and Reporting

- [ ] 5.0 Decide and pin the Hindi/Tamil/code-mix decoder before any WER number is trusted (per 4.4's gate): benchmark `faster-whisper small-int8` (already empirically promising, see 4's follow-up) against AI4Bharat `indic-conformer-600m-multilingual` and `IndicWhisper` on all 11 non-English clean+noisy conditions; read "ASR Under the Stethoscope" (arXiv 2512.10967) and "SamaVaani" (arXiv 2606.26901) first, since they already benchmark several of these exact models on real clinical Indic speech and may shortcut this comparison. Record the choice and its own WER-on-clean baseline (independent of any enhancement backend) before proceeding.
- [ ] 5.1 Bump `schema_version` to `"1.1.0"` in `eval/contracts/report.schema.json` and `eval/contracts/run.schema.json`; add `aggregate.text_by_bucket` (object keyed by `en-US-proxy`/`hi`/`ta`/`hi-en-code-mix`, each a raw-WER object plus a `notApplicable`-shaped normalized-WER sub-object with reason `language_normalizer_not_implemented` for non-English buckets) while preserving `additionalProperties: false`.
- [ ] 5.2 Extend `eval/metrics.py::transcript_metrics` (or add a sibling function) to compute raw-token WER (`_wer` over `evaluation_tokens()`) independent of `frozen_normalize()`, using the decoder pinned in 5.0.
- [ ] 5.3 Extend `eval/reporting.py::build_arm_report` to populate `aggregate.text_by_bucket`, reusing the existing bucket-stratification grouping already used by the bootstrap comparison step (~line 1013) rather than a second, divergent implementation.
- [ ] 5.4 Extend `eval/tests/test_contracts.py` to reject a `1.1.0` report missing `text_by_bucket` and to accept a well-formed one.
- [ ] 5.5 Extend `eval/tests/test_reporting.py` and `eval/tests/test_text_metrics.py` for the new per-bucket aggregation (do not add a new, separate test module).
- [ ] 5.6 Re-run `none`, `dpdfnet2-onnx`, and `sherpa-gtcrn-simple` (the only two arms still real-noise-relevant — `deepfilternet3` already failed real noise and is not owed a re-run) through the updated harness with the 5.0-pinned decoder to produce `1.1.0` reports with real `text_by_bucket` data for every bucket.

## 6. Real-Noise Matrix Extension — SKIPPED (depended on Task Group 3, which found no DeepFilterNet4 to test; `dpdfnet2-onnx`/`sherpa-gtcrn-simple`/`deepfilternet3`/`none` real-noise coverage from `matrix-v2` already stands)

## 7. Comparison Re-Run and Deliverable

- [ ] 7.1 Re-run `eval/reporting.py`'s gated comparison (`cluster_bootstrap`, seed `1729`, 10,000 resamples, `holm_adjust`) across the full arm set and new bucket-stratified WER data.
- [ ] 7.2 Create `evidence/pilot-b2/` mirroring `evidence/pilot-b1/`'s structure (`README.md`, `hypotheses-tested.csv`, `test-data/`, `results/`, `timing/`, `xlsx/`).
- [ ] 7.3 Write `evidence/pilot-b2/README.md` stating, for each of the three gaps: what was found, and whether it changes the Stage-1 `AudioEnhancer` production-default picture — including stating plainly if it does not.
- [ ] 7.4 Update `evidence/README.md`'s "two studies" section to describe `pilot-b2` alongside `pilot-b1` and `matrix-v2`.

## 8. Changelog and Handoff

- [ ] 8.1 Add an entry to the repository `CHANGELOG.md` for this change once landed, per the project's changelog convention.
- [ ] 8.2 Note in `CHANGELOG.md` and `evidence/pilot-b2/README.md` that the next step — picking a Stage-1 `AudioEnhancer` production default and starting the `pipeline/` folder — is a separate, subsequent OpenSpec change blocked on this one's results, not part of this change's scope.

## 9. Follow-Up Change Proposal (not executed here — recommendation only)

- [ ] 9.1 Propose a new, separate OpenSpec change to add `LL-SDR` (arXiv 2603.20242) to the method catalog and evaluate it through the frozen `pilot-15-v1` corpus and `matrix-v2` real-noise methodology, following the same verify-before-pin discipline Task Group 2 just demonstrated (confirm the GitHub/HuggingFace release is real and reproducible before writing a method config — do not assume the paper's numbers transfer to this corpus). **A quick, non-rigorous smoke test (10 below) already argues against this being a quick win** — treat 9.1 as "worth a properly governed evaluation to be sure," not "worth fast-tracking."
- [ ] 9.2 Propose a Stage-2 (STT) OpenSpec change to formally evaluate `faster-whisper small/medium`, AI4Bharat `IndicConformer`, and `IndicWhisper` against the pilot corpus's Hindi/Tamil/code-mix conditions, informed by 5.0's findings and the two clinical-Indic-ASR papers found during this change's research (arXiv 2512.10967, arXiv 2606.26901). This is what actually unblocks a meaningful Hindi/Tamil WER comparison for denoising, and arguably matters more to the product than the Stage-1 decision this change was originally scoped to inform. **`ai4bharat/indic-conformer-600m-multilingual` (the `transformers`-wrapped multilingual repo) is a gated HuggingFace model** — requires an authenticated account that has accepted its license; this change's evaluation must either obtain that access or use the ungated, non-gated per-language NeMo checkpoints instead (`ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large`, `..._ta_hybrid_ctc_rnnt_large`, confirmed ungated via the HuggingFace API), which need `nemo_toolkit`'s ASR collection rather than `transformers` — untested in this change, flag as a real setup cost, not assumed trivial.

## 10. LL-SDR Smoke Test (2026-09-28, exploratory, NOT the frozen harness — do not cite as pilot-grade evidence)

- [x] 10.1 Cloned `github.com/jingyi49/llsdr` (MIT, SLT 2026 accepted paper), downloaded `weights.pth` (297MB, confirmed via HuggingFace API) from `huggingface.co/jingyi49/llsdr`, installed dependencies (`torch`, `torchaudio`, `descript-audiotools`, `argbind`, `einops`, `numba`) in an isolated scratch venv (not `demo/.venv` or any tracked requirements file). Model loads on CPU, 74.26M params — matches the paper's stated size.
- [x] 10.2 Ran raw `VODAC.encode()`/`decode()` against 5 `pilot-15-v1` noisy conditions (`sample-001`, `-002` en; `sample-005` hi; `sample-009` ta; `sample-013` hi-en-code-mix), reusing `eval/metrics.py::si_sdr_db`/`snr_db` for direct comparability with the pilot's own metric functions (not the full frozen harness — no fresh-process isolation, no bootstrap, no hash verification, n=5 not n=15).
- [x] 10.3 **Alignment pitfall found and resolved**: the model exposes `get_delay() == 5264` samples, which looked like a group delay to compensate before scoring (as the frozen harness requires for every catalogued backend). Naively shifting the output by that amount destroyed the signal (normalized cross-correlation ≈ 0.02–0.06, i.e. noise). A full FFT cross-correlation search found the *true* best alignment is lag ≈ 0 (0.90 normalized correlation on `sample-002`) — the raw `encode()`/`decode()` output on a whole-clip batch call is already time-aligned; `get_delay()` evidently describes something else (likely a streaming/chunked-inference lookahead requirement, not applicable to this whole-clip call). The authors' own `dnsmos.py` evaluation script confirms this indirectly — it never aligns against a reference at all, scoring with reference-free DNSMOS instead, so the authors never had to solve this.
- [x] 10.4 **Verified numbers** (lag-0 aligned, cross-correlation-confirmed real reconstruction, not an alignment artifact — per-sample correlation noted):
  | Sample | Bucket | SI-SDR before→after | Δ SI-SDR | SNR before→after | Δ SNR | Alignment corr |
  |---|---|---|---:|---|---:|---:|
  | sample-001 | en | 4.90 → -2.21 | **-7.12** | 4.90 → 2.00 | -2.90 | 0.63 |
  | sample-002 | en | 4.98 → 6.48 | **+1.51** | 4.98 → 7.34 | +2.36 | 0.90 |
  | sample-005 | hi | 4.99 → 2.47 | **-2.52** | 5.00 → 4.41 | -0.59 | 0.80 |
  | sample-009 | ta | 4.81 → 5.57 | **+0.76** | 4.81 → 6.42 | +1.62 | (not measured) |
  | sample-013 | hi-en-code-mix (echo/reverb) | -20.49 → -25.27 | **-4.78** | -3.99 → -1.74 | +2.25 | -0.29 (poor) |
- [x] 10.5 **Finding**: mean SI-SDR delta across these 5 samples ≈ **-2.43 dB (net harmful)**, harmful on 3/5 samples, nowhere near the frozen pilot's synthetic-corpus leaders (`dpdfnet2-onnx` +5.85 dB, `sherpa-gtcrn-simple` +5.30 dB, averaged over all 15 bases with bootstrap CI). Worst on `sample-013`, the code-mix/echo condition, where reconstruction correlation itself was poor (-0.29) — the same failure shape as `deepfilternet3`'s real-noise collapse: a model that looks good on its own paper's benchmark (DNS-Challenge) does not obviously transfer to this corpus's specific noise types (white/pink/hum/babble/echo at 5dB) and multilingual content. **This is a 5-sample, non-bootstrapped smoke test — not evidence LL-SDR is bad, only evidence it is not an obvious win, and a properly governed evaluation (9.1) would be needed before any claim either way.**

## 11. IndicConformer/IndicWhisper License Check and Smoke-Test Attempt (2026-09-28)

- [x] 11.1 **License confirmed clear for proprietary use.** Checked HuggingFace API `cardData.license` directly (not just the model card text) for all three candidates: `ai4bharat/indic-conformer-600m-multilingual` (gated repo) = MIT; `ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large` and `..._ta_hybrid_ctc_rnnt_large` (ungated) = MIT. For IndicWhisper, the official checkpoints ship via the `AI4Bharat/vistaar` GitHub repo, whose `LICENSE` file and README both confirm MIT ("Vistaar is MIT-licensed. The license applies to all the fine-tuned language models"). None of the three restrict commercial/proprietary use. (The underlying FLEURS/Kathbath/CommonVoice/etc. training data each carry their own, separate license — this does not retroactively restrict the released model weights, which is the standard field interpretation, but is noted for completeness.)
- [x] 11.2 **Attempted, blocked twice, stopped rather than trying a third environment.** `ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large`/`..._ta_...` need `nemo_toolkit[asr]`, not `transformers`. First attempt (system Python 3.14, matching this session's original scratch venv): failed — `onnx` (a transitive NeMo dependency) has no prebuilt wheel for 3.14, and building it from source via `cmake` failed (`Failed building wheel for onnx`). Second attempt (Python 3.12, matching `demo/.venv`'s own interpreter — ruling out the Python-version theory): failed differently — `OSError: [Errno 122] Disk quota exceeded` after NeMo's dependency tree (PyTorch, PyTorch Lightning, Hydra, ...) had already written 4.1GB. This is a **per-user filesystem quota on this workstation**, not raw disk space (`df` shows 111GB free on the root filesystem) — `~/.cache/pip` alone was already 4.9GB before this attempt. Both incomplete venvs were deleted to free the quota back up; nothing was left half-installed.
- [x] 11.3 **Retried twice more at the user's explicit request ("ya try running them"), both failed the same way — root cause narrowed to the sandbox, not the filesystem.** Attempt 3: cleared `~/.cache/pip` (4.9GB freed, confirmed via `du`), retried the identical py3.12 install — failed again with the identical `OSError: [Errno 122] Disk quota exceeded`, after `df` confirmed 116GB free on `/` and `$HOME` usage was down to 95GB (of which `du -sh` breakdown showed legitimate large data: `.ollama` 20GB, `.cache` 27GB even post-clear, `.hermes` 12GB, `.local` 9.8GB, `.bun` 6.6GB — nothing safe to unilaterally delete further). Attempt 4: redirected `PIP_CACHE_DIR` and `TMPDIR` both to `/tmp` (a separate tmpfs mount, `df` reports it independently from `/`, 5.5GB free at the time) so neither pip's cache nor its build temp files touched `$HOME`'s filesystem at all — **failed with the exact same `[Errno 122] Disk quota exceeded`**. Two different underlying filesystems (ext4 `/` and tmpfs `/tmp`) producing the identical quota error rules out "wrong directory" as the cause. This strongly indicates the quota is enforced by **the sandbox wrapping these Bash tool calls itself** (a total-bytes-written cap per session, independent of target mount point), not by anything in the workstation's own filesystem configuration. Both incomplete attempts were deleted; nothing left half-installed. **Stopped here per this task's own stated rule (4 distinct attempts, 2 distinct root-cause theories both ruled out) — not attempting a 5th without being asked.**
- [x] 11.4 **What we know instead, from earlier in this change's research (Task Group 4)**: the pinned `faster-whisper base` decoder produces unusable, wrong-script output on Hindi/Tamil/code-mix; bumping one tier to `faster-whisper small` (CPU, no NeMo/quota issues — already proven working in this environment, four separate times, for the LL-SDR test too) immediately produces correct-script, normal-error-rate transcription. **`small` is a concrete, already-verified, zero-additional-setup-cost improvement available today; IndicConformer/IndicWhisper remain the theoretically stronger, purpose-built, MIT-licensed option but installing NeMo to test them is blocked at the sandbox/tool-execution level in this session, not by the workstation's actual disk, not by licensing, and not by model availability. A session with a higher or absent sandbox disk-write cap (or running the install as a plain terminal command outside this tool's sandbox) would very likely succeed unchanged.**

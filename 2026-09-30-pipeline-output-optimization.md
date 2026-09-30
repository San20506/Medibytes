# Optimizing the Pipeline's Output (transcription accuracy)

**Date:** 2026-09-30 · **Scope:** Product FastAPI service + Medibytes demo Stage 1
**Evidence status:** in-repo code audit (verified directly) + deep-research run
`wf_d6e68e88-631` (stalled during Verify; 14 of the 22 queued claims completed
adversarial verification — each item below is labelled with its actual status).

---

## Executive answer

Stop enhancing. The single highest-leverage change is deleting the
`enhance_gtcrn` route, and the external literature now independently supports
what the three internal studies measured. Everything else is second-order.

---

## 1. Disable the GTCRN route — highest leverage, one-line change

`Product/backend/audio_pipeline/routing.py` (read directly):

```python
MIN_SPEECH_FRACTION   = 0.05
SNR_HIGH_THRESHOLD_DB = 20.0
SNR_LOW_THRESHOLD_DB  = 5.0
```

`select_route` returns `direct_asr` only at SNR >= 20 dB or when SNR is
unreliable; **everything from 5-20 dB routes to `enhance_gtcrn`**. Measured from
`Product/evidence/en40-calibration-2026-09-29/results.jsonl`: under
`dynamic_routing`, **51 of the 80 noisy conditions (64%) executed
`enhance_gtcrn`** (29 direct_asr); across all 120, 54 GTCRN / 66 direct. So the
harmful route is the majority route on noisy input — the mechanism behind the
measured 0.201 vs 0.149 weighted-mean WER regression.

**External corroboration (all 3/3 confirmed):**

- *Artifacts, not residual noise, are the dominant cause of ASR degradation from
  single-channel SE.* Scaling down the artifact component greatly improves WER;
  scaling the noise component changes WER much less.
  (Iwamoto et al., "How Bad Are Artifacts?", Interspeech 2022, arXiv:2201.06685)
- *"It is well-known that noise reduction technologies tend to degrade ASR
  performance due to the processing artifacts introduced by non-linear
  transformations... recent ASR systems trained on varied noise tend to be less
  affected by non-speech noise than by the artifacts."*
  (Sato et al., Interspeech 2021, arXiv:2106.00949)
- *Single-channel SE front-ends generally fail to improve ASR in noise*; in that
  paper's own experiments the enhanced signal did not beat the noisy observed
  signal. (arXiv:2201.06685)
- **Nearest-domain match:** MetricGAN+ denoising **raised semantic WER in all 40
  tested configurations** (4 ASR systems x 10 conditions) over **500 Indian
  medical recordings**, degradations 1.1-46.6 pp absolute.
  ("When De-noising Hurts", arXiv:2512.17562). Verifier caveats: arXiv technical
  report, not peer-reviewed; the 46.6 pp extreme is Gemini under *synthetic white
  Gaussian noise* and the hospital-ambient deltas are far smaller; the authors
  explicitly decline to generalize ("this does not imply that speech enhancement
  is inherently detrimental to ASR"). One GAN enhancer, not GTCRN or dpdfnet2.
- On VoiceBank+DEMAND with Whisper large-v3, most enhancers improved PESQ-type
  quality while failing to improve WER — only 1 of the evaluated enhancers
  improved Whisper WER. (2/3 confirmed, arXiv:2607.11157)

**Action:** make `direct_asr` the default everywhere, or remove `enhance_gtcrn`
as a selectable route pending recalibration. Also recovers the 500-820 ms GTCRN
latency.

### Do not expect SNR gating to rescue it

A rule-based SIR/SNR gate (switch to unprocessed when SIR-SNR >= 10 dB) achieved
22% average relative CER reduction (3/3 confirmed, Sato et al.). **But the
verifiers flagged explicitly that the gate beats *always-enhance*, not
*always-direct-ASR*** — the paper never shows gating beating plain unprocessed
input outside overlapping speech. It does not contradict the internal 35%
finding. Tuning thresholds is not a substitute for turning enhancement off.

---

## 2. If enhancement is ever re-enabled: Observation Adding

The one mitigation that survived 3/3 verification and costs nothing to try:

> **Observation Adding (OA):** mix a scaled copy of the raw observed signal back
> into the enhanced signal, `s = ŝ + ω_obs · y`. ~20% relative WER improvement
> over the un-mixed enhanced signal on the CHiME-3 real-recording test set, over
> a broad non-critical ω_obs range of **0.3-0.8**. (arXiv:2201.06685)

Caveats recorded by the verifiers: the 20% is relative to enhanced-without-OA,
not relative to direct ASR on the observation, and the ASR back-end was a Kaldi
LF-MMI hybrid, not Whisper. Treat as a rescue for a low-SNR tail only, never as
a reason to re-enable enhancement broadly.

A second inference-time knob — attenuating the enhancement mask magnitude
(alpha=0.25 lowered Whisper WER 6.44 -> 5.57) — is **undecided**: 1 confirm,
1 refute, third vote never returned. The refuter found the framing overreached
the source. Do not act on it without checking the paper directly.

---

## 3. Decoder settings — what to change and what to leave alone

Both pipelines use identical, untuned settings
(`Product/.../asr.py:66`, `demo/stt_extract.py:183`):
`beam_size=1, temperature=0.0`, no `vad_filter`, no `initial_prompt`/`hotwords`.

- **`beam_size=1` is worth an experiment, not a settled call.** Note both
  pipelines *override* the faster-whisper default (checked in the installed
  library: `beam_size=5`, `condition_on_previous_text=True`). One fetched source
  found beam 1 minimized hallucinations (19.5%/20.3% detected, vs 28.0%/28.3% at
  beam 3 and 28.2%/37.4% at beam 5) with no WER improvement from raising beam —
  but this claim **was never voted on** (the run stalled first), and its scope is
  openai-whisper large-v3 on silence-padded clips whose unprocessed WER exceeded
  100%. That is a hallucination stress test, not normal dictation. Action:
  compare beam 1 vs 5 on the existing en-40 harness rather than assuming.
- **Add medical-vocabulary prompting.** No prompt conditioning exists anywhere,
  though drug names and vitals are the stated critical content. Precedent:
  Whisper-Streaming feeds the last 200 words of confirmed output back as the
  prompt to hold terminology consistent. *(unverified — run stalled before these
  claims were voted)*
- **Set `condition_on_previous_text=False`.** Widely asserted to reduce
  hallucination and repetition; the sources carry no isolating experiment.
  *(assertion-level evidence only)*
- **Insertions are the dangerous error class here.** On Indian clinical speech,
  one system inserted at 39.15% (Hindi) / 36.87% (English). For a discharge note,
  a hallucinated drug name is worse than a dropped word — consider scoring
  insertion rate separately from WER. *(unverified)*

---

## 4. Demo Stage 1 defects (found in code, not in the literature)

**a. Resampler has no anti-aliasing filter.** `demo/audio_clean.py::_resample_linear`
downsamples with bare `np.interp`. Product does this correctly
(`ingestion.py:323`, `scipy.signal.resample_poly`). **Every corpus file checked is
already 16 kHz** (`~/.local/share/medibytes-eval/corpus/pilot-15-v1/*`,
`demo/audio_in/*`), so `source_rate == target_rate` short-circuits and this path
never ran in en_pilot or the n=120 study. A real 44.1/48 kHz phone recording
folds 8-24 kHz content back into the speech band. Fix: use `resample_poly`.

**b. RMS-threshold VAD gates a hard reject.** `_rms_vad` uses a fixed 0.02 RMS
threshold on 30 ms frames, applied *after* normalization to 0.1 RMS — so the
threshold tracks the normalizer, not speech presence — and `vad_ratio < 0.05`
raises a 422. Relevant evidence: VAD *quality* drives the benefit, not VAD
presence — on a *silence-padded augmented set* (clips Whisper already transcribed
correctly, padded until unprocessed WER exceeded 100%), WebRTC VAD reached
68.3%/75.4% WER vs Silero's 8.0%/10.8%; these are stress-test numbers, not
normal-speech WERs. On a 17-hour multi-domain set WebRTC scores ROC-AUC 0.73 vs
Silero v5's 0.96. An energy/RMS VAD was not tested in any source, but it sits
below WebRTC in sophistication. Fix: use Silero here too, as Product does.
Also: published VAD thresholds are pyannote-specific and the authors state the
decision threshold must be tuned per domain — so do not copy constants.

---

## 5. Hindi/Tamil — the largest remaining accuracy gap

- Whisper large-v3 on **real Indian clinical speech** (psychiatric interviews in
  busy wards/OPD): **46.76% WER Indian English, 71.68% Hindi.** Even the largest
  Whisper tier degrades catastrophically in this exact acoustic+accent domain.
- **IndicWhisper** (per-language fine-tunes of Whisper-medium, 769M): **13.6% WER
  Hindi** average across 7 Vistaar benchmarks, beating Google STT (23.9), Azure
  (20.0), Nvidia Conformer (18.6). **Tamil 25.3%** — roughly 2x Hindi, bounding
  what a Tamil-capable release can promise.
  **Do not compare 13.6% against the 71.68% above.** Those are different audio:
  Vistaar is benchmark speech, the 71.68% is noisy clinical ward speech. The
  source itself reports **no code-mixed and no clinical benchmark, and no latency
  figures**, and it is one model per language with the language token forced at
  decode — which does not fit code-mixed input by construction.
- **Fine-tuning on in-domain code-mixed data is the largest lever measured:**
  stock Whisper large-v2 scores **52.0% MER** on Hindi-English code-mixed speech
  (MUCS-2021, 5.2 h test), with code-switch bigram accuracy of 42.9%/36.8%.
  Adding 89.8 h of real in-domain code-switched data + 30 h monolingual, with
  language-specific prompts and on-the-fly CS simulation, reaches **28.8% MER** —
  23% relative over the plain fine-tuned baseline (37.4%). Even so, ~29% MER
  remains. Scope: large-v2 only, no small/medium tier, clean tutorial speech.
- Whisper emits wrong-script output even with `language` explicitly set —
  matching the internal tiny/base observation.

*(All section-5 claims are from the fetched pool but were not reached by the
verification phase before the run stalled.)*

---

## Recommended order

1. Delete/bypass the `enhance_gtcrn` route — quantified harm, one-line fix, also
   recovers 500-820 ms.
2. Fix `_resample_linear` to `resample_poly` — real defect in an untested path.
3. Add `initial_prompt`/`hotwords` for drug names and vitals; set
   `condition_on_previous_text=False`. Keep `beam_size=1`.
4. Replace the demo's RMS VAD with Silero.
5. Evaluate IndicWhisper/IndicConformer for hi/ta (still blocked on the sandbox
   disk-write cap, not on the models).
6. Do not re-open denoising. If ever re-enabled, gate it narrowly and apply
   Observation Adding with ω_obs ≈ 0.3-0.8.

## Research completeness

The deep-research run stalled four times during Verify. Recounted from the
journal: **22 distinct claims entered the verification queue; 14 received votes**
— **7 confirmed** (>=2 of 3 non-refuting), **1 undecided** (the mask-attenuation
claim: 1 confirm, 1 refute, third vote lost), **6 killed**. The 6 kills were all
for *overreach*: verifiers found the underlying directional findings real but the
claim wording inflated or misattributed them. 196 unique claims were fetched
overall, so the large majority were never voted on.

Sections 1-2 are verification-backed. Sections 3-5 rest on the unverified fetched
pool and are labelled as such — treat them as leads to check, not findings.

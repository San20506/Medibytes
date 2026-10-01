# Component-wise accuracy on the MediBytes clinical dataset (2026-10-01)

7 clinician/nurse/caller dictations, 66–87 s each (9.3 min total), 48 kHz stereo
AAC. Six arms: 2 denoisers (`none`, `sherpa-gtcrn-simple`) × 3 decoders
(`small-int8` = the shipped default, `base-int8`, `medasr`) = 42 end-to-end runs.

Run with `clinical_eval/`, a package separate from `en_pilot/`; nothing in the
frozen eval contracts was touched. Raw artefacts in the data root under
`corpus/medibytes-med-7/`.

---

## Headline: ASR is not the bottleneck. The extractor is.

**100% of clinically-critical numbers survive speech recognition. Then Stage 4
throws roughly half of them away.**

| Stage | Best result | What it means |
|---|---|---|
| C1 ingest/clean | **42/42 pass** | no failures, any arm |
| C2 ASR — numbers | **34/34 (100%)** | every dose, BP, pulse, temp, SpO2, glucose is in the transcript |
| C2 ASR — drug names | **7/11 (64%)** | MedASR; `small-int8` 45%, `base-int8` 27% |
| C4/C5 extraction — vitals | **14/26 (54%)** | half the vitals present in the text never reach the record |
| C4/C5 extraction — drugs | **2/11 (18%)** | nine of eleven drugs invisible |
| C4/C5 extraction — doses | **2/2 (100%)** | both doses that exist are correct |
| C4/C5 — allergy polarity | **1/2 (50%)** | **the positive allergy is the one that's dropped** |

Best arm overall is the **current default, `none` + `small-int8`**. Denoising
changed nothing meaningful — expected, since all 7 clips are clean.

---

## First: WER is not computable on this dataset

The shipped reference transcripts are **paraphrases of the audio, not verbatim
transcriptions**. Clip 001, same clinical content, different words:

> **Audio (ASR):** "So I just checked this patient. He is around 52 years old and he's been having fever and cough for about 3 days now."
>
> **`transcripts/MB_MED_001_clean.txt`:** "This is a clinical assessment of a fifty-two-year-old male patient presenting with fever, productive cough, chest discomfort, and shortness of breath for the past three days."

Scored naively that is **WER 0.807** — which measures the rewrite, not the
decoder. So this eval scores **clinical facts**, not words. That is arguably the
better target anyway: the product emits a record, not a transcript.

Gold facts were hand-authored from the reference text in `clinical_eval/gold.py`
**before any decoder was run**, with fixed match rules (numeric equality; BP as a
systolic/diastolic pair; canonical drug names). They are agent-authored and
should be clinician-reviewed before being quoted outside the repo.

---

## C1 — Ingest and cleaning: no failures

42/42 passed, mean VAD ratio 0.907. `clean_audio` ingests 48 kHz stereo AAC in
MP4 natively through its ffmpeg fallback — **the shipped pipeline handles the
dataset's own delivery format with no conversion**. GTCRN produced no
`ENHANCE_FAIL` on these clips.

One unverified observation: the two channels are **not** dual-mono — measured L/R
correlation ≈ 0.50 with different peaks. The pipeline downmixes 0.5·L + 0.5·R.
Left- and right-only WAVs are staged (`--channel left|right`) but not yet run, so
whether the downmix costs accuracy is open.

---

## C2 — Speech recognition: numbers perfect, drug names poor

| Arm | Numbers (doses, vitals) | Drug names |
|---|---|---|
| `none` + `small-int8` *(default)* | **34/34 (100%)** | 5/11 (45%) |
| `none` + `medasr` | **34/34 (100%)** | **7/11 (64%)** |
| `none` + `base-int8` | 32/34 (94%) | 3/11 (27%) |
| `gtcrn` + `small-int8` | 33/34 (97%) | 6/11 (55%) |
| `gtcrn` + `medasr` | 34/34 (100%) | 7/11 (64%) |
| `gtcrn` + `base-int8` | 32/34 (94%) | 2/11 (18%) |

Every number that matters — 650 mg, 130/85, 101 °F, 94%, glucose 54 → 72 — is
transcribed correctly by the default decoder. **Speech recognition is not where
this pipeline loses clinical data.**

Drug names are the weak point, and **MedASR leads** (64% vs 45%), consistent with
the domain argument from earlier in the session. It uniquely got `ceftriaxone`
and the `amoxicillin` in clip 005. But all three decoders fail the hard
compound in 001:

| Decoder | "amoxicillin with clavulanic acid" → |
|---|---|
| `small-int8` | "amoxicin with **chlorveonic acid**" |
| `base-int8` | "amoxilant with **clavonic acid**" |
| `medasr` | "Amoxillin with **clavulic acid**" |

`amlodipine` and `adrenaline` are missed by all three.

---

## C3 — Normalizer: breaks on spoken decimals

`numwords_to_digits` handles integers correctly but has no rule for "point":

| Input | Output | |
|---|---|---|
| "one hundred and thirty over eighty-five" | `130 over 85` | ✅ |
| "one fifty over ninety-five" | `150 over 95` | ✅ |
| "fifty-four milligrams per decilitre" | `54 milligrams per decilitre` | ✅ |
| "ninety-eight point six degrees Fahrenheit" | `98 point 6 degrees Fahrenheit` | ❌ |
| "one hundred point two degrees Fahrenheit" | `100 point 2 degrees Fahrenheit` | ❌ |

Any decimal vital dictated in words is lost. It doesn't bite end-to-end here only
because the decoders already emit `98.6` as digits — it bites whenever upstream
text is spelled out.

*This also means the C4 "ceiling" below is understated: it runs on the
spelled-out reference text, so it inherits this bug. End-to-end scores on
digit-bearing ASR text are the more meaningful number, and in several clips they
exceed the "ceiling".*

---

## C4 / C5 — Extraction: where the data is lost

Perfect transcript in (C4) vs full cascade (C5, best arm):

| | Ceiling (reference text) | End-to-end (`none`+`small-int8`) |
|---|---|---|
| Vitals | 10/26 (38%) | **14/26 (54%)** |
| Drugs | 2/11 (18%) | 2/11 (18%) |
| Doses | 2/2 (100%) | 2/2 (100%) |
| Allergy polarity | 1/2 | 1/2 |

### Four concrete extractor defects

**1. SAFETY — the true allergy is dropped.** Clip 005 is an acute amoxicillin
reaction. The record the pipeline produces contains exactly one allergy:

```json
{"text": "penicillin", "negated": true, "note": "NEGATED - not added to allergy table"}
```

The **amoxicillin allergy — the positive one, the reason for the call — is
absent.** Every arm, including at the ceiling. `ALLERGY_CTX` matches
`allergy to X`; the source sentence is *"document amoxicillin as a suspected drug
allergy in her medical record"*, where the drug precedes the word. So the record
asserts a negated penicillin allergy and stays silent on the drug that actually
caused the reaction. That is the failure mode that gets a patient re-dosed.

**2. Doseless drugs are invisible.** `DRUG_CTX` requires `name + number + unit`.
Nine of eleven gold drugs carry no dose in the dictation, so aspirin, insulin,
metformin, ceftriaxone, adrenaline, amoxicillin, clavulanic acid and the
paracetamol in 007 are all missed **even with a perfect transcript**. The two
that match — paracetamol 650 mg, amlodipine 5 mg — are extracted perfectly.

**3. Whole vital types have no pattern.** `VITALS_PAT` has no rule for
**respiratory rate** or **blood glucose**. Clip 006 — a hypoglycaemia call whose
entire clinical point is glucose 54 → 72 mg/dL — extracts **zero vitals**.

**4. Brittle patterns on the rest.**
- Pulse missed in **6 of 7** clips: the regex wants `pulse 92`, the speech says *"pulse rate is 92"*.
- BP missed in 002 and 004: the pattern allows *"blood pressure is around"* but not *"blood pressure **was** around"*.
- A false positive in 005: pulse 108 emitted as a percentage — `"108% "`.

---

## What I'd fix, in order

1. **The 005 allergy miss.** Safety-critical, and the narrowest fix: match
   `<drug> ... allergy` as well as `allergy to <drug>`, and separate "allergy
   asserted" from "allergy denied" rather than treating allergies as negation-only.
2. **Doseless drug capture.** Match drug-list names standalone, with dose/unit
   optional. Recovers 9 of 11 drugs.
3. **Respiratory rate and glucose patterns**, plus `pulse rate is N` and the
   `was/were` copula. Recovers most of the missing 12 vitals.
4. **`numwords_to_digits` "point" handling.**
5. Re-run this eval after each; it takes about 5 minutes.

I deliberately changed **nothing** mid-eval — these are measurements of the code
as it stands today.

## Limits

- **n=7, single speaker set, English, all clean.** Counts, not rates with
  confidence intervals. No bootstrap would be honest at this size.
- **The dataset ships empty `medium_noise/`, `heavy_noise/`,
  `external_clips/audio/`, `external_clips/transcripts/` and a 0-byte
  `metadata.csv`.** Only the clean condition exists, so the denoising arms had
  nothing to denoise. Synthetic noise could be mixed in from the existing DEMAND
  bank if you want the noisy comparison.
- **Gold is agent-authored** from the reference transcripts, not clinician-reviewed.
- **Extraction is regex-only** (`use_llm=False`) — which is also production today,
  since no Ollama model is pulled. The LLM-primary path is untested.
- Left/right channel arms staged but not run.

## Reproduce

```bash
.venv/bin/python -m clinical_eval.prepare --zip <dataset>.zip
.venv/bin/python -m clinical_eval.run   --channel mix
.venv/bin/python -m clinical_eval.score --channel mix
```

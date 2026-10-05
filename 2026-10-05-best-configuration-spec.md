# Best measured configuration — full specification (2026-10-05)

The highest-scoring pipeline configuration measured on the MediBytes 7-clip
clinical dataset, with everything needed to reproduce it and an honest statement
of what it does and does not establish.

**Status: this is a measured configuration, not the shipped default.** Two of its
four stages differ from what `demo/pipeline.py` runs today. See *Gap to
production*.

---

## 1. Specification

### Stage 1 — ingest and clean

| | |
|---|---|
| module | `demo/audio_clean.py :: clean_audio` |
| denoiser | **`none`** |
| input accepted | 48 kHz stereo AAC in MP4, natively, via the ffmpeg fallback |
| internal format | 16 kHz mono PCM16, downmix `0.5·L + 0.5·R` |
| gain | RMS normalise to 0.1 (≈ −20 dBFS) |
| cost | 0.034 s/clip |

Denoising is **off deliberately**, not by omission. Measured on this corpus:

| denoiser | clinical facts | vs none |
|---|---|---|
| **none** | **36/41** | — |
| dpdfnet2-onnx v0.6.0 | 35/41 | −1 |
| sherpa-gtcrn-simple | 33/41 | −3 |

(greedy decode, so that all three are compared on one footing)

All 7 clips are clean studio recordings. Enhancement has no noise to remove and
only perturbs intact speech — GTCRN destroyed clip 005's amoxicillin allergy.
**This does not generalise to noisy input**, where the 1,440-condition DEMAND
matrix shows both denoisers helping MedASR. On unknown input the right design is
an SNR gate, which does not exist yet.

### Stage 2 — speech recognition

| | |
|---|---|
| model | `google/medasr`, revision `ae1e4845b4b07479735d93e1e591e566435b7104` |
| architecture | Conformer CTC, ~105 M params, 421 MB fp32 |
| **decode** | **CTC beam search with shallow LM fusion** (`pipe.type == "ctc_with_lm"`) |
| language model | `lm_6.kenlm`, 704 MB, shipped in the model repo |
| beam width | 8 |
| chunking | `chunk_length_s=20`, `stride_length_s=2` |
| device | CUDA (RTX 4060 Laptop), fp32 |
| cost | ~0.5 s/clip — **not slower than greedy** |
| determinism | byte-identical across repeated runs |
| post-processing | `demo/stt_extract.py :: medasr_detokenize` resolves `{period}`, `{colon}`, `[SECTION]`, `</s>` |

Validated against the model's own radiology sample before being trusted here:

| decode | WER on `test_audio.wav` |
|---|---|
| greedy CTC | 0.0122 |
| **kenlm beam search** | **0.0000** |

Known limitation: the shipped LM is binary, so pyctcdecode decodes **without a
unigram set** and warns about it. Supplying unigrams could only improve these
numbers — they are a floor for LM fusion.

### Stage 3 — normalise

`demo/stt_extract.py :: normalize_text` — Hinglish map, UCUM unit spacing,
BP/SpO2 shorthand, spoken numerals to digits (including decimals, *"ninety-eight
point six"* → `98.6`, and hundreds said without "hundred", *"one ten"* → `110`).

### Stage 4 — extract

| | |
|---|---|
| module | `demo/stt_extract.py :: extract_entities` |
| mode | **regex only** (`use_llm=False`) |
| formulary | `demo/assets/drug_list_mini.json`, **33 names** |
| ceiling | 41/41 given a perfect transcript — **in-sample**, see §4 |

Covers vitals (BP, pulse/heart rate, respiratory rate, temperature, SpO2,
glucose), drugs with or without a dose, doses with UCUM units, and allergies in
both word orders with clause-scoped polarity.

### Stage 5 — render

`demo/fill_template.py` → `er_discharge` via the premium coords renderer, HTML +
DOCX. Asserted allergies render under *Allergies (active)*; denied ones as
*DENIED (locked)*; doseless drugs as `NIL` + "no dose dictated — physician must
confirm".

### Environment

```
transformers 5.18.0   torch 2.14.1+cu132   CUDA 13.2
kenlm 0.3.0           pyctcdecode 0.5.0    numpy 2.2.6
```

`pyctcdecode` declares `numpy<2`; the pin is **not binding in practice** but a
naive resolve will downgrade numpy and break transformers 5.x. Pin
`numpy==2.2.6`.

The model card's notebook installs a third-party fork of pyctcdecode. **It is not
required** — the decoder uses only `build_ctcdecoder` and `decode_beams`, both
stable upstream; the one difference this code relies on is that `decode_beams`
returns tuples upstream and dataclasses in the fork, handled in four lines.

---

## 2. Performance on the dataset

7 clinician/nurse/caller dictations, 67–87 s each, 9.4 min total, 48 kHz stereo
AAC. Scored against hand-authored clinical gold facts by `clinical_eval`.

### Headline

**37 of 41 clinical facts — 90.2%** · 95% CI **[77.5%, 96.1%]**

| clip | vitals | drugs | doses | allergy | total |
|---|---|---|---|---|---|
| MB_MED_001 | 5/5 | 2/3 | 1/1 | — | 8/9 |
| MB_MED_002 | 4/4 | 0/1 | 0/1 | — | 4/6 |
| MB_MED_003 | 3/3 | — | — | — | 3/3 |
| MB_MED_004 | 4/4 | 1/1 | — | — | 5/5 |
| MB_MED_005 | 3/3 | 1/2 | — | 2/2 | 6/7 |
| MB_MED_006 | 2/2 | 2/2 | — | — | 4/4 |
| MB_MED_007 | 5/5 | 2/2 | — | — | 7/7 |
| **total** | **26/26** | **8/11** | **1/2** | **2/2** | **37/41** |

### By component

| component | result | |
|---|---|---|
| C1 ingest/clean | **7/7 pass** | mean VAD ratio 0.907, no failures |
| C2 — clinical **numbers** in transcript | **34/34 — 100%** | every dose, BP, pulse, temp, SpO2, glucose |
| C2 — **drug names** | **8/11 — 73%** | the weak axis |
| C4 — extractor ceiling | 41/41 | in-sample |
| C5 — **vitals** to record | **26/26 — 100%** | |
| C5 — **drugs** | 8/11 — 73% | |
| C5 — **doses** | 1/2 — 50% | the miss rides on a drug-name miss |
| C5 — **allergy polarity** | **2/2 — 100%** | incl. clip 005's asserted amoxicillin allergy |

**WER is not computable on this corpus** — the shipped reference transcripts are
paraphrases of the audio, not verbatim. Clip 001 scores WER 0.807 against its own
reference while carrying identical clinical facts. Hence facts, not words.

### Precision

Recall alone can be gamed by emitting more. On this configuration the extractor
emits **exactly 26 vitals against 26 gold**, with **no spurious drug rows** — every
row emitted is a row that matched.

### Every remaining miss

All four are one failure mode: **a drug name the decoder spelled wrong.**

| clip | miss | heard as |
|---|---|---|
| 001 | clavulanic acid | "clavulic acid" |
| 002 | amlodipine **and its 5 mg dose** | "Amalify" |
| 005 | adrenaline | "ergenelin" |

Vitals, numbers and allergy polarity are at 100%. **Drug-name recognition is the
entire remaining gap**, and 73% is the number to quote internally — not 90%.

### Cost

~0.55 s/clip of compute for a 67–87 s dictation (0.034 s clean + ~0.5 s decode),
GPU-resident. Real-time factor ≈ 0.007.

---

## 3. Gap to production

Two stages differ from what ships today:

| | spec | `demo/pipeline.py` today |
|---|---|---|
| decoder | medasr + kenlm beam search | `small-int8` (Whisper) default; `--model medasr` routes to **greedy** |
| LM fusion | wired | **not wired into `demo/` at all** — lives in a session scratchpad |
| denoiser | `none` | `none` ✓ already matches |
| extraction | regex | regex ✓ already matches |

Making this the shipped path is a code change that has not been made.

---

## 4. What these numbers do NOT establish

* **n=7, single speaker set, English, all clean.** Counts, not rates.
* **The 95% CI is [77.5%, 96.1%].** One fact is worth 2.4 points. Every
  configuration measured this session has an interval overlapping every other, so
  **a 95% target can be neither demonstrated nor refused on this corpus.**
  Confirming 95% at ±2 points needs a few hundred facts — roughly 50–70 clips.
* **Gold is agent-authored** from the reference transcripts, not clinician-reviewed.
* **The extractor ceiling is in-sample.** Three of its eleven drugs (ceftriaxone,
  adrenaline, clavulanic acid) are recovered because this eval named them as
  misses and they were added to a 33-name formulary stub afterwards. On unseen
  dictation the formulary runs out before the regex does.
* **Noisy clinical audio is untested.** The dataset ships `medium_noise/` and
  `heavy_noise/` empty and a 0-byte `metadata.csv`.
* **English only.** MedASR is an English model; the Hindi/Tamil/code-mix case this
  product cares about is unsupported by this configuration, not merely untested.
* **The LLM extraction path (`--use-llm`) is untested** — Ollama is installed with
  no model pulled.

---

## 5. Reproduce

```bash
uv pip install --python .venv/bin/python kenlm==0.3.0 pyctcdecode
uv pip install --python .venv/bin/python numpy==2.2.6   # undo pyctcdecode's pin

.venv/bin/python -m pytest demo/tests                   # 87 pass
.venv/bin/python -m clinical_eval.prepare --zip ~/Downloads/MediBytes_Medical_Audio_Dataset.zip
.venv/bin/python -m clinical_eval.run   --channel mix   # greedy arms, 36/41 best
.venv/bin/python -m clinical_eval.score --channel mix
```

The LM-fusion decoder and its drivers are in the session scratchpad
(`medasr_lm.py`, `run_lm2.py`), not in the repo. Prior-state artefacts are kept
as `run-mix.BEFORE-extractor-fix.json` / `score-mix.BEFORE-extractor-fix.json`.

Earlier per-session writeups (`2026-09-29`…`2026-10-05`) were consolidated into
this file and removed from the working tree; they remain in git history, and
`CHANGELOG.md` records what each one concluded.

---

## Appendix A — how the extractor got to its ceiling

The 2026-10-01 component eval measured the extractor at **10/26 vitals, 2/11
drugs, 1/2 allergy polarity** and named four defects. All four were fixed on
2026-10-03; the ceiling went to 41/41. For anyone reading the regexes later:

**1. SAFETY — asserted allergies were unreachable by construction.** The whole
allergy block ran under `if neg:`, and `ALLERGY_CTX` matched only
`allergy to <drug>`. Clip 005 — an acute amoxicillin reaction — produced a
record naming *only* a negated penicillin. Added `ALLERGY_ASSERT` for the other
word order (*"document amoxicillin as a suspected drug allergy"*), plus
`allergic to`.

Polarity now comes from `_negated_before`, which runs `NEG_PAT` between the last
contrast marker and the substance — **not** over the sentence:

| sentence | polarity |
|---|---|
| *"document X as a suspected allergy so that it is **not** given again"* | asserted |
| *"she does **not** have asthma **but** has a penicillin allergy"* | asserted |
| *"**no** history of asthma, diabetes, or penicillin allergy"* | denied |

A contrast marker ends a negation's reach; a comma does not. Before this, all
three read as denials — including *"known penicillin allergy, no other drug
allergies"*, the most ordinary phrasing there is.

**2. Doseless drugs were invisible.** `DRUG_CTX` requires name + number + unit,
so 9 of 11 gold drugs were missed even with a perfect transcript. A known
formulary name standing alone is now recorded RED with `dose: null`. Guards: a
name the allergy patterns captured is skipped (span overlap, not a proximity
window — that dropped the insulin from *"takes insulin and has no known drug
allergies"*); a name in a negated clause is skipped (*"cannot take amoxicillin"*
is a prohibition); `_prefer_dosed` keeps one row per drug and lets the dosed row
win.

**3. Whole vital types had no rule.** No respiratory-rate and no glucose pattern
existed, so clip 006 — a hypoglycaemia call about glucose 54 → 72 mg/dL —
extracted zero vitals. Glucose also matches on its unit alone, written or spoken
(*"fifty-four milligrams per decilitre"*).

**4. Cues assumed telegraphic speech.** `pulse\s*\d` missed *"pulse rate is 92"*
in 6 of 7 clips. Cues now use a bounded clause-limited gap, so `pulse`,
`pulse rate` and `heart rate` read with or without a copula, hedge or stutter.
BP tolerates a stray digit run into the copula (*"blood pressure is19 160/90"*).
Guards against the obvious failure: a duration unit after the number is an
interval, not a reading (*"heart failure for 10 years"*), and a comma ends the
cue's scope (*"low sugar diet, review after 15 days"*).

Also: `numwords_to_digits` gained spoken decimals (*"ninety-eight point six"* →
`98.6`) and hundreds said without "hundred" (*"one ten"* → `110`).

**Overfitting guard.** These 7 clips are both the bug report and the scoreboard.
`demo/tests/test_extract_rules.py` adds 54 cases written **for the test**, not
copied from the corpus — including the negative cases that caught three real
misfires during the change.

## Appendix B — LM fusion: unblocking it, and what it does not do

**The fork is not required.** The model card's notebook installs
`git+https://github.com/mediacatch/pyctcdecode.git@ff49fc5`. `LasrCtcBeamSearchDecoder`
touches only `build_ctcdecoder` and `decode_beams`, both stable upstream. The one
difference this code depends on is that `decode_beams` returns tuples upstream
and dataclasses in the fork — four lines, not a dependency to trust. (Not
verified that this is the *only* difference; the fork's source was never
fetched.)

**It does nothing for general English.** 40 clean FLEURS bases, both arms run
identically, scored with `en_pilot.score.word_error_rate`:

| | mean WER |
|---|---|
| MedASR greedy | 0.2748 |
| MedASR + kenlm | 0.2788 |

**Paired Δ +0.0040, 95% CI [−0.0208, +0.0350], p=0.845** (10,000-iteration
bootstrap, seed 1729). Better on 13 bases, worse on 9, identical on 18.

This **corrects a prediction** made in the 2026-10-01 denoise writeup: that the
MedASR-vs-Whisper gap was "a floor, not a verdict" and LM fusion would narrow it.
It does not. MedASR stays at ~0.275 against Whisper `base`'s 0.105 on general
English.

Both results are one result: `lm_6.kenlm` is a *medical dictation* LM. FLEURS is
Wikipedia prose about global warming. A domain LM has nothing to add there and
everything to add to "amoxicillin with clavulanic acid" — **it helps exactly
where the domain matches, which is this product's domain and not the
benchmark's.**

Still open: whether MedASR's *denoising* gain shrinks under LM fusion. That needs
the 1,440-condition noise matrix re-decoded with beam search.

**A lead, not a finding.** Decoding the same clips *bypassing* `clean_audio`
scores 38/41, additionally recovering `clavulanic acid`. The decode is
deterministic and the two inputs genuinely differ (up to 10,322 of 32,767 full
scale, despite the log rounding to −20.0 → −20.0), so the gain stage does
something real under beam search. But one fact sits inside overlapping intervals.
Worth a look, not a conclusion.

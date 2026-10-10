# MediBytes pipeline and 30-clip dataset: accuracy, timing and mistakes

Draft, 2026-10-08. Every number comes from a command run on this machine today (commands are in Parts D and E7). Source data: `runs/v1` (latest run), `runs/v0` (the run before it), `clinical_eval/gold_v2.json` (ground truth).

**Read this first**
- **Which dataset is which.** Parts A–D cover the 30-clip MediBytes clinical-dictation set (`MB_MED_001`..`030`, from `MediBytes_Medical_Audio_Dataset (1).zip`). I first took that to be "the Hugging Face dataset" and was wrong: the Hugging Face dataset on this machine is **`united-we-care/United-Syn-Med`** (synthetic drug-name speech, cached in `~/.cache/huggingface`). It is covered in **Part E**, which also adds the local LLM and the context-repair stage. The older FLEURS corpus belongs to the denoising pilot and is not covered here.
- **The ground truth is agent-authored, not clinician-reviewed.** Three blind annotators, then adjudication, a critic and skeptic verification, all working from reference transcripts only.
- **WER is not reported.** The reference transcripts are paraphrases, not verbatim, so word error rate cannot be computed. "Accuracy" below means fact accuracy: did the pipeline extract each gold vital, drug, dose and allergy correctly?
- **Clips 001–007 are not an honest test.** The extractor was developed against them. Quote the held-out figure (clips 008–030) instead.

---

## Part A. The pipeline

### A1. What it does

Source: `README.md`. The package was ported from `demo/` to `base_pipeline/` (commit `b2f5414`); `stt_extract.py` has since been split into `asr.py`, `extract.py` and the glue.

```
audio -> clean (16 kHz mono, RMS normalise, VAD)  [denoiser: none by default]
      -> ASR (google/medasr on GPU)  + second decoder pass (small-int8) to recover missed items
      -> normalise text -> regex extraction (vitals, drugs, allergies, negation)
      -> fuzzy drug repair against a national drug reference (2,542 substances, 71,872 brands)
      -> optional LLM steps (validator, extraction, discharge note): skipped in these runs
```

Run configuration for both runs: denoiser `none`, MedASR plus the automatic second pass, regex-only extraction. The LLM term validator reported `unavailable (model llama3.2:3b not served)` on all 30 clips and the discharge-note stage was skipped. **Parts A–D do not measure the LLM paths; Part E does.**

### A2. Accuracy

| Slice (run `v1`, borderline items excluded) | Facts correct | Accuracy | 95% interval (clip-level bootstrap) |
|---|---|---|---|
| In-sample, clips 001–007 | 40/40 | 100% | 100–100 (Wilson 91.2–100) |
| **Held-out, clips 008–030** | **111/123** | **90.2%** | **84.0–95.9** (Wilson 83.7–94.3) |
| All 30 clips | 151/163 | 92.6% | not computed |

By fact type, all 30 clips:

| Fact type | Correct | Notes |
|---|---|---|
| Vitals | 117/125 (93.6%) | 8 misses |
| Drug names | 30/34 (88.2%) | 4 misses |
| Doses and allergies | 4/4 | only 4 exist in the gold set, all on in-sample clips |

**Precision.** The pipeline emitted 132 vitals, of which 117 matched gold, so 15 were extra (88.6% precision). It emitted 30 drug rows, all correct, so there were 0 spurious drugs. No drug came out with the wrong polarity (negated or not) in `v1`.

**`v0` to `v1`.** One fact changed, in clip 005: an amoxicillin row that is part of an allergy statement was marked as given in `v0` and is correctly marked negated in `v1`. That is net +1 fact, exact McNemar p = 1.0, so it is not statistically meaningful. The fix matters because wrong polarity on an allergy is the dangerous kind of error.

**What the second decoder pass is worth.** It recovered 10 gold facts the primary decoder missed: 7 drugs (paracetamol in 5 clips, amlodipine, sodium valproate) and 3 vitals. It also added 6 extra vital rows, 5 of them garbled and flagged RED (cluster C3). On balance it helps.

**What noise does and doesn't do.** Fact accuracy by tier (`v1`): clean 55/58 (94.8%), medium 48/53 (90.6%), heavy 48/52 (92.3%). With 52–58 facts per tier these intervals overlap heavily, so I would not claim noise hurts extraction. All 3 clean-tier misses are misheard numbers (C1), so clean audio does not prevent them. The transcripts do visibly degrade in medium and heavy audio (C7). The denoiser question has been tested separately: no denoiser has beaten `none` (`SESSION-SUMMARY.md`, `base_pipeline/README.md`), so `none` stays the default.

### A3. Time taken

Timing is for the regex-only pipeline on `cuda:0`, with the second pass included. Audio averages 79.5 s per clip (2,385 s in total).

| Stage | Mean | Notes |
|---|---|---|
| Clean (resample, normalise, VAD) | 0.08 s | negligible |
| ASR + second pass + extraction | 13.5 s (median 12.7, max 21.5) | 11.6 s mean in `v0` |
| **End to end** | **~13.6 s per clip, real-time factor 0.171** | 0.148 in `v0` |

| Tier (`v1`) | Mean ASR+extraction | Real-time factor |
|---|---|---|
| Clean | 13.1 s | 0.168 |
| Medium | 12.7 s | 0.165 |
| Heavy | 14.8 s | 0.179 |

- Slowest clip: `MB_MED_001` at 21.5 s. It is the first clip in the run, and in `v0` it took 27.8 s, so model warm-up is the likely cause. I did not isolate it.
- `v1` ran about 15% slower than `v0` (13.5 s vs 11.6 s mean) on the same second-pass code. I did not isolate the cause, and the two runs ran back to back on a shared machine.
- The first-pass and second-pass costs are not logged separately, so I can't give a per-pass split.
- LLM steps, if enabled, add time that is not measured here.

### A4. Test status

`python -m pytest base_pipeline clinical_eval`: **334 passed in 18.3 s** (project `.venv`).

### A5. Mistakes by pipeline stage

| Stage | Mistakes | Count (`v1`) |
|---|---|---|
| ASR hears the wrong number | pulse, temperature or BP value replaced by a plausible but wrong one | 6 misses (C1), plus 1 wrong row that the second pass overruled |
| ASR drops a digit | `122/76` read as `22/76` | 1 miss (C2) |
| ASR garbles or drops a drug name | the word is gone or unrecognisable | 3 misses (C5) |
| Second pass adds a garbled duplicate | extra RED or YELLOW rows | 6 extra rows (C3) |
| Extractor misses a mangled label | `respiratory ract is 26` | 1 miss (C4) |
| Drug repair refuses a near-miss | `Paracetamine` | 1 miss (C6) |

The README's claim that "the extractor is at 100% on perfect text; the loss is the decoder not emitting the word" holds up here: 10 of the 12 misses start in the decoder. The two that don't are C4 and C6, and both are cheap to fix.

---

## Part B. The 30-clip dataset

### B1. Composition

30 MP4 clips of roughly 65–94 s each. Tiers: `clean` 001–010, `medium` 011–020, `heavy` 021–030. The gold set holds 163 scored facts: 125 vitals, 34 drugs, 4 doses or allergies. Facts the annotators flagged as doubtful (oxygen, iron, a conditional drug) are excluded from the headline numbers.

### B2. Mistake register: all 12 misses and 15 extra rows

**Misses** (gold fact not found), 12 in total:

| Clip | Tier | Gold | What the pipeline did | Cluster |
|---|---|---|---|---|
| 008 | clean | temp 99.8 °F | heard `99 degrees` (decimal lost) | C1 |
| 009 | clean | temp 101.4 °F | heard `100.4` | C1 |
| 009 | clean | pulse 106 | heard `160` | C1 |
| 011 | medium | pulse 104 | heard `140` | C1 |
| 023 | heavy | temp 100.8 °F | heard `100.9` | C1 |
| 029 | heavy | pulse 88 | heard `98` | C1 |
| 013 | medium | BP 122/76 | heard `22/76` (flagged RED as implausible) | C2 |
| 021 | heavy | resp. rate 26 | transcript said `respiratory ract is 26 breaths per minute`; the extractor didn't match it | C4 |
| 013 | medium | salbutamol | no trace in the transcript | C5 |
| 013 | medium | ipratropium bromide | no trace in the transcript | C5 |
| 015 | medium | ondansetron | `has also been givenaro for vomiting` | C5 |
| 023 | heavy | paracetamol | `Paracetamine has been given for fever`; repair refused | C6 |

**Extra rows** (the pipeline emitted something gold doesn't contain), 15 in total:

| Kind | Rows | Where | Visible to a reviewer? |
|---|---|---|---|
| Wrong value, looks normal (YELLOW) | 7 | 008, 009 ×2, 011, 022, 023, 029 | **No.** Nothing marks them as wrong. |
| Wrong value, implausible (RED) | 1 | 013 `22/76` | Yes: "physiologically implausible - human must confirm" |
| Second-pass garble (RED) | 5 | 015 `400 over 65`, 018 `110 over 17` and `Pulse is 940`, 020 `1-18 over 76`, 022 `110 over 17` | Yes, flagged RED |
| Duplicate of a correct reading | 2 | 003 (BP twice), 004 (pulse twice) | Yes, but as clutter |

The 7 unflagged wrong values are the risky ones. A clinician reading "pulse is 160" or "temperature 99" sees nothing wrong.

### B3. Clusters

Clusters are grouped by root cause, so each one points at a single fix. Counts are from the register above.

| # | Cluster | Misses | Extra rows | Fix location | Effort |
|---|---|---|---|---|---|
| C1 | **ASR misreads a number as another plausible number** | 6 | 7 (unflagged) | Decoder or cross-check | Hard |
| C2 | ASR drops a digit: BP `122` becomes `22` | 1 | 1 (RED) | Repair or re-decode | Medium |
| C3 | Second pass adds garbled or duplicate vitals | 0 | 6 | Second-pass merge rule | Easy |
| C4 | Extractor misses a vital when the label word is mangled | 1 | 0 | `stt_extract.py` | Easy |
| C5 | Drug name lost or garbled by the decoder | 3 | 0 | Decoder or contextual hints | Hard |
| C6 | Fuzzy repair refuses an obvious near-miss | 1 | 0 | `drug_reference.py` | Easy |
| C7 | Garbled transcript text that nothing scores | 0 | 0 | Decoder | Hard, and unmeasured |

**C1. Number misreads (6 misses, 6 clips).** The transcript itself holds the wrong number, so no extractor change can fix it.
- Pulse: `106`→`160` (009), `104`→`140` (011), `88`→`98` (029).
- Temperature, tenths digit: `99.8`→`99` (008), `101.4`→`100.4` (009), `100.8`→`100.9` (023).
- Pulses dictated as "one hundred and N": 2 of 6 were misread (009 and 011); 005, 013, 021 and 024 were read correctly. Six is too few to call that systematic.
- Clip 022 had the same kind of error in the primary decoder (`100.4` for `102.4`). The second pass heard `102.4` correctly. That is the only time the two passes disagreed on a reading.
- The second pass produced no vital rows at all in the other five clips (008, 009, 011, 023, 029), so a "passes disagree" check would not have caught them.

**C2. Digit drops.** `122/76` became `22/76` in 013. The pipeline caught that it was implausible and flagged it RED, so the system worked, but the correct value was lost. The second pass also produced digit-damaged BPs: `110 over 17` for `110/70` (018, 022) and `1-18 over 76` for `118/76` (020). A plausibility failure is a natural trigger for re-decoding just that sentence.

**C3. Second-pass noise.** The second pass was designed to fill gaps, but it also adds rows when the primary decoder already got the reading right. Five of the six extra rows are garbled and flagged RED (015, 018 ×2, 020, 022); one is a clean duplicate (003).
- A blanket rule of "drop a second-pass vital when the primary has a plausible one" would remove all six, but it would also remove the one genuine recovery (022 `102.4`).
- A safer rule is to drop second-pass rows that fail the plausibility check, and to show a second-pass row as an alternative when it disagrees with a plausible primary value.

**C4. Mangled label.** In 021 the transcript reads `His respiratory ract is 26 breaths per minute`. The value and the unit `breaths per minute` are both intact; only the label is wrong. The recent commit "second pass recovers vitals keyed on the reading not the wording" already does this for the second pass. Applying the same unit-anchored rule to the primary extractor would fix this clip. Clip 004's `heartse rate was 96` shows the extractor already copes with some label damage.

**C5. Lost drug names.**
- 013: `nebulisation with salbutamol and ipratropium bromide` came out as `advised treatment`. Both names are gone.
- 015: `ondansetron` came out as `givenaro`. The fuzzy matcher found nothing close enough.
- Ondansetron was misspelled but still repaired in other clips (`odansetron` twice, `ondsetron`), so this is a one-off garble. The sentence structure `given X for vomiting` points to an antiemetic, which could support a context-based candidate list.
- `medasr-lm` (language-model fusion) was measured worse and is off by default, so it is not an easy fix.

**C6. Repair refuses `Paracetamine`.** This one looks systematic, so I checked it:
- Paracetamol appears 8 times in the gold set. In the 2 clean clips (001, 007) the primary decoder gets it right.
- In the 6 medium and heavy mentions (016, 017, 019, 022, 023, 028) the primary decoder mangled the word every time: `Paracetamine` (022, 023), `Parocetamone` (028), and garbled forms I identified from context: `furosemal` (016), `parazema` (017), `fromtain` (019).
- The second pass recovered 5 of those 6. It failed on 023.
- Fuzzy repair scores `paracetamine` against paracetamol at **0.783**, just above the 0.78 floor (`FUZZY_FLOOR` in `drug_reference.py`), and ties with `altretamine`, a chemotherapy drug, at 0.783. A tie means the repair backs off.
- So the primary decoder reliably gets this very common drug wrong, and the pipeline only survives because of the second pass.
- A prior that favours common drugs over rare ones, or a small phonetic alias list for the top 20 drugs, would resolve the tie.

**C7. Garbled text that nothing scores.** The facts the gold set measures are mostly fine, but the surrounding transcript degrades in medium and heavy clips. These errors are not scored, and the discharge-note stage that would use them was skipped, so the real impact is unmeasured:
- Demographics: `veryarriing he -year-old male` (011), `2-year-old man` (021), `a 427 moth` (022).
- Clinical wording: `hemoglobin normal88 g/d` (018), `deingu and maleria` (022).
- Meaning reversed or lost: `I insomnia immediately` for "I will inform you immediately" (011), `I on she is unable to speak normally` (022).
- Whole sentences collapsed: clip 015, from `patient at bedside. She is attoring following up scol in sea.`

### B4. Suggested order of work

*Update:* item 1 (C6, the paracetamol tie) was tried in Part E. The context stage repaired clip 023's `Paracetamine`; it is opt-in and still needs the checks listed there before it becomes a default.

1. **C6**: paracetamol tie-break or alias list. One line of reasoning, protects the most common drug in the set.
2. **C4**: unit-anchored primary extraction. Recovers one fact with no new risk.
3. **C3**: gate second-pass vitals on plausibility. Removes 6 noisy rows without losing the 022 recovery.
4. **C1 and C2**: add a check that makes silent wrong numbers visible, for example re-decoding a vital with a second model and flagging disagreement. This is where most of the remaining accuracy is (7 of 12 misses, and the only errors a clinician can't see). It needs a decoder-level idea, not an extractor patch.
5. **C5 and C7**: need a better or context-biased decoder; evaluate after the above.

---

## Part C. Limits of this report

- The held-out slice has 123 facts, so one fact is about 0.8 points and the 95% interval is roughly ±6.
- The gold set was written by agents from reference transcripts; no clinician has reviewed it. Treat gold errors as possible.
- Facts within a clip share a speaker, a recording and one ASR run, so they are not independent. The bootstrap resamples whole clips for that reason.
- "Spurious" means "not in gold". A few extra rows might be real readings the annotators left out; I did not audit that.
- Every clip in both runs used denoiser `none`; no denoising arm was run on this dataset.
- Timing comes from two runs on one machine with no repeats, so treat differences under about 20% as noise.

## Part D. How to reproduce

```
python -m clinical_eval.fact_score runs/v1 --compare runs/v0 --misses     # accuracy, precision, miss list
.venv/bin/python -m pytest base_pipeline clinical_eval -q                  # 334 passed
python -m base_pipeline.run <clips> --out runs/vN                          # a new run
```

Timing figures are parsed from each clip's `report.md` (Stage 1 and Stage 2 lines) and `clean_meta.json` (`duration_s`). `runs/` is gitignored and exists only on this machine.

## Part E. Local LLM and context-based repair on the Hugging Face dataset (`United-Syn-Med`)

### E1. What was added

- **Local LLM:** `llama3.2:3b` through Ollama (2.0 GB, already supported by `llm_client.py`; nothing in the pipeline was changed to use it). With it served, the existing term validator switches on by itself.
- **New stage `base_pipeline/context_repair.py`** (opt-in, `--context-repair`). For a word in a medicine-related sentence that could be a misheard drug name, it shortlists substances by spelling at the reference module's own `FUZZY_FLOOR` (0.78), then asks the LLM which shortlisted substance, if any, the sentence is about. Between 0.78 and the 0.84 bar for a spelling-only row, the sentence decides. Rules: closed set (the model can only pick from the shortlist), never above RED, "none" is a normal answer, and any failure leaves the entities untouched. It reuses the extractor's lab-request guard so a test name is never treated as a drug.
- **A bug fix found by this dataset:** `dapagliflozin`, spelled correctly, was "repaired" to the different drug `canagliflozin` (0.85). The reference lists dapagliflozin without a dispensing record, so exact lookup refuses it and the sound-alike matcher then ran on it. A word that exactly names a listed substance is no longer sent to the matcher (`drug_reference.is_substance_name`).
- 13 new tests; the full suite is 347 passed.

### E2. The data

`united-we-care/United-Syn-Med` (CC-BY-SA-4.0) is synthetic speech: one carrier sentence per clip with a brand or generic drug in it, plus the sentence as text. Audio ships only in multi-GB tarballs, so `clinical_eval/syn_med.py` streams the start of the test tarball and draws a seeded sample.

| Set | Clips | Use | Gold substances | Gold brand names |
|---|---|---|---|---|
| Development (`united-syn-med-40`) | 40 | where the stage was built and tuned | 15 | 27 |
| **Held-out (`united-syn-med-40b`)** | 40 | disjoint tar members; gold written from the transcriptions **before** any run on it | 12 | 30 |

Gold is single-annotator (me), read off each clip's own dataset text, and not clinician-reviewed. It is in `clinical_eval/syn_med_gold.json` and `syn_med_gold_40b.json`. Botanicals, a sweetener, a biologic and an amino acid are marked borderline and left out of the headline counts. Scoring is in `clinical_eval/syn_med_score.py`: **heard** (the name is in the transcript, fuzzy ratio ≥ 0.80), **emitted** (a drug row for each gold substance), **spurious** (rows naming nothing the clip says). Brands are scored as "heard" only, because the pipeline deliberately never turns a brand into a row.

### E3. Results

All arms use the same ASR and second pass, so only post-processing differs. "Guard" is the dapagliflozin fix; it is on in every arm except the first.

| Arm | Substance rows emitted (dev, 15) | Spurious rows (dev) | Substance rows emitted (held-out, 12) | Spurious rows (held-out) |
|---|---|---|---|---|
| A. baseline, regex only | 10 | 2 (`canagliflozin`, `sucrose`) | n/a | n/a |
| A2. + guard | 10 | 1 (`sucrose`) | 8 | 0 |
| B2. + LLM term validator | 10 | 1 | 8 | 0 |
| C1. + context repair, **first version** | 12 | **12** | not run | not run |
| **C3. + context repair, final** | **12** | **1** | **8** | **0** |

Paired against A2: development +2 facts and no losses (exact McNemar p = 0.5); held-out **0 change**; the 30 clinical clips +1 and no losses (p = 1.0). That is 3 facts gained and none lost across 110 clips, which is too few to call an improvement. It does not show the stage is safe; see the fresh-draw check below.

**What the stage fixed:** `albonazol` → albendazole and `cobisystat` → cobicistat on the development set, and `Paracetamine` → paracetamol on the clinical clips (clip 023, the C6 miss in Part B; the model broke the paracetamol/altretamine tie from "given for fever"). On the held-out set it made **zero LLM calls**, because no word fell in the 0.78–0.84 window, so it neither helped nor hurt there.

**The first version was unsafe, and why that matters.** With a 0.70 floor and no word filter, it added 11 false drug rows on the development set. The 3B model approved ordinary English words with 0.9 confidence and invented justifications: `certain` → ceritinib, `boosting` → bosutinib, `mental` → fentanyl, `treating` → tretinoin, `etabonate` → etidronate. Two filters fixed it: a list of ordinary English words (`assets/english_words.txt`, from the BERT vocabulary, Apache-2.0) and the 0.78 floor. I chose the 0.78 floor after seeing those results, and it is the same constant the dosed path already uses, so treat the development numbers as tuned and the held-out and clinical-clip numbers as the check. The model's confidence is not informative: it said 0.9 for right and wrong answers alike.

**A second false positive turned up on the clinical clips:** `troponin` → tiopronin, from "requested cardiac troponin levels". The extractor already refuses that sentence as a lab request; the new stage had not been told to. After reusing the same guard, a re-run removed it and kept the paracetamol fix. I made that fix after seeing the result, so it is not an independent test.

**Precision on a larger fresh draw: poor.** After seeing the above I drew 200 more clips (tar members 1201–1800, `united-syn-med-200c`), ran the final stage, and checked each repaired row against the clip's own dataset text (no hand annotation needed). It made 13 LLM calls and repaired 5 rows; **2 were right, 3 were wrong**:

| Heard | Stage chose | Clip actually says | Verdict |
|---|---|---|---|
| `Lumeparone` | lumateperone | lumateperone | right |
| `Syinatapride` | cinitapride | cinitapride | right |
| `roxitidine` | **ranitidine** | roxatidine acetate | wrong: a sound-alike drug |
| `Vbuterol` | **salbutamol** | bambuterol | wrong: the model's reason was invented ("Vbuterol is a brand name for salbutamol") |
| `Ethylamino` | **diethylamine** | ethylaminobenzoate | wrong |

All five are RED rows that say they were repaired and need a physician, so none would pass as confirmed, but 60% of the suggestions on fresh data were wrong, and two of the wrong ones are the sound-alike pairs this pipeline's validator was written to prevent. The development-set result (+2, no wrong rows) did not hold up.

**The validator did not change any score.** On the development set it judged 6 rows (4 "corrected", 2 "ambiguous"). Its stated reasons are near-copies of the prompt ("matches dose, unit, frequency and indication" for sentences that have neither). It left `sucrose` for `sucralose` as "ambiguous" rather than rejecting it. A 3B model is not a dependable judge here; it is a flag for a human.

### E4. Time taken

| Arm | Clips | Audio per clip | ASR + extract per clip (mean / median / max) | LLM time |
|---|---|---|---|---|
| A2 regex only, development | 40 | 6.4 s | 3.4 / 3.3 / 9.5 s | none |
| B2 + validator, development | 40 | 6.4 s | 3.5 / 3.3 / 9.8 s | 4.0 s over 6 rows (about 0.7 s each) |
| C3 + context repair, development | 40 | 6.4 s | 4.1 / 3.7 / 13.5 s | validator 3.9 s, context 1.8 s over 3 calls (about 0.6 s each) |
| C3 held-out | 40 | 6.0 s | 4.8 / 3.7 / 26.2 s | validator 0.7 s, context 0 calls |
| Clinical clips, A2 → C3 | 30 | 79.5 s | 9.6 → 10.7 s | validator 4.8 s, context 2.2 s over 4 calls |
| C3 fresh draw (200c) | 200 | 6.3 s | 4.0 / 3.7 / 16.6 s | validator 19.8 s over 30 rows (0.7 s each), context 7.6 s over 13 calls (0.6 s each) |

- The LLM calls themselves are cheap, about 0.6 s each on GPU. The visible extra cost in C3 is mostly the first call of a run: the slowest held-out clip took 26 s against a 3.7 s median, consistent with Ollama cold-loading the model. I did not isolate it.
- GPU memory read 3.8 GB of 8.2 GB once during a context-repair run, with MedASR, the second decoder and the 3B model all loaded. That is a single sample, not a peak.
- Timings vary by about 30% between runs on this machine (the clinical clips averaged 13.5 s per clip in the earlier run and 9.6 s in this one, with the same code path), so only large differences mean anything.

### E5. Mistakes on the Hugging Face data, clustered

Over both sets (57 brand names and 27 substances), after the final stage:

| # | Cluster | Count | Examples | Where to fix |
|---|---|---|---|---|
| H1 | **Brand name respelled by ASR** | 33 of 57 not "heard" at 0.80: 18 near-misses (ratio 0.65–0.80), 15 lost (< 0.65) | `Zenndoplus tab` for Xenda Plus Tab; `Olme trocum` for OLMETRACK-AM; `Imitrex` for Amitrac; `Z` for ZLIMETIN | Decoder. These are invented words, so a stricter spelling test is unfair to ASR; a looser one (≥ 0.65) is met by 42 of 57. Nothing downstream can fix them without a brand list. |
| H2 | **Generic name not heard** | 3 substances | dutasteride → `dude`; feredetate → `ferrititate` (0.67); elvitegravir → `ofvigravir` (0.73) | Decoder; below the 0.78 floor. |
| H3 | **Name absent from the drug reference** | 2 substances | simeprevir (`sprevir`); piperazine (only `piperazine derivative antihistamine` is listed) | Reference coverage. |
| H4 | **Heard exactly, but no row built** | 2 substances | `dapagliflozin` (listed with no dispensing record); `pamidronate` (only `disodium pamidronate` is listed) | Allow an exact name, or one word of a listed salt name, to become a RED row in a medicine-cue sentence. Not done: I did not tune on held-out failures. |
| H5 | **Wrong sibling for a word that is not in the list** | 1 row (plus 1 removed by the guard) | `sucralose` → `sucrose` (0.88); `dapagliflozin` → `canagliflozin` before the guard | Matcher plus validator; the model only marked it "ambiguous". |
| H6 | **Misheard name repaired from context** | 3 fixed | albendazole, cobicistat, paracetamol | Done by the new stage. |

Substance recall after the final stage is 12/15 on development and 8/12 on held-out (74% over the 27). 21 of the 27 were heard, and rows were built for 20, so a repair that works from the transcript has little room above the ASR itself.

### E6. What I would and wouldn't claim

- **Do not enable context repair by default.** On clips it was not tuned on it was idle (held-out, 40 clips: zero calls) or right 2 times in 5 (fresh 200). Its gains (+2 development, +1 clinical, +2 right on the fresh draw) are outweighed by 3 wrong drug suggestions on the fresh draw. It stays opt-in.
- The earlier safety result, "no wrong rows on 110 clips", is only a statement about the data the filters were built on. The fresh 200 is the honest precision figure: 2 of 5.
- The local LLM works as a mechanism (about 0.6 s per call, GPU-resident, closed-set output enforced), but a 3B model cannot tell `roxatidine` from `ranitidine` or `bambuterol` from `salbutamol` from the sentence alone, because both fit the same indication. Spelling plus sentence context is not enough for sound-alike pairs.
- The term validator did not change any score; its reasons are not reliable. Treat both LLM steps as flags for a human.
- The biggest losses on this dataset are upstream (H1, H2) and in the drug reference (H3, H4). A bigger LLM would not fix those.
- Ideas I did not test: require the second decoder to hear the same word before the stage may repair it; cap the stage to candidates with no sound-alike sibling inside the band; use a larger model; add the missing names to the reference.
- Untested: LLM-primary extraction (`--llm`), the discharge note, and any model larger than 3B.

### E7. Reproduce

```
python -m clinical_eval.syn_med fetch --n 40                 # development sample
python -m clinical_eval.syn_med fetch --n 40 --heldout       # disjoint held-out sample
python -m base_pipeline.run <audio dir> --out runs/X --context-repair      # needs: ollama pull llama3.2:3b
python -m clinical_eval.syn_med_score runs/X --gold heldout --compare runs/Y --detail
```

Runs are in `runs/` (gitignored): `sync_c3_context` (the 200-clip precision draw), `syn_a_baseline`, `syn_a2_guard`, `syn_b_validator`, `syn_b2_validator`, `syn_c_context` (first version), `syn_c3_context`, `synb_*`, `mb_a2_guard`, `mb_c3_context`.

## Appendix. Per-clip results (`v1`)

Real-time factor is (clean + ASR) divided by audio duration. "Extra" counts spurious vital rows.

| Clip | Tier | Audio (s) | Clean (s) | ASR+extract (s) | RTF | Facts | Extra | Misses |
|---|---|---|---|---|---|---|---|---|
| MB_MED_001 | clean | 87 | 0.13 | 21.5 | 0.25 | 9/9 | 0 | - |
| MB_MED_002 | clean | 84 | 0.08 | 12.8 | 0.15 | 6/6 | 0 | - |
| MB_MED_003 | clean | 86 | 0.09 | 14.6 | 0.17 | 3/3 | 1 | - |
| MB_MED_004 | clean | 82 | 0.08 | 14.2 | 0.17 | 5/5 | 1 | - |
| MB_MED_005 | clean | 80 | 0.08 | 12.1 | 0.15 | 6/6 | 0 | - |
| MB_MED_006 | clean | 67 | 0.07 | 10.9 | 0.16 | 4/4 | 0 | - |
| MB_MED_007 | clean | 77 | 0.08 | 10.8 | 0.14 | 7/7 | 0 | - |
| MB_MED_008 | clean | 79 | 0.08 | 12.7 | 0.16 | 7/8 | 1 | temp_f |
| MB_MED_009 | clean | 67 | 0.07 | 10.6 | 0.16 | 3/5 | 2 | temp_f, pulse |
| MB_MED_010 | clean | 71 | 0.07 | 10.5 | 0.15 | 5/5 | 0 | - |
| MB_MED_011 | medium | 86 | 0.08 | 11.1 | 0.13 | 5/6 | 1 | pulse |
| MB_MED_012 | medium | 90 | 0.10 | 12.9 | 0.14 | 4/4 | 0 | - |
| MB_MED_013 | medium | 85 | 0.08 | 10.9 | 0.13 | 4/7 | 1 | bp, salbutamol, ipratropium bromide |
| MB_MED_014 | medium | 94 | 0.10 | 11.4 | 0.12 | 5/5 | 0 | - |
| MB_MED_015 | medium | 71 | 0.07 | 14.4 | 0.20 | 5/6 | 1 | ondansetron |
| MB_MED_016 | medium | 76 | 0.07 | 18.2 | 0.24 | 5/5 | 0 | - |
| MB_MED_017 | medium | 72 | 0.07 | 10.7 | 0.15 | 6/6 | 0 | - |
| MB_MED_018 | medium | 66 | 0.06 | 9.4 | 0.14 | 4/4 | 2 | - |
| MB_MED_019 | medium | 67 | 0.06 | 13.5 | 0.20 | 5/5 | 0 | - |
| MB_MED_020 | medium | 67 | 0.06 | 14.1 | 0.21 | 5/5 | 1 | - |
| MB_MED_021 | heavy | 87 | 0.09 | 15.8 | 0.18 | 7/8 | 0 | rr |
| MB_MED_022 | heavy | 82 | 0.08 | 18.8 | 0.23 | 5/5 | 2 | - |
| MB_MED_023 | heavy | 81 | 0.10 | 15.5 | 0.19 | 3/5 | 1 | temp_f, paracetamol |
| MB_MED_024 | heavy | 73 | 0.08 | 12.8 | 0.18 | 6/6 | 0 | - |
| MB_MED_025 | heavy | 88 | 0.09 | 12.6 | 0.14 | 4/4 | 0 | - |
| MB_MED_026 | heavy | 87 | 0.09 | 12.2 | 0.14 | 6/6 | 0 | - |
| MB_MED_027 | heavy | 79 | 0.08 | 20.4 | 0.26 | 4/4 | 0 | - |
| MB_MED_028 | heavy | 84 | 0.10 | 15.3 | 0.18 | 5/5 | 0 | - |
| MB_MED_029 | heavy | 83 | 0.08 | 12.5 | 0.15 | 3/4 | 1 | pulse |
| MB_MED_030 | heavy | 89 | 0.09 | 12.2 | 0.14 | 5/5 | 0 | - |

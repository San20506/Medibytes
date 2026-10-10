# Results: Raw ASR Decoding Presets

Whisper large-v3, GPU float16, raw transcript only (no second pass, validator, LLM or fuzzy repair
in the score). Five agents, one per preset, ran the real pipeline; every number below was
re-scored independently from `transcript.json` by the coordinator and matched the agents' reports.
Tuning sets `dev40` + `held40` (80 clips, already seen by earlier work); held-back `test30`
(30 clips, gold written before any run, scored once per preset).

## WER (%), pooled per set

| preset | dev40 | held40 | test30 |
|---|---|---|---|
| P1 `greedy` (current default) | 15.60 | 33.91 (11.33 without SYN_024) | 9.65 |
| P2 `beam5` | 15.23 | 13.60 | 10.15 |
| P3 `beam5-en` | 15.23 | 13.60 | 10.15 |
| P4 `beam5-en-fallback` | 15.23 | 15.90 | 10.15 |
| P5 `beam5-en-fallback-prompt` | **11.38** | **10.73** | **8.91** |

## Paired against `greedy` (WER difference in points, 95% clip-bootstrap interval)

| preset | test30 (the spec's criterion set) | all 110 clips | 107 clips without the 3 repeat-hallucination clips |
|---|---|---|---|
| `beam5` / `beam5-en` | +0.50 (-0.48, 1.48) n.s. | -7.21 (-23.69, 1.69) n.s. | +0.42 (-0.15, 1.06) n.s. |
| `beam5-en-fallback` | +0.50 (-0.48, 1.48) n.s. | -6.39 (-21.20, 1.69) n.s. | +0.42 (-0.15, 1.06) n.s. |
| `beam5-en-fallback-prompt` | -0.74 (-1.64, 0.00) n.s. | -9.99 (-27.71, -0.41) significant | -0.50 (-1.05, 0.00) n.s. |

## Findings

1. **The WER differences are dominated by a handful of repeat/continuation hallucinations, not by
   across-the-board accuracy.** Greedy decoding of cleaned audio with word timestamps repeats part
   of the sentence on `held40:SYN_024` (136 words for a 19-word reference) and `dev40:SYN_040`
   (39 words for 20). Without the three clips that any preset hallucinates on, the presets are within
   about half a point of each other.
2. **Beam search (P2) is not an accuracy gain.** It removes the SYN_024 runaway but is slightly worse
   on normal clips (+0.42, n.s.) and introduces its own repeat on `held40:SYN_016` (30 words for 19).
3. **`language="en"` changes nothing** (P3 is identical to P2 on all 110 clips): every clip is
   already detected as English.
4. **The temperature fallback does not catch these repeats** (P4): the compression-ratio guard
   (2.4) never fired on SYN_024 (35 words, a doubled phrase) or SYN_040 (33 words). P4 was worse than
   P3 on `held40`.
5. **The medical prompt (P5) is the only preset that removed both hallucinations** (SYN_040: 19 errors
   under greedy, 0 under P5), had the lowest WER on all three sets, raised brand-heard on the two
   tuning sets (dev40 14/27 vs 12/27 for P2-P4 but below greedy's 15/27; held40 21/30 vs 19/30),
   and leaked no prompt text into any of the 110 transcripts. On normal clips its advantage is small
   (-0.50 points, interval touching zero).
6. **Cross-check against the standalone script (task 4.7).** The script first disagreed with the
   pipeline on `dev40` greedy (11.01% vs 15.60%). Cause: the script decoded raw mp3 without word
   timestamps; the pipeline decodes cleaned audio with word timestamps. Re-run with the pipeline's
   settings the script gives 15.60% (85/545), identical to the agent's number, and 19 of the 25-word
   gap is `SYN_040`. So the agents' runs are reproduced; the pipeline path is the one that
   hallucinates more.

## Adoption criterion (spec)

A preset becomes the default only if on `test30` it lowers pooled WER with an interval excluding zero,
does not increase runaway clips, and does not lower substance-heard.

- P2, P3, P4: not met (WER not lower on test30).
- P5: **not met on test30** (-0.74, interval (-1.64, 0.00) touches zero), though it does not increase
  runaway clips and does not lower substance-heard (7/10 = 7/10). Pooled over all 110 clips it is
  significant, but that mixes the tuning clips the criterion excludes.

**Recommendation:** keep `greedy` as the default in this change. P5 is the candidate: consistent
direction on all three sets and the only preset that fixes both hallucinations. Adopting it needs a
larger held-back set (about 30 clips can detect only a few clips' difference), which is a separate decision.

## Follow-ups (not done here)

- The runaway metric (hyp > 3x reference) missed the doubling hallucination on SYN_040 (about 2x).
  A tighter threshold (for example 1.5x) should be specified before the next comparison.
- Test P5 on audio that is not single synthetic sentences; a medical prompt could bias differently on real dictation.

# Design: English-Only Denoising WER Pilot

## Evidence This Design Is Built On

Verified directly against the repository and the workstation's data root before
writing this design, not assumed:

- `evidence/pilot-b1/results/accuracy-probe-english.csv` — 20 rows, columns
  `condition_kind,decoder,backend,mean_wer,bases,conditions,note`. Every row
  carries `bases=4, conditions=8` and the in-file note "4 English bases only; 11
  of 15 bases are hi/ta and are not measurable here". `evidence/README.md:75-80`
  records the probe as run outside the frozen evaluator.
- `evidence/matrix-v2/results/matrix-results.csv` — 1,440 scored rows, columns
  `condition_id,speech,noise,category,snr_target_db,backend,snr_db,si_sdr_db,wer,
  reference_words,reference_tokens,enhance_s,pre_mix_gain`. Reading the shipped
  file directly and pooling `snr_target_db` in {0.0, 5.0} over the four English
  bases: `none` mean WER 0.1589 (n=48), `dpdfnet2-onnx` 0.1685 (+6.1%
  relative), `sherpa-gtcrn-simple` 0.1917 (+20.7%), `deepfilternet3` 2.1667
  (+1264%).
- The same file's `si_sdr_db` column is populated on 23 of 96 English `none`
  rows, 20 of 96 `deepfilternet3` rows, and **0 of 96** rows for both
  `dpdfnet2-onnx` and `sherpa-gtcrn-simple` — the degeneracy
  `evidence/matrix-v2/matrix_eval.py:281` records as `None`.
- `evidence/matrix-v2/results/enhance-all.csv` has columns
  `condition_id,backend,samples,enhance_s,enhanced_path,peak`. There is no
  `actual_backend` column, so the study's dispatch is not auditable from its own
  artefact.
- `/home/sandy/.local/share/medibytes-eval/sources/fleurs/en_us/validation/0000.parquet`
  — 394 rows, columns `id, num_samples, path, audio, transcription,
  raw_transcription, gender, lang_id, language, lang_group_id`; sha256
  `7c3eeb11a9597bd52cdc1b0d637e85389fe094cfd8763913e7bf4fdf7a853959`, the same
  digest `eval/corpus/manifest.jsonl` records for the frozen corpus.
  `eval/corpus/sources.py:20-22` pins `google/fleurs` at revision
  `168de341b3db6859a9bac1c50a2ef5e3b47647e0`, CC-BY-4.0.
- `/home/sandy/.local/share/medibytes-eval/noise-bank/demand-16k/` — 18 DEMAND
  recordings, each with a sha256, DOI, licence and category in
  `evidence/noise-bank/demand-bank-manifest.jsonl` (18 lines).
- `/home/sandy/.local/share/medibytes-eval/models/dpdfnet/dpdf8c40d95cd/dpPDFNet/dpdfnet2.onnx`
  and `.../models/sherpa-onnx/gtcrn_simple.onnx` are present; the per-backend
  venvs exist at `Medibytes-worktrees/eval-{dpdfnet,sherpa-onnx,compare}/.venv/bin/python`.
- `ctranslate2.get_cuda_device_count()` returns 1 on this workstation.
- `eval/reporting.py:27` pins `EXPECTED_GROUP_COUNT = 15`, and
  `cluster_bootstrap` raises `ReportingError` at `:680-681` for any other group
  count — so the frozen bootstrap is unusable over 40 bases and must be
  reimplemented, not called. `holm_adjust` (`:707-724`) takes an explicit
  `family_count` and is generic, so it is reused as-is.
- `eval/corpus/build_corpus.py:235-237` hard-rejects any spec whose
  `corpus_id` is not `pilot-15-v1`, `:84-85` requires exactly 15 resolved groups,
  and `corpus_tool_sha256()` (`:51-60`) binds the frozen manifest to its own
  recipe — so extending that path would invalidate the existing 15-record
  manifest.
- `demo/denoise_backends/` in this repository contains only `none.py` and
  `noisereduce.py`; the `dpdfnet2_onnx` and `sherpa_gtcrn_simple` adapters live
  in their own worktrees. `demo/denoise.py:305-313` dispatches by importing
  `demo.denoise_backends.<module>` from `sys.modules`, which is why one process
  per backend is mandatory and not merely tidy.

## Corpus

`en_pilot/build_corpus.py` calls the frozen primitives instead of extending the
frozen builder:

- `FleursParquetReader` (read) for row access and admission probes;
  `select_fleurs_rows(reader.rows, lambda row: reader.probe(row, decoder),
  limit=40)` for admission. That reuses the frozen privacy gate
  (`privacy_reasons`, `eval/corpus/selection.py:84-103`, which rejects digits,
  URLs, emails, phone numbers and privacy tokens), the frozen 3–12 s duration
  window (`:9-10`), and the frozen ordering by ascending numeric FLEURS `id`.
  The function raises rather than under-filling, so the corpus is fail-closed if
  the split cannot supply 40 bases.
- `decode_bytes` + `canonicalize_clean` (`eval/corpus/media.py:65-83,122-151`)
  for decoding and peak normalisation, including its own hard fail outside
  −3.01..−2.99 dBFS.
- `write_pcm16` for the canonical 44-byte-header WAV.

Because both selection and canonicalisation are the frozen ones, `sample-001`
through `sample-004` are byte-identical to the frozen corpus's first four
bases — verified by digest, and asserted as a gate in Task Group 3. The corpus
is therefore a strict superset of the English bucket already scored, and
`matrix-v2`'s WER numbers remain in scope for comparison rather than being
superseded by an unrelated corpus.

`fleurs_id` alone is not a row identity — 394 rows share only 150 distinct `id`
values across speakers — so the manifest also records `fleurs_path`, `row_group`
and `row_index`, making every admitted base traceable to one exact parquet cell.

## Matrix

`en_pilot/build_matrix.py` follows
`evidence/matrix-v2/build_noise_matrix.py` with three departures: it imports the
frozen audio primitives from this repository instead of a hardcoded sibling
worktree (that file `SystemExit`s at `:46-50` when
`/home/sandy/Projects/Medibytes-worktrees/eval-compare` is absent), it
namespaces its seeds under `medibytes-en-pilot-v1` so no mixture is ever shared
with `matrix-v2`, and it verifies each noise recording against
`evidence/noise-bank/demand-bank-manifest.jsonl` before use rather than
recording whatever digest it happened to read.

40 bases × 18 noises × 2 SNR targets = 1,440 mixtures. Each is built with
`active_mask(clean)` and `mix_at_target_snr(..., target_snr_db=snr)` — the same
masked-SNR definition the frozen corpus uses — then quantised with
`float_to_pcm16` and hard-failed on clipping or on `abs(measured - target) >
SNR_TOLERANCE_DB`. `condition_id` is `f"{sample_id}_{noise_id}_{int(snr)}db"`,
the same format as `matrix-v2`, so condition ids sort identically across the
two studies. 51 of the 1,440 mixtures need a joint SNR-neutral pre-gain to stay
under full scale; that gain is recorded per row, and the reference signal at
scoring time is reconstructed by multiplying the clean base by it, which is what
makes the SNR column comparable across studies.

The count matches `matrix-v2`'s sweep and its per-run disk footprint. What
changes is the independent-cluster count: 10× more base groups, which is the
only axis that tightens the bootstrap interval.

## Process Isolation

One process per backend, each launched with that backend's own venv python.
`en_pilot/enhance.py` prepends exactly one repository root to `sys.path` before
any `demo` import and refuses to continue if a `demo` package is already in
`sys.modules`. Threads are used only *within* a backend — one repository, one
venv, and both adapters construct a fresh model per call, so the pool is
thread-safe — which is the distinction `matrix-v2` documented and then violated.

The per-backend `EnhancementConfig` is reconstructed by reading
`eval/config/methods/<backend>.json` for `variant_id`, `parameters`,
`model_path` and `model_sha256` rather than retyping them, exactly as
`evidence/matrix-v2/matrix_eval.py:97-120` does. A `model_path` that does not
exist on disk is a hard failure naming the file. Both candidate adapters
independently re-verify the model digest and require the exact absolute path,
so a wrong model cannot reach a result.

Every enhance row carries `requested_backend`, `actual_backend` and `variant_id`
read off the returned `EnhancementOutput`. `demo/denoise.py:322-330` already
raises on a mismatch, so a surviving mismatch is impossible by construction —
which is precisely why the column can be trusted as evidence rather than as a
warning. `demo/denoise.py:346-357` also pins the output length, so every
enhanced clip is exactly sample-aligned with its reference; a length mismatch is
still re-checked at scoring time and is a hard failure there too.

## Scoring

`en_pilot/score.py` joins the matrix manifest, three enhance CSVs and two
transcript files. Every metric comes from `eval/metrics.py`:

- WER: `levenshtein_counts` over `evaluation_tokens()`-tokenised text, with the
  reference passed through the same frozen `frozen_normalize` the hypotheses
  were stored with — the frozen normalised-WER path, not a whitespace split and
  not a hand-rolled DP.
- SNR: `eval.metrics.snr_db`.
- SI-SDR: `eval.metrics.si_sdr_db`, whose scale-invariant projection is defined
  for every input and therefore produces a value on every noisy row. The run is
  not considered finished while any noisy row lacks one.
- Reference transcript: the FLEURS `transcription` from this study's own corpus
  manifest, so the study never depends on another worktree's manifest file.

Hard failures, not warnings: an enhance row whose `actual_backend` differs from
`requested_backend`, a missing `(condition_id, backend, decoder)` triple, and a
transcript carrying an error. Any of these prints the offending keys and exits
non-zero rather than dropping the row and quietly shrinking the corpus.

Clean baselines are scored as their own `snr_target_db == ""` block in
`per-condition.csv` and are excluded from every aggregate, so the decoder's WER
floor is visible without contaminating the noisy comparison. They were
transcribed but never scored in `matrix-v2`.

## Statistics

`en_pilot/stats.py` reimplements the paired cluster bootstrap over this study's
40 base groups, using the same estimator, resample count and seed as
`eval/reporting.py::cluster_bootstrap` (10,000 resamples, seed 1729, percentile
2.5/97.5, two-sided p by centred exceedance) and the same return keys. It is a
reimplementation only because the frozen one raises for any group count other
than 15.

For each candidate backend and each decoder, `delta[b]` is that base's mean WER
over its conditions minus the same for `none`, so the comparison is paired
within base and within condition set. Holm correction is applied across the two
candidates per decoder with `family_count=2`, reusing
`eval.reporting.holm_adjust` unchanged.

## Decision Rule

**Stated here, before any run, and implemented verbatim in
`en_pilot/stats.py::decide`.**

For a candidate backend at a decoder tier:

    adopt  iff  ci95_high <= 0.0  and  ci95_low >= -0.02
    reject otherwise

The `none` row is always `baseline`. `mean_delta_snr_db` and `mean_rtf` are
reported but never gate.

This accepts a default which is provably no worse than no denoising, and costs at
most 2 absolute WER points. A backend that trades a small WER regression for a
large SNR gain does not pass, which is the correct trade for a clinical-adjacent
transcription path; a backend that improves WER but costs 3 points does not pass,
because the rule bounds the damage the product would take on for a benefit it
does not gate on.

Two consequences are accepted in advance rather than negotiated after the fact:
`reject` for both candidates is a valid, publishable, and on the current
evidence the *likely* outcome; and the thresholds will not be widened, the
family will not be re-picked, and no backend will be added to obtain an
`adopt`.

## Deliverable

`DATA_ROOT/en-pilot-run/results/` holds `per-condition.csv`, `per-base.csv`,
`decision.csv` and `comparison.json`. `comparison.json` carries the full
bootstrap objects, the Holm map, the rule constants and the sha256 of
`demo/stt_extract.py` and every `demo/denoise_backends/*.py`, so the numbers are
attributable to a code state.

`evidence/en-pilot/README.md` states the rule, the per-backend per-decoder
decision, the confidence interval, the measured RTF, whether the rule was met,
and that no clinical metric was measured over public non-clinical speech.

## Open Questions

None.

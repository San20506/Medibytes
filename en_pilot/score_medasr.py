"""Score the two MedASR variants against each other and against Whisper-small.

Two questions, one scoring rule:

1. **Does the denoising layer help MedASR?**  Paired per-base WER delta of the
   denoised arm against the `none` arm, bootstrapped over whole bases exactly as
   `en_pilot.stats` does for the Whisper study, and judged by the same
   pre-declared rule (`adopt` iff `ci95_high <= 0` and `ci95_low >= -0.02`).
2. **Where does MedASR sit against the incumbent decoder?**  Paired per-base WER
   delta of MedASR against Whisper-`small` on the identical `none` audio, read
   from the n=40 study's own `transcripts-small.jsonl`.  Reported for context;
   it gates nothing.

Every WER here is `en_pilot.score.word_error_rate` — the frozen normalisation and
the frozen tokeniser — so these numbers are directly comparable to
`en-pilot-run/results/per-condition.csv`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Collection, Iterable, Mapping, Sequence

import numpy as np

from en_pilot import config, stats
from en_pilot.score import ScoreError, read_manifest, word_error_rate
from en_pilot.transcribe_medasr import DECODER_ID, MODEL_ID
from eval.corpus.paths import resolve_data_root

PER_CONDITION_COLUMNS = (
    "condition_id",
    "speech",
    "noise",
    "category",
    "snr_target_db",
    "backend",
    "decoder_model",
    "wer",
    "reference_words",
)

PER_BASE_COLUMNS = ("speech", "backend", "decoder_model", "n_conditions", "mean_wer")

CONTEXT_DECODER = config.PRIMARY_DECODER


def _mean(values: Iterable[float]) -> float:
    collected = [float(value) for value in values]
    if not collected:
        raise ScoreError("cannot average an empty collection")
    return float(np.mean(collected))


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _load_transcripts(path: Path, decoder: str) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.is_file():
        raise ScoreError(f"transcript file is absent: {path}")
    loaded: dict[tuple[str, str], dict[str, Any]] = {}
    for record in read_manifest(path):
        if record.get("decoder") != decoder:
            continue
        if record.get("error"):
            raise ScoreError(
                f"{decoder} failed on {record['condition_id']}/{record['backend']}: "
                f"{record['error']}"
            )
        loaded[(record["condition_id"], record["backend"])] = record
    if not loaded:
        raise ScoreError(f"no {decoder} rows in {path}")
    return loaded


def _score_rows(
    *,
    corpus: Mapping[str, Mapping[str, Any]],
    conditions: Sequence[Mapping[str, Any]],
    transcripts: Mapping[tuple[str, str], Mapping[str, Any]],
    decoder_label: str,
    arms: Sequence[str],
    required: bool = True,
) -> list[dict[str, Any]]:
    """One row per scored cell.

    `required=True` is the decoder this study ran itself: a missing cell means an
    incomplete sweep and is a hard failure. `required=False` is a pre-existing
    reference sweep that may not cover every cell; those rows are scored where
    they exist, and the caller reports the coverage it actually got rather than
    pretending to a full grid.
    """

    rows: list[dict[str, Any]] = []
    for sample_id in sorted(corpus):
        key = (sample_id, config.BASELINE_BACKEND)
        if key not in transcripts:
            if required:
                raise ScoreError(
                    f"{decoder_label} is missing the clean floor for {sample_id}"
                )
            continue
        wer, words = word_error_rate(
            corpus[sample_id]["transcript"], transcripts[key]["normalized"]
        )
        rows.append(
            {
                "condition_id": sample_id,
                "speech": sample_id,
                "noise": "",
                "category": "",
                "snr_target_db": "",
                "backend": config.BASELINE_BACKEND,
                "decoder_model": decoder_label,
                "wer": wer,
                "reference_words": words,
            }
        )
    for condition in conditions:
        condition_id = str(condition["condition_id"])
        reference = corpus[str(condition["speech"])]["transcript"]
        for arm in arms:
            key = (condition_id, arm)
            if key not in transcripts:
                if required:
                    raise ScoreError(f"{decoder_label} is missing {condition_id}/{arm}")
                continue
            wer, words = word_error_rate(reference, transcripts[key]["normalized"])
            rows.append(
                {
                    "condition_id": condition_id,
                    "speech": str(condition["speech"]),
                    "noise": condition["noise"],
                    "category": condition["category"],
                    "snr_target_db": condition["snr_db"],
                    "backend": arm,
                    "decoder_model": decoder_label,
                    "wer": wer,
                    "reference_words": words,
                }
            )
    return rows


def _per_base_noisy(
    rows: Sequence[Mapping[str, Any]],
    *,
    backend: str,
    decoder: str,
    only: Collection[str] | None = None,
) -> dict[str, float]:
    """Mean WER per base over the noisy conditions only, for one (arm, decoder).

    `only` restricts the average to a condition set, which is what makes a
    cross-decoder comparison honest when one decoder covers fewer conditions
    than the other: both sides then average the same cells.
    """

    totals: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["decoder_model"] != decoder or row["backend"] != backend:
            continue
        if row["snr_target_db"] == "":  # clean floors are reported, never paired
            continue
        if only is not None and row["condition_id"] not in only:
            continue
        totals[row["speech"]].append(float(row["wer"]))
    return {speech: _mean(values) for speech, values in totals.items()}


def _conditions_covered(
    rows: Sequence[Mapping[str, Any]], *, decoder: str, arms: Sequence[str]
) -> set[str]:
    """Noisy conditions this decoder scored on *every* arm of the comparison."""

    per_arm = [
        {
            row["condition_id"]
            for row in rows
            if row["decoder_model"] == decoder
            and row["backend"] == arm
            and row["snr_target_db"] != ""
        }
        for arm in arms
    ]
    return set.intersection(*per_arm) if per_arm else set()


def _paired_delta(
    left: Mapping[str, float], right: Mapping[str, float], *, expected: int | None = None
) -> dict[str, float]:
    shared = set(left) & set(right)
    if expected is not None and len(shared) != expected:
        raise ScoreError(f"paired comparison needs {expected} shared bases, got {len(shared)}")
    if not shared:
        raise ScoreError("paired comparison has no shared bases")
    return {speech: left[speech] - right[speech] for speech in shared}


def _clean_floor(rows: Sequence[Mapping[str, Any]], *, decoder: str) -> float:
    return _mean(
        float(row["wer"])
        for row in rows
        if row["decoder_model"] == decoder and row["snr_target_db"] == ""
    )


def _bootstrap_block(
    deltas: Mapping[str, float], sentence_of: Mapping[str, str]
) -> dict[str, Any]:
    by_base = stats.bootstrap_paired(deltas, expected_groups=len(deltas))
    by_sentence_values = stats.aggregate_by_sentence(deltas, sentence_of)
    by_sentence = stats.bootstrap_paired(
        by_sentence_values, expected_groups=len(by_sentence_values)
    )
    decision = stats.decide(by_base["ci95_low"], by_base["ci95_high"])
    # `stats.decide`'s lower bound is a plausibility guard written for candidates
    # expected to be neutral or harmful: it rejects any effect whose CI reaches
    # below -0.02 even when the whole interval is an improvement. Say so, rather
    # than let "reject" be read as "did not help".
    note = ""
    if decision == "reject" and by_base["ci95_high"] <= config.WER_DELTA_CI_HIGH_MAX:
        note = (
            "improvement larger than the pre-declared plausibility floor "
            f"({config.WER_DELTA_CI_LOW_MIN}): the whole CI is below zero, so the "
            "rejection is the rule's lower bound firing, not evidence of harm"
        )
    return {
        "per_base": by_base,
        "sentence_stratified": by_sentence,
        "decision": decision,
        "decision_note": note,
    }


def score(data_root: Path, *, denoiser: str) -> dict[str, Any]:
    out_root = data_root / config.OUT_DIR
    results_root = out_root / "results-medasr"
    results_root.mkdir(parents=True, exist_ok=True)
    arms = (config.BASELINE_BACKEND, denoiser)

    corpus = {
        record["sample_id"]: record
        for record in read_manifest(
            data_root / "corpus" / config.CORPUS_ID / "manifest.jsonl"
        )
    }
    if len(corpus) != config.BASE_COUNT:
        raise ScoreError(f"expected {config.BASE_COUNT} bases, found {len(corpus)}")
    conditions = read_manifest(data_root / config.MATRIX_DIR / "noise-matrix.jsonl")
    sentence_of = {
        sample_id: str(record["fleurs_id"]) for sample_id, record in corpus.items()
    }

    medasr_rows = _score_rows(
        corpus=corpus,
        conditions=conditions,
        transcripts=_load_transcripts(
            out_root / f"transcripts-{DECODER_ID}.jsonl", DECODER_ID
        ),
        decoder_label=DECODER_ID,
        arms=arms,
    )
    whisper_rows = _score_rows(
        corpus=corpus,
        conditions=conditions,
        transcripts=_load_transcripts(
            out_root / f"transcripts-{CONTEXT_DECODER}.jsonl", CONTEXT_DECODER
        ),
        decoder_label=CONTEXT_DECODER,
        arms=arms,
        required=False,
    )
    rows = medasr_rows + whisper_rows
    _write_csv(results_root / "per-condition.csv", PER_CONDITION_COLUMNS, rows)

    # The reference sweep may be partial. Everything cross-decoder is computed on
    # the conditions BOTH decoders scored on BOTH arms, so neither side is
    # averaged over an easier or harder subset than the other.
    coverage = {
        decoder: _conditions_covered(rows, decoder=decoder, arms=arms)
        for decoder in (DECODER_ID, CONTEXT_DECODER)
    }
    shared_conditions = coverage[DECODER_ID] & coverage[CONTEXT_DECODER]

    per_base_rows: list[dict[str, Any]] = []
    variants: dict[str, dict[str, Any]] = {}
    for decoder in (DECODER_ID, CONTEXT_DECODER):
        for arm in arms:
            own = _per_base_noisy(rows, backend=arm, decoder=decoder, only=coverage[decoder])
            shared = _per_base_noisy(rows, backend=arm, decoder=decoder, only=shared_conditions)
            variants[f"{decoder}|{arm}"] = {
                "decoder": decoder,
                "denoiser": arm,
                "conditions": len(coverage[decoder]),
                "bases": len(own),
                "noisy_mean_wer": _mean(own.values()),
                "per_base": own,
                "per_base_shared": shared,
            }
            per_base_rows.extend(
                {
                    "speech": speech,
                    "backend": arm,
                    "decoder_model": decoder,
                    "n_conditions": sum(
                        1
                        for row in rows
                        if row["speech"] == speech
                        and row["backend"] == arm
                        and row["decoder_model"] == decoder
                        and row["snr_target_db"] != ""
                    ),
                    "mean_wer": value,
                }
                for speech, value in sorted(own.items())
            )
    _write_csv(results_root / "per-base.csv", PER_BASE_COLUMNS, per_base_rows)

    report: dict[str, Any] = {
        "model": MODEL_ID,
        "corpus": config.CORPUS_ID,
        "bases": config.BASE_COUNT,
        "matrix_conditions": len(conditions),
        "denoiser_arm": denoiser,
        "decoding": "greedy CTC (transformers pipeline default); lm_6.kenlm not used",
        "coverage": {
            "shared_conditions": len(shared_conditions),
            **{f"{decoder}_conditions": len(value) for decoder, value in coverage.items()},
        },
        "rule": {
            "statement": (
                "adopt iff ci95_high <= WER_DELTA_CI_HIGH_MAX and ci95_low >= "
                "WER_DELTA_CI_LOW_MIN, on the paired per-base mean WER delta"
            ),
            "wer_delta_ci_high_max": config.WER_DELTA_CI_HIGH_MAX,
            "wer_delta_ci_low_min": config.WER_DELTA_CI_LOW_MIN,
            "bootstrap_iterations": config.BOOTSTRAP_ITERATIONS,
            "bootstrap_seed": config.BOOTSTRAP_SEED,
        },
        "variants": {
            key: {
                "decoder": value["decoder"],
                "denoiser": value["denoiser"],
                "conditions": value["conditions"],
                "bases": value["bases"],
                "noisy_mean_wer": value["noisy_mean_wer"],
            }
            for key, value in variants.items()
        },
        "clean_floor_wer": {
            decoder: (
                _clean_floor(rows, decoder=decoder)
                if any(
                    row["decoder_model"] == decoder and row["snr_target_db"] == ""
                    for row in rows
                )
                else None
            )
            for decoder in (DECODER_ID, CONTEXT_DECODER)
        },
    }

    # Primary question: does the denoising layer help? Each decoder is paired
    # against itself on its own full coverage, so no decoder is penalised for the
    # other's missing cells.
    report["denoising_effect"] = {
        decoder: _bootstrap_block(
            _paired_delta(
                variants[f"{decoder}|{denoiser}"]["per_base"],
                variants[f"{decoder}|{config.BASELINE_BACKEND}"]["per_base"],
            ),
            sentence_of,
        )
        for decoder in (DECODER_ID, CONTEXT_DECODER)
    }
    # Context: MedASR against the incumbent decoder on identical audio, over the
    # shared conditions only. Reported, never a gate.
    report["decoder_effect"] = {
        arm: _bootstrap_block(
            _paired_delta(
                variants[f"{DECODER_ID}|{arm}"]["per_base_shared"],
                variants[f"{CONTEXT_DECODER}|{arm}"]["per_base_shared"],
            ),
            sentence_of,
        )
        for arm in arms
    }
    report["shared_subset_mean_wer"] = {
        key: _mean(value["per_base_shared"].values())
        for key, value in variants.items()
        if value["per_base_shared"]
    }

    (results_root / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"results": str(results_root), "scored_rows": len(rows), "report": report}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="en_pilot.score_medasr",
        description="Score the denoised and undenoised MedASR variants.",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get(config.ENVIRONMENT_NAME, config.DATA_ROOT_DEFAULT),
    )
    parser.add_argument("--denoiser", default="sherpa-gtcrn-simple")
    arguments = parser.parse_args(argv)
    summary = score(resolve_data_root(arguments.data_root), denoiser=arguments.denoiser)
    report = summary["report"]
    print(f"scored {summary['scored_rows']} cells -> {summary['results']}")
    print("\nnoisy-condition mean WER (lower is better):")
    for key, value in sorted(report["variants"].items()):
        print(f"  {key:<28} {value['noisy_mean_wer']:.4f}")
    print("\nclean-speech floor WER:")
    for decoder, value in sorted(report["clean_floor_wer"].items()):
        shown = f"{value:.4f}" if value is not None else "not in this sweep"
        print(f"  {decoder:<28} {shown}")
    print("\nshared-subset mean WER "
          f"({report['coverage']['shared_conditions']} conditions both decoders scored):")
    for key, value in sorted(report["shared_subset_mean_wer"].items()):
        print(f"  {key:<28} {value:.4f}")
    print("\ndenoising effect (denoised minus none, negative = denoising helps):")
    for decoder, block in sorted(report["denoising_effect"].items()):
        boot = block["per_base"]
        print(
            f"  {decoder:<10} n={boot['group_count']:<3} "
            f"delta {boot['observed_mean_delta']:+.4f} "
            f"CI95 [{boot['ci95_low']:+.4f}, {boot['ci95_high']:+.4f}] "
            f"p={boot['p_value']:.3f} -> {block['decision']}"
        )
        if block["decision_note"]:
            print(f"             note: {block['decision_note']}")
    print("\ndecoder effect (medasr minus whisper-small, shared conditions):")
    for arm, block in sorted(report["decoder_effect"].items()):
        boot = block["per_base"]
        print(
            f"  {arm:<22} n={boot['group_count']:<3} "
            f"delta {boot['observed_mean_delta']:+.4f} "
            f"CI95 [{boot['ci95_low']:+.4f}, {boot['ci95_high']:+.4f}]"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

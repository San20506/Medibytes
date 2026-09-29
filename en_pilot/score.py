"""Score the English pilot and emit the per-condition, per-base and decision tables.

Replaces `evidence/matrix-v2/matrix_eval.py:244-305`'s scoring join, which used a
hand-rolled dynamic-programming WER loop, split its transcripts on whitespace
instead of tokenising, scored SNR and SI-SDR against a reference reconstructed by
a different route, and produced SI-SDR on only 23 of 96 English `none` rows and
**zero** for `dpdfnet2-onnx` and `sherpa-gtcrn-simple`. Everything metric-shaped
here comes from `eval/metrics.py`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from en_pilot import config, stats
from en_pilot.enhance import read_wav_mono16
from eval.corpus.audio import read_pcm16
from eval.corpus.paths import resolve_data_root
from eval.metrics import (
    evaluation_tokens,
    frozen_normalize,
    levenshtein_counts,
    si_sdr_db,
    snr_db,
)

PER_CONDITION_COLUMNS = (
    "condition_id",
    "speech",
    "bucket",
    "noise",
    "category",
    "snr_target_db",
    "measured_snr_db",
    "backend",
    "requested_backend",
    "actual_backend",
    "variant_id",
    "decoder_model",
    "decoder_device",
    "decoder_compute_type",
    "snr_db",
    "si_sdr_db",
    "wer",
    "reference_words",
    "enhance_s",
    "pre_mix_gain",
)

PER_BASE_COLUMNS = (
    "speech",
    "backend",
    "requested_backend",
    "actual_backend",
    "decoder_model",
    "n_conditions",
    "mean_wer",
    "mean_snr_db",
    "mean_si_sdr_db",
    "mean_enhance_s",
)

class ScoreError(RuntimeError):
    """The run cannot be scored as specified."""


def word_error_rate(reference: str, hypothesis: str) -> tuple[float, int]:
    """Frozen Levenshtein counts over `evaluation_tokens`, frozen normalisation."""

    reference_tokens = evaluation_tokens(frozen_normalize(reference))
    if not reference_tokens:
        raise ScoreError("reference transcript tokenises to nothing")
    counts = levenshtein_counts(reference_tokens, evaluation_tokens(hypothesis))
    return counts.edits / len(reference_tokens), len(reference_tokens)


def read_manifest(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def check_completeness(
    expected: set[tuple[str, str, str]], scored: set[tuple[str, str, str]]
) -> list[str]:
    missing = sorted(expected - scored)
    return [f"{condition}|{backend}|{decoder}" for condition, backend, decoder in missing]


def score(data_root: Path, *, decoders: Sequence[str]) -> dict[str, Any]:
    out_root = data_root / config.OUT_DIR
    results_root = out_root / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    corpus = {
        record["sample_id"]: record
        for record in read_manifest(data_root / "corpus" / config.CORPUS_ID / "manifest.jsonl")
    }
    if not corpus:
        raise ScoreError("corpus manifest is empty")
    conditions = read_manifest(data_root / config.MATRIX_DIR / "noise-matrix.jsonl")

    enhance_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for backend in config.BACKENDS:
        csv_path = out_root / f"enhance-{backend}.csv"
        if not csv_path.is_file():
            raise ScoreError(f"enhancement log is absent: {csv_path}")
        for row in csv.DictReader(csv_path.open(encoding="utf-8")):
            if row["error"]:
                raise ScoreError(f"enhancement failed for {row['condition_id']}: {row['error']}")
            if row["actual_backend"] != row["requested_backend"]:
                raise ScoreError(
                    f"mis-dispatch on {row['condition_id']}: requested "
                    f"{row['requested_backend']}, adapter reported {row['actual_backend']}"
                )
            enhance_rows[(row["condition_id"], backend)] = row

    transcripts: dict[tuple[str, str, str], dict[str, Any]] = {}
    for decoder in decoders:
        path = out_root / f"transcripts-{decoder}.jsonl"
        if not path.is_file():
            raise ScoreError(f"transcript file is absent: {path}")
        for record in read_manifest(path):
            transcripts[(record["condition_id"], record["backend"], decoder)] = record

    expected: set[tuple[str, str, str]] = set()
    for condition in conditions:
        for backend in config.BACKENDS:
            for decoder in decoders:
                expected.add((condition["condition_id"], backend, decoder))
    for sample_id in corpus:
        for decoder in decoders:
            expected.add((sample_id, config.BASELINE_BACKEND, decoder))

    scored: set[tuple[str, str, str]] = set()
    rows: list[dict[str, Any]] = []
    for sample_id in sorted(corpus):
        base = corpus[sample_id]
        clean = read_pcm16(data_root / config.CLEAN_DIR / f"{sample_id}.wav").astype(
            np.float64
        ) / 32768.0
        for decoder in decoders:
            key = (sample_id, config.BASELINE_BACKEND, decoder)
            if key not in transcripts:
                continue
            record = transcripts[key]
            wer, words = word_error_rate(base["transcript"], record["normalized"])
            rows.append(
                {
                    "condition_id": sample_id,
                    "speech": sample_id,
                    "bucket": base["bucket"],
                    "noise": "",
                    "category": "",
                    "snr_target_db": "",
                    "measured_snr_db": "",
                    "backend": config.BASELINE_BACKEND,
                    "requested_backend": config.BASELINE_BACKEND,
                    "actual_backend": config.BASELINE_BACKEND,
                    "variant_id": "identity-v1",
                    "decoder_model": decoder,
                    "decoder_device": record["device"],
                    "decoder_compute_type": record["compute_type"],
                    "snr_db": "",
                    "si_sdr_db": "",
                    "wer": round(wer, 6),
                    "reference_words": words,
                    "enhance_s": 0.0,
                    "pre_mix_gain": 1.0,
                }
            )
            scored.add(key)

    for condition in conditions:
        condition_id = str(condition["condition_id"])
        sample_id = str(condition["speech"])
        base = corpus[sample_id]
        gain = float(condition["pre_mix_gain"])
        clean = read_pcm16(data_root / config.CLEAN_DIR / f"{sample_id}.wav").astype(
            np.float64
        ) / 32768.0
        reference = clean * gain
        for backend in config.BACKENDS:
            entry = enhance_rows.get((condition_id, backend))
            if entry is None:
                raise ScoreError(f"no enhancement row for {condition_id}/{backend}")
            enhanced = read_wav_mono16(Path(entry["enhanced_path"])).astype(np.float64)
            if enhanced.size != reference.size:
                raise ScoreError(
                    f"{condition_id}/{backend}: {enhanced.size} enhanced samples "
                    f"against {reference.size} reference samples; refusing to score "
                    f"misaligned audio"
                )
            measured_snr = snr_db(reference, enhanced)
            measured_si_sdr = si_sdr_db(reference, enhanced)
            for decoder in decoders:
                key = (condition_id, backend, decoder)
                if key not in transcripts:
                    continue
                record = transcripts[key]
                if record.get("error"):
                    raise ScoreError(
                        f"transcription failed for {condition_id}/{backend}/{decoder}: "
                        f"{record['error']}"
                    )
                wer, words = word_error_rate(base["transcript"], record["normalized"])
                rows.append(
                    {
                        "condition_id": condition_id,
                        "speech": sample_id,
                        "bucket": condition["bucket"],
                        "noise": condition["noise"],
                        "category": condition["category"],
                        "snr_target_db": condition["snr_db"],
                        "measured_snr_db": condition["measured_snr_db"],
                        "backend": backend,
                        "requested_backend": entry["requested_backend"],
                        "actual_backend": entry["actual_backend"],
                        "variant_id": entry["variant_id"],
                        "decoder_model": decoder,
                        "decoder_device": record["device"],
                        "decoder_compute_type": record["compute_type"],
                        "snr_db": round(measured_snr, 4),
                        "si_sdr_db": round(measured_si_sdr, 4),
                        "wer": round(wer, 6),
                        "reference_words": words,
                        "enhance_s": float(entry["enhance_s"]),
                        "pre_mix_gain": gain,
                    }
                )
                scored.add(key)

    missing = check_completeness(expected, scored)
    if missing:
        raise ScoreError(
            f"{len(missing)} (condition, backend, decoder) triples were never "
            f"scored; first offenders: {missing[:10]}"
        )
    unmeasured_si_sdr = [row["condition_id"] for row in rows if row["si_sdr_db"] == "" and row["snr_target_db"]]
    if unmeasured_si_sdr:
        raise ScoreError(
            f"{len(unmeasured_si_sdr)} noisy rows have no SI-SDR; the run is not "
            f"finished (first: {unmeasured_si_sdr[:5]})"
        )

    _write_csv(results_root / "per-condition.csv", PER_CONDITION_COLUMNS, rows)
    per_base = _per_base(rows)
    _write_csv(results_root / "per-base.csv", PER_BASE_COLUMNS, per_base)

    sentence_of = {
        sample_id: str(record["fleurs_id"]) for sample_id, record in corpus.items()
    }
    decisions, comparison = stats.build_decisions(
        rows,
        decoders=decoders,
        rtf_by_backend=_rtf_by_backend(enhance_rows),
        sentence_of=sentence_of,
    )
    comparison["code_state"] = _code_state()
    comparison["inputs"] = {
        "conditions": len(conditions),
        "bases": len(corpus),
        "distinct_fleurs_sentences": len(set(sentence_of.values())),
        "speakers_per_sentence": sorted(
            Counter(sentence_of.values()).values(), reverse=True
        ),
        "backends": list(config.BACKENDS),
        "decoders": list(decoders),
        "scored_rows": len(rows),
        "noisy_rows": sum(1 for row in rows if row["snr_target_db"] != ""),
        "clean_floor_rows": sum(1 for row in rows if row["snr_target_db"] == ""),
    }
    stats.write_decisions(results_root / "decision.csv", decisions)
    stats.write_comparison(results_root / "comparison.json", comparison)
    return {
        "results": str(results_root),
        "per_condition_rows": len(rows),
        "per_base_rows": len(per_base),
        "decisions": len(decisions),
        "decision_csv": str(results_root / "decision.csv"),
        "adopted": sorted(
            row["backend"]
            for row in decisions
            if row["decision"] == "adopt" and row["decoder_model"] == config.PRIMARY_DECODER
        ),
    }



def _rtf_by_backend(enhance_rows: dict[tuple[str, str], dict[str, Any]]) -> dict[str, float]:
    """Real-time factor per backend: mean enhance seconds per second of audio."""

    totals: dict[str, list[float]] = defaultdict(list)
    for (_, backend), entry in enhance_rows.items():
        duration = float(entry["samples"]) / config.SAMPLE_RATE
        if duration <= 0:
            raise ScoreError(f"{backend} produced a zero-length condition")
        totals[backend].append(float(entry["enhance_s"]) / duration)
    return {backend: float(np.mean(values)) for backend, values in totals.items()}

def _per_base(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"wer": [], "snr_db": [], "si_sdr_db": [], "enhance_s": []}
    )
    identity: dict[tuple[str, str, str], tuple[str, str]] = {}
    for row in rows:
        if row["snr_target_db"] == "":
            continue  # the clean floor is reported per condition, never pooled
        key = (row["speech"], row["backend"], row["decoder_model"])
        identity[key] = (row["requested_backend"], row["actual_backend"])
        for metric in ("wer", "snr_db", "si_sdr_db", "enhance_s"):
            totals[key][metric].append(float(row[metric]))
    out: list[dict[str, Any]] = []
    for (speech, backend, decoder), values in sorted(totals.items()):
        requested, actual = identity[(speech, backend, decoder)]
        out.append(
            {
                "speech": speech,
                "backend": backend,
                "requested_backend": requested,
                "actual_backend": actual,
                "decoder_model": decoder,
                "n_conditions": len(values["wer"]),
                "mean_wer": round(float(np.mean(values["wer"])), 6),
                "mean_snr_db": round(float(np.mean(values["snr_db"])), 4),
                "mean_si_sdr_db": round(float(np.mean(values["si_sdr_db"])), 4),
                "mean_enhance_s": round(float(np.mean(values["enhance_s"])), 4),
            }
        )
    return out


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _code_state() -> dict[str, str]:
    """Bind the numbers to the exact normalisation and adapter code that ran.

    The candidate adapters do not live in this repository — `demo/denoise_backends/`
    here holds only `none.py` and `noisereduce.py`, and each backend was executed
    from its own worktree under its own venv. Hashing this repository's copies
    would attribute the results to code that never ran, so each backend's
    adapter and dispatcher are hashed where they were actually imported from.
    """

    state: dict[str, str] = {}

    def digest(relative: str) -> str | None:
        path = config.REPO_ROOT / relative
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

    for relative in ("demo/stt_extract.py", "eval/metrics.py", "en_pilot/score.py", "en_pilot/stats.py"):
        value = digest(relative)
        if value:
            state[relative] = value
    for backend in config.BACKENDS:
        repo = Path(config.repo_for_backend(backend))
        module = backend.replace("-", "_")
        for relative in (
            "demo/denoise.py",
            f"demo/denoise_backends/{module}.py",
            f"eval/config/methods/{backend}.json",
        ):
            path = repo / relative
            if path.is_file():
                state[f"{backend}:{repo.name}/{relative}"] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    return state


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="en_pilot.score",
        description="Score the English pilot and apply the pre-declared decision rule.",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get(config.ENVIRONMENT_NAME, config.DATA_ROOT_DEFAULT),
    )
    parser.add_argument(
        "--decoders",
        default=",".join([config.PRIMARY_DECODER, *config.CONTINUITY_DECODERS]),
    )
    arguments = parser.parse_args(argv)
    data_root = resolve_data_root(arguments.data_root)
    decoders = [item for item in arguments.decoders.split(",") if item]
    print(json.dumps(score(data_root, decoders=decoders), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

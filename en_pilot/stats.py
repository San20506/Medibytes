"""Paired cluster bootstrap and the pre-declared Stage-1 decision rule.

`eval/reporting.py::cluster_bootstrap` raises unless it is handed exactly
`EXPECTED_GROUP_COUNT == 15` groups (`eval/reporting.py:27,680-681`), which is the
frozen pilot's group count and the wrong one here. This module reimplements the
same estimator over this study's 40 English bases, and states the acceptance rule
in code so a result cannot be rationalised after the fact:

    adopt  iff  ci95_high <= 0.0  and  ci95_low >= -0.02

That accepts a default which is provably no worse than no denoising, and costs at
most 2 absolute WER points. `mean_delta_snr_db` and `mean_rtf` are reported but
never gate.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from en_pilot import config
from eval.reporting import holm_adjust

DECISION_COLUMNS = (
    "backend",
    "decoder_model",
    "group_count",
    "observed_mean_delta_wer",
    "ci95_low",
    "ci95_high",
    "p_value",
    "holm_adjusted_p_value",
    "mean_delta_snr_db",
    "mean_rtf",
    "decision",
)


class StatsError(RuntimeError):
    """The statistics cannot be computed as specified."""


def bootstrap_paired(
    values_by_group: Mapping[str, float],
    *,
    iterations: int = config.BOOTSTRAP_ITERATIONS,
    seed: int = config.BOOTSTRAP_SEED,
    expected_groups: int = config.BASE_COUNT,
) -> dict[str, Any]:
    """Bootstrap the mean of one paired per-base delta, resampling whole bases."""

    if len(values_by_group) != expected_groups:
        raise StatsError(
            f"paired bootstrap requires exactly {expected_groups} base groups, "
            f"got {len(values_by_group)}"
        )
    if iterations <= 0:
        raise StatsError("bootstrap iterations must be positive")
    groups = sorted(values_by_group)
    values = np.asarray([float(values_by_group[group]) for group in groups], dtype=np.float64)
    if not np.isfinite(values).all():
        raise StatsError("bootstrap values must all be finite")
    generator = np.random.Generator(np.random.PCG64(seed))
    indices = generator.integers(0, len(groups), size=(iterations, len(groups)))
    means = values[indices].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    observed = float(np.mean(values))
    p_value = min(1.0, float(np.mean(np.abs(means - observed) >= abs(observed))))
    return {
        "groups": groups,
        "group_count": len(groups),
        "iterations": iterations,
        "seed": seed,
        "observed_mean_delta": observed,
        "ci95_low": float(lower),
        "ci95_high": float(upper),
        "p_value": p_value,
    }


def decide(ci95_low: float, ci95_high: float) -> str:
    """The pre-declared acceptance rule, in one place and testable."""

    if (
        ci95_high <= config.WER_DELTA_CI_HIGH_MAX
        and ci95_low >= config.WER_DELTA_CI_LOW_MIN
    ):
        return "adopt"
    return "reject"


def _mean(values: Iterable[float]) -> float:
    collected = [float(value) for value in values]
    if not collected:
        raise StatsError("cannot average an empty collection")
    return float(np.mean(collected))


def per_base_deltas(
    rows: Sequence[Mapping[str, Any]],
    *,
    backend: str,
    decoder: str,
    metric: str,
) -> dict[str, float]:
    """Mean `metric` per base for one backend, minus the same for `none`."""

    if backend == config.BASELINE_BACKEND:
        raise StatsError("per_base_deltas is a candidate-versus-baseline comparison")
    totals: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["decoder_model"] != decoder or not row["snr_target_db"]:
            continue
        if row[metric] == "":
            continue
        totals[(row["backend"], row["speech"])].append(float(row[metric]))
    baselines = {
        speech: _mean(values) for (backend_id, speech), values in totals.items()
        if backend_id == config.BASELINE_BACKEND
    }
    deltas: dict[str, float] = {}
    for (backend_id, speech), values in totals.items():
        if backend_id != backend or speech not in baselines:
            continue
        deltas[speech] = _mean(values) - baselines[speech]
    return deltas


def aggregate_by_sentence(
    deltas: Mapping[str, float], sentence_of: Mapping[str, str]
) -> dict[str, float]:
    """Collapse per-base deltas onto the FLEURS sentence each base reads.

    FLEURS `en_us` validation records several speakers per sentence, so a base
    is a speaker/utterance pair rather than an independent utterance: 40 bases
    read only 18 distinct sentences. Resampling the 40 pairs therefore treats
    repeated content as independent evidence and produces an interval that is
    too narrow. Averaging each sentence's bases and resampling the sentences is
    the conservative reading of the same data.
    """

    grouped: dict[str, list[float]] = defaultdict(list)
    for speech, delta in deltas.items():
        sentence = sentence_of.get(speech)
        if sentence is None:
            raise StatsError(f"no sentence identity recorded for base {speech}")
        grouped[sentence].append(float(delta))
    return {sentence: _mean(values) for sentence, values in grouped.items()}


def per_base_values(
    rows: Sequence[Mapping[str, Any]], *, backend: str, decoder: str, metric: str
) -> dict[str, float]:
    totals: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["decoder_model"] != decoder or not row["snr_target_db"]:
            continue
        if row[metric] == "":
            continue
        totals[(row["backend"], row["speech"])].append(float(row[metric]))
    return {speech: _mean(values) for (backend_id, speech), values in totals.items() if backend_id == backend}


def check_rtf(rtf_by_backend: Mapping[str, float]) -> None:
    """Every backend in the decision table must carry a measured RTF."""

    missing = sorted(set(config.BACKENDS) - set(rtf_by_backend))
    if missing:
        raise StatsError(f"no measured real-time factor for {missing}")



def build_decisions(
    rows: Sequence[Mapping[str, Any]],
    *,
    decoders: Sequence[str],
    candidates: Sequence[str] = config.CANDIDATE_BACKENDS,
    rtf_by_backend: Mapping[str, float],
    sentence_of: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score every (backend, decoder) cell against the pre-declared rule."""

    if not rows:
        raise StatsError("no scored rows to decide on")
    check_rtf(rtf_by_backend)
    decisions: list[dict[str, Any]] = []
    comparison: dict[str, Any] = {
        "rule": {
            "statement": (
                "adopt iff ci95_high <= WER_DELTA_CI_HIGH_MAX and "
                "ci95_low >= WER_DELTA_CI_LOW_MIN, on the paired per-base "
                "mean WER delta of a candidate against the `none` baseline"
            ),
            "wer_delta_ci_high_max": config.WER_DELTA_CI_HIGH_MAX,
            "wer_delta_ci_low_min": config.WER_DELTA_CI_LOW_MIN,
            "primary_decoder": config.PRIMARY_DECODER,
            "continuity_decoders": list(config.CONTINUITY_DECODERS),
            "bootstrap_iterations": config.BOOTSTRAP_ITERATIONS,
            "bootstrap_seed": config.BOOTSTRAP_SEED,
            "holm_family_count": config.HOLM_FAMILY_COUNT,
            "backends": list(config.BACKENDS),
        },
        "sensitivity_note": (
            "The decision uses the pre-declared 40-base bootstrap. The "
            "sentence-stratified bootstrap below resamples FLEURS sentences "
            "instead of bases and is reported only as a robustness check; it "
            "never changes decision.csv."
        ),
        "decoders": {},
    }
    for decoder in decoders:
        cell: dict[str, Any] = {
            "bootstraps": {},
            "holm": {},
            "sentence_stratified": {},
        }
        p_values: dict[str, float] = {}
        bootstraps: dict[str, dict[str, Any]] = {}
        sentence_bootstraps: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            wer_deltas = per_base_deltas(rows, backend=candidate, decoder=decoder, metric="wer")
            if not wer_deltas:
                raise StatsError(
                    f"no paired WER deltas for {candidate}/{decoder}; the run is incomplete"
                )
            bootstraps[candidate] = bootstrap_paired(wer_deltas)
            p_values[candidate] = bootstraps[candidate]["p_value"]
            by_sentence = aggregate_by_sentence(wer_deltas, sentence_of)
            sentence_bootstraps[candidate] = bootstrap_paired(
                by_sentence, expected_groups=len(by_sentence)
            )
        adjusted = holm_adjust(p_values, family_count=config.HOLM_FAMILY_COUNT)
        cell["bootstraps"] = bootstraps
        cell["holm"] = adjusted
        cell["sentence_stratified"] = sentence_bootstraps
        comparison["decoders"][decoder] = cell

        baseline_snr = per_base_values(rows, backend=config.BASELINE_BACKEND, decoder=decoder, metric="snr_db")
        baseline_rtf = rtf_by_backend[config.BASELINE_BACKEND]
        decisions.append(
            _row(
                backend=config.BASELINE_BACKEND,
                decoder=decoder,
                bootstrap=None,
                holm_p=None,
                snr_delta=0.0,
                rtf=baseline_rtf,
                group_count=len(baseline_snr),
                decision="baseline",
            )
        )
        for candidate, bootstrap in bootstraps.items():
            snr_deltas = per_base_deltas(
                rows, backend=candidate, decoder=decoder, metric="snr_db"
            )
            decisions.append(
                _row(
                    backend=candidate,
                    decoder=decoder,
                    bootstrap=bootstrap,
                    holm_p=adjusted[candidate],
                    snr_delta=_mean(snr_deltas.values()) if snr_deltas else float("nan"),
                    rtf=rtf_by_backend[candidate],
                    group_count=bootstrap["group_count"],
                    decision=decide(bootstrap["ci95_low"], bootstrap["ci95_high"]),
                )
            )
    return decisions, comparison


def _row(
    *,
    backend: str,
    decoder: str,
    bootstrap: Mapping[str, Any] | None,
    holm_p: float | None,
    snr_delta: float,
    rtf: float,
    group_count: int,
    decision: str,
) -> dict[str, Any]:
    if decision not in config.DECISIONS:
        raise StatsError(f"illegal decision {decision!r}")
    return {
        "backend": backend,
        "decoder_model": decoder,
        "group_count": group_count,
        "observed_mean_delta_wer": "" if bootstrap is None else round(bootstrap["observed_mean_delta"], 6),
        "ci95_low": "" if bootstrap is None else round(bootstrap["ci95_low"], 6),
        "ci95_high": "" if bootstrap is None else round(bootstrap["ci95_high"], 6),
        "p_value": "" if bootstrap is None else round(bootstrap["p_value"], 6),
        "holm_adjusted_p_value": "" if holm_p is None else round(holm_p, 6),
        "mean_delta_snr_db": round(snr_delta, 4),
        "mean_rtf": round(rtf, 6),
        "decision": decision,
    }


def write_decisions(path: Path, decisions: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(DECISION_COLUMNS))
        writer.writeheader()
        writer.writerows(decisions)


def write_comparison(path: Path, comparison: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


__all__ = [
    "DECISION_COLUMNS",
    "StatsError",
    "bootstrap_paired",
    "build_decisions",
    "decide",
    "check_rtf",
    "per_base_deltas",
    "per_base_values",
    "write_comparison",
    "write_decisions",
]

"""Run verification, arm aggregation, paired bootstrap, and post-row comparison."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from eval.contracts import (
    ContractValidationError,
    assert_no_absolute_result_paths,
    file_sha256,
    loads_strict,
    load_json,
    resolve_contained_result_path,
    validate_contract,
    write_json_exclusive,
    write_text_exclusive,
)
from eval.metrics import ENTITY_GROUPS, TARGET_SAMPLE_RATE

BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 1729
EXPECTED_GROUP_COUNT = 15
NOT_APPLICABLE = {
    "status": "not_applicable",
    "reason": "non_clinical_source_reference_corpus",
}
AUDIO_METRICS = {
    "snr_db": "higher",
    "delta_snr_db": "higher",
    "si_sdr_db": "higher",
    "stoi": "higher",
    "pesq_mos_lqo": "higher",
    "lsd": "lower",
    "clipping_ratio": "lower",
    "duration_drift_s": "lower",
}
TEXT_METRICS = {
    "wer": "lower",
    "normalized_wer": "lower",
    "exact_match": "higher",
    "false_positive_fields": "lower",
}


class ReportingError(RuntimeError):
    """A run cannot be verified or admitted to comparison."""


def _mean(values: Iterable[float | None]) -> float | None:
    materialized = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(materialized)) if materialized else None


def _metric_value(metrics: Mapping[str, Any], group: str, name: str) -> float | None:
    try:
        section = metrics[group]
        if not isinstance(section, Mapping):
            return None
        if name == "pesq_mos_lqo":
            value = section["pesq_mos_lqo"]
            return float(value["value"]) if isinstance(value, Mapping) and value.get("status") == "observed" else None
        if name == "stoi":
            value = section["stoi"]
            return float(value["value"]) if isinstance(value, Mapping) and value.get("status") == "observed" else None
        if name in {"wer", "normalized_wer", "exact_match"}:
            nested = "raw" if name == "wer" else "normalized"
            value = section[nested][name]
            return float(value) if value is not None else None
        value = section[name]
        return float(value) if value is not None else None
    except (KeyError, TypeError, ValueError):
        return None


def _clinical_aggregate(clinical: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "status": "not_applicable",
        "medical_wer": dict(NOT_APPLICABLE),
        "medication_dose_accuracy": dict(NOT_APPLICABLE),
        "clinical_ner_f1": dict(NOT_APPLICABLE),
        "field_accuracy": dict(NOT_APPLICABLE),
        "critical_error_rate": dict(NOT_APPLICABLE),
    }


def build_arm_report(
    *,
    run_root: Path,
    run_record: Mapping[str, Any],
    statuses: Sequence[Mapping[str, Any]],
    static_outputs_unchanged: bool,
) -> dict[str, Any]:
    """Aggregate terminal condition evidence without upgrading failures to zeroes."""

    expected = int(run_record["counts"]["expected_conditions"])
    status_counts = Counter(str(value.get("status")) for value in statuses)
    ok_count = status_counts["ok"]
    failed_count = sum(
        status_counts[name]
        for name in ("failed", "failed_runtime", "rejected")
    )
    blocked_count = sum(
        status_counts[name]
        for name in ("blocked_prerequisite", "not_comparable", "blocked_target")
    )
    other_count = len(statuses) - ok_count - failed_count - blocked_count - status_counts["oom"] - status_counts["timeout"]
    disposition = str(run_record["method"]["default_disposition"])
    if disposition in {"blocked_prerequisite", "not_comparable", "blocked_target"}:
        outcome = disposition
    elif len(statuses) != expected or ok_count != expected or not static_outputs_unchanged:
        outcome = "invalid"
    else:
        outcome = "observed"

    condition_summaries: list[dict[str, Any]] = []
    audio_values: dict[str, list[float | None]] = {name: [] for name in AUDIO_METRICS}
    resource_rows: list[Mapping[str, Any]] = []
    text_rows: list[Mapping[str, Any]] = []
    status_hashes: dict[str, str] = {}
    for status in sorted(statuses, key=lambda value: str(value["condition_id"])):
        condition_id = str(status["condition_id"])
        status_path = _safe_run_path(run_root, str(status["artifacts"]["status"]), "condition status")
        status_hashes[condition_id] = file_sha256(status_path)
        metrics_available = status.get("status") == "ok" and status_path.is_file()
        condition_summaries.append(
            {
                "condition_id": condition_id,
                "group_id": str(status["group_id"]),
                "bucket": str(status["bucket"]),
                "noise_type": str(status["noise_type"]),
                "status": str(status["status"]),
                "actual_backend": status.get("actual_backend"),
                "metrics_available": metrics_available,
            }
        )
        if not metrics_available:
            continue
        metrics = load_json(
            _safe_run_path(run_root, str(status["artifacts"]["metrics"]), "condition metrics")
        )
        for name in AUDIO_METRICS:
            audio_values[name].append(_metric_value(metrics, "audio", name))
        text_rows.append(metrics)
        resources = load_json(
            _safe_run_path(run_root, str(status["artifacts"]["resources"]), "condition resources")
        )
        if isinstance(resources, Mapping):
            resource_rows.append(resources)

    wer_numerator = wer_denominator = normalized_numerator = normalized_denominator = 0
    exact_matches: list[float] = []
    false_positive_fields: list[float] = []
    for metrics in text_rows:
        text = metrics.get("text", {})
        if not isinstance(text, Mapping) or text.get("status") != "observed":
            continue
        raw = text.get("raw", {})
        normalized = text.get("normalized", {})
        if isinstance(raw, Mapping):
            wer_numerator += int(raw.get("numerator", 0))
            wer_denominator += int(raw.get("denominator", 0))
        if isinstance(normalized, Mapping):
            normalized_numerator += int(normalized.get("numerator", 0))
            normalized_denominator += int(normalized.get("denominator", 0))
            if normalized.get("exact_match") is True:
                exact_matches.append(1.0)
            elif normalized.get("exact_match") is False:
                exact_matches.append(0.0)
        clinical = metrics.get("clinical", {})
        if isinstance(clinical, Mapping):
            false_positive_fields.append(float(clinical.get("false_positive_fields", 0)))
    text_aggregate = None
    if text_rows:
        text_aggregate = {
            "wer_numerator": wer_numerator,
            "wer_denominator": wer_denominator,
            "wer": wer_numerator / wer_denominator if wer_denominator else None,
            "normalized_wer": normalized_numerator / normalized_denominator if normalized_denominator else None,
            "exact_match_rate": _mean(exact_matches),
            "false_positive_fields": int(sum(false_positive_fields)) if false_positive_fields else None,
        }

    inference_p50 = _mean(row.get("inference_p50_ms") for row in resource_rows)
    inference_p95 = _mean(row.get("inference_p95_ms") for row in resource_rows)
    inference_p99 = _mean(row.get("inference_p99_ms") for row in resource_rows)
    cold_values = [
        row.get("cold", {}).get("cold_total_s")
        for row in resource_rows
        if isinstance(row.get("cold"), Mapping)
    ]
    rtf_values = [row.get("realtime_factor_mean") for row in resource_rows]
    rss_values = [row.get("peak_rss_bytes_max") for row in resource_rows]
    vram_values = [
        int(row["peak_vram_bytes"])
        for row in resource_rows
        if isinstance(row.get("peak_vram_bytes"), int)
    ]
    resources_aggregate = {
        "cold_total_s_mean": _mean(cold_values),
        "inference_p50_ms": inference_p50,
        "inference_p95_ms": inference_p95,
        "inference_p99_ms": inference_p99,
        "realtime_factor_mean": _mean(rtf_values),
        "peak_rss_bytes_max": int(max(rss_values)) if rss_values and all(value is not None for value in rss_values) else None,
        "peak_vram_bytes": max(vram_values) if vram_values else "unavailable",
    }
    report = {
        "schema_version": "1.0.0",
        "record_type": "report",
        "run_id": run_record["run_id"],
        "status": outcome,
        "comparison_eligible": outcome == "observed" and expected == 30,
        "method": {
            "backend": run_record["method"]["backend"],
            "variant_id": run_record["method"]["variant_id"],
            "catalog_row": run_record["method"]["catalog_row"],
            "default_disposition": disposition,
            "disposition_reason": run_record["method"].get("disposition_reason"),
            "source": run_record["method"]["source"],
            "source_revision": run_record["method"]["source_revision"],
            "license": run_record["method"]["license"],
            "target_class": run_record["method"]["target_class"],
        },
        "counts": {
            "total": len(statuses),
            "ok": ok_count,
            "failed": failed_count,
            "oom": status_counts["oom"],
            "timeout": status_counts["timeout"],
            "blocked": blocked_count,
            "other": other_count,
        },
        "conditions": condition_summaries,
        "aggregate": {
            "audio": {
                "valid_audio_count": sum(
                    1
                    for status in statuses
                    if status.get("status") == "ok"
                ),
                "snr_db_mean": _mean(audio_values["snr_db"]),
                "delta_snr_db_mean": _mean(audio_values["delta_snr_db"]),
                "si_sdr_db_mean": _mean(audio_values["si_sdr_db"]),
                "pesq_mos_lqo_mean": _mean(audio_values["pesq_mos_lqo"]),
                "stoi_mean": _mean(audio_values["stoi"]),
                "lsd_mean": _mean(audio_values["lsd"]),
                "clipping_ratio_mean": _mean(audio_values["clipping_ratio"]),
                "duration_drift_samples_max": None,
            },
            "text": text_aggregate,
            "clinical": _clinical_aggregate(text_rows),
            "resources": resources_aggregate,
        },
        "artifacts": {
            "run_json": "../../../run.json",
            "report_json": "report.json",
            "report_md": "report.md",
            "status_files_sha256": status_hashes,
        },
        "limitations": list(run_record["limitations"]),
        "production_assessment": "not_assessed_nonclinical_smoke_corpus",
    }
    if not static_outputs_unchanged:
        report["limitations"].append("Static demo output directories changed during evaluation; the run is invalid.")
    drift_values = [
        abs(
            float(
                _metric_value(
                    load_json(
                        _safe_run_path(
                            run_root, str(status["artifacts"]["metrics"]), "condition metrics"
                        )
                    ),
                    "audio",
                    "duration_drift_s",
                )
                or 0.0
            )
        )
        for status in statuses
        if status.get("status") == "ok"
    ]
    if drift_values:
        report["aggregate"]["audio"]["duration_drift_samples_max"] = int(round(max(drift_values) * TARGET_SAMPLE_RATE))
    validate_contract("report", report)
    assert_no_absolute_result_paths(report, "report")
    return report


def render_report_markdown(report: Mapping[str, Any]) -> str:
    method = report["method"]
    counts = report["counts"]
    aggregate = report["aggregate"]
    lines = [
        f"# MediBytes denoising arm: {method['backend']}",
        "",
        f"- Variant: `{method['variant_id']}`",
        f"- Disposition: `{method['default_disposition']}`",
        f"- Outcome: `{report['status']}`",
        f"- Conditions: {counts['ok']}/{counts['total']} ok; failures are retained, never scored as zero.",
        f"- Execution class: `{method['target_class']}`",
        f"- License: {method['license']}",
        "",
        "## Aggregate paired metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in sorted(aggregate["audio"]):
        value = aggregate["audio"][key]
        lines.append(f"| {key} | {value if value is not None else 'not_applicable'} |")
    if aggregate["text"] is not None:
        for key in sorted(aggregate["text"]):
            value = aggregate["text"][key]
            lines.append(f"| {key} | {value if value is not None else 'not_applicable'} |")
    lines.extend(
        [
            "",
            "## Clinical boundary",
            "",
            "Medical WER, medication/dose accuracy, clinical NER F1, field accuracy, and critical-error rate are not applicable because the corpus is non-clinical.",
            "",
            "## Production boundary",
            "",
            "This 15-base smoke corpus does not support a production quality, accuracy, safety, or default-method verdict.",
            "",
        ]
    )
    return "\n".join(lines)

render_arm_report_markdown = render_report_markdown

def _safe_run_path(run_root: Path, value: str, label: str) -> Path:
    try:
        return resolve_contained_result_path(run_root, value, label, must_exist=True)
    except ContractValidationError as error:
        raise ReportingError(str(error)) from error


def _verify_no_static_composition(run: Mapping[str, Any], status: Mapping[str, Any]) -> None:
    if run["composition"] != {
        "mode": "single_backend",
        "backend_count": 1,
        "layered": False,
        "fallback_used": False,
    }:
        raise ReportingError("run composition is not exactly one non-layered, non-fallback backend")
    if status["requested_backend"] != run["method"]["backend"]:
        raise ReportingError("condition requested backend differs from the run method")
    if status["fallback_used"] is not False:
        raise ReportingError("condition contains fallback use")
    if status["status"] == "ok" and status["actual_backend"] != run["method"]["backend"]:
        raise ReportingError("successful condition actual backend differs from the requested backend")


def _verify_real_artifacts(run_root: Path, run: Mapping[str, Any], status: Mapping[str, Any]) -> None:
    from eval.runner import load_downstream_config, validate_real_transcript

    artifacts = status["artifacts"]
    expected_json = {"enhancement", "transcript", "entities", "fields", "resources", "metrics", "status"}
    artifact_prefix = (
        f"arms/{run['method']['backend']}/{run['method']['variant_id']}"
        f"/conditions/{status['condition_id']}"
    )
    expected_artifacts = {
        name: f"{artifact_prefix}/{name}.json" for name in expected_json
    }
    if artifacts != expected_artifacts:
        raise ReportingError("condition artifacts do not match the owning method and condition")
    for name in expected_json:
        path = _safe_run_path(run_root, str(artifacts[name]), f"condition artifacts.{name}")
        if not path.is_file():
            raise ReportingError(f"condition artifact is missing: {name}")
        value = load_json(path)
        assert_no_absolute_result_paths(value, f"condition artifact {name}")
    condition_dir = _safe_run_path(run_root, str(artifacts["status"]), "condition status").parent
    allowed = {f"{name}.json" for name in expected_json}
    if status["status"] == "ok" or status.get("enhanced") is not None:
        allowed.add("enhanced.wav")
    if any(path.is_symlink() for path in condition_dir.iterdir()):
        raise ReportingError("condition directory contains a symlink")
    actual_files = {path.name for path in condition_dir.iterdir() if path.is_file()}
    if actual_files != allowed:
        raise ReportingError("condition directory contains missing or unexpected artifacts")
    if status.get("enhanced") is not None and status["enhanced"]["path"] != (
        f"{artifact_prefix}/enhanced.wav"
    ):
        raise ReportingError("enhanced path does not match the owning condition")
    if status["status"] == "ok":
        enhanced = status["enhanced"]
        if not isinstance(enhanced, Mapping):
            raise ReportingError("successful condition has no enhanced artifact record")
        enhanced_path = _safe_run_path(run_root, str(enhanced["path"]), "enhanced path")
        if file_sha256(enhanced_path) != enhanced["sha256"]:
            raise ReportingError("enhanced WAV digest mismatch")
        from eval.runner import validate_enhancement_metadata

        enhancement_metadata = load_json(
            _safe_run_path(run_root, str(artifacts["enhancement"]), "enhancement")
        )
        from eval.runner import METHOD_CONFIG_ROOT

        method_config = load_json(METHOD_CONFIG_ROOT / f"{run['method']['backend']}.json")
        validate_enhancement_metadata(
            enhancement_metadata,
            method=run["method"]["backend"],
            variant_id=run["method"]["variant_id"],
            output_path=enhanced_path,
            input_samples=int(status["input"]["sample_count"]),
            expected_input_sha256=str(status["input"]["sha256"]),
            expected_model_sha256=run["method"].get("model_sha256"),
            expected_source_revision=run["method"].get("source_revision"),
            expected_config_sha256=run["identity"]["method_config_sha256"],
            expected_parameters=method_config["parameters"],
        )
        if len(status["repeat_hashes"]) != 4 or any(value != enhanced["sha256"] for value in status["repeat_hashes"]):
            raise ReportingError("warm-up and three measured repeats must be byte-identical")
        transcript = load_json(
            _safe_run_path(run_root, str(artifacts["transcript"]), "transcript")
        )
        entities = load_json(_safe_run_path(run_root, str(artifacts["entities"]), "entities"))
        downstream, _ = load_downstream_config()
        validate_real_transcript(
            transcript,
            entities,
            downstream,
            expected_job_id=str(status["condition_id"]),
        )
        metrics = load_json(_safe_run_path(run_root, str(artifacts["metrics"]), "metrics"))
        if metrics.get("status") != "observed":
            raise ReportingError("successful condition metrics must be observed")
        audio_metrics = metrics.get("audio")
        required_audio = {
            "snr_db",
            "delta_snr_db",
            "si_sdr_db",
            "pesq_mos_lqo",
            "stoi",
            "lsd",
            "clipping_ratio",
            "duration_drift_s",
        }
        if not isinstance(audio_metrics, Mapping) or not required_audio.issubset(audio_metrics):
            raise ReportingError("successful condition audio metrics are incomplete")
        text_metrics = metrics.get("text")
        if not isinstance(text_metrics, Mapping) or text_metrics.get("status") != "observed":
            raise ReportingError("successful condition text metrics are incomplete")
        clinical = metrics.get("clinical", {})
        for name in ("medical_wer", "medication_dose_accuracy", "clinical_ner_f1", "field_accuracy", "critical_error_rate"):
            if not isinstance(clinical.get(name), Mapping) or clinical[name].get("status") != "not_applicable":
                raise ReportingError(f"clinical metric {name} must be not_applicable")
        if set(clinical.get("entity_counts", {})) != set(ENTITY_GROUPS):
            raise ReportingError("clinical false-positive diagnostic does not cover all extraction groups")
        resources = load_json(_safe_run_path(run_root, str(artifacts["resources"]), "resources"))
        if resources.get("peak_vram_bytes") == 0:
            raise ReportingError("unavailable VRAM must never be encoded as zero")
        if resources.get("status") != "observed":
            raise ReportingError("successful condition resources are incomplete")
        measured = resources.get("measured")
        if (
            not isinstance(measured, list)
            or [row.get("phase") for row in measured if isinstance(row, Mapping)]
            != ["repeat-1", "repeat-2", "repeat-3"]
        ):
            raise ReportingError("successful condition must retain three measured repeats")
        if not isinstance(resources.get("cold"), Mapping) or not isinstance(
            resources.get("warmup"), Mapping
        ):
            raise ReportingError("successful condition must retain cold and warm-up resources")
    elif status.get("enhanced") is not None:
        enhanced_path = _safe_run_path(
            run_root, str(status["enhanced"]["path"]), "failed enhanced path"
        )
        if not enhanced_path.is_file():
            raise ReportingError("retained failed enhanced output is missing")


def verify_run(run_root: Path) -> dict[str, Any]:
    """Verify hashes, exact layout, strict schemas, and all terminal conditions."""

    expanded_root = run_root.expanduser()
    if expanded_root.is_symlink():
        raise ReportingError("run root must not be a symlink")
    root = expanded_root.resolve(strict=True)
    run_path = _safe_run_path(root, "run.json", "run.json")
    environment_path = _safe_run_path(root, "environment.json", "environment.json")
    if not run_path.is_file() or not environment_path.is_file():
        raise ReportingError("run is missing run.json or environment.json")
    run = load_json(run_path)
    environment = load_json(environment_path)
    validate_contract("run", run)
    assert_no_absolute_result_paths(run, "run")
    assert_no_absolute_result_paths(environment, "environment")
    if run["run_id"] != root.name:
        raise ReportingError("run ID does not match the immutable run directory")
    if run["environment_sha256"] != file_sha256(environment_path):
        raise ReportingError("environment digest mismatch")
    if environment.get("identity") != run["identity"]:
        raise ReportingError("environment identity differs from the finalized run")
    from eval.runner import (
        CATALOG_PATH,
        DOWNSTREAM_CONFIG_PATH,
        MANIFEST_PATH,
        implementation_sha256,
        adapter_source_sha256,
        enhancement_code_sha256,
        METHOD_CONFIG_ROOT,
        downstream_code_sha256,
        load_downstream_config,
        _snapshot_sha,
        load_method_catalog,
        _static_output_snapshot,
    )

    downstream, _ = load_downstream_config()
    if downstream_code_sha256(downstream) != run["identity"]["downstream_code_sha256"]:
        raise ReportingError("downstream source hash changed after this run")
    if file_sha256(DOWNSTREAM_CONFIG_PATH) != run["identity"]["downstream_config_sha256"]:
        raise ReportingError("downstream config hash changed after this run")
    if file_sha256(CATALOG_PATH) != run["identity"]["catalog_sha256"]:
        raise ReportingError("method catalog hash changed after this run")
    if file_sha256(MANIFEST_PATH) != run["identity"]["manifest_sha256"]:
        raise ReportingError("corpus manifest hash changed after this run")
    if implementation_sha256() != run["identity"]["implementation_sha256"]:
        raise ReportingError("shared evaluator implementation changed after this run")
    if enhancement_code_sha256() != run["identity"]["enhancement_code_sha256"]:
        raise ReportingError("shared enhancement code changed after this run")
    if adapter_source_sha256(run["method"]["backend"]) != run["method"]["adapter_sha256"]:
        raise ReportingError("method adapter source changed after this run")
    method_config = METHOD_CONFIG_ROOT / f"{run['method']['backend']}.json"
    if file_sha256(method_config) != run["identity"]["method_config_sha256"]:
        raise ReportingError("method config hash changed after this run")
    method_config_value = load_json(method_config)
    method = run["method"]
    config_identity = {
        "backend": method_config_value.get("backend"),
        "variant_id": method_config_value.get("variant_id"),
        "source": method_config_value.get("source"),
        "source_revision": method_config_value.get("source_revision"),
        "license": method_config_value.get("license"),
        "target_class": method_config_value.get("target_class"),
        "model_path": method_config_value.get("model_path"),
        "model_sha256": method_config_value.get("model_sha256"),
    }
    if any(method.get(key) != value for key, value in config_identity.items()):
        raise ReportingError("run method identity differs from its frozen method config")
    if method["backend"] in {"none", "noisereduce"}:
        expected_disposition = "baseline" if method["backend"] == "none" else "control"
        if (
            method["catalog_row"] is not None
            or method["default_disposition"] != expected_disposition
            or method["disposition_reason"] is not None
        ):
            raise ReportingError("control method disposition differs from the frozen catalog")
    else:
        _, catalog_records = load_method_catalog()
        matches = [
            record for record in catalog_records if record["backend"] == method["backend"]
        ]
        if len(matches) != 1:
            raise ReportingError("run method is not one normalized catalog backend")
        record = matches[0]
        catalog_identity = {
            "catalog_row": record["csv_row"],
            "default_disposition": record["default_disposition"],
            "disposition_reason": record.get("blocked_reason"),
        }
        if any(method.get(key) != value for key, value in catalog_identity.items()):
            raise ReportingError("run method catalog identity mismatch")
    if _snapshot_sha(_static_output_snapshot()) != environment["static_demo_output_snapshot_sha256"]:
        raise ReportingError("static demo outputs changed after this run")
    expected_report_json = f"arms/{run['method']['backend']}/{run['method']['variant_id']}/report.json"
    expected_report_md = f"arms/{run['method']['backend']}/{run['method']['variant_id']}/report.md"
    if run["artifacts"]["report_json"] != expected_report_json:
        raise ReportingError("run report_json path differs from its method identity")
    if run["artifacts"]["report_md"] != expected_report_md:
        raise ReportingError("run report_md path differs from its method identity")
    for directory in (root / "arms", root / "arms" / run["method"]["backend"], root / "arms" / run["method"]["backend"] / run["method"]["variant_id"]):
        if directory.is_symlink():
            raise ReportingError("run layout must not contain symlinked directories")
    if {path.name for path in root.iterdir()} != {"environment.json", "run.json", "arms"}:
        raise ReportingError("run root contains missing or unexpected files")
    arms_root = root / "arms"
    if {path.name for path in arms_root.iterdir()} != {run["method"]["backend"]}:
        raise ReportingError("run must contain exactly one backend arm")
    backend_root = arms_root / run["method"]["backend"]
    if {path.name for path in backend_root.iterdir()} != {run["method"]["variant_id"]}:
        raise ReportingError("backend arm must contain exactly one configured variant")
    variant_root = backend_root / run["method"]["variant_id"]
    if {path.name for path in variant_root.iterdir()} != {"conditions", "report.json", "report.md"}:
        raise ReportingError("variant arm contains missing or unexpected artifacts")
    report_md_path = _safe_run_path(root, str(run["artifacts"]["report_md"]), "report_md")
    report_path = _safe_run_path(root, str(run["artifacts"]["report_json"]), "report_json")
    report = load_json(report_path)
    validate_contract("report", report)
    assert_no_absolute_result_paths(report, "report")
    if report["run_id"] != run["run_id"] or report["method"]["backend"] != run["method"]["backend"]:
        raise ReportingError("report identity differs from run")
    if report["method"]["variant_id"] != run["method"]["variant_id"]:
        raise ReportingError("report variant differs from run")
    if report["method"].get("disposition_reason") != run["method"].get("disposition_reason"):
        raise ReportingError("report disposition reason differs from run")
    statuses: list[dict[str, Any]] = []
    conditions_root = root / "arms" / run["method"]["backend"] / run["method"]["variant_id"] / "conditions"
    if conditions_root.is_symlink() or not conditions_root.is_dir():
        raise ReportingError("conditions must be a real directory")
    condition_directories = {path.name for path in conditions_root.iterdir()}
    if any(path.is_symlink() for path in conditions_root.iterdir()):
        raise ReportingError("conditions must not contain symlinked entries")
    status_paths = sorted(conditions_root.glob("*/status.json"))
    status_condition_ids = {path.parent.name for path in status_paths}
    if condition_directories != status_condition_ids:
        raise ReportingError("condition directories and terminal status files differ")
    for status_path in status_paths:
        safe_status_path = _safe_run_path(
            root,
            f"arms/{run['method']['backend']}/{run['method']['variant_id']}/conditions/{status_path.parent.name}/status.json",
            "condition status",
        )
        status = load_json(safe_status_path)
        validate_contract("condition", status)
        assert_no_absolute_result_paths(status, "condition")
        if status["condition_id"] != status_path.parent.name:
            raise ReportingError("condition ID differs from its terminal directory")
        _verify_no_static_composition(run, status)
        _verify_real_artifacts(root, run, status)
        statuses.append(status)
    if len({status["condition_id"] for status in statuses}) != len(statuses):
        raise ReportingError("condition IDs are not unique")
    expected = int(run["counts"]["expected_conditions"])
    if len(statuses) != expected:
        raise ReportingError(f"run must retain exactly {expected} terminal conditions")
    if len(statuses) != run["counts"]["recorded_conditions"]:
        raise ReportingError("run condition count is inconsistent")
    if sum(status["status"] == "ok" for status in statuses) != run["counts"]["ok"]:
        raise ReportingError("run successful-condition count is inconsistent")
    if sum(status["status"] != "ok" for status in statuses) != run["counts"]["failed"]:
        raise ReportingError("run failed-condition count is inconsistent")
    if report["counts"]["total"] != len(statuses):
        raise ReportingError("report condition count is inconsistent")
    rebuilt_report = build_arm_report(
        run_root=root,
        run_record=run,
        statuses=statuses,
        static_outputs_unchanged=True,
    )
    if report != rebuilt_report:
        raise ReportingError("report differs from a strict rebuild over terminal evidence")
    rebuilt_status_hashes = {
        str(status["condition_id"]): file_sha256(
            _safe_run_path(root, str(status["artifacts"]["status"]), "condition status")
        )
        for status in statuses
    }
    if report["artifacts"]["status_files_sha256"] != rebuilt_status_hashes:
        raise ReportingError("report terminal status hashes differ from persisted status files")
    expected_markdown = render_report_markdown(report)
    try:
        actual_markdown = report_md_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ReportingError("report markdown cannot be read") from error
    if actual_markdown != expected_markdown:
        raise ReportingError("report markdown differs from the strict report rendering")
    return {"run": run, "report": report, "statuses": statuses}


def cluster_bootstrap(
    values_by_group: Mapping[str, float],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap paired group deltas over exactly 15 independent base groups."""

    if len(values_by_group) != EXPECTED_GROUP_COUNT:
        raise ReportingError(f"cluster bootstrap requires exactly {EXPECTED_GROUP_COUNT} groups")
    if iterations <= 0:
        raise ReportingError("bootstrap iterations must be positive")
    groups = sorted(values_by_group)
    values = np.asarray([float(values_by_group[group]) for group in groups], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ReportingError("bootstrap values must all be finite")
    generator = np.random.Generator(np.random.PCG64(seed))
    indices = generator.integers(0, EXPECTED_GROUP_COUNT, size=(iterations, EXPECTED_GROUP_COUNT))
    means = values[indices].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    observed = float(np.mean(values))
    centered_means = means - observed
    p_value = min(1.0, float(np.mean(np.abs(centered_means) >= abs(observed))))
    return {
        "groups": groups,
        "group_count": EXPECTED_GROUP_COUNT,
        "iterations": iterations,
        "seed": seed,
        "observed_mean_delta": observed,
        "ci95_low": float(lower),
        "ci95_high": float(upper),
        "p_value": p_value,
    }


def holm_adjust(
    p_values: Mapping[str, float], *, family_count: int | None = None
) -> dict[str, float]:
    """Holm step-down adjustment with a predeclared total family count."""

    if family_count is None:
        family_count = len(p_values)
    if family_count < len(p_values) or family_count <= 0:
        raise ReportingError("Holm family count is invalid")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (key, p_value) in enumerate(ordered):
        if not math.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
            raise ReportingError("Holm p-values must be finite values in [0, 1]")
        running = max(running, min(1.0, (family_count - index) * p_value))
        adjusted[key] = running
    return adjusted


def _metric_observations(
    run_root: Path,
    run: Mapping[str, Any],
    statuses: Sequence[Mapping[str, Any]],
    group: str,
    name: str,
) -> dict[str, float]:
    observations: dict[str, float] = {}
    for status in statuses:
        if status["status"] != "ok":
            continue
        metrics = load_json(
            _safe_run_path(run_root, str(status["artifacts"]["metrics"]), "metrics")
        )
        section_name = "clinical" if name == "false_positive_fields" else group
        value = _metric_value(metrics, section_name, name)
        if value is not None and math.isfinite(value):
            observations[str(status["condition_id"])] = value
    return observations


def _group_means(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    statuses: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for status in statuses:
        condition_id = str(status["condition_id"])
        if condition_id not in baseline or condition_id not in candidate:
            continue
        groups.setdefault(str(status["group_id"]), []).append(candidate[condition_id] - baseline[condition_id])
    return {
        group: float(np.mean(deltas))
        for group, deltas in groups.items()
        if deltas
    }


def _favorable(bootstrap: Mapping[str, Any], direction: str) -> bool:
    low = float(bootstrap["ci95_low"])
    high = float(bootstrap["ci95_high"])
    return high < 0.0 if direction == "lower" else low > 0.0


def validate_comparison_inputs(
    baseline_root: Path,
    candidate_roots: Sequence[Path],
) -> dict[str, Any]:
    """Require one verified none run plus all 12 terminal catalog-row runs."""

    baseline = verify_run(baseline_root)
    if baseline["run"]["method"]["backend"] != "none" or baseline["run"]["counts"]["ok"] != 30:
        raise ReportingError("comparison baseline must be a verified 30/30 none run")
    if baseline["run"]["method"].get("adapter_sha256") is None:
        raise ReportingError("comparison baseline must identify a concrete none adapter")
    baseline_identity_fields = {
        "backend": "none",
        "variant_id": "identity-v1",
        "catalog_row": None,
        "default_disposition": "baseline",
        "disposition_reason": None,
        "source": "MediBytes shared baseline",
        "source_revision": "b0",
        "license": "MediBytes shared baseline",
        "target_class": "baseline",
        "model_path": None,
        "model_sha256": None,
    }
    if any(
        baseline["run"]["method"].get(key) != value
        for key, value in baseline_identity_fields.items()
    ):
        raise ReportingError("comparison baseline method identity is not the frozen none control")
    from eval.runner import load_method_catalog

    _, catalog = load_method_catalog()
    expected = {record["backend"]: record for record in catalog}
    if len(candidate_roots) != 12:
        raise ReportingError("comparison requires exactly 12 candidate run roots")
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    identity_keys = ("catalog_sha256", "corpus_id", "downstream_code_sha256", "downstream_config_sha256", "enhancement_code_sha256", "implementation_sha256", "manifest_sha256")
    baseline_identity = baseline["run"]["identity"]
    for root in candidate_roots:
        verified = verify_run(root)
        run = verified["run"]
        if run.get("composition") != {
            "mode": "single_backend",
            "backend_count": 1,
            "layered": False,
            "fallback_used": False,
        }:
            raise ReportingError("comparison candidates must be one non-layered, non-fallback backend")
        backend = run["method"]["backend"]
        if backend in seen or backend not in expected:
            raise ReportingError("candidate runs must contain each normalized catalog backend exactly once")
        seen.add(backend)
        record = expected[backend]
        catalog_identity = {
            "catalog_row": record["csv_row"],
            "variant_id": record["variant_id"],
            "default_disposition": record["default_disposition"],
            "disposition_reason": record.get("blocked_reason"),
            "source": record["upstream_source"],
            "source_revision": record["upstream_revision"],
            "license": record["license"],
            "target_class": record["target_class"],
            "model_path": record["model_path"],
            "model_sha256": record["model_sha256"],
        }
        if any(run["method"].get(key) != value for key, value in catalog_identity.items()):
            raise ReportingError("candidate method catalog identity mismatch")
        if any(run["identity"][key] != baseline_identity[key] for key in identity_keys):
            raise ReportingError("candidate shared corpus/downstream/catalog/implementation hashes differ from baseline")
        disposition = run["method"]["default_disposition"]
        expected_count = 0 if disposition in {"blocked_prerequisite", "not_comparable", "blocked_target"} else 30
        if expected_count == 30 and run["method"].get("adapter_sha256") is None:
            raise ReportingError("attempted candidate row has no concrete adapter source hash")
        if run["counts"]["expected_conditions"] != expected_count:
            raise ReportingError("candidate run has an invalid condition expectation for its disposition")
        eligible_observation = (
            expected_count == 30
            and run["counts"]["ok"] == expected_count
            and verified["report"]["counts"]["ok"] == expected_count
            and verified["report"]["counts"]["total"] == expected_count
        )
        if verified["report"]["comparison_eligible"] != eligible_observation:
            raise ReportingError("candidate comparison_eligible does not match complete observations")
        if eligible_observation and verified["report"]["status"] != "observed":
            raise ReportingError("complete candidate observations must have an observed report")
        candidates.append(verified)
    if seen != set(expected):
        missing = sorted(set(expected) - seen)
        raise ReportingError(f"comparison is missing terminal row reports: {missing}")
    return {"baseline": baseline, "candidates": candidates, "catalog": catalog}




def compare_runs(
    *,
    baseline_root: Path,
    candidate_roots: Sequence[Path],
    output: Path,
) -> dict[str, Any]:
    """Compare only after all 12 row reports exist; never select a production default."""

    admitted = validate_comparison_inputs(baseline_root, candidate_roots)
    if output.is_absolute():
        raise ReportingError("comparison output must be repository-relative under eval/reports")
    relative = PurePosixPath(str(output))
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ReportingError("comparison output contains an unsafe path")
    try:
        output_root = resolve_contained_result_path(
            Path(__file__).resolve().parents[1],
            f"eval/reports/{relative.as_posix()}",
            "comparison output",
        )
    except ContractValidationError as error:
        raise ReportingError("comparison output must be symlink-free under eval/reports") from error
    if output_root.exists():
        raise ReportingError("comparison output already exists and cannot be overwritten")
    baseline = admitted["baseline"]
    baseline_run_root = baseline_root.resolve(strict=True)
    per_metric_p_values: dict[str, dict[str, float]] = {name: {} for name in (*AUDIO_METRICS, *TEXT_METRICS)}
    rows: list[dict[str, Any]] = []
    for candidate in admitted["candidates"]:
        run = candidate["run"]
        report = candidate["report"]
        backend = run["method"]["backend"]
        disposition = run["method"]["default_disposition"]
        if disposition in {"blocked_prerequisite", "not_comparable", "blocked_target"}:
            rows.append(
                {
                    "backend": backend,
                    "variant_id": run["method"]["variant_id"],
                    "csv_row": run["method"]["catalog_row"],
                    "target_class": run["method"]["target_class"],
                    "license": run["method"]["license"],
                    "disposition": disposition,
                    "outcome": disposition,
                    "comparison_metrics": {},
                    "reason": run["method"].get("disposition_reason"),
                    "rank_eligible": False,
                }
            )
            continue
        candidate_root = next(
            root.resolve(strict=True)
            for root in candidate_roots
            if load_json(root / "run.json")["method"]["backend"] == backend
        )
        baseline_clean_by_group = {
            str(status["group_id"]): {
                "input_sha256": str(status["input"]["sha256"]),
                "enhanced_sha256": str(status["enhanced"]["sha256"]),
            }
            for status in baseline["statuses"]
            if status["condition_id"].endswith("-clean") and status["status"] == "ok"
        }
        for condition_status in candidate["statuses"]:
            if condition_status["status"] != "ok":
                continue
            group_id = str(condition_status["group_id"])
            if group_id not in baseline_clean_by_group:
                raise ReportingError("candidate group is absent from the canonical none run")
            condition_metrics = load_json(
                _safe_run_path(candidate_root, str(condition_status["artifacts"]["metrics"]), "metrics")
            )
            reference_input_hash = condition_metrics.get("audio", {}).get("reference_input_sha256")
            if reference_input_hash != baseline_clean_by_group[group_id]["input_sha256"]:
                raise ReportingError("candidate metrics do not use the canonical clean source recording")
            canonical_none_hash = condition_metrics.get("audio", {}).get("none_reference_sha256")
            if canonical_none_hash != baseline_clean_by_group[group_id]["enhanced_sha256"]:
                raise ReportingError("candidate metrics do not bind the canonical none-run clean output")
            if condition_status["condition_kind"] == "clean":
                enhancement = load_json(
                    _safe_run_path(
                        candidate_root,
                        str(condition_status["artifacts"]["enhancement"]),
                        "enhancement",
                    )
                )
                if enhancement.get("none_reference_sha256") != baseline_clean_by_group[group_id]["enhanced_sha256"]:
                    raise ReportingError("candidate clean preprocessing differs from the canonical none run")
        comparison_metrics: dict[str, Any] = {}
        metric_outcomes: list[bool] = []
        for group, metric_names in (("audio", AUDIO_METRICS), ("text", TEXT_METRICS)):
            for name, direction in metric_names.items():
                metric_key = f"{group}.{name}"
                if not report["comparison_eligible"]:
                    comparison_metrics[metric_key] = {
                        "status": "invalid",
                        "reason": "candidate does not contain 30 complete terminal observations",
                        "paired_condition_count": 0,
                    }
                    continue
                baseline_values = _metric_observations(
                    baseline_run_root,
                    baseline["run"],
                    baseline["statuses"],
                    group,
                    name,
                )
                candidate_values = _metric_observations(
                    candidate_root,
                    run,
                    candidate["statuses"],
                    group,
                    name,
                )
                groups = _group_means(baseline_values, candidate_values, candidate["statuses"])
                if len(groups) != EXPECTED_GROUP_COUNT:
                    comparison_metrics[metric_key] = {
                        "status": "invalid" if report["status"] == "invalid" else "not_comparable",
                        "reason": "fewer than 15 complete paired groups",
                        "paired_condition_count": sum(
                            1
                            for status in candidate["statuses"]
                            if status["condition_id"] in baseline_values and status["condition_id"] in candidate_values
                        ),
                    }
                    continue
                bootstrap = cluster_bootstrap(groups)
                favorable = _favorable(bootstrap, direction)
                comparison_metrics[metric_key] = {
                    **bootstrap,
                    "favorable_direction": direction,
                    "favorable_ci": favorable,
                }
                per_metric_p_values[metric_key][backend] = float(bootstrap["p_value"])
                metric_outcomes.append(favorable)
        baseline_normalized_wer = _metric_observations(
            baseline_run_root, baseline["run"], baseline["statuses"], "text", "normalized_wer"
        )
        candidate_normalized_wer = _metric_observations(
            candidate_root, run, candidate["statuses"], "text", "normalized_wer"
        )
        baseline_snr = _metric_observations(
            baseline_run_root, baseline["run"], baseline["statuses"], "audio", "snr_db"
        )
        candidate_snr = _metric_observations(
            candidate_root, run, candidate["statuses"], "audio", "snr_db"
        )
        strata: dict[str, dict[str, dict[str, float | int | None]]] = {"bucket": {}, "noise_condition": {}}
        for condition_status in candidate["statuses"]:
            condition_id = str(condition_status["condition_id"])
            if condition_status["status"] != "ok":
                continue
            for stratum_name, stratum_value in (
                ("bucket", condition_status["bucket"]),
                ("noise_condition", condition_status["noise_type"]),
            ):
                row_value = strata[stratum_name].setdefault(
                    str(stratum_value),
                    {"paired_conditions": 0, "normalized_wer_delta": None, "snr_db_delta": None},
                )
                row_value["paired_conditions"] = int(row_value["paired_conditions"]) + 1
                if condition_id in baseline_normalized_wer and condition_id in candidate_normalized_wer:
                    row_value["normalized_wer_delta"] = float(
                        candidate_normalized_wer[condition_id] - baseline_normalized_wer[condition_id]
                    )
                if condition_id in baseline_snr and condition_id in candidate_snr:
                    row_value["snr_db_delta"] = float(candidate_snr[condition_id] - baseline_snr[condition_id])
        rows.append(
            {
                "backend": backend,
                "variant_id": run["method"]["variant_id"],
                "csv_row": run["method"]["catalog_row"],
                "target_class": run["method"]["target_class"],
                "license": run["method"]["license"],
                "disposition": disposition,
                "outcome": "invalid" if report["status"] == "invalid" else ("observed" if any(metric_outcomes) else "inconclusive"),
                "comparison_metrics": comparison_metrics,
                "strata": strata,
                "rank_eligible": report["comparison_eligible"],
                "resources": report["aggregate"]["resources"],
                "counts": report["counts"],
            }
        )
    holm: dict[str, dict[str, float]] = {}
    for metric_name, p_values in per_metric_p_values.items():
        if p_values:
            holm[metric_name] = holm_adjust(p_values, family_count=12)
    for row in rows:
        for metric_name, adjusted in holm.items():
            metric = row["comparison_metrics"].get(metric_name)
            if isinstance(metric, Mapping) and row["backend"] in adjusted:
                metric["holm_adjusted_p_value"] = adjusted[row["backend"]]
                if adjusted[row["backend"]] < 0.05 and metric.get("favorable_ci") is True:
                    metric["outcome"] = "observed"
                else:
                    metric["outcome"] = "inconclusive"
        if row["comparison_metrics"]:
            row["outcome"] = (
                "invalid"
                if row.get("counts", {}).get("ok") != 30
                else (
                    "observed"
                    if any(
                        isinstance(metric, Mapping) and metric.get("outcome") == "observed"
                        for metric in row["comparison_metrics"].values()
                    )
                    else "inconclusive"
                )
            )
    ranked = sorted(
        (
            row
            for row in rows
            if row["rank_eligible"] and isinstance(row["comparison_metrics"].get("text.normalized_wer"), Mapping)
            and "observed_mean_delta" in row["comparison_metrics"]["text.normalized_wer"]
        ),
        key=lambda row: float(row["comparison_metrics"]["text.normalized_wer"]["observed_mean_delta"]),
    )
    for rank, row in enumerate(ranked, start=1):
        row["paired_rank"] = rank
    result = {
        "schema_version": "1.0.0",
        "record_type": "comparison",
        "comparison_id": relative.name,
        "baseline_run_id": baseline["run"]["run_id"],
        "candidate_run_ids": [candidate["run"]["run_id"] for candidate in admitted["candidates"]],
        "bootstrap": {
            "unit": "base_group_id",
            "group_count": 15,
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed": BOOTSTRAP_SEED,
            "holm_family_count": 12,
        },
        "rows": rows,
        "execution_classes": {
            target_class: [row["backend"] for row in rows if row["target_class"] == target_class]
            for target_class in sorted({row["target_class"] for row in rows})
        },
        "production_assessment": "not_assessed_nonclinical_smoke_corpus",
        "limitations": [
            "Only rows with a single-backend 30/30 run are rank eligible.",
            "Blocked and not-comparable rows receive no accuracy score.",
            "No production default is selected and no layered or fallback composition is evaluated.",
        ],
    }
    output_root.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(output_root / "comparison.json", result)
    write_text_exclusive(output_root / "comparison.md", render_comparison_markdown(result))
    return result


def render_comparison_markdown(comparison: Mapping[str, Any]) -> str:
    lines = [
        f"# MediBytes denoising comparison {comparison['comparison_id']}",
        "",
        "Single-backend rows only. Candidate metrics are paired deltas against the verified `none` baseline over 15 base groups (10,000 resamples, seed 1729, Holm family count 12).",
        "",
        "| CSV row | Backend | Class | Disposition | Outcome | Normalized WER delta (95% CI) |",
        "|---:|---|---|---|---|---:|",
    ]
    for row in comparison["rows"]:
        metric = row["comparison_metrics"].get("text.normalized_wer", {})
        if "ci95_low" in metric:
            metric_text = f"{metric['observed_mean_delta']:.4f} [{metric['ci95_low']:.4f}, {metric['ci95_high']:.4f}]"
        else:
            metric_text = "not_comparable"
        lines.append(
            f"| {row['csv_row']} | {row['backend']} | {row['target_class']} | {row['disposition']} | {row['outcome']} | {metric_text} |"
        )
    lines.extend(
        [
            "",
            "Blocked/not-comparable/target-blocked rows are not assigned quality scores. This 15-base non-clinical smoke corpus cannot establish production medical performance, safety, or a default backend.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "AUDIO_METRICS",
    "BOOTSTRAP_ITERATIONS",
    "BOOTSTRAP_SEED",
    "TEXT_METRICS",
    "ReportingError",
    "build_arm_report",
    "cluster_bootstrap",
    "compare_runs",
    "holm_adjust",
    "render_arm_report_markdown",
    "render_comparison_markdown",
    "validate_comparison_inputs",
    "verify_run",
]

from __future__ import annotations

from pathlib import Path

import pytest

from eval.contracts import (
    atomic_write_json,
    file_sha256,
    load_json,
    validate_contract,
    write_json_exclusive,
    write_text_exclusive,
)
from eval.reporting import (
    ReportingError,
    build_arm_report,
    cluster_bootstrap,
    _verify_real_artifacts,
    holm_adjust,
    render_report_markdown,
    validate_comparison_inputs,
    verify_run,
)
from eval.runner import (
    CATALOG_PATH,
    DOWNSTREAM_CONFIG_PATH,
    MANIFEST_PATH,
    METHOD_CONFIG_ROOT,
    _snapshot_sha,
    _static_output_snapshot,
    adapter_source_sha256,
    downstream_code_sha256,
    enhancement_code_sha256,
    implementation_sha256,
    load_downstream_config,
)

SHA = "d" * 64


def _identity() -> dict[str, str]:
    return {
        "catalog_sha256": SHA,
        "corpus_id": "pilot-15-v1",
        "downstream_code_sha256": SHA,
        "downstream_config_sha256": SHA,
        "enhancement_code_sha256": SHA,
        "implementation_sha256": SHA,
        "manifest_sha256": SHA,
        "method_config_sha256": SHA,
    }


def _baseline_run() -> dict[str, object]:
    return {
        "run_id": "baseline",
        "method": {
            "backend": "none",
            "adapter_sha256": SHA,
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
        },
        "identity": _identity(),
        "composition": {
            "mode": "single_backend",
            "backend_count": 1,
            "layered": False,
            "fallback_used": False,
        },
        "counts": {"expected_conditions": 30, "recorded_conditions": 30, "ok": 30, "failed": 0},
    }


def _candidate_run(record: dict[str, object]) -> dict[str, object]:
    disposition = str(record["default_disposition"])
    expected = 0 if disposition in {"blocked_prerequisite", "not_comparable", "blocked_target"} else 30
    return {
        "run_id": f"run-{record['backend']}",
        "method": {
            "adapter_sha256": SHA,
            "backend": record["backend"],
            "variant_id": record["variant_id"],
            "catalog_row": record["csv_row"],
            "default_disposition": disposition,
            "disposition_reason": record.get("blocked_reason"),
            "source": record["upstream_source"],
            "source_revision": record["upstream_revision"],
            "license": record["license"],
            "target_class": record["target_class"],
            "model_path": record["model_path"],
            "model_sha256": record["model_sha256"],
        },
        "identity": {**_identity(), "method_config_sha256": "e" * 64},
        "composition": {
            "mode": "single_backend",
            "backend_count": 1,
            "layered": False,
            "fallback_used": False,
        },
        "counts": {
            "expected_conditions": expected,
            "recorded_conditions": expected,
            "ok": expected,
            "failed": 0,
        },
    }


def test_cluster_bootstrap_uses_exactly_15_base_groups() -> None:
    values = {f"group-{index:02d}": 1.0 for index in range(1, 16)}
    result = cluster_bootstrap(values, iterations=1_000, seed=1729)
    assert result["group_count"] == 15
    assert result["observed_mean_delta"] == 1.0
    assert result["ci95_low"] == 1.0
    assert result["ci95_high"] == 1.0
    with pytest.raises(ReportingError, match="15"):
        cluster_bootstrap({f"group-{index}": 0.0 for index in range(14)})


def test_holm_adjustment_is_step_down_and_capped() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


def test_comparison_refuses_incomplete_twelve_row_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("eval.runner.load_method_catalog", lambda: ({"records": []}, load_json(CATALOG_PATH)["records"]))
    monkeypatch.setattr(
        "eval.reporting.verify_run",
        lambda _root: {"run": _baseline_run(), "report": {}, "statuses": []},
    )
    with pytest.raises(ReportingError, match="exactly 12"):
        validate_comparison_inputs(Path("baseline"), [Path(f"candidate-{index}") for index in range(11)])


@pytest.mark.parametrize(
    ("field", "value"),
    [("fallback_used", True), ("layered", True), ("backend_count", 2)],
)
def test_comparison_refuses_fallback_and_layered_candidates(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    records = load_json(CATALOG_PATH)["records"]
    candidates = {str(record["backend"]): _candidate_run(record) for record in records}
    monkeypatch.setattr("eval.runner.load_method_catalog", lambda: ({"records": []}, records))

    def verify(root: Path) -> dict[str, object]:
        run = _baseline_run() if root.name == "baseline" else candidates[root.name.removeprefix("candidate-")]
        run["composition"][field] = value
        return {
            "run": run,
            "report": {"comparison_eligible": True, "status": "observed"},
            "statuses": [],
        }

    monkeypatch.setattr("eval.reporting.verify_run", verify)
    roots = [Path("baseline"), *[Path(f"candidate-{backend}") for backend in candidates]]
    with pytest.raises(ReportingError, match="non-layered, non-fallback"):
        validate_comparison_inputs(roots[0], roots[1:])


def test_comparison_rejects_catalog_method_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = load_json(CATALOG_PATH)["records"]
    candidates = {str(record["backend"]): _candidate_run(record) for record in records}
    candidates["rnnoise"]["method"]["variant_id"] = "unreviewed-variant"
    monkeypatch.setattr("eval.runner.load_method_catalog", lambda: ({"records": []}, records))

    def verify(root: Path) -> dict[str, object]:
        run = _baseline_run() if root.name == "baseline" else candidates[root.name.removeprefix("candidate-")]
        return {
            "run": run,
            "report": {
                "comparison_eligible": True,
                "status": "observed",
                "counts": {"ok": run["counts"]["ok"], "total": run["counts"]["expected_conditions"]},
            },
            "statuses": [],
        }

    monkeypatch.setattr("eval.reporting.verify_run", verify)
    roots = [Path("baseline"), *[Path(f"candidate-{backend}") for backend in candidates]]
    with pytest.raises(ReportingError, match="catalog identity"):
        validate_comparison_inputs(roots[0], roots[1:])


def test_comparison_rejects_forged_complete_eligibility_on_incomplete_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = load_json(CATALOG_PATH)["records"]
    candidates = {str(record["backend"]): _candidate_run(record) for record in records}
    candidates["rnnoise"]["counts"]["ok"] = 29
    candidates["rnnoise"]["counts"]["failed"] = 1
    monkeypatch.setattr("eval.runner.load_method_catalog", lambda: ({"records": []}, records))

    def verify(root: Path) -> dict[str, object]:
        run = _baseline_run() if root.name == "baseline" else candidates[root.name.removeprefix("candidate-")]
        return {
            "run": run,
            "report": {
                "comparison_eligible": True,
                "status": "observed",
                "counts": {"ok": run["counts"]["ok"], "total": run["counts"]["expected_conditions"]},
            },
            "statuses": [],
        }

    monkeypatch.setattr("eval.reporting.verify_run", verify)
    roots = [Path("baseline"), *[Path(f"candidate-{backend}") for backend in candidates]]
    with pytest.raises(ReportingError, match="comparison_eligible"):
        validate_comparison_inputs(roots[0], roots[1:])




def _write_blocked_run(root: Path) -> tuple[Path, Path]:
    record = next(
        record for record in load_json(CATALOG_PATH)["records"] if record["csv_row"] == 6
    )
    backend = str(record["backend"])
    variant = str(record["variant_id"])
    variant_root = root / "arms" / backend / variant
    (variant_root / "conditions").mkdir(parents=True)
    downstream, downstream_path = load_downstream_config()
    method_config_path = METHOD_CONFIG_ROOT / f"{backend}.json"
    identity = {
        "catalog_sha256": file_sha256(CATALOG_PATH),
        "corpus_id": "pilot-15-v1",
        "downstream_code_sha256": downstream_code_sha256(downstream),
        "enhancement_code_sha256": enhancement_code_sha256(),
        "downstream_config_sha256": file_sha256(downstream_path),
        "implementation_sha256": implementation_sha256(),
        "manifest_sha256": file_sha256(MANIFEST_PATH),
        "method_config_sha256": file_sha256(method_config_path),
    }
    environment = {
        "schema_version": "1.0.0",
        "record_type": "environment",
        "run_id": root.name,
        "identity": identity,
        "static_demo_output_snapshot_sha256": _snapshot_sha(_static_output_snapshot()),
    }
    environment_path = root / "environment.json"
    write_json_exclusive(environment_path, environment)
    method = {
        "adapter_sha256": adapter_source_sha256(backend),
        "backend": backend,
        "variant_id": variant,
        "catalog_row": record["csv_row"],
        "default_disposition": record["default_disposition"],
        "disposition_reason": record.get("blocked_reason"),
        "source": record["upstream_source"],
        "source_revision": record["upstream_revision"],
        "license": record["license"],
        "target_class": record["target_class"],
        "model_path": record["model_path"],
        "model_sha256": record["model_sha256"],
    }
    run = {
        "schema_version": "1.0.0",
        "record_type": "run",
        "run_id": root.name,
        "status": "complete",
        "environment_sha256": file_sha256(environment_path),
        "method": method,
        "identity": identity,
        "composition": {
            "mode": "single_backend",
            "backend_count": 1,
            "layered": False,
            "fallback_used": False,
        },
        "counts": {
            "expected_conditions": 0,
            "recorded_conditions": 0,
            "ok": 0,
            "failed": 0,
        },
        "artifacts": {
            "environment": "environment.json",
            "report_json": f"arms/{backend}/{variant}/report.json",
            "report_md": f"arms/{backend}/{variant}/report.md",
        },
        "limitations": ["non-clinical smoke corpus"],
    }
    validate_contract("run", run)
    write_json_exclusive(root / "run.json", run)
    report = build_arm_report(
        run_root=root,
        run_record=run,
        statuses=[],
        static_outputs_unchanged=True,
    )
    report_path = root / str(run["artifacts"]["report_json"])
    write_json_exclusive(report_path, report)
    write_text_exclusive(
        root / str(run["artifacts"]["report_md"]), render_report_markdown(report)
    )
    return root, report_path


def test_verify_run_rebuilds_report_and_exact_markdown(tmp_path: Path) -> None:
    root, report_path = _write_blocked_run(tmp_path / "blocked-run")
    verified = verify_run(root)
    assert verified["report"]["status"] == "blocked_prerequisite"

    tampered = load_json(report_path)
    tampered["limitations"] = [*tampered["limitations"], "tampered"]
    atomic_write_json(report_path, tampered)
    with pytest.raises(ReportingError, match="strict rebuild"):
        verify_run(root)


def test_verify_run_rejects_modified_report_markdown(tmp_path: Path) -> None:
    root, _ = _write_blocked_run(tmp_path / "blocked-run")
    markdown_path = root / "arms" / "percepnet" / "unofficial-8ffae433" / "report.md"
    markdown_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ReportingError, match="markdown"):
        verify_run(root)


def test_report_rebuild_binds_terminal_status_hashes(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    status_relative = (
        "arms/none/identity-v1/conditions/sample-001-noisy/status.json"
    )
    status = {
        "schema_version": "1.0.0",
        "record_type": "condition",
        "condition_id": "sample-001-noisy",
        "sample_id": "sample-001",
        "group_id": "group-001",
        "bucket": "en-US-proxy",
        "condition_kind": "noisy",
        "noise_type": "white",
        "requested_backend": "none",
        "actual_backend": None,
        "variant_id": "identity-v1",
        "fallback_used": False,
        "status": "failed",
        "stage": "reference",
        "downstream_status": "not_run",
        "started_at": "2026-09-25T00:00:00Z",
        "completed_at": "2026-09-25T00:00:01Z",
        "input": {
            "path": "corpus/pilot-15-v1/sample-001-noisy.wav",
            "sha256": SHA,
            "sample_count": 16_000,
            "sample_rate": 16_000,
        },
        "enhanced": None,
        "repeat_hashes": [],
        "error": None,
        "artifacts": {
            name: f"arms/none/identity-v1/conditions/sample-001-noisy/{name}.json"
            for name in (
                "enhancement",
                "transcript",
                "entities",
                "fields",
                "resources",
                "metrics",
                "status",
            )
        },
    }
    validate_contract("condition", status)
    status_path = run_root / status_relative
    atomic_write_json(status_path, status)
    method = {
        **_baseline_run()["method"],
        "source": "MediBytes shared baseline",
    }
    run = {
        "run_id": "run",
        "method": method,
        "counts": {
            "expected_conditions": 1,
            "recorded_conditions": 1,
            "ok": 0,
            "failed": 1,
        },
        "limitations": ["non-clinical smoke corpus"],
    }
    first = build_arm_report(
        run_root=run_root,
        run_record=run,
        statuses=[status],
        static_outputs_unchanged=True,
    )
    first_hash = first["artifacts"]["status_files_sha256"]["sample-001-noisy"]
    status["completed_at"] = "2026-09-25T00:00:02Z"
    atomic_write_json(status_path, status)
    second = build_arm_report(
        run_root=run_root,
        run_record=run,
        statuses=[status],
        static_outputs_unchanged=True,
    )
    assert first_hash != second["artifacts"]["status_files_sha256"]["sample-001-noisy"]


def test_condition_artifact_inventory_requires_json_filenames(tmp_path: Path) -> None:
    prefix = "arms/none/identity-v1/conditions/sample-001-noisy"
    artifacts = {
        name: f"{prefix}/{name}.json"
        for name in (
            "enhancement",
            "transcript",
            "entities",
            "fields",
            "resources",
            "metrics",
            "status",
        )
    }
    for name, relative in artifacts.items():
        if name != "status":
            atomic_write_json(tmp_path / relative, {})
    status = {
        "condition_id": "sample-001-noisy",
        "status": "failed",
        "enhanced": None,
        "artifacts": artifacts,
    }
    atomic_write_json(tmp_path / artifacts["status"], status)
    run = {
        "method": {
            "backend": "none",
            "variant_id": "identity-v1",
            "model_sha256": None,
            "source_revision": "b0",
        },
        "identity": {"method_config_sha256": SHA},
    }

    _verify_real_artifacts(tmp_path, run, status)

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft7Validator

from eval.contracts import (
    ContractValidationError,
    assert_relative_result_path,
    load_schema,
    loads_strict,
    resolve_contained_result_path,
    validate_contract,
)
from eval.reporting import build_arm_report

REPO_ROOT = Path(__file__).resolve().parents[2]
SHA = "a" * 64


def _run_fixture() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "record_type": "run",
        "run_id": "contract-run",
        "status": "complete",
        "environment_sha256": SHA,
        "method": {
            "adapter_sha256": None,
            "backend": "percepnet",
            "variant_id": "unofficial-8ffae433",
            "catalog_row": 6,
            "default_disposition": "blocked_prerequisite",
            "disposition_reason": "no_authoritative_pretrained_checkpoint_or_reproducible_training_recipe",
            "source": "https://github.com/jzi040941/PercepNet",
            "source_revision": "8ffae4337d23f920176ac2a7426e84610fa338ab",
            "license": "unverified",
            "target_class": "realtime_laptop",
            "model_path": None,
            "model_sha256": None,
        },
        "identity": {
            "catalog_sha256": SHA,
            "corpus_id": "pilot-15-v1",
            "downstream_code_sha256": SHA,
            "enhancement_code_sha256": SHA,
            "downstream_config_sha256": SHA,
            "implementation_sha256": SHA,
            "manifest_sha256": SHA,
            "method_config_sha256": SHA,
        },
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
            "report_json": "arms/percepnet/unofficial-8ffae433/report.json",
            "report_md": "arms/percepnet/unofficial-8ffae433/report.md",
        },
        "limitations": ["non-clinical smoke corpus"],
    }


@pytest.mark.parametrize("kind", ["corpus", "run", "condition", "report"])
def test_all_contracts_are_strict_draft_07(kind: str) -> None:
    schema = load_schema(kind)
    assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
    assert schema["additionalProperties"] is False
    Draft7Validator.check_schema(schema)


def test_real_corpus_record_matches_contract_and_rejects_extra_member() -> None:
    line = (REPO_ROOT / "eval/corpus/manifest.jsonl").read_text(encoding="utf-8").splitlines()[0]
    record = json.loads(line)
    validate_contract("corpus", record)
    record["unexpected"] = True
    with pytest.raises(ContractValidationError):
        validate_contract("corpus", record)


def test_run_schema_rejects_fallback_and_layering() -> None:
    run = _run_fixture()
    validate_contract("run", run)
    run["composition"]["fallback_used"] = True
    with pytest.raises(ContractValidationError):
        validate_contract("run", run)


def test_blocked_report_has_explicit_disposition_and_no_conditions(tmp_path: Path) -> None:
    report = build_arm_report(
        run_root=tmp_path,
        run_record=_run_fixture(),
        statuses=[],
        static_outputs_unchanged=True,
    )
    assert report["status"] == "blocked_prerequisite"
    assert report["counts"]["total"] == 0
    assert report["method"]["disposition_reason"].startswith("no_authoritative")
    assert report["production_assessment"] == "not_assessed_nonclinical_smoke_corpus"


def test_serialized_result_paths_must_be_relative_and_non_traversing() -> None:
    assert str(assert_relative_result_path("arms/none/identity/report.json", "path")) == (
        "arms/none/identity/report.json"
    )
    for value in ("/tmp/result.wav", "../escape.json", "file:///tmp/result.wav"):
        with pytest.raises(ContractValidationError):
            assert_relative_result_path(value, "path")


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_strict_json_rejects_non_finite_numbers(constant: str) -> None:
    with pytest.raises(ContractValidationError):
        loads_strict(f'{{"value": {constant}}}')


def test_contained_result_path_rejects_symlinked_parent(tmp_path: Path) -> None:
    root = tmp_path / "run"
    outside = tmp_path / "outside"
    (root / "arms").mkdir(parents=True)
    outside.mkdir()
    (root / "arms" / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ContractValidationError, match="symlink"):
        resolve_contained_result_path(root, "arms/linked/status.json", "status")


def test_condition_contract_accepts_requested_backend_and_every_repeat_stage() -> None:
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
    for stage in ("warmup", "repeat-1", "repeat-2", "repeat-3"):
        status["stage"] = stage
        validate_contract("condition", status)

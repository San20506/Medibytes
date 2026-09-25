"""Command-line entry point for the MediBytes denoising evaluation pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from eval.contracts import ContractValidationError, load_json, resolve_contained_result_path, validate_contract
from eval.reporting import ReportingError, compare_runs, verify_run
from eval.runner import (
    BACKEND_IDS,
    DEFAULT_DATA_ROOT,
    RunnerError,
    _load_and_verify_corpus,
    preflight_method,
    resolve_data_root,
    run_evaluation,
)

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "downstream-v1.json"
MANIFEST_PATH = Path(__file__).resolve().parent / "corpus" / "manifest.jsonl"


def _validate_corpus_command(manifest: Path, data_root_raw: str) -> dict[str, object]:
    data_root = resolve_data_root(data_root_raw)
    records = _load_and_verify_corpus(manifest, data_root)
    return {
        "status": "ok",
        "corpus_id": "pilot-15-v1",
        "groups": len({record["group_id"] for record in records}),
        "conditions": len(records) * 2,
        "contract": "corpus-record.schema.json",
    }


def _run_root_under_data_root(raw: str) -> Path:
    supplied = Path(raw)
    if not supplied.is_absolute():
        raise RunnerError("run root must be an absolute path")
    if ".." in supplied.parts:
        raise RunnerError("run root must not contain parent traversal")
    data_root = resolve_data_root()
    runs_root = data_root / "runs"
    if supplied.parent != runs_root:
        raise RunnerError("verify/compare accepts a direct child of the data root runs directory")
    try:
        return resolve_contained_result_path(
            data_root, f"runs/{supplied.name}", "run root", must_exist=True
        )
    except ContractValidationError as error:
        raise RunnerError(str(error)) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eval.harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-corpus")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--data-root", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--method", choices=BACKEND_IDS, required=True)
    preflight.add_argument("--config", type=Path, default=CONFIG_PATH)

    run = subparsers.add_parser("run")
    run.add_argument("--method", choices=BACKEND_IDS, required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--data-root", required=True)
    run.add_argument("--results-root", type=Path, required=True)

    verify = subparsers.add_parser("verify-run")
    verify.add_argument("--run-root", required=True)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--baseline-run", type=Path, required=True)
    compare.add_argument("--candidate-run", type=Path, action="append", required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "validate-corpus":
            result = _validate_corpus_command(arguments.manifest, arguments.data_root)
        elif arguments.command == "preflight":
            result = preflight_method(
                arguments.method,
                resolve_data_root(),
                arguments.config,
            )
        elif arguments.command == "run":
            result = run_evaluation(
                method=arguments.method,
                run_id=arguments.run_id,
                data_root_value=arguments.data_root,
                results_root=arguments.results_root,
                manifest_path=MANIFEST_PATH,
                downstream_path=CONFIG_PATH,
            )
        elif arguments.command == "verify-run":
            root = _run_root_under_data_root(arguments.run_root)
            result = verify_run(root)
            result = {
                "status": "ok",
                "run_id": result["run"]["run_id"],
                "method": result["run"]["method"]["backend"],
                "arm_status": result["report"]["status"],
                "counts": result["run"]["counts"],
            }
        elif arguments.command == "compare":
            baseline = _run_root_under_data_root(str(arguments.baseline_run))
            candidates = [
                _run_root_under_data_root(str(candidate))
                for candidate in arguments.candidate_run
            ]
            result = compare_runs(
                baseline_root=baseline,
                candidate_roots=candidates,
                output=arguments.output,
            )
        else:
            raise AssertionError(f"unhandled command {arguments.command}")
    except (OSError, ValueError, RuntimeError, AssertionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

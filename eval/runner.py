"""Fresh-process evaluation runner for the frozen MediBytes denoising pilot."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import importlib
from importlib import metadata as importlib_metadata
import json
import math
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from eval.contracts import (
    ContractValidationError,
    assert_no_absolute_result_paths,
    atomic_write_json,
    file_sha256,
    load_json,
    sha256_bytes,
    validate_contract,
    write_json_exclusive,
    write_text_exclusive,
    loads_strict,
    resolve_contained_result_path,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = REPO_ROOT / "eval"
CONFIG_ROOT = EVAL_ROOT / "config"
METHOD_CONFIG_ROOT = CONFIG_ROOT / "methods"
DOWNSTREAM_CONFIG_PATH = CONFIG_ROOT / "downstream-v1.json"
CATALOG_PATH = CONFIG_ROOT / "method-catalog.json"
MANIFEST_PATH = EVAL_ROOT / "corpus" / "manifest.jsonl"
DEFAULT_DATA_ROOT = Path("/home/sandy/.local/share/medibytes-eval")
TARGET_SAMPLE_RATE = 16_000

BACKEND_IDS = (
    "none",
    "noisereduce",
    "rnnoise",
    "deepfilternet3",
    "dpdfnet2-onnx",
    "sherpa-gtcrn-simple",
    "percepnet",
    "nemo-se-den-sb-16k",
    "speecht5",
    "speex-denoise",
    "speechbrain-metricgan-plus",
    "voicefixer-mode0",
    "cmsis-dsp",
    "spectral-subtraction",
)
BLOCKED_DISPOSITIONS = {
    "blocked_prerequisite",
    "not_comparable",
    "blocked_target",
}
EXPECTED_CATALOG_ROWS = {
    2: ("rnnoise", "v0.2-0b50c45"),
    3: ("deepfilternet3", "v0.5.6-dfn3"),
    4: ("dpdfnet2-onnx", "v0.6.0"),
    5: ("sherpa-gtcrn-simple", "gtcrn-simple-1.13.8"),
    6: ("percepnet", "unofficial-8ffae433"),
    7: ("nemo-se-den-sb-16k", "se-den-sb-16k-small-2.7.3"),
    8: ("speecht5", "official-checkpoint-audit"),
    9: ("speex-denoise", "1.2.1-denoise-only"),
    10: ("speechbrain-metricgan-plus", "metricgan-plus-voicebank-1.1.1"),
    11: ("voicefixer-mode0", "0.1.3-mode0"),
    12: ("cmsis-dsp", "1.18.0"),
    13: ("spectral-subtraction", "pyroomacoustics-0.10.1-ss"),
}
EXPECTED_CATALOG_TARGET_CLASSES = {
    2: "realtime_laptop",
    3: "realtime_laptop",
    4: "realtime_laptop",
    5: "realtime_laptop",
    6: "realtime_laptop",
    7: "research_gpu",
    8: "research_gpu",
    9: "realtime_laptop",
    10: "offline_laptop",
    11: "offline_laptop",
    12: "target_only",
    13: "realtime_laptop",
}
ALLOWED_TARGET_CLASSES = {
    "realtime_laptop",
    "offline_laptop",
    "research_gpu",
    "target_only",
}
FORBIDDEN_COMPOSITION_KEYS = {
    "fallback",
    "fallbacks",
    "fallback_used",
    "cascade",
    "cascades",
    "candidates",
    "layer",
    "layers",
    "layered",
    "sequential",
}
STATIC_DEMO_DIRECTORIES = (
    "cleaned",
    "transcripts",
    "entities",
    "exports",
    "_state",
)
IMPLEMENTATION_FILES = (
    EVAL_ROOT / "harness.py",
    EVAL_ROOT / "runner.py",
    EVAL_ROOT / "reporting.py",
    EVAL_ROOT / "metrics.py",
    EVAL_ROOT / "contracts" / "__init__.py",
    EVAL_ROOT / "contracts" / "run.schema.json",
    EVAL_ROOT / "contracts" / "condition.schema.json",
    EVAL_ROOT / "contracts" / "report.schema.json",
    EVAL_ROOT / "contracts" / "corpus-record.schema.json",
)
_WORKER_INVOKER: Callable[..., dict[str, Any]] | None = None


class RunnerError(RuntimeError):
    """An evaluation request or terminal worker result is invalid."""


class BackendUnavailable(RunnerError):
    """The requested single backend cannot be executed in this environment."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sanitize_text(value: str, roots: Sequence[Path] = ()) -> str:
    sanitized = value
    for root in sorted(
        {path.resolve(strict=False) for path in roots},
        key=lambda path: len(str(path)),
        reverse=True,
    ):
        sanitized = sanitized.replace(str(root), "<evaluation-root>")
    sanitized = re.sub(
        r"(?<![A-Za-z0-9_])/(?:home|tmp|var|usr|opt)/[^\s'\"\)]+",
        "<absolute-path>",
        sanitized,
    )
    return sanitized


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RunnerError(f"{label} must be a JSON object")
    return value


def _require_text(mapping: Mapping[str, Any], key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RunnerError(f"{label}.{key} must be a nonempty string")
    return value


def _require_finite_number(mapping: Mapping[str, Any], key: str, label: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunnerError(f"{label}.{key} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise RunnerError(f"{label}.{key} must be finite")
    return number


def _reject_composition_keys(value: Any, label: str = "parameters") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in FORBIDDEN_COMPOSITION_KEYS:
                raise RunnerError(f"{label} contains forbidden composition key {key!r}")
            _reject_composition_keys(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_composition_keys(child, f"{label}[{index}]")


def resolve_data_root(value: str | os.PathLike[str] | None = None) -> Path:
    raw = os.environ.get("MEDIBYTES_EVAL_DATA_ROOT") if value is None else os.fspath(value)
    if raw is None or not raw.strip():
        raise RunnerError("MEDIBYTES_EVAL_DATA_ROOT must be a nonempty absolute path")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise RunnerError(f"evaluation data root must be absolute: {raw!r}")
    resolved = path.resolve(strict=False)
    if resolved == Path(resolved.anchor):
        raise RunnerError("evaluation data root must not be the filesystem root")
    return resolved


def validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id):
        raise RunnerError("run ID must be a safe 1..128 character identifier")
    if run_id in {".", ".."} or ".." in run_id:
        raise RunnerError("run ID must not contain parent traversal")
    return run_id


def _relative_data_path(data_root: Path, raw: str, label: str) -> Path:
    try:
        return resolve_contained_result_path(data_root, raw, label)
    except ContractValidationError as error:
        raise RunnerError(f"{label} must be a symlink-free data-root-relative POSIX path") from error


def validate_method_catalog(catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    if catalog.get("schema_version") != "1.0.0" or catalog.get("catalog_id") != "denoise-pilot-12-v1":
        raise RunnerError("method catalog identity is not denoise-pilot-12-v1")
    records_value = catalog.get("records")
    if not isinstance(records_value, list) or len(records_value) != 12:
        raise RunnerError("method catalog must contain exactly 12 records")
    records: list[dict[str, Any]] = []
    seen_rows: set[int] = set()
    seen_backends: set[str] = set()
    required = {
        "csv_row",
        "method_name",
        "backend",
        "variant_id",
        "upstream_source",
        "upstream_revision",
        "license",
        "model_path",
        "model_sha256",
        "alternate_digests",
        "target_class",
        "prerequisites",
        "default_disposition",
    }
    for index, value in enumerate(records_value):
        record = dict(_require_mapping(value, f"catalog records[{index}]"))
        missing = required - record.keys()
        if missing:
            raise RunnerError(f"catalog row {index + 1} is missing {sorted(missing)}")
        row = record.get("csv_row")
        if not isinstance(row, int) or row not in EXPECTED_CATALOG_ROWS or row in seen_rows:
            raise RunnerError("catalog CSV rows must be unique and exactly 2..13")
        expected_backend, expected_variant = EXPECTED_CATALOG_ROWS[row]
        if record.get("backend") != expected_backend or record.get("variant_id") != expected_variant:
            raise RunnerError(f"catalog row {row} does not match the approved execution identity")
        backend = str(record["backend"])
        if backend in seen_backends or backend in {"none", "noisereduce"}:
            raise RunnerError("catalog backend IDs must be unique and exclude controls")
        seen_rows.add(row)
        seen_backends.add(backend)
        if record.get("target_class") != EXPECTED_CATALOG_TARGET_CLASSES[row]:
            raise RunnerError(f"catalog row {row} has an invalid target class")
        if not isinstance(record.get("prerequisites"), list) or not record["prerequisites"]:
            raise RunnerError(f"catalog row {row} must declare prerequisites")
        if not isinstance(record.get("alternate_digests"), Mapping):
            raise RunnerError(f"catalog row {row} alternate_digests must be an object")
        digest = record.get("model_sha256")
        if digest is not None and (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise RunnerError(f"catalog row {row} model_sha256 must be null or a SHA-256")
        disposition = record.get("default_disposition")
        if disposition not in {"run_required", *BLOCKED_DISPOSITIONS}:
            raise RunnerError(f"catalog row {row} has an invalid default disposition")
        if row == 6 and disposition != "blocked_prerequisite":
            raise RunnerError("PercepNet must remain blocked_prerequisite")
        if row == 8 and disposition != "not_comparable":
            raise RunnerError("SpeechT5 must remain not_comparable")
        if row == 12 and disposition != "blocked_target":
            raise RunnerError("CMSIS-DSP must remain blocked_target")
        _reject_composition_keys(record.get("parameters", {}), f"catalog row {row}")
        records.append(record)
    if seen_rows != set(range(2, 14)):
        raise RunnerError("method catalog does not cover every authoritative CSV row")
    return records


def load_method_catalog() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = _repository_file(CATALOG_PATH, "method catalog")
    value = _require_mapping(load_json(path), "method catalog")
    records = validate_method_catalog(value)
    return dict(value), records


def validate_downstream_config(config: Mapping[str, Any]) -> dict[str, Any]:
    expected_stt = {
        "provider": "faster-whisper",
        "model": "base-int8",
        "actual_model": "base",
        "device": "cpu",
        "compute_type": "int8",
        "beam_size": 1,
        "word_timestamps": True,
        "temperature": 0,
    }
    if config.get("schema_version") != "1.0.0" or config.get("config_id") != "downstream-v1":
        raise RunnerError("downstream config identity must be downstream-v1")
    if config.get("strict") is not True:
        raise RunnerError("downstream strict mode must be true")
    if _require_mapping(config.get("stt"), "downstream.stt") != expected_stt:
        raise RunnerError("downstream STT settings differ from the frozen contract")
    extraction = _require_mapping(config.get("extraction"), "downstream.extraction")
    if extraction != {
        "mode": "regex",
        "use_llm": False,
        "allow_corrected_artifacts": False,
    }:
        raise RunnerError("downstream extraction must be regex-only with no LLM")
    cache = _require_mapping(config.get("cache"), "downstream.cache")
    if cache != {"output_cache_enabled": False, "idempotency_reuse_enabled": False}:
        raise RunnerError("downstream output cache and idempotency reuse must be disabled")
    source_files = config.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise RunnerError("downstream source_files must be a nonempty list")
    for value in source_files:
        relative = _require_text({"path": value}, "path", "downstream.source_files")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            raise RunnerError("downstream source files must be safe repository-relative paths")
    return dict(config)



def _repository_file(path: Path, label: str) -> Path:
    candidate = path.expanduser() if path.is_absolute() else REPO_ROOT / path
    if ".." in candidate.parts:
        raise RunnerError(f"{label} must not contain parent traversal")
    try:
        relative = candidate.relative_to(REPO_ROOT).as_posix()
        return resolve_contained_result_path(
            REPO_ROOT, relative, label, must_exist=True
        )
    except (ContractValidationError, ValueError) as error:
        raise RunnerError(f"{label} must be a symlink-free repository file") from error

def load_downstream_config(path: Path = DOWNSTREAM_CONFIG_PATH) -> tuple[dict[str, Any], Path]:
    resolved = _repository_file(path, "downstream config")
    value = _require_mapping(load_json(resolved), "downstream config")
    return validate_downstream_config(value), resolved


def _combined_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(REPO_ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256(resolved)))
    return digest.hexdigest()


def implementation_sha256() -> str:
    return _combined_sha256(IMPLEMENTATION_FILES)


def downstream_code_sha256(config: Mapping[str, Any]) -> str:
    paths = [REPO_ROOT / value for value in config["source_files"]]
    return _combined_sha256(paths)

def enhancement_code_sha256() -> str:
    return _combined_sha256((REPO_ROOT / "demo/audio_clean.py", REPO_ROOT / "demo/denoise.py"))

def adapter_source_sha256(method: str) -> str | None:
    path = REPO_ROOT / "demo" / "denoise_backends" / f"{method.replace('-', '_')}.py"
    if path.is_symlink():
        raise RunnerError("method adapter source must not be a symlink")
    if not path.is_file():
        return None
    return file_sha256(_repository_file(path, f"{method} adapter source"))



def _method_identity(method: str, config: Mapping[str, Any]) -> dict[str, Any]:
    if method in {"none", "noisereduce"}:
        disposition = "baseline" if method == "none" else "control"
        return {
            "backend": method,
            "variant_id": str(config["variant_id"]),
            "catalog_row": None,
            "default_disposition": disposition,
            "disposition_reason": None,
            "source_revision": str(config["source_revision"]),
            "source": str(config["source"]),
            "license": str(config["license"]),
            "target_class": str(config["target_class"]),
            "model_path": config.get("model_path"),
            "model_sha256": config.get("model_sha256"),
            "adapter_sha256": adapter_source_sha256(method),
        }
    catalog, records = load_method_catalog()
    del catalog
    matches = [record for record in records if record["backend"] == method]
    if len(matches) != 1:
        raise RunnerError(f"method {method!r} is not one normalized catalog row")
    record = matches[0]
    expected_config = {
        "variant_id": record["variant_id"],
        "source": record["upstream_source"],
        "source_revision": record["upstream_revision"],
        "model_path": record["model_path"],
        "model_sha256": record["model_sha256"],
        "target_class": record["target_class"],
        "license": record["license"],
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise RunnerError(f"method config {key} differs from the normalized catalog")
    if record["default_disposition"] in BLOCKED_DISPOSITIONS and config.get("reason") != record.get("blocked_reason"):
        raise RunnerError(f"blocked reason for {method} differs from the normalized catalog")
    return {
        "backend": method,
        "variant_id": record["variant_id"],
        "catalog_row": record["csv_row"],
        "default_disposition": record["default_disposition"],
        "disposition_reason": record.get("blocked_reason"),
        "source_revision": record["upstream_revision"],
        "source": record["upstream_source"],
        "license": record["license"],
        "target_class": record["target_class"],
        "model_path": record["model_path"],
        "model_sha256": record["model_sha256"],
        "adapter_sha256": adapter_source_sha256(method),
    }


def load_method_spec(method: str, data_root: Path) -> dict[str, Any]:
    if method not in BACKEND_IDS:
        raise RunnerError(f"unknown backend ID: {method}")
    path = METHOD_CONFIG_ROOT / f"{method}.json"
    resolved = _repository_file(path, f"method config {method}")
    config = dict(_require_mapping(load_json(resolved), f"method config {method}"))
    if config.get("schema_version") != "1.0.0" or config.get("backend") != method:
        raise RunnerError(f"method config identity mismatch for {method}")
    if not isinstance(config.get("executable"), bool):
        raise RunnerError(f"method config {method} must declare executable")
    for key in ("variant_id", "source", "source_revision", "license", "target_class"):
        _require_text(config, key, f"method config {method}")
    model_sha256 = config.get("model_sha256")
    if model_sha256 is not None and (
        not isinstance(model_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", model_sha256)
    ):
        raise RunnerError(f"method config {method}.model_sha256 must be null or a SHA-256")
    if not isinstance(config.get("parameters"), Mapping):
        raise RunnerError(f"method config {method}.parameters must be an object")
    _reject_composition_keys(config["parameters"], f"method config {method}.parameters")
    timeout = _require_finite_number(config, "timeout_s", f"method config {method}")
    if timeout <= 0:
        raise RunnerError(f"method config {method}.timeout_s must be positive")
    model_path_raw = config.get("model_path")
    resolved_model: Path | None = None
    if model_path_raw is not None:
        if not isinstance(model_path_raw, str):
            raise RunnerError(f"method config {method}.model_path must be a string or null")
        resolved_model = _relative_data_path(data_root, model_path_raw, f"{method}.model_path")
    identity = _method_identity(method, config)
    if identity["default_disposition"] in BLOCKED_DISPOSITIONS and config["executable"]:
        raise RunnerError(f"blocked catalog row {method} cannot enable an executable adapter")
    return {
        "backend": method,
        "config": config,
        "config_path": resolved,
        "config_sha256": file_sha256(resolved),
        "model_path": resolved_model,
        "model_path_relative": model_path_raw,
        "identity": identity,
        "executable": bool(config["executable"]),
        "disposition": identity["default_disposition"],
        "reason": config.get("reason"),
    }


def _enhancement_config(method_spec: Mapping[str, Any]):
    from demo.denoise import BackendId, EnhancementConfig

    config = method_spec["config"]
    return EnhancementConfig(
        backend=BackendId(method_spec["backend"]),
        variant_id=str(config["variant_id"]),
        model_path=method_spec["model_path"],
        model_sha256=config.get("model_sha256"),
        parameters=dict(config["parameters"]),
        timeout_s=float(config["timeout_s"]),
    )


def _backend_probe(method_spec: Mapping[str, Any]):
    module_name = "demo.denoise_backends." + method_spec["backend"].replace("-", "_")
    try:
        module = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as error:
        raise BackendUnavailable(
            f"BACKEND_UNAVAILABLE: cannot import {module_name}: {type(error).__name__}"
        ) from error
    probe_function = getattr(module, "probe", None)
    if not callable(probe_function):
        raise BackendUnavailable(f"BACKEND_UNAVAILABLE: {module_name}.probe is unavailable")
    return probe_function(_enhancement_config(method_spec))


def preflight_method(
    method: str, data_root_value: str | os.PathLike[str], config_path: Path = DOWNSTREAM_CONFIG_PATH
) -> dict[str, Any]:
    """Probe and execute one real deterministic adapter without writing artifacts."""

    data_root = resolve_data_root(data_root_value)
    downstream, resolved_config = load_downstream_config(config_path)
    del downstream
    method_spec = load_method_spec(method, data_root)
    identity = method_spec["identity"]
    if not method_spec["executable"]:
        return {
            "status": method_spec["disposition"],
            "reason": method_spec.get("reason"),
            "method": identity,
            "preflight_config_sha256": file_sha256(resolved_config),
        }
    probe = _backend_probe(method_spec)
    probe_data = dataclasses.asdict(probe) if dataclasses.is_dataclass(probe) else dict(probe)
    if probe_data.get("available") is not True:
        raise BackendUnavailable(
            f"BACKEND_UNAVAILABLE: {method}: {probe_data.get('reason') or 'probe unavailable'}"
        )
    from demo.denoise import run_backend

    time_axis = np.arange(TARGET_SAMPLE_RATE, dtype=np.float64) / TARGET_SAMPLE_RATE
    smoke_audio = (0.1 * np.sin(2.0 * np.pi * 220.0 * time_axis)).astype(np.float32)
    output = run_backend(smoke_audio, TARGET_SAMPLE_RATE, _enhancement_config(method_spec))
    if output.actual_backend != method:
        raise BackendUnavailable(f"backend identity mismatch during preflight: {output.actual_backend}")
    return {
        "status": "ok",
        "method": identity,
        "probe": probe_data,
        "smoke_output_samples": int(np.asarray(output.audio).size),
        "preflight_config_sha256": file_sha256(resolved_config),
    }


def validate_real_transcript(
    transcript: Mapping[str, Any],
    entities: Mapping[str, Any],
    downstream: Mapping[str, Any],
    *,
    expected_job_id: str | None = None,
) -> None:
    """Fail closed on every mock, fallback, cache, corrected, or LLM artifact."""

    required_transcript = {
        "job_id",
        "text",
        "language",
        "segments",
        "normalized_en",
        "normalizations",
        "stt_engine",
        "stt_provenance",
    }
    if set(transcript) != required_transcript:
        raise RunnerError("strict downstream transcript fields differ from the frozen contract")
    provenance = _require_mapping(transcript.get("stt_provenance"), "stt_provenance")
    required_provenance = {
        "provider",
        "requested_model",
        "actual_model",
        "device",
        "compute_type",
        "beam_size",
        "runtime_version",
        "model_hash",
        "model_snapshot",
        "is_mock",
        "job_id",
        "word_timestamps",
        "temperature",
    }
    if set(provenance) != required_provenance:
        raise RunnerError("stt_provenance fields differ from the frozen downstream contract")
    if type(provenance.get("word_timestamps")) is not bool:
        raise RunnerError("STT word_timestamps must be a boolean")
    temperature = provenance.get("temperature")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
    ):
        raise RunnerError("STT temperature must be a finite number")
    if type(provenance.get("beam_size")) is not int:
        raise RunnerError("STT beam_size must be an integer")
    stt = downstream["stt"]
    expected = {
        "provider": stt["provider"],
        "requested_model": stt["model"],
        "actual_model": stt["actual_model"],
        "device": stt["device"],
        "compute_type": stt["compute_type"],
        "beam_size": stt["beam_size"],
        "word_timestamps": stt["word_timestamps"],
        "temperature": float(stt["temperature"]),
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise RunnerError(f"STT provenance mismatch for {key}")
    if provenance.get("is_mock") is not False:
        raise RunnerError("mock STT is forbidden in evaluation")
    job_id = transcript.get("job_id")
    if not isinstance(job_id, str) or not job_id.strip():
        raise RunnerError("transcript job_id must be a nonempty string")
    if expected_job_id is not None and job_id != expected_job_id:
        raise RunnerError("transcript job_id differs from the evaluated condition")
    if provenance.get("job_id") != job_id:
        raise RunnerError("STT provenance job_id differs from transcript job_id")
    if not isinstance(provenance.get("runtime_version"), str) or not provenance["runtime_version"]:
        raise RunnerError("strict STT runtime_version must be recorded")
    expected_engine = (
        f"faster-whisper:{stt['actual_model']}-{stt['device']}-{stt['compute_type']}"
    )
    if transcript.get("stt_engine") != expected_engine:
        raise RunnerError("strict STT engine identity differs from the frozen downstream contract")
    if not isinstance(transcript.get("language"), str) or not transcript["language"]:
        raise RunnerError("strict transcript language must be a nonempty string")
    if not isinstance(transcript.get("segments"), list) or not isinstance(
        transcript.get("normalizations"), list
    ):
        raise RunnerError("strict transcript segments and normalizations must be arrays")
    text = transcript.get("text")
    normalized_text = transcript.get("normalized_en")
    if not isinstance(text, str) or not isinstance(normalized_text, str):
        raise RunnerError("strict transcript text and normalized_en must be strings")
    from demo.stt_extract import normalize_text

    expected_normalization = normalize_text(text)
    if normalized_text != expected_normalization["normalized_en"]:
        raise RunnerError("strict transcript normalized_en differs from the frozen normalizer")
    if transcript["normalizations"] != expected_normalization["normalizations"]:
        raise RunnerError("strict transcript normalizations differ from the frozen normalizer")

    forbidden_keys = {
        "cache_hit",
        "cached",
        "cached_artifact",
        "corrected",
        "human_corrected",
        "fallback",
        "from_cache",
        "llm_engine",
        "tidy_engine",
        "tidy_dropped",
    }

    def inspect(value: Any, label: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key).casefold() in forbidden_keys:
                    raise RunnerError(f"forbidden downstream artifact key at {label}.{key}")
                inspect(child, f"{label}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                inspect(child, f"{label}[{index}]")

    inspect(transcript, "transcript")
    inspect(entities, "entities")
    expected_entity_keys = {"job_id", "drugs", "symptoms", "vitals", "allergies", "negations", "diagnosis", "followup"}
    if set(entities) != expected_entity_keys:
        raise RunnerError("entity extraction groups differ from the strict regex contract")
    if entities.get("job_id") != transcript.get("job_id"):
        raise RunnerError("entity job_id differs from transcript job_id")
    for group in ("drugs", "symptoms", "vitals", "allergies", "negations", "diagnosis", "followup"):
        value = entities.get(group)
        if not isinstance(value, (list, dict)):
            raise RunnerError(f"entity group {group} must be a list or object")
        values = value if isinstance(value, list) else ([] if not value else [value])
        for index, entity in enumerate(values):
            if not isinstance(entity, Mapping):
                raise RunnerError(f"entity {group}[{index}] must be an object")
            for key in ("start_char", "end_char", "span_status"):
                if key not in entity:
                    raise RunnerError(f"entity {group}[{index}] is missing {key}")
            start = entity["start_char"]
            end = entity["end_char"]
            span_status = entity["span_status"]
            if isinstance(start, bool) or not isinstance(start, int) or isinstance(end, bool) or not isinstance(end, int):
                raise RunnerError(f"entity {group}[{index}] character offsets must be integers")
            if span_status not in {"exact", "ambiguous", "not_found"}:
                raise RunnerError(f"entity {group}[{index}] has invalid span_status")
            if span_status == "not_found":
                if start != -1 or end != -1:
                    raise RunnerError(f"entity {group}[{index}] not_found offsets must be -1/-1")
            elif start < 0 or end <= start:
                raise RunnerError(f"entity {group}[{index}] located offsets are invalid")


def validate_enhancement_metadata(
    metadata: Mapping[str, Any],
    *,
    method: str,
    variant_id: str,
    output_path: Path,
    input_samples: int,
    expected_input_sha256: str | None = None,
    expected_model_sha256: str | None = None,
    expected_source_revision: str | None = None,
    expected_config_sha256: str | None = None,
    expected_parameters: Mapping[str, Any] | None = None,
) -> None:
    required = {
        "input",
        "sample_rate",
        "channels",
        "input_sample_count",
        "output_sample_count",
        "sample_count_before",
        "sample_count_after",
        "sample_count_drift",
        "input_duration_s",
        "output_duration_s",
        "duration_drift_s",
        "vad_ratio",
        "rms_dbfs_before",
        "rms_dbfs_after",
        "requested_backend",
        "actual_backend",
        "variant_id",
        "model_sha256",
        "runtime_version",
        "source_revision",
        "native_sample_rate",
        "delay_samples",
        "fallback_used",
        "input_sha256",
        "output_sha256",
        "processing_time_s",
        "preprocess_gain_linear",
        "preprocess_rms_gain_linear",
        "preprocess_peak_scale",
        "preprocess_total_gain_linear",
        "preprocess_peak_before",
        "preprocess_peak_after",
        "output_peak",
        "output_clipping_count",
        "enhancement_metadata",
        "provenance",
        "preprocess_reference_sha256",
        "none_reference_sha256",
    }
    missing = required - metadata.keys()
    if missing:
        raise RunnerError(f"enhancement metadata is missing {sorted(missing)}")
    input_info = _require_mapping(metadata.get("input"), "enhancement input")
    size = input_info.get("size")
    source_duration = input_info.get("duration")
    if type(size) is not int or size <= 0:
        raise RunnerError("enhancement input size must be a positive integer")
    if (
        isinstance(source_duration, bool)
        or not isinstance(source_duration, (int, float))
        or not math.isfinite(float(source_duration))
        or float(source_duration) <= 0
    ):
        raise RunnerError("enhancement input duration must be measured and finite")
    if metadata.get("requested_backend") != method or metadata.get("actual_backend") != method:
        raise RunnerError("enhancement requested/actual backend identity mismatch")
    if metadata.get("variant_id") != variant_id:
        raise RunnerError("enhancement variant identity mismatch")
    if metadata.get("fallback_used") is not False:
        raise RunnerError("enhancement fallback is forbidden")
    if metadata.get("sample_rate") != TARGET_SAMPLE_RATE or metadata.get("channels") != 1:
        raise RunnerError("enhanced output must be 16 kHz mono")
    native_rate = metadata.get("native_sample_rate")
    delay_samples = metadata.get("delay_samples")
    if type(native_rate) is not int or native_rate <= 0:
        raise RunnerError("enhancement native_sample_rate must be a positive integer")
    if type(delay_samples) is not int or delay_samples < 0:
        raise RunnerError("enhancement delay_samples must be a non-negative integer")
    if metadata.get("input_sample_count") != input_samples or metadata.get("output_sample_count") != input_samples:
        raise RunnerError("enhanced output sample count must exactly match the common input")
    if metadata.get("sample_count_before") != input_samples or metadata.get("sample_count_after") != input_samples:
        raise RunnerError("enhancement before/after sample counts must exactly match the common input")
    if metadata.get("sample_count_drift") != 0:
        raise RunnerError("enhanced output sample-count drift must be exactly compensated")
    input_duration = _require_finite_number(metadata, "input_duration_s", "enhancement metadata")
    output_duration = _require_finite_number(metadata, "output_duration_s", "enhancement metadata")
    duration_drift = _require_finite_number(metadata, "duration_drift_s", "enhancement metadata")
    if input_duration <= 0 or output_duration <= 0 or abs(duration_drift) > 1e-12:
        raise RunnerError("enhanced output duration drift must be exactly compensated")
    if not math.isclose(output_duration, input_duration, rel_tol=0.0, abs_tol=1e-12):
        raise RunnerError("enhancement output duration differs from its input duration")
    if metadata.get("output_sha256") != file_sha256(output_path):
        raise RunnerError("enhancement output hash does not match enhanced.wav")
    if "snr_before_db" in metadata or "snr_after_db" in metadata:
        raise RunnerError("nominal SNR metadata is forbidden")
    for key in ("rms_dbfs_before", "rms_dbfs_after"):
        value = metadata.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise RunnerError(f"enhancement metadata {key} must be measured and finite")
    vad_ratio = _require_finite_number(metadata, "vad_ratio", "enhancement metadata")
    output_peak = _require_finite_number(metadata, "output_peak", "enhancement metadata")
    clipping_count = metadata.get("output_clipping_count")
    if not 0.0 <= vad_ratio <= 1.0:
        raise RunnerError("enhancement VAD ratio must be between zero and one")
    if not 0.0 < output_peak <= 1.0 or type(clipping_count) is not int or not 0 <= clipping_count <= input_samples:
        raise RunnerError("enhancement peak/clipping observations are invalid")
    if _require_finite_number(metadata, "processing_time_s", "enhancement metadata") < 0:
        raise RunnerError("enhancement processing time must not be negative")
    if not isinstance(metadata.get("runtime_version"), str) or not metadata["runtime_version"]:
        raise RunnerError("enhancement runtime_version must be recorded")
    if not isinstance(metadata.get("source_revision"), str) or not metadata["source_revision"]:
        raise RunnerError("enhancement source_revision must be recorded")
    gain = _require_finite_number(metadata, "preprocess_gain_linear", "enhancement metadata")
    total_gain = _require_finite_number(metadata, "preprocess_total_gain_linear", "enhancement metadata")
    peak_scale = _require_finite_number(metadata, "preprocess_peak_scale", "enhancement metadata")
    if gain <= 0 or not 0 < peak_scale <= 1:
        raise RunnerError("preprocessing gain/peak scale must be positive and bounded")
    if not math.isclose(total_gain, gain * peak_scale, rel_tol=1e-7, abs_tol=1e-9):
        raise RunnerError("preprocess_total_gain_linear must equal RMS gain times peak scale")
    duplicate_rms_gain = _require_finite_number(
        metadata, "preprocess_rms_gain_linear", "enhancement metadata"
    )
    if not math.isclose(duplicate_rms_gain, gain, rel_tol=0.0, abs_tol=1e-9):
        raise RunnerError("preprocess RMS gain observations must agree")
    for key in ("preprocess_peak_before", "preprocess_peak_after"):
        if _require_finite_number(metadata, key, "enhancement metadata") <= 0:
            raise RunnerError(f"enhancement metadata {key} must be positive")
    if expected_input_sha256 is not None and metadata["input_sha256"] != expected_input_sha256:
        raise RunnerError("enhancement input hash differs from the persisted condition")
    model_sha256 = metadata.get("model_sha256")
    model_free = method in {"none", "noisereduce", "speex-denoise", "spectral-subtraction"}
    if model_free and model_sha256 is not None:
        raise RunnerError("model-free enhancement must report model_sha256=null")
    if not model_free and (
        not isinstance(model_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", model_sha256)
    ):
        raise RunnerError("model-backed enhancement must report a SHA-256 model")
    if expected_model_sha256 is not None and model_sha256 != expected_model_sha256:
        raise RunnerError("enhancement model hash differs from the frozen method config")
    provenance = _require_mapping(metadata.get("provenance"), "enhancement provenance")
    required_provenance = {
        "actual_backend",
        "backend",
        "config_sha256",
        "device",
        "fallback_used",
        "model_sha256",
        "parameters",
        "provider",
        "requested_backend",
        "runtime_version",
        "source_revision",
        "variant_id",
    }
    if not required_provenance.issubset(provenance):
        raise RunnerError("enhancement provenance is incomplete")
    if (
        provenance["backend"] != method
        or provenance["actual_backend"] != method
        or provenance["requested_backend"] != method
    ):
        raise RunnerError("enhancement provenance backend mismatch")
    if provenance["variant_id"] != variant_id:
        raise RunnerError("enhancement provenance variant mismatch")
    if provenance["fallback_used"] is not False:
        raise RunnerError("enhancement provenance fallback must be boolean false")
    for key in ("provider", "device", "runtime_version", "source_revision"):
        if not isinstance(provenance[key], str) or not provenance[key]:
            raise RunnerError(f"enhancement provenance {key} must be a nonempty string")
    if not isinstance(provenance["parameters"], Mapping):
        raise RunnerError("enhancement provenance parameters must be an object")
    if expected_parameters is not None and dict(provenance["parameters"]) != dict(expected_parameters):
        raise RunnerError("enhancement provenance parameters differ from the frozen config")
    if not isinstance(provenance["config_sha256"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", provenance["config_sha256"]
    ):
        raise RunnerError("enhancement provenance config_sha256 must be a SHA-256")
    if expected_config_sha256 is not None and provenance["config_sha256"] != expected_config_sha256:
        raise RunnerError("enhancement provenance config hash differs from the frozen method config")
    if provenance["runtime_version"] != metadata["runtime_version"]:
        raise RunnerError("enhancement runtime version differs from adapter provenance")
    if provenance["model_sha256"] != model_sha256:
        raise RunnerError("enhancement model hash differs from adapter provenance")
    provenance_revision = provenance["source_revision"]
    if provenance_revision != metadata["source_revision"]:
        raise RunnerError("enhancement source revision differs from adapter provenance")
    if expected_source_revision is not None and provenance_revision != expected_source_revision:
        raise RunnerError("enhancement source revision differs from the frozen method config")
    if not isinstance(metadata.get("enhancement_metadata"), Mapping):
        raise RunnerError("enhancement metadata detail must be an object")
    _reject_composition_keys(metadata["enhancement_metadata"], "enhancement_metadata")
    for key in ("preprocess_reference_sha256", "none_reference_sha256"):
        value = metadata.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise RunnerError(f"enhancement metadata {key} must be a SHA-256")
    if method == "none" and metadata["none_reference_sha256"] != metadata["output_sha256"]:
        raise RunnerError("none backend reference hash must identify its saved PCM16 output")


def _nvml_peak_vram_bytes() -> int | str:
    try:
        import pynvml
    except (ImportError, ModuleNotFoundError):
        return "unavailable"
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        values = [int(process.usedGpuMemory) for process in processes if process.pid == os.getpid()]
        return max(values) if values else "unavailable"
    except Exception:
        return "unavailable"
    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()


def _resource_record(start_wall: float, stage1_wall: float, metadata: Mapping[str, Any]) -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    duration = float(metadata.get("processing_time_s", stage1_wall))
    model_load = metadata.get("model_load_s")
    if isinstance(model_load, bool) or not isinstance(model_load, (int, float)) or not math.isfinite(model_load):
        model_load_value: float | None = None
    else:
        model_load_value = float(model_load)
    audio_duration = float(metadata["output_sample_count"]) / TARGET_SAMPLE_RATE
    return {
        "stage1_wall_s": stage1_wall,
        "cold_total_s": max(0.0, time.perf_counter() - start_wall),
        "model_load_s": model_load_value,
        "inference_s": duration,
        "child_cpu_time_s": float(usage.ru_utime + usage.ru_stime),
        "peak_rss_bytes": int(usage.ru_maxrss) * 1024,
        "peak_vram_bytes": _nvml_peak_vram_bytes(),
        "audio_duration_s": audio_duration,
        "real_time_factor": duration / audio_duration if audio_duration > 0 else None,
    }


def _error_record(error: BaseException, roots: Sequence[Path]) -> dict[str, str]:
    return {
        "type": type(error).__name__,
        "message": _sanitize_text(str(error) or type(error).__name__, roots),
        "traceback": _sanitize_text("".join(traceback.format_exception(error)), roots),
    }


def _worker_payload(spec: Mapping[str, Any]) -> dict[str, Any]:
    from demo.audio_clean import clean_audio
    from demo.denoise import BackendId, EnhancementConfig
    from demo.stt_extract import run_stt_extract

    method = str(spec["method"])
    variant_id = str(spec["variant_id"])
    input_path = Path(spec["input_path"])
    output_path = Path(spec["output_path"])
    config = _require_mapping(spec["enhancement_config"], "enhancement_config")
    model_path_raw = config.get("model_path")
    enhancement_config = EnhancementConfig(
        backend=BackendId(method),
        variant_id=variant_id,
        model_path=Path(model_path_raw) if isinstance(model_path_raw, str) else None,
        model_sha256=config.get("model_sha256"),
        parameters=dict(config.get("parameters", {})),
        timeout_s=float(config.get("timeout_s", 30.0)),
    )
    start_wall = time.perf_counter()
    stage1_start = time.perf_counter()
    try:
        metadata = clean_audio(
            input_path,
            output_path,
            backend=BackendId(method),
            config=enhancement_config,
        )
        if not isinstance(metadata, Mapping):
            raise RunnerError("clean_audio must return an enhancement metadata object")
        metadata = dict(metadata)
        provenance = dict(_require_mapping(metadata.get("provenance"), "enhancement provenance"))
        config_sha256 = config.get("config_sha256")
        if not isinstance(config_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", config_sha256):
            raise RunnerError("worker enhancement config_sha256 must be a SHA-256")
        provenance["config_sha256"] = config_sha256
        metadata["provenance"] = provenance
        input_samples = int(spec["input_samples"])
        validate_enhancement_metadata(
            metadata,
            method=method,
            variant_id=variant_id,
            output_path=output_path,
            input_samples=input_samples,
            expected_input_sha256=file_sha256(input_path),
            expected_model_sha256=config.get("model_sha256"),
            expected_source_revision=config.get("source_revision"),
            expected_config_sha256=config_sha256,
            expected_parameters=_require_mapping(config.get("parameters"), "enhancement parameters"),
        )
        if metadata.get("input_sha256") != file_sha256(input_path):
            raise RunnerError("enhancement input hash does not match the condition WAV")
        from eval.metrics import read_pcm16_mono

        output_samples = read_pcm16_mono(output_path)
        if output_samples.size != input_samples or not np.any(output_samples):
            raise RunnerError("enhanced PCM16 output is empty, all-zero, or has incorrect length")
        stage1_wall = time.perf_counter() - stage1_start
        payload: dict[str, Any] = {
            "status": "ok",
            "phase": spec["phase"],
            "enhanced_path": str(output_path),
            "enhancement": metadata,
            "resources": _resource_record(start_wall, stage1_wall, metadata),
            "downstream_status": "not_run",
        }
        if spec.get("downstream"):
            try:
                transcript, entities = run_stt_extract(
                    str(output_path),
                    job_id=str(spec["condition_id"]),
                    use_llm=False,
                    model=str(spec["downstream"]["stt"]["model"]),
                    strict=True,
                )
                if not isinstance(transcript, Mapping) or not isinstance(entities, Mapping):
                    raise RunnerError("strict downstream must return transcript and entities objects")
                transcript = dict(transcript)
                entities = dict(entities)
                validate_real_transcript(
                    transcript,
                    entities,
                    spec["downstream"],
                    expected_job_id=str(spec["condition_id"]),
                )
            except BaseException as error:
                payload["status"] = "failed"
                payload["downstream_status"] = "failed"
                payload["error"] = _error_record(
                    error,
                    [Path(spec["result_path"]).parent, Path(spec.get("data_root", ""))],
                )
            else:
                payload["transcript"] = transcript
                payload["entities"] = entities
                payload["downstream_status"] = "ok"
        return payload
    except BaseException as error:
        roots = [Path(spec["result_path"]).parent, Path(spec.get("data_root", ""))]
        return {
            "status": "failed_runtime",
            "phase": spec["phase"],
            "enhanced_path": str(output_path) if output_path.is_file() else None,
            "downstream_status": "not_run",
            "error": _error_record(error, roots),
        }


def _worker_main(spec: Mapping[str, Any]) -> int:
    result_path = Path(spec["result_path"])
    try:
        result = _worker_payload(spec)
    except BaseException as error:
        result = {
            "status": "failed_runtime",
            "phase": spec.get("phase", "unknown"),
            "error": _error_record(error, [result_path.parent]),
        }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return 0


def _invoke_worker(spec: Mapping[str, Any], work_dir: Path, timeout_s: float) -> dict[str, Any]:
    result_path = work_dir / f"result-{spec['phase']}.json"
    child_spec = dict(spec)
    child_spec["result_path"] = str(result_path)
    command = [sys.executable, "-m", "eval.runner", "--worker"]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(REPO_ROOT), environment.get("PYTHONPATH", "")) if value
    )
    environment["MEDIBYTES_EVAL_NO_CACHE"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            input=json.dumps(child_spec, ensure_ascii=False, allow_nan=False),
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return {
            "status": "timeout",
            "phase": spec["phase"],
            "error": {
                "type": "TimeoutExpired",
                "message": f"fresh worker exceeded {timeout_s:.3f} seconds",
                "traceback": "",
            },
        }
    if result_path.is_file():
        try:
            value = load_json(result_path)
            if isinstance(value, dict):
                return value
        except ContractValidationError:
            pass
    stderr = _sanitize_text(completed.stderr or "", [REPO_ROOT, work_dir])
    stdout = _sanitize_text(completed.stdout or "", [REPO_ROOT, work_dir])
    diagnostic = (stderr or stdout or f"worker exited with code {completed.returncode}").strip()
    status = "oom" if completed.returncode in {-9, 137} or "MemoryError" in diagnostic else "failed_runtime"
    if completed.returncode < 0 and completed.returncode != -9:
        status = "failed_runtime"
    return {
        "status": status,
        "phase": spec["phase"],
        "error": {
            "type": "WorkerProcessFailure" if status == "oom" else "WorkerProcessExit",
            "message": diagnostic[-4000:],
            "traceback": "",
        },
    }


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _wav_properties(path: Path) -> dict[str, object]:
    from eval.metrics import read_pcm16_mono

    samples = read_pcm16_mono(path)
    return {
        "path": path,
        "sha256": file_sha256(path),
        "sample_count": int(samples.size),
        "sample_rate": TARGET_SAMPLE_RATE,
    }


def _condition_relative_paths(backend: str, variant_id: str, condition_id: str) -> dict[str, str]:
    base = f"arms/{backend}/{variant_id}/conditions/{condition_id}"
    return {name: f"{base}/{name}.json" for name in ("enhancement", "transcript", "entities", "fields", "resources", "metrics", "status")}


def _status_record(
    *,
    condition_id: str,
    record: Mapping[str, Any],
    noise_type: str,
    method: str,
    variant_id: str,
    status: str,
    stage: str,
    downstream_status: str,
    started_at: str,
    input_path: str,
    input_sha256: str,
    input_samples: int,
    enhanced: Mapping[str, Any] | None,
    repeat_hashes: Sequence[str],
    error: Mapping[str, Any] | None,
    relative_paths: Mapping[str, str],
) -> dict[str, Any]:
    condition_kind = "clean" if noise_type == "clean" else "noisy"
    resolved_noise_type = "clean" if condition_kind == "clean" else str(record["noisy"]["noise_type"])
    return {
        "schema_version": "1.0.0",
        "record_type": "condition",
        "condition_id": condition_id,
        "sample_id": record["sample_id"],
        "group_id": record["group_id"],
        "bucket": record["bucket"],
        "condition_kind": condition_kind,
        "noise_type": resolved_noise_type,
        "requested_backend": method,
        "actual_backend": enhanced.get("actual_backend") if enhanced else None,
        "variant_id": variant_id,
        "fallback_used": False,
        "status": status,
        "stage": stage,
        "downstream_status": downstream_status,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "input": {
            "path": input_path,
            "sha256": input_sha256,
            "sample_count": input_samples,
            "sample_rate": TARGET_SAMPLE_RATE,
        },
        "enhanced": enhanced,
        "repeat_hashes": list(repeat_hashes),
        "error": dict(error) if error else None,
        "artifacts": dict(relative_paths),
    }


def _failure_record(status: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "record_type": status,
        "status": status,
        "reason": reason,
    }


def _write_condition_failure(
    *,
    run_root: Path,
    paths: Mapping[str, str],
    result: Mapping[str, Any],
    existing_enhancement: Mapping[str, Any] | None = None,
    existing_transcript: Mapping[str, Any] | None = None,
    existing_entities: Mapping[str, Any] | None = None,
    text_metrics: Mapping[str, Any] | None = None,
) -> None:
    reason = "terminal failure retained"
    if isinstance(result.get("error"), Mapping):
        reason = str(result["error"].get("message") or reason)
    if existing_enhancement is None:
        enhancement = _failure_record("not_available", reason)
    else:
        enhancement = dict(existing_enhancement)
    atomic_write_json(run_root / paths["enhancement"], enhancement)
    transcript = dict(existing_transcript) if existing_transcript is not None else _failure_record("not_available", reason)
    entities = dict(existing_entities) if existing_entities is not None else _failure_record("not_available", reason)
    atomic_write_json(run_root / paths["transcript"], transcript)
    atomic_write_json(run_root / paths["entities"], entities)
    from eval.metrics import clinical_metrics

    clinical = clinical_metrics(entities if "drugs" in entities else {})
    fields = {
        "schema_version": "1.0.0",
        "record_type": "fields",
        "status": "not_available" if "drugs" not in entities else "observed",
        "reason": reason if "drugs" not in entities else None,
        "fields": {group: entities.get(group, []) for group in ("drugs", "symptoms", "vitals", "allergies", "diagnosis", "followup")} if "drugs" in entities else {},
        "clinical_gold": "empty_nonclinical",
    }
    atomic_write_json(run_root / paths["fields"], fields)
    resources = {
        "schema_version": "1.0.0",
        "record_type": "resources",
        "status": "incomplete_after_terminal_failure",
        "reason": reason,
        "retained_phases": [],
    }
    atomic_write_json(run_root / paths["resources"], resources)
    metrics = {
        "schema_version": "1.0.0",
        "record_type": "metrics",
        "status": "partial" if existing_enhancement is not None else "not_available",
        "reason": reason,
        "audio": {"status": "not_available", "reason": reason},
        "text": dict(text_metrics) if text_metrics is not None else {"status": "not_available", "reason": reason},
        "clinical": clinical,
    }
    atomic_write_json(run_root / paths["metrics"], metrics)


def _resource_summary(resources: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    measured = [value for value in resources if value.get("phase") in {"repeat-1", "repeat-2", "repeat-3"}]
    inference_ms = [float(value["inference_s"]) * 1000.0 for value in measured]
    rtf = [float(value["real_time_factor"]) for value in measured if value.get("real_time_factor") is not None]
    rss = [int(value["peak_rss_bytes"]) for value in resources]
    vram_values = [value.get("peak_vram_bytes") for value in resources if isinstance(value.get("peak_vram_bytes"), int)]
    percentiles = np.percentile(np.asarray(inference_ms), [50, 95, 99]) if inference_ms else [None, None, None]
    return {
        "schema_version": "1.0.0",
        "record_type": "resources",
        "status": "observed" if len(measured) == 3 else "incomplete",
        "cold": resources[0] if resources else None,
        "warmup": resources[1] if len(resources) > 1 else None,
        "measured": measured,
        "inference_ms_samples": inference_ms,
        "inference_p50_ms": float(percentiles[0]) if inference_ms else None,
        "inference_p95_ms": float(percentiles[1]) if inference_ms else None,
        "inference_p99_ms": float(percentiles[2]) if inference_ms else None,
        "realtime_factor_mean": float(np.mean(rtf)) if rtf else None,
        "peak_rss_bytes_max": max(rss) if rss else None,
        "peak_vram_bytes": max(vram_values) if vram_values else "unavailable",
    }


def _resource_with_phase(result: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(result.get("resources", {}))
    value["phase"] = result.get("phase")
    return value


def _worker_spec(
    *,
    method_spec: Mapping[str, Any],
    phase: str,
    condition_id: str,
    input_path: Path,
    output_path: Path,
    input_samples: int,
    downstream: Mapping[str, Any] | None,
    data_root: Path,
) -> dict[str, Any]:
    config = method_spec["config"]
    return {
        "method": method_spec["backend"],
        "variant_id": str(config["variant_id"]),
        "phase": phase,
        "condition_id": condition_id,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_samples": input_samples,
        "data_root": str(data_root),
        "downstream": dict(downstream) if downstream is not None else None,
        "enhancement_config": {
            "backend": method_spec["backend"],
            "variant_id": str(config["variant_id"]),
            "source_revision": str(config["source_revision"]),
            "config_sha256": str(method_spec["config_sha256"]),
            "model_path": str(method_spec["model_path"]) if method_spec["model_path"] else None,
            "model_sha256": config.get("model_sha256"),
            "parameters": dict(config["parameters"]),
            "timeout_s": float(config["timeout_s"]),
        },
    }


def _worker_timeout(config: Mapping[str, Any], downstream: bool) -> float:
    return max(60.0, float(config["timeout_s"]) * 1.25 + (300.0 if downstream else 30.0))


def _failure_status(result: Mapping[str, Any]) -> str:
    status = str(result.get("status", "failed_runtime"))
    return status if status in {"failed", "oom", "timeout", "rejected", "failed_runtime", *BLOCKED_DISPOSITIONS} else "failed_runtime"


def run_condition(
    record: Mapping[str, Any],
    noise_type: str,
    *,
    run_root: Path,
    data_root: Path,
    method_spec: Mapping[str, Any],
    downstream: Mapping[str, Any],
    invoke_worker: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one condition with reference, accuracy, warm-up, and three fresh repeats."""

    invoke = invoke_worker or _invoke_worker
    method = str(method_spec["backend"])
    variant_id = str(method_spec["config"]["variant_id"])
    condition_id = f"{record['sample_id']}-{noise_type}"
    relative_paths = _condition_relative_paths(method, variant_id, condition_id)
    try:
        condition_dir = resolve_contained_result_path(
            run_root,
            f"arms/{method}/{variant_id}/conditions/{condition_id}",
            "condition directory",
        )
        status_path = resolve_contained_result_path(
            run_root, relative_paths["status"], "condition status"
        )
    except ContractValidationError as error:
        raise RunnerError(str(error)) from error
    if status_path.is_file():
        existing = load_json(status_path)
        validate_contract("condition", existing)
        expected_identity = {
            "condition_id": condition_id,
            "sample_id": record["sample_id"],
            "group_id": record["group_id"],
            "bucket": record["bucket"],
            "condition_kind": "clean" if noise_type == "clean" else "noisy",
            "noise_type": "clean" if noise_type == "clean" else record["noisy"]["noise_type"],
            "requested_backend": method,
            "variant_id": variant_id,
            "fallback_used": False,
            "input": {
                "path": record[noise_type]["path"],
                "sha256": record[noise_type]["sha256"],
                "sample_count": record[noise_type]["sample_count"],
                "sample_rate": TARGET_SAMPLE_RATE,
            },
            "artifacts": relative_paths,
        }
        if any(existing.get(key) != value for key, value in expected_identity.items()):
            raise RunnerError("existing terminal status does not match current condition evidence")
        return existing
    condition_dir.mkdir(parents=True, exist_ok=True)
    input_meta = record[noise_type]
    input_path = _relative_data_path(data_root, str(input_meta["path"]), f"{condition_id}.input")
    input_samples = int(input_meta["sample_count"])
    started_at = _utc_now()
    method_error = {
        "type": "RunnerError",
        "message": "worker process did not produce a result",
        "traceback": "",
    }

    with tempfile.TemporaryDirectory(prefix="medibytes-eval-condition-") as temporary_name:
        temporary = Path(temporary_name)
        none_spec = load_method_spec("none", data_root)
        clean_input_path = _relative_data_path(
            data_root,
            str(record["clean"]["path"]),
            f"{condition_id}.clean",
        )
        reference_results: dict[str, dict[str, Any]] = {}
        for phase, reference_input, reference_samples in (
            ("clean-reference", clean_input_path, int(record["clean"]["sample_count"])),
            ("input-reference", input_path, input_samples),
        ):
            reference_spec = _worker_spec(
                method_spec=none_spec,
                phase=phase,
                condition_id=condition_id,
                input_path=reference_input,
                output_path=temporary / f"{phase}.wav",
                input_samples=reference_samples,
                downstream=None,
                data_root=data_root,
            )
            reference_result = invoke(
                reference_spec,
                temporary,
                _worker_timeout(none_spec["config"], False),
            )
            if reference_result.get("status") != "ok":
                _write_condition_failure(run_root=run_root, paths=relative_paths, result=reference_result)
                status = _status_record(
                    condition_id=condition_id,
                    record=record,
                    noise_type=noise_type,
                    method=method,
                    variant_id=variant_id,
                    status=_failure_status(reference_result),
                    stage="reference",
                    downstream_status="not_run",
                    started_at=started_at,
                    input_path=input_meta["path"],
                    input_sha256=input_meta["sha256"],
                    input_samples=input_samples,
                    enhanced=None,
                    repeat_hashes=[],
                    error=reference_result.get("error") or method_error,
                    relative_paths=relative_paths,
                )
                validate_contract("condition", status)
                atomic_write_json(status_path, status)
                return status
            reference_results[phase] = reference_result
        clean_reference_result = reference_results["clean-reference"]
        input_reference_result = reference_results["input-reference"]

        accuracy_spec = _worker_spec(
            method_spec=method_spec,
            phase="accuracy",
            condition_id=condition_id,
            input_path=input_path,
            output_path=temporary / "accuracy.wav",
            input_samples=input_samples,
            downstream=downstream,
            data_root=data_root,
        )
        accuracy_result = invoke(
            accuracy_spec,
            temporary,
            _worker_timeout(method_spec["config"], True),
        )
        accuracy_output_raw = accuracy_result.get("enhanced_path")
        accuracy_output = Path(accuracy_output_raw) if isinstance(accuracy_output_raw, str) else temporary / "missing.wav"
        if accuracy_result.get("status") != "ok" and not accuracy_output.is_file():
            _write_condition_failure(run_root=run_root, paths=relative_paths, result=accuracy_result)
            status = _status_record(
                condition_id=condition_id,
                record=record,
                noise_type=noise_type,
                method=method,
                variant_id=variant_id,
                status=_failure_status(accuracy_result),
                stage="accuracy",
                downstream_status="not_run",
                started_at=started_at,
                input_path=input_meta["path"],
                input_sha256=input_meta["sha256"],
                input_samples=input_samples,
                enhanced=None,
                repeat_hashes=[],
                error=accuracy_result.get("error") or method_error,
                relative_paths=relative_paths,
            )
            validate_contract("condition", status)
            atomic_write_json(status_path, status)
            return status

        from eval.metrics import (
            audio_quality_metrics,
            clinical_metrics,
            read_pcm16_mono,
            transcript_metrics,
            unavailable_text_metrics,
        )

        final_output = condition_dir / "enhanced.wav"
        _atomic_copy(accuracy_output, final_output)
        enhancement = dict(accuracy_result.get("enhancement", {}))
        enhanced_properties = _wav_properties(final_output)
        enhanced_properties["actual_backend"] = enhancement.get("actual_backend")
        clean_reference_path = Path(clean_reference_result["enhanced_path"])
        input_reference_path = Path(input_reference_result["enhanced_path"])
        clean_reference_metadata = dict(clean_reference_result.get("enhancement", {}))
        input_reference_metadata = dict(input_reference_result.get("enhancement", {}))
        for hash_key in ("preprocess_reference_sha256", "none_reference_sha256"):
            if enhancement.get(hash_key) != input_reference_metadata.get(hash_key):
                raise RunnerError(f"candidate {hash_key} differs from the same-input none preparation")
        for gain_key in ("preprocess_total_gain_linear", "preprocess_peak_scale"):
            if enhancement.get(gain_key) != input_reference_metadata.get(gain_key):
                raise RunnerError(f"candidate {gain_key} differs from the same-input none preparation")
        canonical_clean_hash = file_sha256(clean_reference_path)
        if noise_type == "clean" and enhancement.get("none_reference_sha256") != canonical_clean_hash:
            raise RunnerError("clean candidate does not preserve the canonical none reference")
        atomic_write_json(run_root / relative_paths["enhancement"], enhancement)
        try:
            reference_samples = read_pcm16_mono(clean_reference_path)
            input_samples_array = read_pcm16_mono(input_reference_path)
            enhanced_samples = read_pcm16_mono(final_output)
            audio = audio_quality_metrics(
                reference_samples,
                input_samples_array,
                enhanced_samples,
                sample_rate=TARGET_SAMPLE_RATE,
            )
            audio["aligned_reference_sha256"] = sha256_bytes(
                np.asarray(reference_samples, dtype="<f4").tobytes()
            )
            audio["reference_input_sha256"] = record["clean"]["sha256"]
            audio["reference_preprocess_total_gain_linear"] = clean_reference_metadata[
                "preprocess_total_gain_linear"
            ]
            audio["input_preprocess_total_gain_linear"] = input_reference_metadata[
                "preprocess_total_gain_linear"
            ]
            audio["none_reference_sha256"] = canonical_clean_hash
            audio["candidate_none_reference_sha256"] = enhancement["none_reference_sha256"]
            audio["input_sha256"] = input_meta["sha256"]
            audio["enhanced_sha256"] = enhanced_properties["sha256"]
        except Exception as error:
            audio = {"status": "not_available", "reason": _sanitize_text(str(error), [data_root, REPO_ROOT])}
        transcript = accuracy_result.get("transcript")
        entities = accuracy_result.get("entities")
        if isinstance(transcript, Mapping) and isinstance(entities, Mapping):
            transcript = dict(transcript)
            entities = dict(entities)
            atomic_write_json(run_root / relative_paths["transcript"], transcript)
            atomic_write_json(run_root / relative_paths["entities"], entities)
            fields = {
                "schema_version": "1.0.0",
                "record_type": "fields",
                "status": "observed",
                "fields": {
                    group: entities.get(group, [])
                    for group in ("drugs", "symptoms", "vitals", "allergies", "diagnosis", "followup")
                },
                "clinical_gold": "empty_nonclinical",
            }
            atomic_write_json(run_root / relative_paths["fields"], fields)
            text = transcript_metrics(str(record["source"]["transcript"]), str(transcript["text"]))
            clinical = clinical_metrics(entities)
        else:
            reason = "strict downstream did not return transcript and entities"
            transcript = None
            entities = None
            text = unavailable_text_metrics(reason)
            clinical = clinical_metrics({})
            _write_condition_failure(
                run_root=run_root,
                paths=relative_paths,
                result=accuracy_result,
                existing_enhancement=enhancement,
                existing_transcript=None,
                existing_entities=None,
                text_metrics=text,
            )

        resource_records = [_resource_with_phase(accuracy_result)]
        repeat_hashes: list[str] = []
        repeat_failure: dict[str, Any] | None = None
        if enhancement.get("actual_backend") == method:
            phase_names = ("warmup", "repeat-1", "repeat-2", "repeat-3")
            for phase in phase_names:
                phase_spec = _worker_spec(
                    method_spec=method_spec,
                    phase=phase,
                    condition_id=condition_id,
                    input_path=input_path,
                    output_path=temporary / f"{phase}.wav",
                    input_samples=input_samples,
                    downstream=None,
                    data_root=data_root,
                )
                phase_result = invoke(
                    phase_spec,
                    temporary,
                    _worker_timeout(method_spec["config"], False),
                )
                resource_records.append(_resource_with_phase(phase_result))
                if phase_result.get("status") != "ok":
                    repeat_failure = phase_result
                    break
                phase_output = Path(phase_result["enhanced_path"])
                phase_hash = file_sha256(phase_output)
                if phase_hash != enhanced_properties["sha256"]:
                    repeat_failure = {
                        "status": "failed",
                        "phase": phase,
                        "error": {
                            "type": "NondeterministicOutput",
                            "message": f"{phase} output hash differs from the accuracy output",
                            "traceback": "",
                        },
                    }
                    break
                repeat_hashes.append(phase_hash)
        if repeat_failure is None and accuracy_result.get("status") != "ok":
            repeat_failure = accuracy_result
        resources = _resource_summary(resource_records)
        atomic_write_json(run_root / relative_paths["resources"], resources)
        metrics = {
            "schema_version": "1.0.0",
            "record_type": "metrics",
            "condition_id": condition_id,
            "status": "observed" if repeat_failure is None else "partial",
            "audio": audio,
            "text": text,
            "clinical": clinical,
        }
        if repeat_failure is None:
            atomic_write_json(run_root / relative_paths["metrics"], metrics)

        if repeat_failure is not None:
            if repeat_failure is not accuracy_result:
                _write_condition_failure(
                    run_root=run_root,
                    paths=relative_paths,
                    result=repeat_failure,
                    existing_enhancement=enhancement,
                    existing_transcript=transcript,
                    existing_entities=entities,
                    text_metrics=text,
                )
            enhanced_status = {
                "channels": 1,
                "path": str(final_output.relative_to(run_root)),
                "sha256": enhanced_properties["sha256"],
                "sample_count": enhanced_properties["sample_count"],
                "sample_rate": TARGET_SAMPLE_RATE,
                "actual_backend": enhancement.get("actual_backend"),
            }
            status = _status_record(
                condition_id=condition_id,
                record=record,
                noise_type=noise_type,
                method=method,
                variant_id=variant_id,
                status=_failure_status(repeat_failure),
                stage=str(repeat_failure.get("phase", "measurement")),
                downstream_status=str(accuracy_result.get("downstream_status", "failed")),
                started_at=started_at,
                input_path=input_meta["path"],
                input_sha256=input_meta["sha256"],
                input_samples=input_samples,
                enhanced=enhanced_status,
                repeat_hashes=repeat_hashes,
                error=repeat_failure.get("error") or method_error,
                relative_paths=relative_paths,
            )
            validate_contract("condition", status)
            atomic_write_json(status_path, status)
            return status

        enhanced_status = {
            "channels": 1,
            "path": str(final_output.relative_to(run_root)),
            "sha256": enhanced_properties["sha256"],
            "sample_count": enhanced_properties["sample_count"],
            "sample_rate": TARGET_SAMPLE_RATE,
            "actual_backend": enhancement.get("actual_backend"),
        }
        status = _status_record(
            condition_id=condition_id,
            record=record,
            noise_type=noise_type,
            method=method,
            variant_id=variant_id,
            status="ok",
            stage="complete",
            downstream_status="ok",
            started_at=started_at,
            input_path=input_meta["path"],
            input_sha256=input_meta["sha256"],
            input_samples=input_samples,
            enhanced=enhanced_status,
            repeat_hashes=repeat_hashes,
            error=None,
            relative_paths=relative_paths,
        )
        validate_contract("condition", status)
        atomic_write_json(status_path, status)
        return status


def _static_output_snapshot() -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for directory_name in STATIC_DEMO_DIRECTORIES:
        directory = REPO_ROOT / "demo" / directory_name
        if not directory.exists():
            continue
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            relative = path.relative_to(REPO_ROOT).as_posix()
            snapshot[relative] = file_sha256(path)
    return snapshot


def _snapshot_sha(snapshot: Mapping[str, str]) -> str:
    return sha256_bytes("\n".join(f"{key}:{value}" for key, value in sorted(snapshot.items())).encode("utf-8"))


def _load_and_verify_corpus(manifest_path: Path, data_root: Path) -> list[dict[str, Any]]:
    from eval.corpus.build_corpus import command_verify
    from eval.corpus.manifest import load_manifest_jsonl

    resolved_manifest = manifest_path.resolve(strict=True)
    command_verify(resolved_manifest, str(data_root))
    records = load_manifest_jsonl(resolved_manifest)
    for record in records:
        validate_contract("corpus", record)
    return records


def _method_environment_record(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "record_type": "method_environment",
        "method": dict(identity),
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "device": "cpu",
    }

def _runtime_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in (
        "faster-whisper",
        "imageio-ffmpeg",
        "jsonschema",
        "numpy",
        "pesq",
        "pystoi",
        "pyarrow",
    ):
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _build_identity(
    *,
    manifest_path: Path,
    downstream: Mapping[str, Any],
    downstream_path: Path,
    method_spec: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "catalog_sha256": file_sha256(CATALOG_PATH),
        "corpus_id": "pilot-15-v1",
        "downstream_code_sha256": downstream_code_sha256(downstream),
        "enhancement_code_sha256": enhancement_code_sha256(),
        "downstream_config_sha256": file_sha256(downstream_path),
        "implementation_sha256": implementation_sha256(),
        "manifest_sha256": file_sha256(manifest_path),
        "method_config_sha256": str(method_spec["config_sha256"]),
    }


def _run_limitations() -> list[str]:
    return [
        "The 15-base corpus is a non-clinical smoke/screening corpus and cannot establish production medical accuracy or safety.",
        "Clinical WER, medication/dose accuracy, clinical NER F1, field accuracy, and critical-error rate are not applicable.",
        "No production verdict is derived from this pilot.",
    ]


def _finalize_report(
    *,
    run_root: Path,
    method_spec: Mapping[str, Any],
    run_record: Mapping[str, Any],
    static_outputs_unchanged: bool,
) -> dict[str, Any]:
    from eval.reporting import build_arm_report, render_report_markdown

    statuses: list[dict[str, Any]] = []
    condition_relative = (
        f"arms/{method_spec['backend']}/{method_spec['config']['variant_id']}/conditions"
    )
    try:
        condition_root = resolve_contained_result_path(
            run_root, condition_relative, "condition root"
        )
    except ContractValidationError as error:
        raise RunnerError(str(error)) from error
    condition_root.mkdir(parents=True, exist_ok=True)
    if any(path.is_symlink() for path in condition_root.iterdir()):
        raise RunnerError("condition entries must not be symlinks")
    for status_path in sorted(condition_root.glob("*/status.json")):
        if status_path.is_symlink():
            raise RunnerError("condition status must not be a symlink")
        status = load_json(status_path)
        validate_contract("condition", status)
        statuses.append(status)
    report = build_arm_report(
        run_root=run_root,
        run_record=dict(run_record),
        statuses=statuses,
        static_outputs_unchanged=static_outputs_unchanged,
    )
    try:
        report_json = resolve_contained_result_path(
            run_root, str(run_record["artifacts"]["report_json"]), "report_json"
        )
        report_md = resolve_contained_result_path(
            run_root, str(run_record["artifacts"]["report_md"]), "report_md"
        )
    except ContractValidationError as error:
        raise RunnerError(str(error)) from error
    if report_json.exists():
        existing = load_json(report_json)
        if existing != report:
            raise RunnerError("existing finalized report differs from resumed evidence")
    else:
        write_json_exclusive(report_json, report)
    markdown = render_report_markdown(report)
    if report_md.exists():
        if report_md.read_text(encoding="utf-8") != markdown:
            raise RunnerError("existing finalized report markdown differs from resumed evidence")
    else:
        write_text_exclusive(report_md, markdown)
    return report


def run_evaluation(
    *,
    method: str,
    run_id: str,
    data_root_value: str | os.PathLike[str],
    results_root: Path,
    manifest_path: Path = MANIFEST_PATH,
    downstream_path: Path = DOWNSTREAM_CONFIG_PATH,
) -> dict[str, Any]:
    """Execute or resume one immutable, single-backend evaluation run."""

    data_root = resolve_data_root(data_root_value)
    run_id = validate_run_id(run_id)
    if not results_root.is_absolute():
        raise RunnerError("--results-root must be absolute")
    if ".." in results_root.parts:
        raise RunnerError("--results-root must not contain parent traversal")
    try:
        results_relative = results_root.relative_to(data_root).as_posix()
        expected_root = resolve_contained_result_path(
            data_root, f"runs/{run_id}", "run root"
        )
        resolved_results = resolve_contained_result_path(
            data_root, results_relative, "results root"
        )
    except (ContractValidationError, ValueError) as error:
        raise RunnerError("run/results roots must be symlink-free under the evaluation data root") from error
    if resolved_results != expected_root:
        raise RunnerError("--results-root must exactly equal <data-root>/runs/<run-id>")
    records = _load_and_verify_corpus(manifest_path, data_root)
    downstream, resolved_downstream = load_downstream_config(downstream_path)
    method_spec = load_method_spec(method, data_root)
    identity = _build_identity(
        manifest_path=manifest_path.resolve(strict=True),
        downstream=downstream,
        downstream_path=resolved_downstream,
        method_spec=method_spec,
    )
    run_root = expected_root
    run_root.mkdir(parents=True, exist_ok=True)
    try:
        run_root = resolve_contained_result_path(
            data_root, f"runs/{run_id}", "run root", must_exist=True
        )
        run_json_path = resolve_contained_result_path(run_root, "run.json", "run.json")
    except ContractValidationError as error:
        raise RunnerError(str(error)) from error
    if run_json_path.is_file():
        existing = load_json(run_json_path)
        validate_contract("run", existing)
        if existing.get("run_id") != run_id or existing.get("identity") != identity:
            raise RunnerError("finalized run ID cannot be reused with changed identity hashes")
        from eval.reporting import verify_run

        verified = verify_run(run_root)
        return {
            "status": "already_finalized",
            "run_id": run_id,
            "method": method,
            "run": verified["run"],
        }

    snapshot_before = _static_output_snapshot()
    try:
        environment_path = resolve_contained_result_path(
            run_root, "environment.json", "environment.json"
        )
    except ContractValidationError as error:
        raise RunnerError(str(error)) from error
    environment = {
        "schema_version": "1.0.0",
        "record_type": "environment",
        "run_id": run_id,
        "created_at": _utc_now(),
        "identity": identity,
        "data_root_configured": True,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": platform.platform(),
        "packages": _runtime_versions(),
        "downstream": _method_environment_record(method_spec["identity"]),
        "static_demo_output_snapshot_sha256": _snapshot_sha(snapshot_before),
    }
    environment_snapshot_digest = _snapshot_sha(snapshot_before)
    if environment_path.exists():
        existing_environment = load_json(environment_path)
        if existing_environment.get("identity") != identity:
            raise RunnerError("resume requires identical manifest/config/downstream/corpus hashes")
        environment_snapshot_digest = str(existing_environment["static_demo_output_snapshot_sha256"])
    else:
        environment["static_demo_output_snapshot_sha256"] = environment_snapshot_digest
        write_json_exclusive(environment_path, environment)
    environment_hash = file_sha256(environment_path)

    statuses: list[dict[str, Any]] = []
    if method_spec["disposition"] not in BLOCKED_DISPOSITIONS:
        for record in records:
            for noise_type in ("clean", "noisy"):
                try:
                    status = run_condition(
                        record,
                        noise_type,
                        run_root=run_root,
                        data_root=data_root,
                        method_spec=method_spec,
                        downstream=downstream,
                        invoke_worker=_WORKER_INVOKER,
                    )
                except BaseException as error:
                    condition_id = f"{record['sample_id']}-{noise_type}"
                    method = str(method_spec["backend"])
                    variant_id = str(method_spec["config"]["variant_id"])
                    condition_dir = run_root / "arms" / method / variant_id / "conditions" / condition_id
                    status_path = condition_dir / "status.json"
                    if status_path.is_file():
                        status = load_json(status_path)
                        validate_contract("condition", status)
                    else:
                        paths = _condition_relative_paths(method, variant_id, condition_id)
                        failure = {
                            "status": "failed_runtime",
                            "phase": "verification",
                            "error": _error_record(error, [data_root, REPO_ROOT, run_root]),
                        }
                        _write_condition_failure(run_root=run_root, paths=paths, result=failure)
                        input_meta = record[noise_type]
                        status = _status_record(
                            condition_id=condition_id,
                            record=record,
                            noise_type=noise_type,
                            method=method,
                            variant_id=variant_id,
                            status="failed_runtime",
                            stage="verification",
                            downstream_status="not_run",
                            started_at=_utc_now(),
                            input_path=str(input_meta["path"]),
                            input_sha256=str(input_meta["sha256"]),
                            input_samples=int(input_meta["sample_count"]),
                            enhanced=None,
                            repeat_hashes=[],
                            error=failure["error"],
                            relative_paths=paths,
                        )
                        validate_contract("condition", status)
                        atomic_write_json(status_path, status)
                    if isinstance(error, (KeyboardInterrupt, SystemExit)):
                        raise
                statuses.append(status)

    snapshot_after = _static_output_snapshot()
    static_outputs_unchanged = (
        snapshot_before == snapshot_after
        and environment_snapshot_digest == _snapshot_sha(snapshot_before)
    )
    counts = {
        "expected_conditions": 0 if method_spec["disposition"] in BLOCKED_DISPOSITIONS else 30,
        "recorded_conditions": len(statuses),
        "ok": sum(status.get("status") == "ok" for status in statuses),
        "failed": sum(status.get("status") != "ok" for status in statuses),
    }
    report_relative = (
        f"arms/{method_spec['backend']}/{method_spec['config']['variant_id']}/report."
    )
    run_record = {
        "schema_version": "1.0.0",
        "record_type": "run",
        "run_id": run_id,
        "status": "complete",
        "environment_sha256": environment_hash,
        "method": method_spec["identity"],
        "identity": identity,
        "composition": {
            "mode": "single_backend",
            "backend_count": 1,
            "layered": False,
            "fallback_used": False,
        },
        "counts": counts,
        "artifacts": {
            "environment": "environment.json",
            "report_json": f"{report_relative}json",
            "report_md": f"{report_relative}md",
        },
        "limitations": _run_limitations(),
    }
    validate_contract("run", run_record)
    report = _finalize_report(
        run_root=run_root,
        method_spec=method_spec,
        run_record=run_record,
        static_outputs_unchanged=static_outputs_unchanged,
    )
    if run_json_path.exists():
        existing = load_json(run_json_path)
        if existing != run_record:
            raise RunnerError("run.json changed during finalization")
    else:
        write_json_exclusive(run_json_path, run_record)
    from eval.reporting import verify_run

    verified = verify_run(run_root)
    if verified["report"] != report:
        raise RunnerError("final report changed during strict verification")
    return {
        "status": "complete",
        "run_id": run_id,
        "method": method,
        "counts": counts,
        "arm_status": report["status"],
        "run": run_record,
    }


def set_worker_invoker_for_tests(
    invoker: Callable[..., dict[str, Any]] | None,
) -> Callable[..., dict[str, Any]] | None:
    """Inject a deterministic worker only for isolated runner contract tests."""

    global _WORKER_INVOKER
    previous = _WORKER_INVOKER
    _WORKER_INVOKER = invoker
    return previous


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eval.runner")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.worker:
        try:
            spec = loads_strict(sys.stdin.read())
        except (ContractValidationError, json.JSONDecodeError, OSError) as error:
            print(f"worker specification error: {error}", file=sys.stderr)
            return 2
        if not isinstance(spec, dict):
            print("worker specification must be a JSON object", file=sys.stderr)
            return 2
        return _worker_main(spec)
    print("eval.runner is an internal worker; use python -m eval.harness", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

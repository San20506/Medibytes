"""Draft-07 contracts and strict JSON persistence for MediBytes evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from jsonschema import Draft7Validator, FormatChecker
from jsonschema.exceptions import SchemaError

SCHEMA_VERSION = "1.0.0"
CONTRACT_DIR = Path(__file__).resolve().parent
SCHEMA_FILES = {
    "corpus": "corpus-record.schema.json",
    "run": "run.schema.json",
    "condition": "condition.schema.json",
    "report": "report.schema.json",
}


class ContractValidationError(ValueError):
    """A persisted evaluation record violates its frozen schema or path rule."""


class DuplicateKeyError(ValueError):
    """A JSON object contained the same member more than once."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result



def _reject_nonfinite_constant(value: str) -> None:
    raise ContractValidationError(f"non-finite JSON number is forbidden: {value}")



def loads_strict(text: str) -> Any:
    """Decode JSON while rejecting duplicate members and non-finite numbers."""

    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite_constant,
    )


def load_json(path: Path) -> Any:
    try:
        return loads_strict(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, DuplicateKeyError) as error:
        raise ContractValidationError(f"cannot load strict JSON {path.name}: {error}") from error


@lru_cache(maxsize=None)
def load_schema(kind: str) -> dict[str, Any]:
    try:
        filename = SCHEMA_FILES[kind]
    except KeyError as error:
        raise KeyError(f"unknown contract schema {kind!r}") from error
    schema = load_json(CONTRACT_DIR / filename)
    if schema.get("$schema") != "http://json-schema.org/draft-07/schema#":
        raise ContractValidationError(f"{filename} is not Draft-07")
    try:
        Draft7Validator.check_schema(schema)
    except SchemaError as error:
        raise ContractValidationError(f"invalid Draft-07 schema {filename}: {error}") from error
    return schema


@lru_cache(maxsize=None)
def validator_for(kind: str) -> Draft7Validator:
    return Draft7Validator(load_schema(kind), format_checker=FormatChecker())


def contract_errors(kind: str, value: Any) -> list[str]:
    """Return stable human-readable validation errors for a contract record."""

    errors = sorted(validator_for(kind).iter_errors(value), key=lambda item: list(item.path))
    messages: list[str] = []
    for error in errors:
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        messages.append(f"{location}: {error.message}")
    return messages


def validate_contract(kind: str, value: Any) -> None:
    errors = contract_errors(kind, value)
    if errors:
        raise ContractValidationError(f"{kind} contract failed: " + "; ".join(errors[:8]))


def assert_relative_result_path(value: str, label: str) -> PurePosixPath:
    """Validate a serialized POSIX result path without consulting the host FS."""

    if not isinstance(value, str) or not value or value.startswith("/") or value.startswith("\\"):
        raise ContractValidationError(f"{label} must be a nonempty relative POSIX path")
    if "://" in value:
        raise ContractValidationError(f"{label} must not be a URL")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ContractValidationError(f"{label} contains an unsafe path component")
    return path


def resolve_contained_result_path(
    root: Path,
    value: str,
    label: str,
    *,
    must_exist: bool = False,
) -> Path:
    """Resolve one serialized path under ``root`` while rejecting symlink components."""

    relative = assert_relative_result_path(value, label)
    expanded_root = root.expanduser()
    if expanded_root.is_symlink():
        raise ContractValidationError(f"{label} root must not be a symlink")
    try:
        base = expanded_root.resolve(strict=must_exist)
    except (OSError, RuntimeError) as error:
        raise ContractValidationError(f"{label} root cannot be resolved") from error
    current = base
    for part in relative.parts:
        current = current / part
        try:
            current.lstat()
        except FileNotFoundError:
            if must_exist:
                raise ContractValidationError(f"{label} does not exist") from None
            continue
        if current.is_symlink():
            raise ContractValidationError(f"{label} contains a symlink component")
    candidate = base.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=must_exist)
    except (OSError, RuntimeError) as error:
        raise ContractValidationError(f"{label} cannot be resolved") from error
    try:
        resolved.relative_to(base)
    except ValueError as error:
        raise ContractValidationError(f"{label} escapes its declared root") from error
    return candidate


def assert_no_absolute_result_paths(value: Any, label: str = "result") -> None:
    """Reject absolute paths anywhere in a persisted result document."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            child_label = f"{label}.{key}"
            if "path" in key.lower() and isinstance(child, str):
                assert_relative_result_path(child, child_label)
            assert_no_absolute_result_paths(child, child_label)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_no_absolute_result_paths(child, f"{label}[{index}]")
    elif isinstance(value, str) and (value.startswith("/home/") or value.startswith("file://")):
        raise ContractValidationError(f"{label} contains an absolute host path")


def canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ContractValidationError(f"value is not canonical JSON: {error}") from error
    return text.encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ContractValidationError(f"cannot hash {path.name}: {error}") from error
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    """Atomically replace one unfinished artifact with strict finite JSON."""

    assert_no_absolute_result_paths(value, path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_json_exclusive(path: Path, value: Any) -> None:
    """Create a finalized JSON artifact once; never overwrite it."""

    assert_no_absolute_result_paths(value, path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise ContractValidationError(f"finalized artifact already exists: {path.name}") from error


def write_text_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise ContractValidationError(f"finalized artifact already exists: {path.name}") from error


__all__ = [
    "CONTRACT_DIR",
    "ContractValidationError",
    "SCHEMA_VERSION",
    "assert_no_absolute_result_paths",
    "assert_relative_result_path",
    "atomic_write_json",
    "canonical_json_bytes",
    "contract_errors",
    "file_sha256",
    "load_json",
    "resolve_contained_result_path",
    "load_schema",
    "loads_strict",
    "sha256_bytes",
    "sha256_json",
    "validate_contract",
    "validator_for",
    "write_json_exclusive",
    "write_text_exclusive",
]

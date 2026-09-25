"""Strict manifest validation, admission, and rebuild identity checks."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import numpy as np

from eval.corpus.audio import (
    CHANNELS,
    CODEC,
    SAMPLE_RATE,
    WAV_HEADER_BYTES,
    active_mask,
    apply_echo,
    clipping_count,
    pcm16_peak,
    read_pcm16,
    seed_hex,
    validate_pcm16,
)

SCHEMA_VERSION = "1.0.0"
CORPUS_ID = "pilot-15-v1"
NOISE_TYPES = ("white", "pink_voss16", "hum50", "am_babble_proxy", "echo")
EXPECTED_BUCKETS = {
    "en-US-proxy": 4,
    "hi": 4,
    "ta": 4,
    "hi-en-code-mix": 3,
}
BUCKET_PROFILES = {
    "en-US-proxy": ("en-US-proxy", "en_us", False),
    "hi": ("hi", "hi_in", False),
    "ta": ("ta", "ta_in", False),
    "hi-en-code-mix": ("hi-en-code-mix", "hi-en", True),
}
EXPECTED_NOISES = {name: 3 for name in NOISE_TYPES}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def noise_type_for_index(index: int) -> str:
    if index < 0 or index >= 15:
        raise ValueError("the pilot corpus contains exactly 15 sample indices (0..14)")
    return NOISE_TYPES[index // 3]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transcript_sha256(transcript: str) -> str:
    return hashlib.sha256(transcript.encode("utf-8")).hexdigest()


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_text(record: Mapping[str, Any], key: str, label: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}.{key} must be a nonempty string")
    return value


def _require_sha256(record: Mapping[str, Any], key: str, label: str) -> str:
    value = _require_text(record, key, label)
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{label}.{key} must be a lowercase SHA-256 digest")
    return value


def _safe_data_path(data_root: Path, raw_path: object, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label}.path must be a nonempty relative path")
    relative = PurePosixPath(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label}.path must stay under the evaluation data root")
    path = data_root.joinpath(*relative.parts)
    try:
        path.resolve(strict=False).relative_to(data_root.resolve(strict=False))
    except ValueError as error:
        raise ValueError(f"{label}.path escapes the evaluation data root") from error
    return path


def _validate_audio_metadata(
    audio: Mapping[str, Any],
    *,
    label: str,
    data_root: Path,
) -> Path:
    path = _safe_data_path(data_root, audio.get("path"), label)
    if not path.is_file():
        raise ValueError(f"{label} file is missing: {audio.get('path')}")
    expected_hash = _require_sha256(audio, "sha256", label)
    actual_hash = file_sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"{label} SHA-256 mismatch for {audio.get('path')}: "
            f"expected {expected_hash}, got {actual_hash}"
        )
    expected_bytes = audio.get("bytes")
    if not isinstance(expected_bytes, int) or expected_bytes != path.stat().st_size:
        raise ValueError(f"{label}.bytes does not match the canonical file")
    if audio.get("sample_rate") != SAMPLE_RATE:
        raise ValueError(f"{label}.sample_rate must be {SAMPLE_RATE}")
    if audio.get("channels") != CHANNELS:
        raise ValueError(f"{label}.channels must be 1")
    if audio.get("codec") != CODEC:
        raise ValueError(f"{label}.codec must be {CODEC}")
    if audio.get("wav_header_bytes") not in (None, WAV_HEADER_BYTES):
        raise ValueError(f"{label}.wav_header_bytes must be 44")
    samples = read_pcm16(path)
    expected_count = audio.get("sample_count")
    if expected_count != samples.size:
        raise ValueError(f"{label}.sample_count does not match decoded PCM16")
    expected_duration = audio.get("duration_s")
    if not isinstance(expected_duration, (int, float)) or not math.isclose(
        float(expected_duration), samples.size / SAMPLE_RATE, abs_tol=1e-12
    ):
        raise ValueError(f"{label}.duration_s does not match decoded PCM16")
    peak = audio.get("peak")
    if not isinstance(peak, (int, float)) or not math.isclose(
        float(peak), pcm16_peak(samples), abs_tol=1e-12
    ):
        raise ValueError(f"{label}.peak does not match decoded PCM16")
    return path


def _validate_source(record: Mapping[str, Any]) -> None:
    source = _require_mapping(record.get("source"), "source")
    for key in ("dataset", "release", "digest", "split", "source_id", "member_path"):
        _require_text(source, key, "source")
    _require_sha256(source, "digest", "source")
    transcript = _require_text(source, "transcript", "source")
    expected_transcript_hash = _require_sha256(source, "transcript_sha256", "source")
    if transcript_sha256(transcript) != expected_transcript_hash:
        raise ValueError("source transcript digest mismatch")
    _require_text(source, "license_spdx", "source")
    _require_text(source, "attribution", "source")
    if not isinstance(source.get("row"), Mapping):
        raise ValueError("source.row must preserve the selected source row")


def _validate_measured_condition(
    record: Mapping[str, Any], clean_path: Path, noisy_path: Path
) -> None:
    sample_id = _require_text(record, "sample_id", "record")
    noisy = _require_mapping(record.get("noisy"), "noisy")
    noise_type = _require_text(noisy, "noise_type", "noisy")
    if noisy.get("seed") != seed_hex(sample_id, noise_type):
        raise ValueError("noisy.seed does not match the approved SHA-256/PCG64 derivation")
    clean = read_pcm16(clean_path).astype(np.float64) / 32768.0
    noisy_float = read_pcm16(noisy_path).astype(np.float64) / 32768.0
    mask, _ = active_mask(clean)
    active = mask > 0
    if noise_type == "echo":
        reference, _ = apply_echo(clean, sample_id)
        reference *= float(noisy.get("post_mix_gain"))
    else:
        reference = clean
    reference_energy = float(np.dot(reference[active], reference[active]))
    error = noisy_float - reference
    error_energy = float(np.dot(error[active], error[active]))
    if reference_energy <= 0 or error_energy <= 0:
        raise ValueError("condition has nonpositive active reference/error energy")
    measured = 10.0 * math.log10(reference_energy / error_energy)
    if not math.isclose(
        float(noisy.get("measured_snr_db")), measured, abs_tol=1e-9
    ):
        raise ValueError("noisy.measured_snr_db does not match decoded audio")
    active_rms = math.sqrt(reference_energy / float(np.count_nonzero(active)))
    if not math.isclose(float(noisy.get("active_rms")), active_rms, abs_tol=1e-12):
        raise ValueError("noisy.active_rms does not match decoded audio")
    if noise_type != "echo" and not math.isclose(
        float(noisy.get("post_mix_gain")), 1.0, abs_tol=1e-15
    ):
        raise ValueError("additive conditions must use post_mix_gain=1.0")


def _validate_record(record: Mapping[str, Any], data_root: Path) -> tuple[Path, Path]:
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"record schema_version must be {SCHEMA_VERSION}")
    if record.get("record_status") != "complete":
        raise ValueError("admission requires record_status=complete")
    if record.get("corpus_id") != CORPUS_ID:
        raise ValueError(f"record corpus_id must be {CORPUS_ID}")
    for key in ("sample_id", "group_id"):
        _require_text(record, key, "record")
    if record.get("split_role") != "smoke" or record.get("tuning_allowed") is not False:
        raise ValueError("record must be smoke with tuning_allowed=false")
    for key in ("bucket", "language_profile", "locale_observed"):
        _require_text(record, key, "record")
    if not isinstance(record.get("code_mix"), bool):
        raise ValueError("record.code_mix must be boolean")
    if record.get("reference_status") != "source_reference":
        raise ValueError("record.reference_status must be source_reference")
    bucket = str(record["bucket"])
    if bucket not in BUCKET_PROFILES:
        raise ValueError(f"unknown corpus bucket: {bucket}")
    expected_profile, expected_locale, expected_code_mix = BUCKET_PROFILES[bucket]
    if (
        record["language_profile"] != expected_profile
        or record["locale_observed"] != expected_locale
        or record["code_mix"] is not expected_code_mix
    ):
        raise ValueError("bucket language/code-mix identity is inconsistent")
    privacy = _require_mapping(record.get("privacy_decision"), "privacy_decision")
    if privacy.get("eligible") is not True or privacy.get("reasons") != []:
        raise ValueError("privacy decision must be eligible with no rejection reasons")
    _require_sha256(privacy, "policy_sha256", "privacy_decision")
    build = _require_mapping(record.get("build"), "build")
    _require_text(build, "tool", "build")
    _require_sha256(build, "tool_sha256", "build")
    _require_text(build, "decoder", "build")
    _require_sha256(build, "decoder_sha256", "build")
    _validate_source(record)
    clean = _require_mapping(record.get("clean"), "clean")
    noisy = _require_mapping(record.get("noisy"), "noisy")
    clean_path = _validate_audio_metadata(clean, label="clean", data_root=data_root)
    noisy_path = _validate_audio_metadata(noisy, label="noisy", data_root=data_root)
    if clean.get("sample_count") != noisy.get("sample_count"):
        raise ValueError("clean/noisy sample counts differ")
    clean_duration = float(clean["duration_s"])
    if not 3.0 <= clean_duration <= 12.0:
        raise ValueError("clean duration must be within 3..12 seconds")
    clean_peak_dbfs = 20.0 * math.log10(float(clean["peak"]))
    if not -3.01 <= clean_peak_dbfs <= -2.99:
        raise ValueError("clean peak is outside -3.00 +/- 0.01 dBFS")
    noise_type = noisy.get("noise_type")
    if noise_type not in NOISE_TYPES:
        raise ValueError("noisy.noise_type is not admitted")
    seed = noisy.get("seed")
    if not isinstance(seed, str) or not re.fullmatch(r"[0-9a-f]{16}", seed):
        raise ValueError("noisy.seed must contain exactly 16 hexadecimal characters")
    if not math.isclose(float(noisy.get("target_snr_db")), 5.0, abs_tol=1e-12):
        raise ValueError("noisy.target_snr_db must be 5.0")
    measured_snr = float(noisy.get("measured_snr_db"))
    if not math.isfinite(measured_snr) or not 4.9 <= measured_snr <= 5.1:
        raise ValueError("noisy.measured_snr_db must be 5.0 +/- 0.1 dB")
    active_rms = float(noisy.get("active_rms"))
    post_mix_gain = float(noisy.get("post_mix_gain"))
    if not math.isfinite(active_rms) or active_rms <= 0:
        raise ValueError("noisy.active_rms must be positive and finite")
    if not math.isfinite(post_mix_gain) or post_mix_gain <= 0:
        raise ValueError("noisy.post_mix_gain must be positive and finite")
    if noisy.get("clipping_count") != 0 or clipping_count(read_pcm16(noisy_path)):
        raise ValueError("noisy audio contains clipping")
    if not isinstance(noisy.get("parameters"), Mapping):
        raise ValueError("noisy.parameters must be an object")
    validate_pcm16(read_pcm16(clean_path))
    validate_pcm16(read_pcm16(noisy_path))
    _validate_measured_condition(record, clean_path, noisy_path)
    return clean_path, noisy_path


def validate_complete_manifest(
    records: Iterable[Mapping[str, Any]], data_root: Path
) -> dict[str, object]:
    """Validate exact counts, geometry, provenance, files, and measurements."""

    materialized = list(records)
    if len(materialized) != 15:
        raise ValueError(f"corpus must contain exactly 15 records, got {len(materialized)}")
    sample_ids = [str(record.get("sample_id")) for record in materialized]
    group_ids = [str(record.get("group_id")) for record in materialized]
    if len(set(sample_ids)) != 15 or len(set(group_ids)) != 15:
        raise ValueError("sample_id and group_id must each be unique across 15 records")
    expected_sample_ids = [f"sample-{index:03d}" for index in range(1, 16)]
    if sample_ids != expected_sample_ids:
        raise ValueError("manifest sample IDs must be sample-001..sample-015 in order")
    datasets = Counter(
        str(_require_mapping(record.get("source"), "source").get("dataset"))
        for record in materialized
    )
    if datasets != Counter(
        {"Google FLEURS": 12, "MUCS 2021 Hindi-English test": 3}
    ):
        raise ValueError(f"source dataset distribution must be 12 FLEURS/3 MUCS: {dict(datasets)}")
    buckets = Counter(str(record.get("bucket")) for record in materialized)
    noises = Counter(
        str(_require_mapping(record.get("noisy"), "noisy").get("noise_type"))
        for record in materialized
    )
    if dict(buckets) != EXPECTED_BUCKETS:
        raise ValueError(f"bucket distribution must be 4/4/4/3, got {dict(buckets)}")
    if dict(noises) != EXPECTED_NOISES:
        raise ValueError(f"noise distribution must be 3/3/3/3/3, got {dict(noises)}")
    audio_paths: list[Path] = []
    corpus_roots: set[Path] = set()
    for index, record in enumerate(materialized):
        expected_noise = noise_type_for_index(index)
        actual_noise = _require_mapping(record.get("noisy"), "noisy").get("noise_type")
        if actual_noise != expected_noise:
            raise ValueError(
                f"record {index} must use {expected_noise}, got {actual_noise}"
            )
        try:
            clean_path, noisy_path = _validate_record(record, data_root)
        except ValueError as error:
            raise ValueError(f"record {index} ({sample_ids[index]}) failed: {error}") from error
        audio_paths.extend((clean_path, noisy_path))
        corpus_roots.update((clean_path.parent.parent, noisy_path.parent.parent))
    if len(set(audio_paths)) != 30:
        raise ValueError("corpus must contain exactly 30 unique clean/noisy files")
    if len(corpus_roots) != 1:
        raise ValueError("all corpus audio must use one output root")
    corpus_root = next(iter(corpus_roots))
    actual_wavs = sorted(corpus_root.rglob("*.wav"))
    if len(actual_wavs) != 30:
        raise ValueError(
            f"corpus output root must contain exactly 30 WAV files, got {len(actual_wavs)}"
        )
    if set(actual_wavs) != set(audio_paths):
        raise ValueError("corpus output root contains unmanifested or mislocated WAV files")
    return {
        "groups": 15,
        "files": 30,
        "buckets": dict(buckets),
        "noises": dict(noises),
    }


def _without_audio_paths(record: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(record))
    for condition in ("clean", "noisy"):
        normalized[condition].pop("path", None)
    return normalized


def assert_rebuild_identity(
    original: Iterable[Mapping[str, Any]], rebuilt: Iterable[Mapping[str, Any]]
) -> None:
    """Require every hash and metadata field to match across output roots."""

    original_records = sorted(original, key=lambda record: str(record.get("sample_id")))
    rebuilt_records = sorted(rebuilt, key=lambda record: str(record.get("sample_id")))
    if len(original_records) != len(rebuilt_records):
        raise ValueError("rebuild identity mismatch: record counts differ")
    for original_record, rebuilt_record in zip(
        original_records, rebuilt_records, strict=True
    ):
        sample_id = original_record.get("sample_id")
        if sample_id != rebuilt_record.get("sample_id"):
            raise ValueError("rebuild identity mismatch: sample ordering differs")
        if _without_audio_paths(original_record) != _without_audio_paths(rebuilt_record):
            raise ValueError(f"rebuild identity mismatch for {sample_id}")


def manifest_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    canonical = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def write_manifest_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_manifest_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read manifest {path}: {error}") from error
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("manifest JSONL must contain nonempty records")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid manifest JSON at line {line_number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"manifest line {line_number} must be a JSON object")
        records.append(value)
    return records


def verify_checksum_ledger(
    ledger_path: Path,
    data_root: Path,
    sources: Mapping[str, Path],
) -> None:
    """Require every pinned source digest to match its data-root artifact."""

    if not ledger_path.is_file():
        raise ValueError(f"source checksum ledger is missing: {ledger_path}")
    recorded: dict[str, str] = {}
    for line_number, line in enumerate(
        ledger_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or not _SHA256_RE.fullmatch(parts[0]):
            raise ValueError(f"malformed source checksum line {line_number}")
        digest, raw_path = parts
        relative = PurePosixPath(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe source checksum path on line {line_number}")
        if relative.as_posix() in recorded:
            raise ValueError(f"duplicate source checksum path: {relative.as_posix()}")
        recorded[relative.as_posix()] = digest
    for raw_path, source_path in sources.items():
        relative = PurePosixPath(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe source path: {raw_path!r}")
        if relative.as_posix() not in recorded:
            raise ValueError(f"source checksum is missing for {raw_path}")
        path = data_root.joinpath(*relative.parts)
        if not path.is_file():
            raise ValueError(f"required source file is missing: {raw_path}")
        actual = file_sha256(path)
        expected = recorded[relative.as_posix()]
        if actual != expected:
            raise ValueError(
                f"source checksum mismatch for {raw_path}: expected {expected}, got {actual}"
            )

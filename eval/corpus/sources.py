"""Pinned official source downloads, checksums, and FLEURS resolution."""

from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import pyarrow.parquet as pq

from eval.corpus.audio import validate_pcm16
from eval.corpus.manifest import file_sha256, transcript_sha256
from eval.corpus.media import Decoder, decode_bytes, decode_file
from eval.corpus.selection import select_fleurs_rows, select_mucs_segments

FLEURS_REPO_ID = "google/fleurs"
FLEURS_REPO_TYPE = "dataset"
FLEURS_REVISION = "168de341b3db6859a9bac1c50a2ef5e3b47647e0"
MUCS_URL = "https://openslr.elda.org/resources/104/Hindi-English_test.tar.gz"
MUCS_REQUIRED_BYTES = 443_929_204
FLEURS_LICENSE = "CC-BY-4.0"
MUCS_LICENSE = "CC-BY-SA-4.0"


@dataclass(frozen=True)
class DownloadedSources:
    files: dict[str, Path]
    checksums: dict[str, str]


def load_policy(path: Path) -> dict[str, Any]:
    try:
        policy = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load selection policy {path}: {error}") from error
    if not isinstance(policy, dict):
        raise ValueError("selection policy must be a JSON object")
    fleurs = policy.get("sources", {}).get("fleurs", {})
    if (
        fleurs.get("repo_id") != FLEURS_REPO_ID
        or fleurs.get("repo_type") != FLEURS_REPO_TYPE
        or fleurs.get("revision") != FLEURS_REVISION
    ):
        raise ValueError("selection policy does not contain the pinned FLEURS revision")
    mucs = policy.get("sources", {}).get("mucs", {})
    if (
        mucs.get("archive_url") != MUCS_URL
        or mucs.get("required_archive_bytes") != MUCS_REQUIRED_BYTES
        or mucs.get("license_spdx") != MUCS_LICENSE
    ):
        raise ValueError("selection policy does not contain the pinned MUCS archive")
    return policy


def policy_sha256(path: Path) -> str:
    return file_sha256(path)


def source_relative_paths(policy: Mapping[str, Any]) -> list[str]:
    paths = [
        f"sources/fleurs/{entry['locale']}/validation/0000.parquet"
        for entry in policy["sources"]["fleurs"]["locales"]
    ]
    paths.append("sources/mucs/Hindi-English_test.tar.gz")
    return paths


def _read_ledger(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    recorded: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"malformed source checksum line {line_number}")
        if parts[1] in recorded:
            raise ValueError(f"duplicate source checksum path: {parts[1]}")
        recorded[parts[1]] = parts[0]
    return recorded


def _write_ledger(path: Path, checksums: Mapping[str, str]) -> None:
    body = "".join(
        f"{checksums[relative]}  {relative}\n" for relative in sorted(checksums)
    )
    path.write_text(body, encoding="utf-8")


def _validate_existing_digest(
    relative: str, path: Path, recorded: Mapping[str, str]
) -> None:
    if not path.is_file():
        return
    actual = file_sha256(path)
    if relative in recorded and recorded[relative] != actual:
        raise ValueError(
            f"source checksum mismatch for {relative}: "
            f"expected {recorded[relative]}, got {actual}"
        )


def _download_fleurs(
    *,
    locale: str,
    data_root: Path,
    recorded: Mapping[str, str],
) -> Path:
    relative = f"sources/fleurs/{locale}/validation/0000.parquet"
    destination = data_root / relative
    _validate_existing_digest(relative, destination, recorded)
    if destination.is_file():
        return destination
    from huggingface_hub import hf_hub_download

    filename = f"{locale}/validation/0000.parquet"
    downloaded = Path(
        hf_hub_download(
            repo_id=FLEURS_REPO_ID,
            repo_type=FLEURS_REPO_TYPE,
            revision=FLEURS_REVISION,
            filename=filename,
            cache_dir=str(data_root / "cache" / "huggingface"),
        )
    ).resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".parquet.part")
    shutil.copyfile(downloaded, temporary)
    temporary.replace(destination)
    return destination


def _download_mucs(data_root: Path, recorded: Mapping[str, str]) -> Path:
    relative = "sources/mucs/Hindi-English_test.tar.gz"
    destination = data_root / relative
    _validate_existing_digest(relative, destination, recorded)
    if destination.exists():
        if destination.stat().st_size != MUCS_REQUIRED_BYTES:
            raise ValueError(
                f"MUCS archive size mismatch: expected {MUCS_REQUIRED_BYTES}, "
                f"got {destination.stat().st_size}"
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tar.gz.part")
    command = [
        "curl",
        "--fail",
        "--location",
        "--retry",
        "3",
        "--silent",
        "--show-error",
        "--output",
        str(temporary),
        MUCS_URL,
    ]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if process.returncode != 0:
        message = process.stderr.decode("utf-8", errors="replace").strip()
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"official MUCS download failed: {message}")
    actual_size = temporary.stat().st_size
    if actual_size != MUCS_REQUIRED_BYTES:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"MUCS archive size mismatch: expected {MUCS_REQUIRED_BYTES}, got {actual_size}"
        )
    temporary.replace(destination)
    return destination


def download_sources(
    policy: Mapping[str, Any], data_root: Path, ledger_path: Path
) -> DownloadedSources:
    """Download only official pinned artifacts and atomically update checksums."""

    data_root.mkdir(parents=True, exist_ok=True)
    recorded = _read_ledger(ledger_path)
    files: dict[str, Path] = {}
    for entry in policy["sources"]["fleurs"]["locales"]:
        locale = str(entry["locale"])
        relative = f"sources/fleurs/{locale}/validation/0000.parquet"
        files[relative] = _download_fleurs(
            locale=locale, data_root=data_root, recorded=recorded
        )
    files["sources/mucs/Hindi-English_test.tar.gz"] = _download_mucs(
        data_root, recorded
    )
    checksums: dict[str, str] = {}
    for relative, path in files.items():
        actual = file_sha256(path)
        if relative in recorded and recorded[relative] != actual:
            raise ValueError(
                f"source checksum mismatch for {relative}: "
                f"expected {recorded[relative]}, got {actual}"
            )
        checksums[relative] = actual
    _write_ledger(ledger_path, checksums)
    return DownloadedSources(files=files, checksums=checksums)


class FleursParquetReader:
    """Read lightweight rows globally, decoding audio only for admission probes."""

    def __init__(self, path: Path):
        self.path = path
        self._parquet = pq.ParquetFile(path)
        columns = set(self._parquet.schema_arrow.names)
        required = {"id", "num_samples", "path", "transcription", "audio"}
        if not required.issubset(columns):
            raise ValueError(
                f"FLEURS parquet schema mismatch; missing {sorted(required - columns)}"
            )
        lightweight_columns = [
            name for name in self._parquet.schema_arrow.names if name != "audio"
        ]
        rows: list[dict[str, Any]] = []
        for row_group_index in range(self._parquet.num_row_groups):
            group_rows = self._parquet.read_row_group(
                row_group_index, columns=lightweight_columns
            ).to_pylist()
            for row_index, row in enumerate(group_rows):
                row["_row_group"] = row_group_index
                row["_row_index"] = row_index
                rows.append(row)
        self.rows = rows
        self._audio_cache: dict[int, list[bytes | None]] = {}

    def audio_bytes(self, row: Mapping[str, Any]) -> bytes:
        row_group = int(row["_row_group"])
        if row_group not in self._audio_cache:
            audio_rows = self._parquet.read_row_group(
                row_group, columns=["audio"]
            ).column("audio")
            decoded = audio_rows.to_pylist()
            values: list[bytes | None] = []
            for value in decoded:
                if isinstance(value, Mapping):
                    value = value.get("bytes")
                if not isinstance(value, bytes):
                    values.append(None)
                else:
                    values.append(value)
            self._audio_cache[row_group] = values
        values = self._audio_cache[row_group]
        row_index = int(row["_row_index"])
        value = values[row_index]
        if not isinstance(value, bytes) or not value:
            raise ValueError(f"FLEURS row {row.get('id')} has no embedded audio bytes")
        return value

    def probe(self, row: Mapping[str, Any], decoder: Decoder) -> float:
        samples = decode_bytes(self.audio_bytes(row), decoder)
        expected = int(row["num_samples"])
        if samples.size != expected:
            raise ValueError(
                f"FLEURS row {row.get('id')} duration mismatch: "
                f"expected {expected}, decoded {samples.size}"
            )
        return samples.size / 16_000


def _json_safe_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if not key.startswith("_") and not isinstance(value, (bytes, bytearray))
    }


def resolve_fleurs_sources(
    policy: Mapping[str, Any],
    files: Mapping[str, Path],
    decoder: Decoder,
    policy_digest: str,
) -> list[dict[str, Any]]:
    """Resolve the twelve FLEURS literal identities in numeric-ID order."""

    resolved: list[dict[str, Any]] = []
    sample_number = 1
    for locale_entry in policy["sources"]["fleurs"]["locales"]:
        locale = str(locale_entry["locale"])
        relative = f"sources/fleurs/{locale}/validation/0000.parquet"
        source_path = files[relative]
        reader = FleursParquetReader(source_path)
        selected = select_fleurs_rows(
            reader.rows,
            lambda row: reader.probe(row, decoder),
            limit=int(locale_entry["limit"]),
        )
        source_digest = file_sha256(source_path)
        for item in selected:
            row = item.row
            transcript = str(row["transcription"]).strip()
            source_row_id = str(int(row["id"]))
            member_audio_path = str(row.get("path") or f"id={source_row_id}")
            source_audio_id = PurePosixPath(member_audio_path).stem
            source_id = f"{source_row_id}@{source_audio_id}"
            resolved.append(
                {
                    "schema_version": "1.0.0",
                    "record_status": "resolved",
                    "corpus_id": "pilot-15-v1",
                    "sample_id": f"sample-{sample_number:03d}",
                    "group_id": f"fleurs:{locale}:{source_id}",
                    "split_role": "smoke",
                    "tuning_allowed": False,
                    "bucket": str(locale_entry["bucket"]),
                    "language_profile": str(locale_entry["language_profile"]),
                    "code_mix": bool(locale_entry["code_mix"]),
                    "locale_observed": locale,
                    "source": {
                        "dataset": "Google FLEURS",
                        "release": f"google/fleurs@{FLEURS_REVISION}",
                        "revision": FLEURS_REVISION,
                        "archive_url": (
                            "https://huggingface.co/datasets/google/fleurs/blob/"
                            f"{FLEURS_REVISION}/{locale}/validation/0000.parquet"
                        ),
                        "digest": source_digest,
                        "split": "validation",
                        "source_id": source_id,
                        "member_path": (
                            f"{relative}#row_group={row['_row_group']};"
                            f"row_index={row['_row_index']};id={source_row_id};"
                            f"audio={member_audio_path}"
                        ),
                        "client_id": None,
                        "recording_id": None,
                        "speaker_id": None,
                        "row": _json_safe_row(row),
                        "transcript": transcript,
                        "transcript_sha256": transcript_sha256(transcript),
                        "license_spdx": FLEURS_LICENSE,
                        "attribution": "The FLEURS dataset by Google Research and contributors",
                    },
                    "privacy_decision": {
                        "eligible": True,
                        "reasons": [],
                        "policy_sha256": policy_digest,
                    },
                    "reference_status": "source_reference",
                }
            )
            sample_number += 1
    return resolved


def _archive_lines(bundle: tarfile.TarFile, member_name: str) -> list[str]:
    member = bundle.getmember(member_name)
    if not member.isfile():
        raise ValueError(f"MUCS metadata member is not a file: {member_name}")
    source = bundle.extractfile(member)
    if source is None:
        raise ValueError(f"cannot read MUCS metadata member: {member_name}")
    return source.read().decode("utf-8").splitlines()


def _keyed_lines(
    lines: list[str], *, value_count: int, label: str
) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        fields = line.split(maxsplit=value_count - 1)
        if len(fields) != value_count or not fields[0]:
            raise ValueError(f"malformed {label} line {line_number}")
        if fields[0] in values:
            raise ValueError(f"duplicate {label} ID: {fields[0]}")
        values[fields[0]] = fields
    return values


def _extract_mucs_audio(
    archive: Path, member_name: str, data_root: Path, archive_digest: str
) -> Path:
    relative = PurePosixPath(member_name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"unsafe MUCS member path: {member_name!r}")
    destination = data_root / "cache" / "corpus" / "mucs" / archive_digest / relative
    if destination.is_file():
        return destination
    with tarfile.open(archive, mode="r:gz") as bundle:
        member = bundle.getmember(member_name)
        if not member.isfile():
            raise ValueError(f"MUCS audio member is not a regular file: {member_name}")
        source = bundle.extractfile(member)
        if source is None:
            raise ValueError(f"cannot extract MUCS audio member: {member_name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".part")
        with temporary.open("wb") as output:
            shutil.copyfileobj(source, output)
        if temporary.stat().st_size != member.size:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"MUCS extracted member size mismatch: {member_name}")
        temporary.replace(destination)
    return destination


def resolve_mucs_sources(
    policy: Mapping[str, Any],
    archive: Path,
    data_root: Path,
    decoder: Decoder,
    policy_digest: str,
) -> list[dict[str, Any]]:
    """Resolve three Kaldi segments from the first three eligible recordings."""

    archive_digest = file_sha256(archive)
    with tarfile.open(archive, mode="r:gz") as bundle:
        wav_entries = _keyed_lines(
            _archive_lines(bundle, "test/transcripts/wav.scp"),
            value_count=2,
            label="MUCS wav.scp",
        )
        segment_entries = _keyed_lines(
            _archive_lines(bundle, "test/transcripts/segments"),
            value_count=4,
            label="MUCS segment",
        )
        text_entries = _keyed_lines(
            _archive_lines(bundle, "test/transcripts/text"),
            value_count=2,
            label="MUCS text",
        )
        speaker_entries = _keyed_lines(
            _archive_lines(bundle, "test/transcripts/utt2spk"),
            value_count=2,
            label="MUCS speaker",
        )
    if set(segment_entries) != set(text_entries):
        raise ValueError("MUCS segment/text IDs do not match")
    rows: list[dict[str, Any]] = []
    for utterance_id in sorted(segment_entries):
        _, recording_id, start_raw, end_raw = segment_entries[utterance_id]
        if recording_id not in wav_entries:
            raise ValueError(f"MUCS recording is absent from wav.scp: {recording_id}")
        if utterance_id not in speaker_entries:
            raise ValueError(f"MUCS utterance is absent from utt2spk: {utterance_id}")
        wav_name = wav_entries[recording_id][1]
        member_path = f"test/{wav_name}"
        start_sample = round(float(start_raw) * 16_000)
        end_sample = round(float(end_raw) * 16_000)
        transcript = text_entries[utterance_id][1].strip()
        speaker_id = speaker_entries[utterance_id][1]
        rows.append(
            {
                "utterance_id": utterance_id,
                "recording_id": recording_id,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "start_s": float(start_raw),
                "end_s": float(end_raw),
                "transcript": transcript,
                "member_path": member_path,
                "speaker_id": speaker_id,
            }
        )
    decoded_recordings: dict[str, Any] = {}

    def duration_probe(row: Mapping[str, Any]) -> float | None:
        recording_id = str(row["recording_id"])
        if recording_id not in decoded_recordings:
            member_path = _extract_mucs_audio(
                archive, str(row["member_path"]), data_root, archive_digest
            )
            decoded_recordings[recording_id] = decode_file(member_path, decoder)
        samples = decoded_recordings[recording_id]
        start_sample = int(row["start_sample"])
        end_sample = int(row["end_sample"])
        if end_sample > samples.size:
            raise ValueError(
                f"MUCS segment exceeds recording {recording_id}: "
                f"{end_sample}>{samples.size}"
            )
        segment = samples[start_sample:end_sample]
        validate_pcm16(segment)
        return segment.size / 16_000

    mucs_policy = policy["sources"]["mucs"]
    selected = select_mucs_segments(
        rows,
        duration_probe,
        limit_recordings=int(mucs_policy["limit_recordings"]),
    )
    resolved: list[dict[str, Any]] = []
    for offset, item in enumerate(selected, 13):
        row = item.row
        transcript = str(row["transcript"])
        utterance_id = str(row["utterance_id"])
        recording_id = str(row["recording_id"])
        speaker_id = str(row["speaker_id"])
        resolved.append(
            {
                "schema_version": "1.0.0",
                "record_status": "resolved",
                "corpus_id": "pilot-15-v1",
                "sample_id": f"sample-{offset:03d}",
                "group_id": f"mucs:{recording_id}:{utterance_id}",
                "split_role": "smoke",
                "tuning_allowed": False,
                "bucket": str(mucs_policy["bucket"]),
                "language_profile": str(mucs_policy["language_profile"]),
                "code_mix": bool(mucs_policy["code_mix"]),
                "locale_observed": "hi-en",
                "source": {
                    "dataset": "MUCS 2021 Hindi-English test",
                    "release": "OpenSLR SLR104 Hindi-English_test",
                    "revision": None,
                    "archive_url": MUCS_URL,
                    "digest": archive_digest,
                    "split": "test",
                    "source_id": utterance_id,
                    "member_path": str(row["member_path"]),
                    "client_id": speaker_id,
                    "recording_id": recording_id,
                    "speaker_id": speaker_id,
                    "row": dict(row),
                    "transcript": transcript,
                    "transcript_sha256": transcript_sha256(transcript),
                    "license_spdx": MUCS_LICENSE,
                    "attribution": "MUCS 2021 challenge contributors",
                },
                "privacy_decision": {
                    "eligible": True,
                    "reasons": [],
                    "policy_sha256": policy_digest,
                },
                "reference_status": "source_reference",
            }
        )
    return resolved


def write_third_party_notices(
    path: Path, checksums: Mapping[str, str], decoder_sha256: str
) -> None:
    """Write source/tool attribution bound to the exact downloaded artifacts."""

    ledger_lines = "\n".join(
        f"- `{relative}`: `{checksums[relative]}`"
        for relative in sorted(checksums)
    )
    content = f"""# Third-Party Notices

## Google FLEURS

The `pilot-15-v1` corpus includes 12 English (`en-US-proxy`), Hindi, and Tamil
recordings from The FLEURS dataset by Google Research and contributors. FLEURS
is licensed under Creative Commons Attribution 4.0 International (`CC-BY-4.0`).
The immutable dataset revision is `{FLEURS_REVISION}`. See the FLEURS dataset
card and license at <https://huggingface.co/datasets/google/fleurs>.

## MUCS 2021 Hindi-English Test

The `pilot-15-v1` corpus includes three code-mixed segments from the official
SLR104 Hindi-English test archive. MUCS is licensed under Creative Commons
Attribution-ShareAlike 4.0 International (`CC-BY-SA-4.0`). Source:
<{MUCS_URL}>; dataset description: <https://www.openslr.org/104/>.

## Build Tools

- NumPy 2.2.6, BSD-3-Clause.
- PyArrow 21.0.0, Apache-2.0.
- huggingface-hub 0.36.0, Apache-2.0.
- imageio-ffmpeg 0.6.0, BSD-2-Clause, and its bundled FFmpeg executable.
- Bundled FFmpeg SHA-256: `{decoder_sha256}`. FFmpeg licensing and source
  details are available from <https://ffmpeg.org/legal.html>.

## Pinned Artifact Digests

{ledger_lines}

The generated corpus retains the source attribution and share-alike terms.
Source transcripts are marked `reference_status=source_reference`; this
non-clinical smoke corpus is not clinical gold and does not establish
production medical performance.
"""
    path.write_text(content, encoding="utf-8")

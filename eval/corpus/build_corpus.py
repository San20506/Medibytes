"""CLI for deterministic MediBytes pilot-corpus download/build/verify/rebuild."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from eval.corpus.builder import build_corpus_records
from eval.corpus.manifest import (
    assert_rebuild_identity,
    file_sha256,
    load_manifest_jsonl,
    validate_complete_manifest,
    verify_checksum_ledger,
    write_manifest_jsonl,
)
from eval.corpus.media import bundled_decoder
from eval.corpus.paths import resolve_data_root
from eval.corpus.sources import (
    download_sources,
    load_policy,
    policy_sha256,
    resolve_fleurs_sources,
    resolve_mucs_sources,
    write_third_party_notices,
)

PACKAGE_DIR = Path(__file__).resolve().parent
SELECTION_PATH = PACKAGE_DIR / "selection.json"
MANIFEST_PATH = PACKAGE_DIR / "manifest.jsonl"
CHECKSUM_PATH = PACKAGE_DIR / "source-checksums.sha256"
NOTICES_PATH = PACKAGE_DIR / "THIRD_PARTY_NOTICES.md"
TOOL_FILES = (
    "selection.json",
    "audio.py",
    "builder.py",
    "build_corpus.py",
    "manifest.py",
    "media.py",
    "paths.py",
    "selection.py",
    "sources.py",
)


def corpus_tool_sha256() -> str:
    """Bind the manifest to every executable corpus recipe and the policy."""

    digest = hashlib.sha256()
    for name in TOOL_FILES:
        path = PACKAGE_DIR / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def _resolve_sources(
    policy: Mapping[str, Any],
    source_files: Mapping[str, Path],
    data_root: Path,
    selection_path: Path,
) -> list[dict[str, Any]]:
    decoder = bundled_decoder()
    policy_digest = policy_sha256(selection_path)
    resolved = resolve_fleurs_sources(
        policy, source_files, decoder, policy_digest
    )
    resolved.extend(
        resolve_mucs_sources(
            policy,
            source_files["sources/mucs/Hindi-English_test.tar.gz"],
            data_root,
            decoder,
            policy_digest,
        )
    )
    resolved.sort(key=lambda record: str(record["sample_id"]))
    if len(resolved) != 15 or len({record["group_id"] for record in resolved}) != 15:
        raise ValueError("source resolution did not produce 15 unique literal groups")
    return resolved


def _source_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    source = record["source"]
    return {
        key: record[key]
        for key in (
            "sample_id",
            "group_id",
            "bucket",
            "language_profile",
            "code_mix",
            "locale_observed",
        )
    } | {
        "source": {
            key: source[key]
            for key in (
                "dataset",
                "release",
                "revision",
                "archive_url",
                "digest",
                "split",
                "source_id",
                "member_path",
                "client_id",
                "recording_id",
                "speaker_id",
                "row",
                "transcript",
                "transcript_sha256",
                "license_spdx",
                "attribution",
            )
        },
        "privacy_decision": record["privacy_decision"],
    }


def command_download() -> dict[str, Any]:
    data_root = resolve_data_root()
    policy = load_policy(SELECTION_PATH)
    downloaded = download_sources(policy, data_root, CHECKSUM_PATH)
    resolved = _resolve_sources(
        policy, downloaded.files, data_root, SELECTION_PATH
    )
    if MANIFEST_PATH.exists():
        existing = load_manifest_jsonl(MANIFEST_PATH)
        if existing and all(record.get("record_status") == "complete" for record in existing):
            if [_source_identity(record) for record in existing] != [
                _source_identity(record) for record in resolved
            ]:
                raise ValueError(
                    "deterministic source resolution differs from the complete manifest"
                )
            manifest_status = "complete-manifest-retained"
        else:
            write_manifest_jsonl(MANIFEST_PATH, resolved)
            manifest_status = "resolved-manifest-written"
    else:
        write_manifest_jsonl(MANIFEST_PATH, resolved)
        manifest_status = "resolved-manifest-written"
    decoder = bundled_decoder()
    write_third_party_notices(
        NOTICES_PATH, downloaded.checksums, decoder.sha256
    )
    return {
        "status": "ok",
        "manifest": manifest_status,
        "data_root": str(data_root),
        "sources": len(downloaded.checksums),
        "groups": len(resolved),
        "source_ids": [
            f"{record['source']['dataset']}:{record['source']['source_id']}"
            for record in resolved
        ],
    }


def _absolute_output(raw: str, data_root: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError("--out must be an absolute path")
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(data_root)
    except ValueError as error:
        raise ValueError("--out must stay under MEDIBYTES_EVAL_DATA_ROOT") from error
    if resolved == data_root or resolved == data_root / "sources":
        raise ValueError("--out cannot replace the evaluation data root or sources")
    return resolved


def _relocate_record_paths(
    records: list[dict[str, Any]], staging: Path, output: Path, data_root: Path
) -> list[dict[str, Any]]:
    old_prefix = staging.resolve(strict=False).relative_to(data_root).as_posix()
    new_prefix = output.resolve(strict=False).relative_to(data_root).as_posix()
    for record in records:
        for condition in ("clean", "noisy"):
            raw = str(record[condition]["path"])
            if not raw.startswith(old_prefix + "/"):
                raise ValueError(f"built path is outside staging root: {raw}")
            record[condition]["path"] = new_prefix + raw[len(old_prefix) :]
    return records


def _build_at(
    records: list[Mapping[str, Any]],
    *,
    data_root: Path,
    output: Path,
) -> list[dict[str, Any]]:
    if output.exists():
        raise FileExistsError(f"corpus output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".building")
    if staging.exists():
        raise FileExistsError(f"stale corpus staging root exists: {staging}")
    try:
        built = build_corpus_records(
            records,
            data_root=data_root,
            output_root=staging,
            decoder=bundled_decoder(),
            build_tool_sha256=corpus_tool_sha256(),
        )
        built = _relocate_record_paths(built, staging, output, data_root)
        staging.replace(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    try:
        validate_complete_manifest(built, data_root)
    except Exception:
        shutil.rmtree(output)
        raise
    return built


def command_build(spec_path: Path, output_raw: str) -> dict[str, Any]:
    data_root = resolve_data_root()
    spec = spec_path.resolve(strict=True)
    if spec != SELECTION_PATH.resolve():
        raise ValueError(f"build --spec must be the frozen {SELECTION_PATH} policy")
    policy = load_policy(spec)
    if policy.get("corpus_id") != "pilot-15-v1":
        raise ValueError("selection policy corpus_id must be pilot-15-v1")
    if not MANIFEST_PATH.is_file():
        raise ValueError("run the download command to resolve literal source IDs first")
    records = load_manifest_jsonl(MANIFEST_PATH)
    if not records or any(
        record.get("record_status") not in {"resolved", "complete"}
        for record in records
    ):
        raise ValueError("manifest does not contain resolved literal sources")
    output = _absolute_output(output_raw, data_root)
    built = _build_at(records, data_root=data_root, output=output)
    write_manifest_jsonl(MANIFEST_PATH, built)
    return {
        "status": "ok",
        "data_root": str(data_root),
        "output": str(output),
        "manifest": str(MANIFEST_PATH),
        **validate_complete_manifest(built, data_root),
    }


def _verify_sources(records: list[Mapping[str, Any]], data_root: Path) -> None:
    files: dict[str, Path] = {}
    declared: dict[str, str] = {}
    for record in records:
        source = record["source"]
        if source["dataset"] == "Google FLEURS":
            relative = str(source["member_path"]).split("#", 1)[0]
        elif source["dataset"] == "MUCS 2021 Hindi-English test":
            relative = "sources/mucs/Hindi-English_test.tar.gz"
        else:
            raise ValueError(f"unknown source dataset: {source['dataset']}")
        files[relative] = data_root / relative
        previous = declared.setdefault(relative, str(source["digest"]))
        if previous != source["digest"]:
            raise ValueError(f"manifest contains conflicting digests for {relative}")
    verify_checksum_ledger(CHECKSUM_PATH, data_root, files)
    for relative, digest in declared.items():
        if file_sha256(data_root / relative) != digest:
            raise ValueError(f"source digest changed for {relative}")


def command_verify(manifest_path: Path, data_root_raw: str) -> dict[str, Any]:
    data_root = resolve_data_root(data_root_raw)
    records = load_manifest_jsonl(manifest_path.resolve(strict=True))
    summary = validate_complete_manifest(records, data_root)
    _verify_sources(records, data_root)
    decoder = bundled_decoder()
    tool_hash = corpus_tool_sha256()
    selection_digest = policy_sha256(SELECTION_PATH)
    for record in records:
        if record["privacy_decision"]["policy_sha256"] != selection_digest:
            raise ValueError("manifest selection-policy hash changed")
        build = record["build"]
        if build["tool_sha256"] != tool_hash:
            raise ValueError("corpus build-tool hash changed")
        if build["decoder_sha256"] != decoder.sha256:
            raise ValueError("corpus FFmpeg decoder hash changed")
    if not NOTICES_PATH.is_file() or not CHECKSUM_PATH.is_file():
        raise ValueError("corpus notices or source checksum ledger is missing")
    return {
        "status": "ok",
        "data_root": str(data_root),
        "manifest": str(manifest_path),
        "tool_sha256": tool_hash,
        "decoder_sha256": decoder.sha256,
        **summary,
    }


def command_rebuild(output_raw: str) -> dict[str, Any]:
    data_root = resolve_data_root()
    records = load_manifest_jsonl(MANIFEST_PATH)
    if len(records) != 15 or any(
        record.get("record_status") != "complete" for record in records
    ):
        raise ValueError("rebuild requires a complete 15-record manifest")
    output = _absolute_output(output_raw, data_root)
    rebuilt = _build_at(records, data_root=data_root, output=output)
    assert_rebuild_identity(records, rebuilt)
    rebuild_manifest = output / "rebuild-manifest.jsonl"
    write_manifest_jsonl(rebuild_manifest, rebuilt)
    return {
        "status": "ok",
        "data_root": str(data_root),
        "output": str(output),
        "manifest": str(rebuild_manifest),
        "rebuild_identity": "byte-identical",
        **validate_complete_manifest(rebuilt, data_root),
    }


def parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(prog="eval.corpus.build_corpus")
    subcommands = command_parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("download")
    build = subcommands.add_parser("build")
    build.add_argument("--spec", type=Path, required=True)
    build.add_argument("--out", required=True)
    verify = subcommands.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--data-root", required=True)
    rebuild = subcommands.add_parser("rebuild")
    rebuild.add_argument("--out", required=True)
    return command_parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "download":
            result = command_download()
        elif arguments.command == "build":
            result = command_build(arguments.spec, arguments.out)
        elif arguments.command == "verify":
            result = command_verify(arguments.manifest, arguments.data_root)
        elif arguments.command == "rebuild":
            result = command_rebuild(arguments.out)
        else:
            raise AssertionError(f"unhandled command: {arguments.command}")
    except (OSError, RuntimeError, ValueError, AssertionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

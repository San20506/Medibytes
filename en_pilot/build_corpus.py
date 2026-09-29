"""Build the 40-base English-only corpus through the frozen corpus primitives.

`eval/corpus/build_corpus.py` cannot build this corpus: it hard-rejects any spec
whose `corpus_id` is not `pilot-15-v1` (`eval/corpus/build_corpus.py:235-237`),
requires exactly 15 resolved groups (`:84-85`), and its manifest record shape is
pinned by `eval/contracts/*.schema.json`. This module therefore calls the frozen
primitives directly — the same privacy gate, duration window, decoder and
canonicalisation the frozen corpus uses — so the English bases it admits are
admitted by identical rules, and `sample-001`..`sample-004` are byte-identical
to the frozen corpus's first four bases.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

from en_pilot import config
from eval.corpus.manifest import file_sha256, transcript_sha256, write_manifest_jsonl
from eval.corpus.media import (
    bundled_decoder,
    canonicalize_clean,
    decode_bytes,
)
from eval.corpus.paths import resolve_data_root
from eval.corpus.selection import select_fleurs_rows
from eval.corpus.sources import FleursParquetReader


class CorpusError(RuntimeError):
    """The English corpus cannot be built as specified."""


def corpus_root(data_root: Path) -> Path:
    return data_root / "corpus" / config.CORPUS_ID


def build(data_root: Path, *, bases: int = config.BASE_COUNT) -> dict[str, Any]:
    """Decode, canonicalize and admit `bases` English FLEURS bases."""

    if bases <= 0:
        raise CorpusError("base count must be positive")
    parquet_path = data_root / config.FLEURS_PARQUET_RELATIVE
    if not parquet_path.is_file():
        raise CorpusError(f"FLEURS en_us validation parquet is absent: {parquet_path}")

    decoder = bundled_decoder()
    reader = FleursParquetReader(parquet_path)
    # Selection runs to completion before anything is written, so an
    # under-populated split leaves no half-built corpus behind.
    selected = select_fleurs_rows(
        reader.rows,
        lambda row: reader.probe(row, decoder),
        limit=bases,
    )
    if len(selected) != bases:
        raise CorpusError(f"selected {len(selected)} bases, required {bases}")

    output = corpus_root(data_root)
    if output.exists():
        raise CorpusError(f"corpus output already exists: {output}")
    staging = output.with_name(output.name + ".building")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    records: list[dict[str, Any]] = []
    try:
        for index, entry in enumerate(selected, 1):
            records.append(_build_base(reader, entry, decoder, staging, index))
        write_manifest_jsonl(staging / "manifest.jsonl", records)
        (staging / "sources.json").write_text(
            json.dumps(
                {
                    "corpus_id": config.CORPUS_ID,
                    "parquet_path": str(parquet_path),
                    "parquet_sha256": file_sha256(parquet_path),
                    "parquet_rows_available": len(reader.rows),
                    "source_repo": config.FLEURS_REPO,
                    "source_revision": config.FLEURS_REVISION,
                    "split": config.FLEURS_SPLIT,
                    "license": config.FLEURS_LICENSE,
                    "python_version": sys.version,
                    "ffmpeg_provider": decoder.provider,
                    "ffmpeg_executable": str(decoder.executable),
                    "ffmpeg_sha256": decoder.sha256,
                    "audio_primitives": "eval/corpus/media.py + eval/corpus/audio.py",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "status": "ok",
        "bases": len(records),
        "clean_dir": str(output / "clean"),
        "manifest": str(output / "manifest.jsonl"),
        "sources": str(output / "sources.json"),
        "parquet_sha256": file_sha256(parquet_path),
    }


def _build_base(
    reader: FleursParquetReader,
    entry: Any,
    decoder: Any,
    staging: Path,
    index: int,
) -> dict[str, Any]:
    row = entry.row
    sample_id = f"sample-{index:03d}"
    target = staging / "clean" / f"{sample_id}.wav"
    decoded = decode_bytes(reader.audio_bytes(row), decoder)
    metrics = canonicalize_clean(decoded, target)
    transcript = str(row["transcription"])
    return {
        "sample_id": sample_id,
        "bucket": config.BUCKET,
        "language_profile": config.LANGUAGE_PROFILE,
        "code_mix": False,
        "fleurs_id": row["id"],
        # `fleurs_id` repeats across speakers, so the exact source row is only
        # recoverable with its recording path and row coordinates.
        "fleurs_path": row["path"],
        "row_group": int(row["_row_group"]),
        "row_index": int(row["_row_index"]),
        "speaker_id": Path(str(row["path"])).stem,
        "speaker_gender": row.get("gender"),
        "source_repo": config.FLEURS_REPO,
        "source_revision": config.FLEURS_REVISION,
        "split": config.FLEURS_SPLIT,
        "license": config.FLEURS_LICENSE,
        "transcript": transcript,
        "transcript_sha256": transcript_sha256(transcript),
        "privacy_reasons": [],
        "clean_path": str(target.relative_to(staging)),
        "clean_sha256": file_sha256(target),
        "sample_count": int(metrics["sample_count"]),
        "sample_rate": config.SAMPLE_RATE,
        "duration_s": float(metrics["duration_s"]),
        "peak_dbfs": float(metrics["peak_dbfs"]),
        "source_peak": float(metrics["source_peak"]),
        "normalization_gain": float(metrics["normalization_gain"]),
        "privacy_reasons": [],
    }




def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="en_pilot.build_corpus",
        description="Build the 40-base English-only pilot corpus.",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get(config.ENVIRONMENT_NAME, config.DATA_ROOT_DEFAULT),
    )
    parser.add_argument("--bases", type=int, default=config.BASE_COUNT)
    arguments = parser.parse_args(argv)
    data_root = resolve_data_root(arguments.data_root)
    print(json.dumps(build(data_root, bases=arguments.bases), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

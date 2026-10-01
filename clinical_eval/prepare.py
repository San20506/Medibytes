"""Stage the MediBytes 7-clip dataset into the eval data root.

The dataset ships 48 kHz stereo AAC in MP4 containers. `demo/audio_clean.py`
ingests that directly through its ffmpeg fallback, so conversion is not required
for the pipeline to run — it is done here only so every arm reads byte-identical
PCM and the decode step cannot vary between runs.

The two channels are **not** a dual-mono copy (measured correlation ~0.50 with
different peaks), so each channel is also written separately. Whether the
pipeline's downmix costs accuracy against the better single channel is then a
measurement rather than an assumption.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

CORPUS_ID = "medibytes-med-7"
DATA_ROOT_DEFAULT = Path("/home/sandy/.local/share/medibytes-eval")
SAMPLE_RATE = 16_000
CHANNEL_MODES = ("mix", "left", "right")


class PrepareError(RuntimeError):
    """The clinical corpus cannot be staged as specified."""


def corpus_root(data_root: Path) -> Path:
    return data_root / "corpus" / CORPUS_ID


def _ffmpeg(args: Sequence[str]) -> bytes:
    result = subprocess.run(["ffmpeg", "-v", "error", *args], capture_output=True)
    if result.returncode != 0:
        raise PrepareError(f"ffmpeg failed: {result.stderr.decode()[:400]}")
    return result.stdout


def _convert(source: Path, destination: Path, mode: str) -> dict[str, Any]:
    """One 16 kHz mono PCM16 WAV per channel mode, recorded with its command."""

    pan = {
        "mix": "pan=mono|c0=0.5*c0+0.5*c1",
        "left": "pan=mono|c0=c0",
        "right": "pan=mono|c0=c1",
    }[mode]
    args = ["-y", "-i", str(source), "-af", pan, "-ar", str(SAMPLE_RATE),
            "-ac", "1", "-c:a", "pcm_s16le", str(destination)]
    _ffmpeg(args)
    return {"mode": mode, "ffmpeg_filter": pan, "command": "ffmpeg -v error " + " ".join(args)}


def _channel_stats(source: Path) -> dict[str, Any]:
    raw = _ffmpeg(["-i", str(source), "-f", "s16le", "-acodec", "pcm_s16le",
                   "-ar", "48000", "-ac", "2", "-"])
    stereo = np.frombuffer(raw, dtype="<i2").astype(np.float64).reshape(-1, 2)
    left, right = stereo[:, 0], stereo[:, 1]
    return {
        "lr_correlation": float(np.corrcoef(left, right)[0, 1]),
        "identical_channels": bool(np.array_equal(left, right)),
        "peak_left": float(np.abs(left).max()),
        "peak_right": float(np.abs(right).max()),
        "rms_left": float(np.sqrt(np.mean(left ** 2))),
        "rms_right": float(np.sqrt(np.mean(right ** 2))),
    }


def prepare(zip_path: Path, data_root: Path) -> dict[str, Any]:
    root = corpus_root(data_root)
    if root.exists():
        shutil.rmtree(root)
    staging = root / "_src"
    staging.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(staging)
    base = staging / "MediBytes_Medical_Audio_Dataset"
    if not base.is_dir():
        raise PrepareError(f"unexpected archive layout under {staging}")

    sources = sorted((base / "recorded_samples" / "clean").glob("*.mp4"))
    if not sources:
        raise PrepareError("no clips under recorded_samples/clean")

    # Record what the archive promises but does not contain, so a later reader
    # does not assume the noisy conditions were tested and found equivalent.
    empty: list[str] = []
    for relative in ("recorded_samples/medium_noise", "recorded_samples/heavy_noise",
                     "external_clips/audio", "external_clips/transcripts"):
        directory = base / relative
        if not directory.is_dir() or not any(directory.iterdir()):
            empty.append(relative)
    metadata_csv = base / "metadata.csv"
    metadata_bytes = metadata_csv.stat().st_size if metadata_csv.is_file() else -1

    records: list[dict[str, Any]] = []
    for mode in CHANNEL_MODES:
        (root / mode).mkdir(parents=True, exist_ok=True)
    for source in sources:
        clip_id = source.stem
        transcript_path = base / "transcripts" / f"{clip_id}.txt"
        if not transcript_path.is_file():
            raise PrepareError(f"no reference transcript for {clip_id}")
        record: dict[str, Any] = {
            "clip_id": clip_id,
            "source_mp4": str(source),
            "reference_text": transcript_path.read_text(encoding="utf-8").strip(),
            "channels": _channel_stats(source),
            "conversions": [],
        }
        for mode in CHANNEL_MODES:
            destination = root / mode / f"{clip_id}.wav"
            record["conversions"].append(_convert(source, destination, mode))
            record[f"wav_{mode}"] = str(destination)
        records.append(record)

    manifest = {
        "corpus_id": CORPUS_ID,
        "source_zip": str(zip_path),
        "sample_rate": SAMPLE_RATE,
        "clips": records,
        "archive_gaps": {
            "empty_directories": empty,
            "metadata_csv_bytes": metadata_bytes,
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "corpus_root": str(root),
        "clips": len(records),
        "channel_modes": list(CHANNEL_MODES),
        "empty_directories": empty,
        "metadata_csv_bytes": metadata_bytes,
    }


def load_manifest(data_root: Path) -> dict[str, Any]:
    path = corpus_root(data_root) / "manifest.json"
    if not path.is_file():
        raise PrepareError(f"corpus is not staged: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clinical_eval.prepare")
    parser.add_argument(
        "--zip",
        default="/home/sandy/Downloads/MediBytes_Medical_Audio_Dataset.zip",
    )
    parser.add_argument("--data-root", default=str(DATA_ROOT_DEFAULT))
    arguments = parser.parse_args(argv)
    summary = prepare(Path(arguments.zip), Path(arguments.data_root))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

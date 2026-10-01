"""Transcribe the already-enhanced pilot conditions with `google/medasr`.

This is `en_pilot.transcribe`'s sibling for a decoder that is not Whisper and
not CTranslate2, so none of that module's device probe applies: MedASR is a
CTC model driven by torch, and `torch.cuda.is_available()` is the honest test.

The audio is not re-enhanced.  Every arm reads the exact PCM16 files the n=40
study already wrote under `en-pilot-run/enhance-<backend>/`, so a MedASR row and
a Whisper row for the same `(condition_id, backend)` are the same bytes decoded
twice.  Hypotheses go through `demo.stt_extract.normalize_text`, the same
function `en_pilot.transcribe` uses, so the two decoders are scored by one rule.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from en_pilot import config
from en_pilot.enhance import read_wav_mono16
from eval.corpus.paths import resolve_data_root

DECODER_ID = "medasr"
MODEL_ID = "google/medasr"
DEFAULT_ARMS = ("none", "sherpa-gtcrn-simple")


class TranscribeError(RuntimeError):
    """MedASR transcription cannot be run as specified."""


def load_jobs(data_root: Path, arms: Sequence[str]) -> list[dict[str, Any]]:
    """One job per (matrix condition, arm) plus one clean floor per base.

    Identical job set to `en_pilot.transcribe.load_jobs`, restricted to the arms
    this comparison needs, so the scored cells line up row for row.
    """

    manifest = data_root / config.MATRIX_DIR / "noise-matrix.jsonl"
    if not manifest.is_file():
        raise TranscribeError(f"noise matrix manifest is absent: {manifest}")
    conditions = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    jobs: list[dict[str, Any]] = []
    for condition in conditions:
        condition_id = str(condition["condition_id"])
        for arm in arms:
            path = data_root / config.OUT_DIR / f"enhance-{arm}" / f"{condition_id}.wav"
            if not path.is_file():
                raise TranscribeError(f"enhanced audio is absent: {path}")
            jobs.append(
                {
                    "condition_id": condition_id,
                    "backend": arm,
                    "source": "matrix",
                    "path": str(path),
                }
            )
    clean_root = data_root / config.CLEAN_DIR
    bases = sorted(clean_root.glob("*.wav"))
    if not bases:
        raise TranscribeError(f"no clean bases found in {clean_root}")
    for base in bases:
        jobs.append(
            {
                "condition_id": base.stem,
                "backend": config.BASELINE_BACKEND,
                "source": "clean",
                "path": str(base),
            }
        )
    return jobs


def completed_keys(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("decoder") == DECODER_ID and not record.get("error"):
            done.add(f"{record['condition_id']}|{record['backend']}")
    return done


def build_pipeline(device: str) -> Any:
    import torch
    from transformers import pipeline as hf_pipeline

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise TranscribeError("cuda requested but torch reports no CUDA device")
    index = 0 if device == "cuda" else -1
    return hf_pipeline("automatic-speech-recognition", model=MODEL_ID, device=index), device


def run(
    data_root: Path, *, arms: Sequence[str], device: str, batch_size: int
) -> dict[str, Any]:
    from demo.stt_extract import medasr_detokenize, normalize_text

    jobs = load_jobs(data_root, arms)
    out_path = data_root / config.OUT_DIR / f"transcripts-{DECODER_ID}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_keys(out_path)
    pending = [
        job for job in jobs if f"{job['condition_id']}|{job['backend']}" not in done
    ]
    print(
        f"medasr: {len(pending)}/{len(jobs)} clips to transcribe "
        f"(arms: {', '.join(arms)})",
        flush=True,
    )
    if not pending:
        return {
            "decoder": DECODER_ID,
            "model": MODEL_ID,
            "transcribed": 0,
            "failed": 0,
            "output": str(out_path),
        }

    pipe, resolved_device = build_pipeline(device)
    sink = out_path.open("a", encoding="utf-8")
    written = 0
    failed = 0
    errors: list[str] = []
    started = time.perf_counter()
    audio_seconds = 0.0

    # The HF pipeline batches a list of arrays in one call; a batch that raises
    # is retried one clip at a time so a single bad file cannot lose 31 others.
    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        arrays = [
            np.asarray(read_wav_mono16(Path(job["path"])), dtype=np.float32)
            for job in chunk
        ]
        audio_seconds += sum(a.size for a in arrays) / config.SAMPLE_RATE
        try:
            outputs = pipe(list(arrays), batch_size=len(arrays))
            texts = [str(item["text"]) for item in outputs]
            failures = [""] * len(chunk)
        except Exception as batch_error:  # fall back to one clip at a time
            texts, failures = [], []
            for array in arrays:
                try:
                    texts.append(str(pipe(array)["text"]))
                    failures.append("")
                except Exception as error:
                    texts.append("")
                    failures.append(f"{type(error).__name__}: {error}")
            del batch_error
        for job, raw_text, failure in zip(chunk, texts, failures):
            # `raw_text` keeps MedASR's dictation markup so a scoring question
            # can be re-asked later without re-running the model.
            text = medasr_detokenize(raw_text) if raw_text else ""
            record = {
                "condition_id": job["condition_id"],
                "backend": job["backend"],
                "source": job["source"],
                "decoder": DECODER_ID,
                "model": MODEL_ID,
                "device": resolved_device,
                "raw_text": raw_text,
                "text": text,
                "normalized": normalize_text(text)["normalized_en"] if text else "",
                "error": failure,
            }
            if failure:
                failed += 1
                errors.append(f"{job['backend']}/{job['condition_id']}: {failure}")
            sink.write(json.dumps(record, sort_keys=True) + "\n")
            written += 1
        sink.flush()
        if written % 200 < batch_size:
            elapsed = time.perf_counter() - started
            print(
                f"  {written}/{len(pending)} "
                f"({elapsed:.0f}s, rtf {elapsed / max(audio_seconds, 1e-9):.3f})",
                flush=True,
            )
    sink.close()
    if errors:
        (data_root / config.OUT_DIR / f"transcribe-errors-{DECODER_ID}.log").write_text(
            "\n".join(errors), encoding="utf-8"
        )
    elapsed = time.perf_counter() - started
    return {
        "decoder": DECODER_ID,
        "model": MODEL_ID,
        "device": resolved_device,
        "batch_size": batch_size,
        "transcribed": written,
        "failed": failed,
        "wall_s": round(elapsed, 1),
        "audio_s": round(audio_seconds, 1),
        "rtf": round(elapsed / max(audio_seconds, 1e-9), 4),
        "output": str(out_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="en_pilot.transcribe_medasr",
        description="Transcribe the enhanced pilot conditions with google/medasr.",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get(config.ENVIRONMENT_NAME, config.DATA_ROOT_DEFAULT),
    )
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--batch-size", type=int, default=1)
    arguments = parser.parse_args(argv)
    arms = tuple(part.strip() for part in arguments.arms.split(",") if part.strip())
    if config.BASELINE_BACKEND not in arms:
        raise TranscribeError("the `none` baseline arm is required for a paired delta")
    summary = run(
        resolve_data_root(arguments.data_root),
        arms=arms,
        device=arguments.device,
        batch_size=max(1, arguments.batch_size),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

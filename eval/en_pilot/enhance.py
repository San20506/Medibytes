"""Enhance every matrix condition with exactly one backend, in its own process.

One backend per invocation, so a run never mixes adapters. The earlier matrix evaluator
(`evidence/matrix-v2/matrix_eval.py`) had no `actual_backend` column, so a mis-dispatch would
have been invisible in its results; this module records `actual_backend`/`variant_id`, read off
the returned `EnhancementOutput`, for every row so one is visible in the artefact.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from eval.en_pilot import config
from eval.corpus.paths import resolve_data_root

CSV_COLUMNS = (
    "condition_id",
    "speech",
    "backend",
    "requested_backend",
    "actual_backend",
    "variant_id",
    "samples",
    "enhance_s",
    "enhanced_path",
    "peak",
    "error",
)


class EnhanceError(RuntimeError):
    """Enhancement cannot be run as specified."""


@dataclass(frozen=True)
class MethodSpec:
    backend: str
    variant_id: str
    parameters: dict[str, Any]
    model_path: Path | None
    model_sha256: str | None
    timeout_s: float


def read_wav_mono16(path: Path) -> np.ndarray:
    """Read a canonical 16 kHz mono PCM16 WAV into float in [-1, 1)."""

    raw = path.read_bytes()
    if len(raw) < 44:
        raise EnhanceError(f"truncated WAV: {path}")
    channels = struct.unpack_from("<H", raw, 22)[0]
    rate = struct.unpack_from("<I", raw, 24)[0]
    if rate != config.SAMPLE_RATE or channels != 1:
        raise EnhanceError(
            f"{path} is {channels}ch at {rate} Hz, expected mono at "
            f"{config.SAMPLE_RATE} Hz"
        )
    return np.frombuffer(raw, dtype="<i2", offset=44).astype(np.float32) / 32768.0


def write_wav_mono16(path: Path, samples: np.ndarray) -> None:
    clipped = np.clip(
        np.asarray(samples, dtype=np.float64), -1.0, config.CLIPPING_CEILING
    )
    pcm = np.rint(clipped * 32768.0).astype("<i2")
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + pcm.nbytes,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        config.SAMPLE_RATE,
        config.SAMPLE_RATE * 2,
        2,
        16,
        b"data",
        pcm.nbytes,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + pcm.tobytes())


def load_method_spec(backend: str, data_root: Path) -> MethodSpec:
    """Reconstruct the frozen EnhancementConfig request from the method config."""

    path = config.METHOD_CONFIG_ROOT / f"{backend}.json"
    if not path.is_file():
        raise EnhanceError(f"method config does not exist: {path}")
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("backend") != backend:
        raise EnhanceError(f"{path} declares backend {spec.get('backend')!r}")
    raw_model_path = spec.get("model_path")
    model_path: Path | None = None
    if raw_model_path:
        model_path = data_root / raw_model_path
        if not model_path.is_file():
            # Never silently substitute a different model file.
            raise EnhanceError(f"pinned model file is absent: {model_path}")
    return MethodSpec(
        backend=backend,
        variant_id=str(spec["variant_id"]),
        parameters=dict(spec.get("parameters") or {}),
        model_path=model_path,
        model_sha256=spec.get("model_sha256"),
        timeout_s=float(spec.get("timeout_s") or 30.0),
    )


def check_backend(backend: str) -> None:
    """Fail closed before any work on an id the study does not run."""

    if backend not in config.BACKENDS:
        raise EnhanceError(
            f"unknown backend {backend!r}; the study runs exactly {list(config.BACKENDS)}"
        )


def enhance_one(
    condition: dict[str, Any],
    spec: MethodSpec,
    output_dir: Path,
    denoise: Any,
) -> dict[str, Any]:
    from base_pipeline.denoise import BackendId, EnhancementConfig

    request = EnhancementConfig(
        backend=BackendId(spec.backend),
        variant_id=spec.variant_id,
        model_path=spec.model_path,
        model_sha256=spec.model_sha256,
        parameters=spec.parameters,
        timeout_s=spec.timeout_s,
    )
    condition_id = str(condition["condition_id"])
    target = output_dir / f"{condition_id}.wav"
    started = time.perf_counter()
    try:
        audio = read_wav_mono16(Path(condition["output_path"]))
        result = denoise.run_backend(audio, config.SAMPLE_RATE, request)
        enhanced = np.asarray(result.audio, dtype=np.float32)
        if enhanced.size != audio.size:
            raise EnhanceError(
                f"{condition_id}: enhanced {enhanced.size} samples from "
                f"{audio.size}; the reference would not align"
            )
        write_wav_mono16(target, enhanced)
        return {
            "condition_id": condition_id,
            "speech": condition["speech"],
            "backend": spec.backend,
            "requested_backend": spec.backend,
            "actual_backend": str(result.actual_backend),
            "variant_id": str(result.variant_id),
            "samples": int(enhanced.size),
            "enhance_s": round(time.perf_counter() - started, 4),
            "enhanced_path": str(target),
            "peak": round(float(np.max(np.abs(enhanced))), 6),
            "error": "",
        }
    except Exception as error:  # recorded, never silently dropped
        return {
            "condition_id": condition_id,
            "speech": condition["speech"],
            "backend": spec.backend,
            "requested_backend": spec.backend,
            "actual_backend": "",
            "variant_id": "",
            "samples": 0,
            "enhance_s": round(time.perf_counter() - started, 4),
            "enhanced_path": "",
            "peak": "",
            "error": f"{type(error).__name__}: {error}",
        }


def load_conditions(data_root: Path) -> list[dict[str, Any]]:
    manifest = data_root / config.MATRIX_DIR / "noise-matrix.jsonl"
    if not manifest.is_file():
        raise EnhanceError(f"noise matrix manifest is absent: {manifest}")
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise EnhanceError(f"noise matrix manifest is empty: {manifest}")
    return rows


def run(data_root: Path, backend: str, *, workers: int) -> dict[str, Any]:
    check_backend(backend)
    from base_pipeline import denoise

    spec = load_method_spec(backend, data_root)
    conditions = load_conditions(data_root)
    output_dir = data_root / config.OUT_DIR / f"enhance-{backend}"
    csv_path = data_root / config.OUT_DIR / f"enhance-{backend}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for index, row in enumerate(
            pool.map(
                lambda condition: enhance_one(condition, spec, output_dir, denoise),
                conditions,
            ),
            1,
        ):
            rows.append(row)
            if index % 100 == 0:
                print(f"  enhanced {index}/{len(conditions)}", flush=True)
    elapsed = time.perf_counter() - started

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    failed = [row for row in rows if row["error"]]
    mismatched = [
        row for row in rows if not row["error"] and row["actual_backend"] != backend
    ]
    return {
        "backend": backend,
        "variant_id": spec.variant_id,
        "conditions": len(rows),
        "failed": len(failed),
        "actual_backend_mismatches": len(mismatched),
        "elapsed_s": round(elapsed, 2),
        "csv": str(csv_path),
        "enhanced_dir": str(output_dir),
        "errors": [f"{row['condition_id']}: {row['error']}" for row in failed[:10]],
        "mismatches": [row["condition_id"] for row in mismatched[:10]],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval.en_pilot.enhance",
        description="Enhance every matrix condition with one pinned backend.",
    )
    parser.add_argument("--backend", required=True, choices=list(config.BACKENDS))
    parser.add_argument(
        "--data-root",
        default=os.environ.get(config.ENVIRONMENT_NAME, config.DATA_ROOT_DEFAULT),
    )
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args(argv)
    data_root = resolve_data_root(arguments.data_root)
    summary = run(data_root, arguments.backend, workers=arguments.workers)
    print(json.dumps(summary, sort_keys=True))
    return 1 if summary["failed"] or summary["actual_backend_mismatches"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Measure the two variants through the *shipped* chain, not the eval harness.

`en_pilot.transcribe_medasr` decodes the pilot's pre-enhanced files, which is the
right way to isolate an enhancement backend but is not what `demo/pipeline.py`
does. The demo calls `demo.audio_clean.clean_audio`, which RMS-normalises the
signal to 0.1 before handing it to the backend, resamples and validates after,
and can refuse a clip outright (`422 NO_AUDIO`). A 5-clip spot check showed that
normalisation alone recovers most of the undenoised arm's loss, so the harness
delta and the product delta are not the same quantity.

This module therefore runs the real thing: `clean_audio` with the variant's own
backend config, then `demo.stt_extract.transcribe(..., model="medasr", strict=True)`,
over the whole matrix. A refused clip is recorded as an outcome, never dropped.

Enhancement is CPU-bound and parallel; decoding is GPU-bound and sequential, so
the two phases are split and the decode phase runs in one process to load MedASR
once.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from en_pilot import config
from eval.corpus.paths import resolve_data_root

DECODER_ID = "medasr-pipeline"
MODEL_ID = "google/medasr"
# The adapter fails closed on every one of these, so the variant is pinned here
# exactly as `demo/denoise_backends/sherpa-gtcrn-simple.config.json` pins it.
BACKEND_CONFIGS: dict[str, dict[str, Any]] = {
    "none": {"variant_id": "none"},
    "sherpa-gtcrn-simple": {
        "variant_id": "gtcrn-simple-1.13.8",
        "model_path": (
            "/home/sandy/.local/share/medibytes-eval/models/sherpa-onnx/gtcrn_simple.onnx"
        ),
        "model_sha256": (
            "e77603ac0c23dac3227dd2d7135b3a585cbee2679048aecfa886657d3ae1b534"
        ),
        "parameters": {"sample_rate": 16000, "provider": "cpu", "num_threads": 1},
    },
}


class PipelineRunError(RuntimeError):
    """The shipped-chain sweep cannot be run as specified."""


def _clean_one(task: tuple[str, str, str, str]) -> dict[str, Any]:
    """Run `clean_audio` for one (clip, arm). Returns the outcome, never raises."""

    from demo.audio_clean import BackendId, EnhancementConfig, clean_audio

    condition_id, arm, in_path, out_path = task
    raw = dict(BACKEND_CONFIGS[arm])
    if "model_path" in raw:
        raw["model_path"] = Path(raw["model_path"])
    backend = BackendId(arm)
    started = time.perf_counter()
    try:
        clean_audio(
            Path(in_path),
            Path(out_path),
            backend=backend,
            config=EnhancementConfig(backend=backend, **raw),
        )
    except Exception as error:
        return {
            "condition_id": condition_id,
            "backend": arm,
            "clean_error": f"{type(error).__name__}: {error}",
            "clean_s": round(time.perf_counter() - started, 4),
        }
    return {
        "condition_id": condition_id,
        "backend": arm,
        "clean_error": "",
        "clean_s": round(time.perf_counter() - started, 4),
    }


def _tasks(data_root: Path, arms: Sequence[str], work_root: Path) -> list[tuple[str, str, str, str]]:
    manifest = data_root / config.MATRIX_DIR / "noise-matrix.jsonl"
    if not manifest.is_file():
        raise PipelineRunError(f"noise matrix manifest is absent: {manifest}")
    sources: list[tuple[str, Path]] = [
        (str(json.loads(line)["condition_id"]), data_root / config.MATRIX_DIR / f"{json.loads(line)['condition_id']}.wav")
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    clean_root = data_root / config.CLEAN_DIR
    sources.extend((path.stem, path) for path in sorted(clean_root.glob("*.wav")))
    tasks: list[tuple[str, str, str, str]] = []
    for condition_id, source in sources:
        for arm in arms:
            out = work_root / arm / f"{condition_id}.wav"
            out.parent.mkdir(parents=True, exist_ok=True)
            tasks.append((condition_id, arm, str(source), str(out)))
    return tasks


def run(
    data_root: Path, *, arms: Sequence[str], workers: int, work_root: Path
) -> dict[str, Any]:
    from demo.stt_extract import normalize_text, transcribe

    clean_bases = {path.stem for path in sorted((data_root / config.CLEAN_DIR).glob("*.wav"))}
    tasks = _tasks(data_root, arms, work_root)
    print(f"{DECODER_ID}: cleaning {len(tasks)} (clip, arm) pairs on {workers} workers", flush=True)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        cleaned = list(pool.map(_clean_one, tasks, chunksize=8))
    refused = sum(1 for row in cleaned if row["clean_error"])
    print(
        f"  cleaned in {time.perf_counter() - started:.0f}s; {refused} refused by clean_audio",
        flush=True,
    )

    out_path = data_root / config.OUT_DIR / f"transcripts-{DECODER_ID}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sink = out_path.open("w", encoding="utf-8")
    started = time.perf_counter()
    failed = 0
    for index, (task, outcome) in enumerate(zip(tasks, cleaned), start=1):
        condition_id, arm, _, cleaned_path = task
        record: dict[str, Any] = {
            "condition_id": condition_id,
            "backend": arm,
            "source": "clean" if condition_id in clean_bases else "matrix",
            "decoder": DECODER_ID,
            "model": MODEL_ID,
            "clean_error": outcome["clean_error"],
            "clean_s": outcome["clean_s"],
        }
        if outcome["clean_error"]:
            # A refused clip is an outcome of the variant, not a missing row: the
            # product would have produced no transcript at all here.
            record.update(raw_text="", text="", normalized="", error=outcome["clean_error"])
            failed += 1
        else:
            try:
                result = transcribe(cleaned_path, job_id=condition_id, model="medasr", strict=True)
                record.update(
                    raw_text=result["raw_text"],
                    text=result["text"],
                    normalized=normalize_text(result["text"])["normalized_en"],
                    error="",
                )
            except Exception as error:
                record.update(raw_text="", text="", normalized="", error=f"{type(error).__name__}: {error}")
                failed += 1
        sink.write(json.dumps(record, sort_keys=True) + "\n")
        if index % 400 == 0:
            sink.flush()
            print(f"  decoded {index}/{len(tasks)} ({time.perf_counter() - started:.0f}s)", flush=True)
    sink.close()
    return {
        "decoder": DECODER_ID,
        "model": MODEL_ID,
        "arms": list(arms),
        "pairs": len(tasks),
        "refused_by_clean_audio": refused,
        "no_transcript": failed,
        "decode_s": round(time.perf_counter() - started, 1),
        "output": str(out_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="en_pilot.transcribe_pipeline",
        description="Measure the two variants through demo/pipeline.py's own chain.",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get(config.ENVIRONMENT_NAME, config.DATA_ROOT_DEFAULT),
    )
    parser.add_argument("--arms", default="none,sherpa-gtcrn-simple")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument(
        "--work-root",
        default=None,
        help="where cleaned audio is written (default: a temporary directory)",
    )
    arguments = parser.parse_args(argv)
    arms = tuple(part.strip() for part in arguments.arms.split(",") if part.strip())
    unknown = set(arms) - set(BACKEND_CONFIGS)
    if unknown:
        raise PipelineRunError(f"no pinned backend config for {sorted(unknown)}")
    if config.BASELINE_BACKEND not in arms:
        raise PipelineRunError("the `none` baseline arm is required for a paired delta")
    data_root = resolve_data_root(arguments.data_root)
    if arguments.work_root:
        summary = run(
            data_root, arms=arms, workers=arguments.workers, work_root=Path(arguments.work_root)
        )
    else:
        with tempfile.TemporaryDirectory(prefix="medasr-pipeline-") as scratch:
            summary = run(data_root, arms=arms, workers=arguments.workers, work_root=Path(scratch))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

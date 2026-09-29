"""Transcribe every enhanced condition with one pinned Whisper decoder tier.

The device probe is `evidence/matrix-v2/matrix_eval.py:172-174`'s, verbatim, and
the reasoning is load-bearing: CTranslate2 drives Whisper and carries its own
CUDA runtime, so a CPU-only torch in the driver venv says nothing about whether
GPU decoding is available. The resolved `device`/`compute_type` is recorded on
every row, so the numbers are attributable to the tier that produced them.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from en_pilot import config
from en_pilot.enhance import read_wav_mono16
from eval.corpus.paths import resolve_data_root

MAX_STREAMS = 6


class TranscribeError(RuntimeError):
    """Transcription cannot be run as specified."""


def resolve_device(decoder: str) -> dict[str, str]:
    """Pick a device, then prove the runtime actually loads on it.

    `ctranslate2.get_cuda_device_count()` only asks the driver how many devices
    exist; it says nothing about whether this CTranslate2 build's CUDA runtime
    can load its own libraries. On a host whose CUDA toolkit is newer than the
    one CTranslate2 was built against, the count is 1 and every transcribe call
    then fails with "libcublas.so.NN is not found". Counting alone is therefore
    not evidence, so the choice is verified against a real decode before the
    sweep commits to it, and the fallback is recorded on every row.
    """

    import ctranslate2
    from faster_whisper import WhisperModel

    requested = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    for device, compute_type in (("cuda", "float16"), ("cpu", "int8")):
        if device != requested and device == "cuda":
            continue
        try:
            WhisperModel(decoder, device=device, compute_type=compute_type).transcribe(
                np.zeros(config.SAMPLE_RATE // 2, dtype=np.float32),
                **config.STT_DECODE_KWARGS,
            )
        except Exception as error:
            if device == "cpu":
                raise TranscribeError(
                    f"no usable transcription device: cuda failed with "
                    f"{type(error).__name__}: {error}"
                ) from error
            continue
        return {"device": device, "compute_type": compute_type, "device_requested": requested}
    raise TranscribeError("device probe found no usable CTranslate2 runtime")


def load_jobs(data_root: Path, backends: Sequence[str]) -> list[dict[str, Any]]:
    """One job per (matrix condition, backend) plus one clean floor per base."""

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
        for backend in backends:
            path = data_root / config.OUT_DIR / f"enhance-{backend}" / f"{condition_id}.wav"
            if not path.is_file():
                raise TranscribeError(f"enhanced audio is absent: {path}")
            jobs.append(
                {
                    "condition_id": condition_id,
                    "backend": backend,
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


def completed_keys(path: Path, decoder: str) -> set[str]:
    if not path.is_file():
        return set()
    done: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("decoder") == decoder:
            done.add(f"{record['condition_id']}|{record['backend']}|{decoder}")
    return done


def run(data_root: Path, decoder: str, *, streams: int) -> dict[str, Any]:
    if decoder not in (config.PRIMARY_DECODER, *config.CONTINUITY_DECODERS):
        raise TranscribeError(f"unknown decoder tier: {decoder!r}")
    from faster_whisper import WhisperModel

    from demo.stt_extract import normalize_text

    placement = resolve_device(decoder)
    device = placement["device"]
    compute_type = placement["compute_type"]
    cpu_threads = max(1, (os.cpu_count() or 8) // max(1, min(streams, MAX_STREAMS)))
    jobs = load_jobs(data_root, config.BACKENDS)
    out_path = data_root / config.OUT_DIR / f"transcripts-{decoder}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_keys(out_path, decoder)
    pending = [
        job
        for job in jobs
        if f"{job['condition_id']}|{job['backend']}|{decoder}" not in done
    ]
    width = max(1, min(streams, MAX_STREAMS))
    print(
        f"decoder {decoder}: {device}/{compute_type} (requested "
        f"{placement['device_requested']}), {width} streams x {cpu_threads} cpu "
        f"threads, {len(pending)}/{len(jobs)} clips to transcribe",
        flush=True,
    )
    if not pending:
        return {
            "decoder": decoder,
            "device": device,
            "device_requested": placement["device_requested"],
            "compute_type": compute_type,
            "streams": width,
            "cpu_threads": cpu_threads,
            "transcribed": 0,
            "failed": 0,
            "output": str(out_path),
        }

    models = [
        WhisperModel(
            decoder,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads if device == "cpu" else 0,
        )
        for _ in range(width)
    ]
    work: queue.Queue[dict[str, Any]] = queue.Queue()
    for job in pending:
        work.put(job)
    lock = threading.Lock()
    counter = {"written": 0, "failed": 0}
    errors: list[str] = []
    sink = out_path.open("a", encoding="utf-8")

    def worker(slot: int) -> None:
        model = models[slot]
        while True:
            try:
                job = work.get_nowait()
            except queue.Empty:
                return
            record: dict[str, Any] = {
                "condition_id": job["condition_id"],
                "backend": job["backend"],
                "source": job["source"],
                "decoder": decoder,
                "device": device,
                "compute_type": compute_type,
                "device_requested": placement["device_requested"],
            }
            try:
                audio = read_wav_mono16(Path(job["path"]))
                segments, _ = model.transcribe(
                    np.asarray(audio, dtype=np.float32), **config.STT_DECODE_KWARGS
                )
                text = "".join(segment.text for segment in segments)
                record["text"] = text
                record["normalized"] = normalize_text(text)["normalized_en"]
                record["error"] = ""
            except Exception as error:  # keep the sweep alive, record the failure
                record["text"] = ""
                record["normalized"] = ""
                record["error"] = f"{type(error).__name__}: {error}"
                with lock:
                    counter["failed"] += 1
                    errors.append(
                        f"{job['backend']}/{job['condition_id']}: "
                        f"{type(error).__name__}: {error}"
                    )
            with lock:
                sink.write(json.dumps(record, sort_keys=True) + "\n")
                counter["written"] += 1
                if counter["written"] % 200 == 0:
                    sink.flush()
                    print(f"  {counter['written']}/{len(pending)}", flush=True)

    threads = [threading.Thread(target=worker, args=(index,), daemon=True) for index in range(width)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    sink.close()
    if errors:
        (data_root / config.OUT_DIR / f"transcribe-errors-{decoder}.log").write_text(
            "\n".join(errors), encoding="utf-8"
        )
    return {
        "decoder": decoder,
        "device": device,
        "device_requested": placement["device_requested"],
        "compute_type": compute_type,
        "streams": width,
        "cpu_threads": cpu_threads,
        "transcribed": counter["written"],
        "failed": counter["failed"],
        "output": str(out_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="en_pilot.transcribe",
        description="Transcribe enhanced conditions with one Whisper decoder tier.",
    )
    parser.add_argument(
        "--decoder",
        required=True,
        choices=[config.PRIMARY_DECODER, *config.CONTINUITY_DECODERS],
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get(config.ENVIRONMENT_NAME, config.DATA_ROOT_DEFAULT),
    )
    parser.add_argument("--streams", type=int, default=4)
    arguments = parser.parse_args(argv)
    data_root = resolve_data_root(arguments.data_root)
    print(json.dumps(run(data_root, arguments.decoder, streams=arguments.streams), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

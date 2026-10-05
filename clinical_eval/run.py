"""Run every pipeline component over the clinical corpus, stage by stage.

Each component is exercised on the input it would get in production, and also on
*perfect* input where that is possible, so a failure can be attributed to the
stage that caused it rather than to the cascade:

  C1 ingest/clean   `demo.audio_clean.clean_audio`, per denoiser arm
  C2 STT            `demo.stt_extract.transcribe`, per decoder, on C1's output
  C3 normalize      `demo.stt_extract.normalize_text` on the REFERENCE text
  C4 extract        `demo.stt_extract.extract_entities` on the REFERENCE text
                    (the extractor's ceiling: perfect transcript in)
  C5 end to end     the real cascade, audio -> clean -> STT -> normalize -> extract

C4 minus C5 is the ASR-induced loss; C4 on its own is the extractor's own ceiling.
Extraction is regex-only (`use_llm=False`), which is also what production does
today, because no Ollama model is pulled.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from clinical_eval.prepare import DATA_ROOT_DEFAULT, corpus_root, load_manifest

DENOISERS = ("none", "sherpa-gtcrn-simple")
# The default sweep. `demo/server.py` falls back to this for `/api/eval/run`,
# so anything added here is paid for by every future full run.
DECODERS = ("small-int8", "base-int8", "medasr")
# Accepted but opt-in: `medasr-lm` pulls a 704 MB kenlm language model on top
# of the MedASR weights and beam-searches instead of taking the argmax, so it
# is minutes-per-clip slower. Ask for it explicitly: `--decoders medasr-lm`.
OPTIONAL_DECODERS = ("medasr-lm",)
# `mock` is accepted so a plumbing run needs no weights at all.
ACCEPTED_DECODERS = DECODERS + OPTIONAL_DECODERS + ("tiny-int8", "medium", "large-v3", "mock")
GTCRN = {
    "variant_id": "gtcrn-simple-1.13.8",
    "model_path": "/home/sandy/.local/share/medibytes-eval/models/sherpa-onnx/gtcrn_simple.onnx",
    "model_sha256": "e77603ac0c23dac3227dd2d7135b3a585cbee2679048aecfa886657d3ae1b534",
    "parameters": {"sample_rate": 16000, "provider": "cpu", "num_threads": 1},
}


def _clean(source: Path, destination: Path, denoiser: str) -> dict[str, Any]:
    from demo.audio_clean import BackendId, EnhancementConfig, clean_audio

    raw = dict(GTCRN) if denoiser != "none" else {"variant_id": "none"}
    if "model_path" in raw:
        raw["model_path"] = Path(raw["model_path"])
    backend = BackendId(denoiser)
    started = time.perf_counter()
    try:
        meta = clean_audio(source, destination, backend=backend,
                           config=EnhancementConfig(backend=backend, **raw))
    except Exception as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error}",
                "seconds": round(time.perf_counter() - started, 3)}
    return {
        "ok": True,
        "error": "",
        "seconds": round(time.perf_counter() - started, 3),
        "vad_ratio": meta.get("vad_ratio"),
        "rms_dbfs_before": meta.get("rms_dbfs_before"),
        "rms_dbfs_after": meta.get("rms_dbfs_after"),
        "actual_backend": str(meta.get("actual_backend")),
    }


def _ensure_demo_on_path() -> None:
    """`demo/` itself, not just the repo root, must be importable.

    `demo.stt_extract` reaches `medasr-lm` with a bare `from medasr_lm import
    ...`, and `demo/medasr_lm.py` imports `stt_extract` the same flat way, so
    without this every `medasr-lm` arm dies as `ModuleNotFoundError` — and
    under `strict=True` that is a failed arm on every clip, not a fallback.
    """
    demo = str(Path(__file__).resolve().parents[1] / "demo")
    if demo not in sys.path:
        sys.path.insert(0, demo)


def _transcribe(wav: Path, decoder: str, clip_id: str) -> dict[str, Any]:
    _ensure_demo_on_path()
    from demo.stt_extract import transcribe

    started = time.perf_counter()
    try:
        result = transcribe(str(wav), job_id=clip_id, model=decoder, strict=True)
    except Exception as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error}", "text": "",
                "seconds": round(time.perf_counter() - started, 3)}
    return {
        "ok": True,
        "error": "",
        "text": result["text"],
        "engine": result["engine"],
        "seconds": round(time.perf_counter() - started, 3),
    }


def _extract(text: str) -> dict[str, Any]:
    from demo.stt_extract import extract_entities, normalize_text

    normalized = normalize_text(text)
    entities = extract_entities(text, normalized["normalized_en"], [])
    return {"normalized_en": normalized["normalized_en"], "entities": entities}


def run(data_root: Path, *, denoisers: Sequence[str], decoders: Sequence[str],
        channel: str) -> dict[str, Any]:
    unknown = [d for d in decoders if d not in ACCEPTED_DECODERS]
    if unknown:
        # Unvalidated names used to reach faster-whisper's `.get(model, "tiny")`
        # and silently become Whisper tiny, so `medasr_lm` scored as tiny.
        raise ValueError(
            f"unknown decoder(s) {unknown}; accepted: {list(ACCEPTED_DECODERS)}")
    manifest = load_manifest(data_root)
    work = corpus_root(data_root) / "_runs"
    work.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "channel": channel,
        "denoisers": list(denoisers),
        "decoders": list(decoders),
        "clips": {},
    }
    for record in manifest["clips"]:
        clip_id = record["clip_id"]
        source = Path(record[f"wav_{channel}"])
        reference = record["reference_text"]
        entry: dict[str, Any] = {
            "reference_text": reference,
            # C3 + C4: the normalizer and the extractor on a perfect transcript.
            "reference_pass": _extract(reference),
            "arms": {},
        }
        for denoiser in denoisers:
            cleaned = work / f"{clip_id}.{channel}.{denoiser}.wav"
            clean_outcome = _clean(source, cleaned, denoiser)   # C1
            for decoder in decoders:
                key = f"{denoiser}|{decoder}"
                arm: dict[str, Any] = {"clean": clean_outcome}
                if not clean_outcome["ok"]:
                    arm.update(stt={"ok": False, "error": clean_outcome["error"], "text": ""},
                               pipeline=None)
                    entry["arms"][key] = arm
                    continue
                stt = _transcribe(cleaned, decoder, clip_id)    # C2
                arm["stt"] = stt
                arm["pipeline"] = _extract(stt["text"]) if stt["ok"] else None  # C5
                entry["arms"][key] = arm
                print(f"  {clip_id} {key}: {stt['seconds']}s", flush=True)
        results["clips"][clip_id] = entry

    out = corpus_root(data_root) / f"run-{channel}.json"
    out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results["output"] = str(out)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clinical_eval.run")
    parser.add_argument("--data-root", default=str(DATA_ROOT_DEFAULT))
    parser.add_argument("--channel", default="mix", choices=("mix", "left", "right"))
    parser.add_argument("--denoisers", default=",".join(DENOISERS))
    parser.add_argument(
        "--decoders", default=",".join(DECODERS),
        help=("comma-separated; default %(default)s. Also accepted, opt-in: "
              + ", ".join(OPTIONAL_DECODERS)
              + " (downloads a 704 MB LM and beam-searches; much slower)."))
    arguments = parser.parse_args(argv)
    summary = run(
        Path(arguments.data_root),
        denoisers=tuple(p.strip() for p in arguments.denoisers.split(",") if p.strip()),
        decoders=tuple(p.strip() for p in arguments.decoders.split(",") if p.strip()),
        channel=arguments.channel,
    )
    print(f"wrote {summary['output']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

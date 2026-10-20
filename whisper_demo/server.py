"""HTTP backend for the Whisper large-v3 demo.

One process, one inference worker (the model is heavy and not known to be thread-safe),
file-backed job output. There is no mock: a decode or STT failure marks the job `failed`
with the reason.

Run:  python -m whisper_demo        (http://127.0.0.1:8001/docs)

Env:  WHISPER_DEMO_JOB_ROOT        job and upload directory (default: whisper_demo/jobs)
      WHISPER_DEMO_CORS_ORIGINS    comma-separated origins allowed to call the API (default: none)
      WHISPER_DEMO_PRELOAD         "0" skips loading the model at startup (default "1")
      WHISPER_DEMO_MAX_UPLOAD_BYTES, WHISPER_DEMO_HOST, WHISPER_DEMO_PORT
"""
from __future__ import annotations

import hashlib
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import pipeline, stt
from .llm_client import llm_available

JOB_ROOT = Path(os.environ.get("WHISPER_DEMO_JOB_ROOT", Path(__file__).resolve().parent / "jobs"))
UPLOAD_ROOT = JOB_ROOT / "uploads"
MAX_UPLOAD_BYTES = int(os.environ.get("WHISPER_DEMO_MAX_UPLOAD_BYTES", 64 * 1024 * 1024))
ALLOWED_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4", ".aac"}
LANGUAGE_CHOICES = ("auto", *stt.LANGUAGES)

executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper-demo-worker")
_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()
_startup = {"error": None}


def _set(job_id: str, **fields: Any) -> None:
    with _lock:
        _jobs.setdefault(job_id, {}).update(fields)


def _get(job_id: str) -> dict[str, Any]:
    with _lock:
        record = _jobs.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id}")
        return dict(record)


def _preload() -> None:
    try:
        stt.load()
    except Exception as error:  # the server stays up; health and every job report the reason
        _startup["error"] = f"{type(error).__name__}: {error}"[:300]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if os.environ.get("WHISPER_DEMO_PRELOAD", "1") != "0":
        executor.submit(_preload)  # on the worker thread, so jobs queue behind the load
    yield


app = FastAPI(title="MediBytes Whisper large-v3 demo",
              description="Whisper large-v3 -> extraction -> optional LLM note.",
              version="0.1.0", lifespan=lifespan)

_origins = [o.strip() for o in os.environ.get("WHISPER_DEMO_CORS_ORIGINS", "").split(",") if o.strip()]
if _origins:
    app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["*"],
                       allow_headers=["*"])


def _run(job_id: str, source: Path, params: dict[str, Any]) -> None:
    _set(job_id, status="running")
    out_dir = JOB_ROOT / "jobs" / job_id
    try:
        ran = pipeline.run(source, out_dir, job_id=job_id, language=params["language"],
                           use_llm=params["use_llm"], validate_terms=params["validate_terms"],
                           context_repair=params["context_repair"])
    except Exception as error:  # a failed job must not take the worker down
        _set(job_id, status="failed", error=f"{type(error).__name__}: {error}")
        return
    provenance = ran["transcript"].get("stt_provenance") or {}
    _set(job_id, status="succeeded", result=ran, output_dir=str(out_dir),
         observed={"stt_engine": ran["transcript"].get("stt_engine"),
                   "device": provenance.get("device"),
                   "compute_type": provenance.get("compute_type"),
                   "fallback_reason": provenance.get("fallback_reason"),
                   "detected_language": provenance.get("detected_language"),
                   "is_mock": provenance.get("is_mock", False),
                   "llm_engine": (ran["entities"] or {}).get("llm_engine", "")})


def _bool(name: str, value: str) -> bool:
    if value not in ("true", "false"):
        raise HTTPException(status_code=400, detail=f"{name} must be true|false")
    return value == "true"


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Model, device and LLM state. Never loads the model itself."""
    ok, why = llm_available(None)
    with _lock:
        counts: dict[str, int] = {}
        for record in _jobs.values():
            counts[record.get("status", "?")] = counts.get(record.get("status", "?"), 0) + 1
    return {"status": "ok", "stt": {**stt.status(), "startup_error": _startup["error"]},
            "llm": {"available": bool(ok), "reason": "" if ok else str(why)},
            "languages": list(LANGUAGE_CHOICES), "job_root": str(JOB_ROOT), "jobs": counts}


@app.post("/api/jobs", status_code=202)
async def create_job(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    use_llm: str = Form("false"),
    validate_terms: str = Form("auto"),
    context_repair: str = Form("false"),
) -> JSONResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"unsupported format: {suffix or '(none)'}")
    if language not in LANGUAGE_CHOICES:
        raise HTTPException(status_code=400,
                            detail=f"unknown language: {language}; valid: {', '.join(LANGUAGE_CHOICES)}")
    if validate_terms not in ("auto", "true", "false"):
        raise HTTPException(status_code=400, detail="validate_terms must be auto|true|false")
    params = {"language": None if language == "auto" else language,
              "use_llm": _bool("use_llm", use_llm),
              "validate_terms": {"auto": "auto", "true": True, "false": False}[validate_terms],
              "context_repair": _bool("context_repair", context_repair)}

    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    job_id = "job-" + uuid.uuid4().hex[:12]
    staged = UPLOAD_ROOT / f"{job_id}{suffix}"
    digest, written = hashlib.sha256(), 0
    with open(staged, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                out.close()
                staged.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="upload exceeds limit")
            digest.update(chunk)
            out.write(chunk)
    if written == 0:
        staged.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="empty upload")

    _set(job_id, status="queued", params=params, source_filename=file.filename,
         sha256=digest.hexdigest())
    executor.submit(_run, job_id, staged, params)
    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "queued",
                                                  "status_url": f"/api/jobs/{job_id}"})


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    record = _get(job_id)
    result = record.get("result") or {}
    return {"job_id": job_id, "status": record.get("status"), "error": record.get("error", ""),
            "requested": record.get("params"), "observed": record.get("observed"),
            "audio": result.get("audio"), "transcript": result.get("transcript"),
            "entities": result.get("entities"), "discharge": result.get("discharge"),
            "seconds": result.get("seconds"), "output_dir": record.get("output_dir")}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict[str, str]:
    _get(job_id)
    with _lock:
        _jobs.pop(job_id, None)
    return {"job_id": job_id, "status": "deleted"}

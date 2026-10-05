"""HTTP backend over the demo pipeline, for verification and evaluation.

One process, one inference worker, file-backed results.  Every job goes
through `pipeline.run_pipeline`, the same function `python demo/pipeline.py`
calls, so what this serves is the shipped chain and not a second copy of it.

Run:  python demo/server.py            (or: uvicorn demo.server:app --reload)
Docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse

from denoise import BackendId, EnhancementConfig, _module_name
from pipeline import run_pipeline

# Server state lives outside the committed demo/ output tree.
JOB_ROOT = Path(os.environ.get("MEDIBYTES_JOB_ROOT", os.path.join(_HERE, "_server_jobs")))
UPLOAD_ROOT = JOB_ROOT / "uploads"
MAX_UPLOAD_BYTES = int(os.environ.get("MEDIBYTES_MAX_UPLOAD_BYTES", 64 * 1024 * 1024))
ALLOWED_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}

# One inference worker: the STT and denoiser models are heavy and are not
# known to be thread-safe, and the idempotency ledger is a read-modify-write
# on a JSON file.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="medibytes-worker")
_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()

app = FastAPI(
    title="MediBytes pipeline backend",
    description="Verification and evaluation surface over the demo pipeline.",
    version="0.1.0",
)


def _set(job_id: str, **fields: Any) -> None:
    with _lock:
        _jobs.setdefault(job_id, {})
        _jobs[job_id].update(fields)


def _get(job_id: str) -> dict[str, Any]:
    with _lock:
        record = _jobs.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id}")
        return dict(record)


def _observed(result: dict[str, Any]) -> dict[str, Any]:
    """What actually ran, next to what was asked for.

    `clean_audio` and `run_stt_extract` both fall back silently, so a result
    that does not report the actual backend and engine cannot be trusted as
    evidence for the configuration it was requested under.
    """
    clean = result.get("clean") or {}
    transcript = result.get("transcript") or {}
    requested = result.get("requested") or {}
    actual_backend = str(clean.get("actual_backend", ""))
    engine = str(transcript.get("stt_engine", ""))
    mismatches = []
    if actual_backend and actual_backend != requested.get("denoiser"):
        mismatches.append(f"denoiser requested={requested.get('denoiser')} actual={actual_backend}")
    # The engine string decorates the model id ("base-int8" ->
    # "faster-whisper:base-cpu-int8"), so compare on its parts, not equality.
    model = str(requested.get("model", ""))
    if engine and model and not all(part in engine for part in model.split("-")):
        mismatches.append(f"model requested={model} engine={engine}")
    # Authoritative: the mock fallback labels itself "mock (<reason>: <Type>)",
    # and the reason can contain the requested model id, which would let the
    # string check above pass a fallback as clean.
    provenance = transcript.get("stt_provenance") or {}
    if provenance.get("is_mock") and model != "mock":
        mismatches.append(f"model requested={model} but STT fell back to mock: {engine}")
    return {
        "actual_denoiser": actual_backend,
        "variant_id": str(clean.get("variant_id", "")),
        "stt_engine": engine,
        "stt_provenance": transcript.get("stt_provenance"),
        "language": transcript.get("language"),
        # Set by run_stt_extract only when LLM extraction was asked for and
        # silently fell back to regex.
        "llm_engine": (result.get("entities") or {}).get("llm_engine", ""),
        "mismatches": mismatches,
    }


def _run(job_id: str, source: Path, params: dict[str, Any]) -> None:
    _set(job_id, status="running")
    lines: list[str] = []
    try:
        result = run_pipeline(
            str(source),
            key=job_id,
            use_llm=params["use_llm"],
            model=params["model"],
            template=params["template"],
            denoiser=params["denoiser"],
            denoiser_config=params["denoiser_config"],
            root=str(JOB_ROOT),
            log=lines.append,
        )
    except Exception as error:  # a failed job must not take the worker down
        _set(job_id, status="failed", error=f"{type(error).__name__}: {error}", log=lines)
        return
    _set(
        job_id,
        status="succeeded",
        result=result,
        observed=_observed(result),
        log=lines,
    )


def _probe(backend: BackendId) -> str:
    """Whether a denoiser would actually run, via the adapter's own probe().

    Importing the adapter is not enough -- it imports cleanly while the package
    it wraps is missing, and the job then fails at run time instead.  Model
    backends are probed under their shipped config, since they refuse a bare one.
    """
    try:
        module = importlib.import_module(f"denoise_backends.{_module_name(backend)}")
    except Exception as error:
        return f"unavailable: {type(error).__name__}: {error}"
    raw: dict[str, Any] = {"variant_id": backend.value}
    shipped = Path(_HERE) / "denoise_backends" / f"{backend.value}.config.json"
    if shipped.is_file():
        try:
            raw = json.loads(shipped.read_text(encoding="utf-8"))
            raw.pop("backend", None)
            if "model_path" in raw:
                raw["model_path"] = Path(raw["model_path"])
        except Exception as error:
            return f"unavailable: bad shipped config: {error}"
    try:
        probe = module.probe(EnhancementConfig(backend=backend, **raw))
    except Exception as error:
        return f"unavailable: {error}"
    return "available" if getattr(probe, "available", False) else (
        f"unavailable: {getattr(probe, 'reason', 'no reason given')}")


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Which denoiser backends and decoders this process can actually load."""
    backends = {backend.value: _probe(backend) for backend in BackendId}
    with _lock:
        counts: dict[str, int] = {}
        for record in _jobs.values():
            counts[record.get("status", "?")] = counts.get(record.get("status", "?"), 0) + 1
    return {
        "status": "ok",
        "job_root": str(JOB_ROOT),
        "backend_probe": backends,
        "models": ["medasr", "small-int8", "base-int8", "tiny-int8", "mock"],
        "jobs": counts,
    }


@app.post("/api/audio/jobs", status_code=202)
async def create_job(
    file: UploadFile = File(...),
    denoiser: str = Form(BackendId.NONE.value),
    model: str = Form("medasr"),
    template: str = Form("er_discharge"),
    use_llm: str = Form("auto"),
    denoiser_config: str | None = Form(None),
) -> JSONResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"unsupported format: {suffix or '(none)'}")
    try:
        BackendId(denoiser)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"unknown denoiser: {denoiser}")
    if use_llm not in ("auto", "true", "false"):
        raise HTTPException(status_code=400, detail="use_llm must be auto|true|false")

    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    staged = UPLOAD_ROOT / f"{uuid.uuid4().hex}{suffix}"
    digest = hashlib.sha256()
    written = 0
    with open(staged, "wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
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

    config_path = None
    if denoiser_config:
        try:
            parsed = json.loads(denoiser_config)
        except json.JSONDecodeError as error:
            staged.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"denoiser_config is not JSON: {error}")
        config_path = str(staged.with_suffix(".denoiser.json"))
        Path(config_path).write_text(json.dumps(parsed), encoding="utf-8")

    params = {
        "denoiser": denoiser,
        "model": model,
        "template": template,
        "use_llm": {"auto": "auto", "true": True, "false": False}[use_llm],
        "denoiser_config": config_path,
    }
    # The job id keys on the bytes AND the configuration, so two runs of the
    # same clip under different denoisers do not collide in the idempotency
    # ledger and overwrite each other's outputs.
    fingerprint = digest.hexdigest() + json.dumps(params, sort_keys=True, default=str)
    job_id = "job-" + hashlib.sha256(fingerprint.encode()).hexdigest()[:12]

    final = UPLOAD_ROOT / f"{job_id}{suffix}"
    shutil.move(str(staged), final)
    _set(job_id, status="queued", params=params,
         source_filename=file.filename, sha256=digest.hexdigest())
    _executor.submit(_run, job_id, final, params)
    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "status": "queued", "status_url": f"/api/audio/jobs/{job_id}"},
    )


@app.get("/api/audio/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    """Status plus every stage's output: clean meta, transcript, entities, exports."""
    record = _get(job_id)
    result = record.get("result") or {}
    paths = result.get("paths") or {}
    return {
        "job_id": job_id,
        "status": record.get("status"),
        "error": record.get("error", ""),
        "requested": record.get("params"),
        "observed": record.get("observed"),
        "clean": result.get("clean"),
        "transcript": result.get("transcript"),
        "entities": result.get("entities"),
        "template_error": result.get("template_error", ""),
        "exports": {
            name: f"/api/audio/jobs/{job_id}/export.{name}"
            for name in ("html", "docx") if name in paths
        },
        "paths": paths,
        "log": record.get("log", []),
    }


@app.delete("/api/audio/jobs/{job_id}")
def delete_job(job_id: str) -> dict[str, str]:
    _get(job_id)
    with _lock:
        _jobs.pop(job_id, None)
    return {"job_id": job_id, "status": "deleted"}


@app.get("/api/audio/jobs/{job_id}/export.{kind}")
def get_export(job_id: str, kind: str):
    if kind not in ("html", "docx"):
        raise HTTPException(status_code=404, detail="export must be html or docx")
    record = _get(job_id)
    path = ((record.get("result") or {}).get("paths") or {}).get(kind)
    if not path or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"no {kind} export for {job_id}")
    media = "text/html" if kind == "html" else (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    return FileResponse(path, media_type=media, filename=f"{job_id}.{kind}")


@app.post("/api/eval/run")
def eval_run(
    channel: str = Form("mix"),
    denoisers: str = Form(""),
    decoders: str = Form(""),
    data_root: str = Form(""),
) -> dict[str, Any]:
    """Run the 7-clip component-wise clinical eval and score it.

    Wraps `clinical_eval.run` + `clinical_eval.score` rather than defining new
    metrics, so the API and `python -m clinical_eval.run` report the same thing.
    Synchronous and slow by design: this is an evaluation trigger, not a job API.
    """
    repo_root = os.path.dirname(_HERE)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        from clinical_eval import run as eval_runner, score as eval_scorer
        from clinical_eval.prepare import DATA_ROOT_DEFAULT, PrepareError
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"clinical_eval unavailable: {error}")

    root = Path(data_root) if data_root else DATA_ROOT_DEFAULT
    arms = {
        "denoisers": tuple(d for d in denoisers.split(",") if d) or eval_runner.DENOISERS,
        "decoders": tuple(d for d in decoders.split(",") if d) or eval_runner.DECODERS,
    }
    def _work():
        run_report = eval_runner.run(root, channel=channel, **arms)
        return run_report, eval_scorer.score(root, channel=channel)

    try:
        # Through the same single worker as jobs: a sync route otherwise runs in
        # Starlette's threadpool and would load models concurrently with a job.
        run_report, score_report = _executor.submit(_work).result()
    except PrepareError as error:
        # Not a server fault: the corpus has to be staged first with
        # `python -m clinical_eval.prepare`.
        raise HTTPException(status_code=503, detail=f"corpus not staged: {error}")
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"{type(error).__name__}: {error}")
    return {"channel": channel, **arms, "run_output": run_report.get("output"),
            "score": score_report}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("MEDIBYTES_HOST", "127.0.0.1"),
                port=int(os.environ.get("MEDIBYTES_PORT", 8000)))

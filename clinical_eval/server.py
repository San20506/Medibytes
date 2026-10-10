"""The pipeline server plus an evaluation trigger.

    uvicorn clinical_eval.server:app

`base_pipeline.server` stays free of evaluation code; this module imports it and
adds `/api/eval/run`, so the dependency points from the evaluation to the pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Form, HTTPException

from base_pipeline.server import app, executor

from . import run as eval_runner, score as eval_scorer
from .prepare import DATA_ROOT_DEFAULT, PrepareError


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
        run_report, score_report = executor.submit(_work).result()
    except PrepareError as error:
        # Not a server fault: the corpus has to be staged first with
        # `python -m clinical_eval.prepare`.
        raise HTTPException(status_code=503, detail=f"corpus not staged: {error}")
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"{type(error).__name__}: {error}")
    return {"channel": channel, **arms, "run_output": run_report.get("output"),
            "score": score_report}


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(app, host=os.environ.get("MEDIBYTES_HOST", "127.0.0.1"),
                port=int(os.environ.get("MEDIBYTES_PORT", 8000)))

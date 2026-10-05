"""The HTTP backend must serve the same chain `pipeline.py` runs."""
import json
import os
import sys
import time

import pytest

_DEMO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DEMO not in sys.path:
    sys.path.insert(0, _DEMO)

fastapi_testclient = pytest.importorskip("fastapi.testclient")
SAMPLE = os.path.join(_DEMO, "audio_in", "sample1_hinglish_fever.wav")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDIBYTES_JOB_ROOT", str(tmp_path))
    for name in ("server",):
        sys.modules.pop(name, None)
    import server

    server.JOB_ROOT = tmp_path
    server.UPLOAD_ROOT = tmp_path / "uploads"
    with fastapi_testclient.TestClient(server.app) as c:
        yield c, server


def _submit(client, **form):
    with open(SAMPLE, "rb") as handle:
        return client.post(
            "/api/audio/jobs",
            files={"file": ("sample1.wav", handle, "audio/wav")},
            data={"model": "mock", "denoiser": "none", "template": "none",
                  "use_llm": "false", **form},
        )


def _await(client, job_id, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/audio/jobs/{job_id}").json()
        if body["status"] in ("succeeded", "failed"):
            return body
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")


def test_health_reports_real_backend_availability(client):
    c, _ = client
    body = c.get("/api/health").json()
    assert body["status"] == "ok"
    # `none` always runs; backends whose package is missing must say so rather
    # than claim availability the job would then contradict at run time.
    assert body["backend_probe"]["none"] == "available"
    for name, state in body["backend_probe"].items():
        assert state == "available" or state.startswith("unavailable: "), (name, state)


def test_job_runs_every_stage_and_reports_what_actually_ran(client):
    c, _ = client
    created = _submit(c)
    assert created.status_code == 202
    body = _await(c, created.json()["job_id"])
    assert body["status"] == "succeeded", body.get("error")
    assert body["clean"]["actual_backend"] == "none"
    assert body["transcript"]["normalized_en"] is not None
    for bucket in ("drugs", "symptoms", "vitals", "allergies"):
        assert bucket in body["entities"]
    assert body["observed"]["mismatches"] == []


def test_distinct_configs_do_not_collide_on_one_job_id(client):
    c, _ = client
    one = _submit(c, denoiser="none").json()["job_id"]
    two = _submit(c, denoiser="noisereduce").json()["job_id"]
    assert one != two


def test_rejects_unsupported_format_and_unknown_denoiser(client):
    c, _ = client
    bad = c.post("/api/audio/jobs", files={"file": ("x.txt", b"nope", "text/plain")},
                 data={"model": "mock"})
    assert bad.status_code == 415
    with open(SAMPLE, "rb") as handle:
        unknown = c.post("/api/audio/jobs",
                         files={"file": ("s.wav", handle, "audio/wav")},
                         data={"denoiser": "not-a-backend", "model": "mock"})
    assert unknown.status_code == 400


def test_unknown_job_is_404(client):
    c, _ = client
    assert c.get("/api/audio/jobs/job-missing").status_code == 404


def test_api_result_matches_the_cli_for_the_same_clip_and_config(client, tmp_path):
    """The point of the backend: it must not drift from the shipped chain."""
    c, _ = client
    body = _await(c, _submit(c).json()["job_id"])
    assert body["status"] == "succeeded", body.get("error")

    import pipeline

    direct = pipeline.run_pipeline(
        SAMPLE, key="cli-parity", use_llm=False, model="mock", template="none",
        denoiser="none", root=str(tmp_path / "cli"), log=lambda *a: None,
    )
    def _comparable(entities):
        # job_id differs by construction; every extracted field must not.
        return json.dumps({k: v for k, v in entities.items() if k != "job_id"},
                          sort_keys=True)

    assert direct["transcript"]["normalized_en"] == body["transcript"]["normalized_en"]
    assert _comparable(direct["entities"]) == _comparable(body["entities"])


def test_mock_fallback_is_flagged_even_when_the_engine_string_looks_right(client):
    """The fallback label embeds its reason, which can contain the model id."""
    _, server = client
    observed = server._observed({
        "requested": {"denoiser": "none", "model": "small-int8"},
        "clean": {"actual_backend": "none", "variant_id": "none"},
        "transcript": {
            "stt_engine": "mock (small-int8 load failed: OSError)",
            "stt_provenance": {"is_mock": True},
        },
        "entities": {},
    })
    assert any("fell back to mock" in m for m in observed["mismatches"]), observed


def test_a_real_engine_is_not_flagged(client):
    _, server = client
    observed = server._observed({
        "requested": {"denoiser": "none", "model": "base-int8"},
        "clean": {"actual_backend": "none", "variant_id": "none"},
        "transcript": {
            "stt_engine": "faster-whisper:base-cpu-int8",
            "stt_provenance": {"is_mock": False},
        },
        "entities": {},
    })
    assert observed["mismatches"] == []

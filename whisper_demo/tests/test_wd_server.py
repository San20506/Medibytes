import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from whisper_demo import server, stt

TEXT = "Give azithromycin 500 mg once daily 3 days. BP 130 by 80. No penicillin allergy."


def _fake(audio, language=None, job_id="job"):
    return {"text": TEXT, "language": "en", "engine": "fake",
            "segments": [{"id": 0, "text": TEXT, "start": 0.0, "end": 5.0, "lang": "en",
                          "confidence": 0.9, "words": []}],
            "stt_provenance": {"job_id": job_id, "is_mock": False, "device": "cpu",
                               "compute_type": "int8", "forced_language": language}}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("WHISPER_DEMO_PRELOAD", "0")
    monkeypatch.setattr(server, "JOB_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(server, "UPLOAD_ROOT", tmp_path / "jobs" / "uploads")
    monkeypatch.setattr(stt, "transcribe", _fake)
    with TestClient(server.app) as c:
        yield c


@pytest.fixture
def wav_bytes(tmp_path):
    t = np.arange(16000 * 2) / 16000
    audio = 0.3 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 3 * t) > 0)
    path = tmp_path / "in.wav"
    sf.write(path, audio.astype("float32"), 16000)
    return path.read_bytes()


def _wait(client, job_id):
    for _ in range(100):
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("succeeded", "failed"):
            return body
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_health_shape(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert {"loaded", "model", "device", "compute_type", "fallback_reason"} <= set(body["stt"])
    assert body["stt"]["model"] == "large-v3"
    assert "available" in body["llm"]


def test_job_round_trip(client, wav_bytes):
    r = client.post("/api/jobs", files={"file": ("clip.wav", wav_bytes, "audio/wav")},
                    data={"validate_terms": "false"})
    assert r.status_code == 202
    body = _wait(client, r.json()["job_id"])
    assert body["status"] == "succeeded", body["error"]
    assert body["observed"]["is_mock"] is False
    assert any("azithromycin" in d["name"].lower() for d in body["entities"]["drugs"])
    assert "skipped" in body["discharge"]["provenance"]["llm_engine"]
    assert client.delete(f"/api/jobs/{r.json()['job_id']}").status_code == 200
    assert client.get(f"/api/jobs/{r.json()['job_id']}").status_code == 404


@pytest.mark.parametrize("name,data,code", [
    ("clip.txt", {}, 415),
    ("clip.wav", {"language": "fr"}, 400),
    ("clip.wav", {"use_llm": "maybe"}, 400),
])
def test_bad_requests(client, wav_bytes, name, data, code):
    r = client.post("/api/jobs", files={"file": (name, wav_bytes)}, data=data)
    assert r.status_code == code


def test_unknown_job_is_404(client):
    assert client.get("/api/jobs/job-nope").status_code == 404


def test_stt_failure_marks_the_job_failed_without_a_mock(client, wav_bytes, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("model could not load")

    monkeypatch.setattr(stt, "transcribe", boom)
    r = client.post("/api/jobs", files={"file": ("clip.wav", wav_bytes)})
    body = _wait(client, r.json()["job_id"])
    assert body["status"] == "failed"
    assert "model could not load" in body["error"]
    assert body["transcript"] is None


def test_health_reports_the_form_llm_without_the_key(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    body = client.get("/api/health").json()
    assert body["form_llm"]["key_present"] is True
    assert body["templates"] == ["discharge", "admission"]
    assert "sk-secret-value" not in str(body)


def test_unknown_template_is_400(client, wav_bytes):
    r = client.post("/api/jobs", files={"file": ("clip.wav", wav_bytes)}, data={"template": "opd"})
    assert r.status_code == 400


def test_default_template_is_discharge_and_has_no_form(client, wav_bytes):
    r = client.post("/api/jobs", files={"file": ("clip.wav", wav_bytes)}, data={"validate_terms": "false"})
    body = _wait(client, r.json()["job_id"])
    assert body["template"] == "discharge" and body["form"] is None and body["discharge"]


def test_admission_template_returns_the_form_and_no_discharge(client, wav_bytes, monkeypatch):
    import json

    from whisper_demo import admission

    sent = "Give azithromycin 500 mg once daily 3 days."
    reply = {"admission_plan": {"orders": [{"drug": "azithromycin", "dose": 500, "unit": "mg",
                                            "source_sentence": sent, "confidence": 0.9}]}}
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(admission, "_chat", lambda prompt, **kw: (json.dumps(reply), {}))
    r = client.post("/api/jobs", files={"file": ("clip.wav", wav_bytes)},
                    data={"template": "admission", "validate_terms": "false"})
    body = _wait(client, r.json()["job_id"])
    assert body["status"] == "succeeded", body["error"]
    assert body["template"] == "admission" and body["discharge"] is None
    assert body["form"]["form"]["admission_plan"]["orders"][0]["drug"] == "azithromycin"
    assert body["observed"]["form_engine"].startswith("form:")
    assert (server.JOB_ROOT / "jobs" / r.json()["job_id"] / "admission.json").is_file()


def test_admission_without_a_key_still_succeeds_with_a_labelled_empty_form(client, wav_bytes, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CHATGPT_API_KEY", raising=False)
    r = client.post("/api/jobs", files={"file": ("clip.wav", wav_bytes)}, data={"template": "admission"})
    body = _wait(client, r.json()["job_id"])
    assert body["status"] == "succeeded"
    assert "form-skipped" in body["form"]["provenance"]["llm_engine"]
    assert body["form"]["form"]["chief_complaints"] == []

import json

import pytest

from whisper_demo import pipeline, stt

TEXT = "Give azithromycin 500 mg once daily 3 days. BP 130 by 80. No penicillin allergy."


def _fake_stt(audio, language=None, job_id="job"):
    return {"text": TEXT, "language": "en", "engine": "fake",
            "segments": [{"id": 0, "text": TEXT, "start": 0.0, "end": 5.0, "lang": "en",
                          "confidence": 0.9, "words": []}],
            "stt_provenance": {"job_id": job_id, "is_mock": False, "forced_language": language}}


@pytest.fixture
def clip(tmp_path):
    import numpy as np
    import soundfile as sf

    t = np.arange(16000 * 2) / 16000
    audio = 0.3 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 3 * t) > 0)
    path = tmp_path / "clip.wav"
    sf.write(path, audio.astype("float32"), 16000)
    return path


def test_run_extracts_and_writes_the_four_files(monkeypatch, clip, tmp_path):
    monkeypatch.setattr(stt, "transcribe", _fake_stt)
    result = pipeline.run(clip, tmp_path / "out", use_llm=False, validate_terms=False)
    names = [d["name"].lower() for d in result["entities"]["drugs"]]
    assert any("azithromycin" in n for n in names)
    assert result["entities"]["vitals"]
    assert any(a.get("negated") for a in result["entities"]["allergies"])
    for name in ("transcript.json", "entities.json", "verify.json", "discharge.json"):
        assert json.loads((tmp_path / "out" / name).read_text(encoding="utf-8"))
    note = result["discharge"]
    assert "skipped" in note["provenance"]["llm_engine"]
    assert all(not s["items"] for s in note["sections"].values())


def test_stt_failure_propagates_and_is_never_mocked(monkeypatch, clip, tmp_path):
    def boom(*a, **k):
        raise RuntimeError("no gpu, no cpu")

    monkeypatch.setattr(stt, "transcribe", boom)
    with pytest.raises(RuntimeError, match="no gpu"):
        pipeline.run(clip, tmp_path / "out")


def test_silence_is_rejected_before_the_model_runs(monkeypatch, tmp_path):
    import numpy as np
    import soundfile as sf

    called = []
    monkeypatch.setattr(stt, "transcribe", lambda *a, **k: called.append(1))
    path = tmp_path / "silence.wav"
    sf.write(path, np.zeros(16000 * 2, dtype="float32"), 16000)
    with pytest.raises(ValueError, match="NO_AUDIO"):
        pipeline.run(path, tmp_path / "out")
    assert not called

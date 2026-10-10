"""The Streamlit page takes its choices from base_pipeline.presets and runs through run.run_clip."""
import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "fixtures" / "audio" / "sample1_hinglish_fever.wav"


def _page():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _checkbox(at, label_start):
    return next(c for c in at.sidebar.checkbox if c.label.startswith(label_start))


def test_page_offers_the_runnable_presets_and_starts_on_the_baseline():
    from base_pipeline import presets

    box = _page().sidebar.selectbox[0]
    assert box.label == "Preset" and box.value == presets.DEFAULT
    # whisper-medasr replaces the whole chain, which this page cannot drive.
    assert "whisper-medasr" not in box.options and "context-repair" in box.options


def test_picking_a_preset_resets_the_knobs_to_its_values():
    at = _page()
    assert _checkbox(at, "LLM context repair").value is False
    assert _checkbox(at, "skip term validation").value is False

    at.sidebar.selectbox[0].select("context-repair").run()
    assert not at.exception
    assert _checkbox(at, "LLM context repair").value is True

    at.sidebar.selectbox[0].select("regex-only").run()
    assert _checkbox(at, "skip term validation").value is True
    assert _checkbox(at, "LLM context repair").value is False

    at.sidebar.selectbox[0].select("no-second-pass").run()
    assert _checkbox(at, "second decoder pass").value is False


def test_run_wrapper_goes_through_run_clip_and_writes_the_arm_outputs(tmp_path):
    import app

    result = app._run_one_arm(SAMPLE, tmp_path / "arm", "none", sherpa_path="", model="mock",
                              use_llm=False, validate_terms=False, second_pass=None, llm_model=None,
                              context_repair=False, decode="greedy")
    assert result["meta"]["actual_backend"] == "none"
    assert result["transcript"]["stt_provenance"]["is_mock"] is True
    for name in ("transcript.json", "entities.json", "clean_meta.json", "verify.json",
                 "discharge.json", "report.md"):
        assert (tmp_path / "arm" / name).is_file(), name
    assert json.loads((tmp_path / "arm" / "entities.json").read_text(encoding="utf-8"))["drugs"]

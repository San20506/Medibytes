"""Strict STT provenance and deterministic entity-span contracts."""
import copy
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
from jsonschema import Draft7Validator, ValidationError

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEMO = os.path.dirname(_HERE)
sys.path.insert(0, _DEMO)

import stt_extract


def _fake_faster_whisper(monkeypatch, whisper_model, version="1.2.3"):
    module = ModuleType("faster_whisper")
    module.WhisperModel = whisper_model
    module.__version__ = version
    monkeypatch.setitem(sys.modules, "faster_whisper", module)


@pytest.mark.parametrize("failure_phase", ["model_load", "transcription"])
def test_strict_real_model_failures_propagate_without_mock_leakage(monkeypatch, failure_phase):
    class FailingWhisperModel:
        def __init__(self, *_args, **_kwargs):
            if failure_phase == "model_load":
                raise RuntimeError("forced model-load failure")

        def transcribe(self, *_args, **_kwargs):
            raise RuntimeError("forced transcription failure")

    _fake_faster_whisper(monkeypatch, FailingWhisperModel)

    with pytest.raises(RuntimeError, match="forced"):
        stt_extract.transcribe("input.wav", model="base-int8", strict=True)

    with pytest.raises(RuntimeError, match="forced"):
        stt_extract.run_stt_extract(
            "input.wav", use_llm=False, model="base-int8", strict=True
        )


def test_non_strict_real_model_failure_remains_explicit_mock_fallback(monkeypatch):
    class FailingWhisperModel:
        def __init__(self, *_args, **_kwargs):
            raise ImportError("forced missing runtime")

    _fake_faster_whisper(monkeypatch, FailingWhisperModel)

    result = stt_extract.transcribe("input.wav", model="base-int8", strict=False)

    assert result["text"] in stt_extract.MOCK_TEXTS.values()
    assert result["stt_provenance"] == {
        "job_id": "demo-001",
        "provider": "mock",
        "requested_model": "base-int8",
        "actual_model": "mock",
        "device": None,
        "compute_type": None,
        "beam_size": None,
        "word_timestamps": True,
        "temperature": 0.0,
        "runtime_version": None,
        "model_hash": None,
        "model_snapshot": None,
        "is_mock": True,
    }
    assert "mock (no faster-whisper" in result["engine"]


def test_explicit_mock_remains_usable_and_labeled():
    result = stt_extract.transcribe("anything.wav", job_id="demo-002", model="mock")

    assert result["text"] == stt_extract.MOCK_TEXTS["demo-002"]
    assert result["engine"] == "mock (forced --model mock)"
    assert result["stt_provenance"]["is_mock"] is True
    assert result["stt_provenance"]["provider"] == "mock"
    assert result["stt_provenance"]["requested_model"] == "mock"
    assert result["stt_provenance"]["job_id"] == "demo-002"
    assert result["stt_provenance"]["word_timestamps"] is True
    assert result["stt_provenance"]["temperature"] == 0.0


def test_real_transcription_records_complete_provenance(monkeypatch):
    calls = {}

    class Segment:
        text = " Paracetamol 500 mg once daily. "
        start = 0.0
        end = 2.0
        avg_logprob = -0.1
        words = ()

    class WhisperModel:
        model_sha256 = "sha256:model"
        model_snapshot = "snapshot-123"

        def __init__(self, model, *, device, compute_type):
            calls["model"] = model
            calls["device"] = device
            calls["compute_type"] = compute_type

        def transcribe(self, audio, **kwargs):
            calls["audio"] = audio
            calls.update(kwargs)
            return iter([Segment()]), SimpleNamespace(language="en")

    _fake_faster_whisper(monkeypatch, WhisperModel, version="1.2.3")

    result = stt_extract.transcribe("clean.wav", model="base-int8", strict=True)

    assert calls["model"] == "base"
    assert calls["device"] == "cpu"
    assert calls["compute_type"] == "int8"
    assert calls["beam_size"] == 1
    assert calls["word_timestamps"] is True
    assert calls["temperature"] == 0
    assert result["stt_provenance"] == {
        "job_id": "demo-001",
        "provider": "faster-whisper",
        "requested_model": "base-int8",
        "actual_model": "base",
        "device": "cpu",
        "compute_type": "int8",
        "beam_size": 1,
        "word_timestamps": True,
        "temperature": 0.0,
        "runtime_version": "1.2.3",
        "model_hash": "sha256:model",
        "model_snapshot": "snapshot-123",
        "is_mock": False,
    }
    assert result["engine"] == "faster-whisper:base-cpu-int8"


def test_runner_forwards_strict_and_keeps_evaluator_regex_only(monkeypatch):
    class Segment:
        text = "Paracetamol 500 mg once daily."
        start = 0.0
        end = 2.0
        avg_logprob = -0.1
        words = ()

    class WhisperModel:
        def __init__(self, *_args, **_kwargs):
            pass

        def transcribe(self, *_args, **_kwargs):
            return iter([Segment()]), SimpleNamespace(language="en")

    _fake_faster_whisper(monkeypatch, WhisperModel)

    transcript, entities = stt_extract.run_stt_extract(
        "clean.wav", job_id="contract-run", use_llm=False, model="base-int8", strict=True
    )

    assert transcript["stt_provenance"]["provider"] == "faster-whisper"
    assert transcript["stt_provenance"]["job_id"] == "contract-run"
    assert transcript["stt_provenance"]["word_timestamps"] is True
    assert transcript["stt_provenance"]["temperature"] == 0.0
    assert transcript["stt_provenance"]["is_mock"] is False
    assert "llm_engine" not in entities
    assert "tidy_engine" not in entities


def _extract(text):
    normalized = stt_extract.normalize_text(text)
    segments = [{
        "id": 0,
        "text": text,
        "start": 0.0,
        "end": 1.0,
        "lang": "en",
        "confidence": 0.9,
        "words": [],
    }]
    return normalized, stt_extract.extract_entities(text, normalized["normalized_en"], segments)


def test_entity_spans_distinguish_unique_ambiguous_and_not_found():
    normalized, unique = _extract("Paracetamol 500 mg once daily.")
    drug = unique["drugs"][0]
    assert drug["name"] == "paracetamol"
    assert drug["dose"] == 500.0
    assert drug["span_status"] == "exact"
    assert normalized["normalized_en"][drug["start_char"]:drug["end_char"]] == "Paracetamol"

    _, ambiguous = _extract(
        "Fever noted. Paracetamol 500 mg once daily. Fever returned."
    )
    fever_spans = [item["span_status"] for item in ambiguous["symptoms"] if item["text"] == "fever"]
    assert fever_spans == ["ambiguous", "ambiguous"]
    assert all(
        item["start_char"] == -1 and item["end_char"] == -1
        for item in ambiguous["symptoms"]
    )

    _, missing = _extract("Give asitromaisin 500 mg once daily.")
    fuzzy_drug = missing["drugs"][0]
    assert fuzzy_drug["name"] == "azithromycin"
    assert fuzzy_drug["span_status"] == "not_found"
    assert fuzzy_drug["start_char"] == -1
    assert fuzzy_drug["end_char"] == -1


def test_every_extracted_entity_has_deterministic_spans_and_schema_accepts_them():
    text = "No penicillin allergy. Diagnosis: pneumonia. Review after three days."
    normalized, entities = _extract(text)
    again = _extract(text)[1]

    for group in ("drugs", "symptoms", "vitals", "allergies", "negations"):
        for entity in entities[group]:
            assert set(("start_char", "end_char", "span_status")) <= entity.keys()
    for group in ("diagnosis", "followup"):
        if entities[group]:
            assert set(("start_char", "end_char", "span_status")) <= entities[group].keys()
    assert entities == again

    schema = json.loads(
        (Path(_DEMO) / "templates" / "er_discharge.schema.json").read_text(encoding="utf-8")
    )
    Draft7Validator.check_schema(schema)
    payload = {"job_id": "contract", **entities}
    Draft7Validator(schema).validate(payload)

    invalid = copy.deepcopy(payload)
    invalid["allergies"][0]["span_status"] = "exact"
    invalid["allergies"][0]["start_char"] = -1
    invalid["allergies"][0]["end_char"] = -1
    with pytest.raises(ValidationError):
        Draft7Validator(schema).validate(invalid)


# ---- medasr is the default decoder, so its failure path is load-bearing ----

def test_medasr_degrades_to_a_real_decoder_not_to_the_mock(monkeypatch):
    """A missing MedASR must not hand back invented clinical text.

    `_mock_result` returns a fixture transcript carrying vitals and drugs. That
    is a reasonable last resort for an explicitly-requested backend, but MedASR
    is now the default, so on any machine without the weights the default path
    would have produced a chart full of facts nobody dictated. It degrades to
    faster-whisper first, and says so in the engine string.
    """
    import stt_extract

    class Segment:
        text = " Paracetamol 500 mg once daily. "
        start, end, avg_logprob, words = 0.0, 2.0, -0.1, ()

    class WhisperModel:
        model_sha256 = "sha256:model"
        model_snapshot = "snapshot-123"

        def __init__(self, model, *, device, compute_type):
            pass

        def transcribe(self, audio, **kwargs):
            return iter([Segment()]), SimpleNamespace(language="en")

    _fake_faster_whisper(monkeypatch, WhisperModel)
    monkeypatch.setattr(stt_extract, "transcribe_medasr",
                        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("no weights")))
    result = stt_extract.transcribe("a.wav", job_id="x", model="medasr", strict=False)

    assert result["stt_provenance"]["is_mock"] is False
    assert "faster-whisper" in result["engine"]
    assert "medasr unavailable" in result["engine"]


def test_medasr_still_raises_under_strict(monkeypatch):
    import stt_extract

    monkeypatch.setattr(stt_extract, "transcribe_medasr",
                        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("no weights")))
    with pytest.raises(RuntimeError):
        stt_extract.transcribe("b.wav", job_id="x", model="medasr", strict=True)

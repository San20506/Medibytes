"""Spec: openspec/changes/raw-asr-decoding-presets (presets, model reuse, medasr rejection)."""
from types import ModuleType, SimpleNamespace
import sys

import pytest

from base_pipeline import decode_presets, stt_extract


def _fake_whisper(monkeypatch):
    seen = {"built": 0, "calls": []}

    class Segment:
        text = " Paracetamol 500 mg. "
        start, end, avg_logprob, words = 0.0, 1.0, -0.1, ()

    class WhisperModel:
        def __init__(self, *_a, **_k):
            seen["built"] += 1

        def transcribe(self, audio, **kwargs):
            seen["calls"].append(kwargs)
            return iter([Segment()]), SimpleNamespace(language="en")

    mod = ModuleType("faster_whisper")
    mod.WhisperModel, mod.__version__ = WhisperModel, "1.0"
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)
    return seen


def test_each_preset_adds_exactly_one_change_over_its_predecessor():
    names = list(decode_presets.PRESETS)
    assert names[0] == decode_presets.DEFAULT == "greedy"
    for prev, cur in zip(names, names[1:]):
        a, b = decode_presets.get(prev), decode_presets.get(cur)
        changed = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
        assert len(changed) == 1, (prev, cur, changed)


@pytest.mark.parametrize("name", list(decode_presets.PRESETS))
def test_transcribe_uses_exactly_the_presets_arguments(monkeypatch, name):
    seen = _fake_whisper(monkeypatch)
    res = stt_extract.transcribe("x.wav", model="base-int8", strict=True, decode=name)
    assert seen["calls"][-1] == decode_presets.kwargs(name)
    assert res["stt_provenance"]["decode_preset"] == name
    assert res["stt_provenance"]["decode_kwargs"] == decode_presets.kwargs(name)


def test_unknown_preset_fails_before_any_decode_and_names_the_valid_ones(monkeypatch):
    seen = _fake_whisper(monkeypatch)
    with pytest.raises(ValueError, match="beam5-en-fallback-prompt"):
        stt_extract.transcribe("x.wav", model="base-int8", decode="turbo")
    assert seen["built"] == 0 and seen["calls"] == []


@pytest.mark.parametrize("model", ["medasr", "medasr-lm"])
def test_a_non_default_preset_is_rejected_for_medasr_not_ignored(model):
    with pytest.raises(ValueError, match="faster-whisper models only"):
        stt_extract.transcribe("x.wav", model=model, decode="beam5")


def test_the_model_is_built_once_per_process_and_reused(monkeypatch):
    seen = _fake_whisper(monkeypatch)
    stt_extract.transcribe("a.wav", model="tiny-int8", strict=True)
    stt_extract.transcribe("b.wav", model="tiny-int8", strict=True, decode="beam5")
    assert seen["built"] == 1 and len(seen["calls"]) == 2

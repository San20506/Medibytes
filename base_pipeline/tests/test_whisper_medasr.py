"""The whisper -> medical clips -> MedASR variant: SNR gate, clip isolation, stitching."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from base_pipeline import whisper_medasr as wm
from base_pipeline.audio_clean import estimate_snr

SR = 16000


def _bursty(noise_rms: float, seconds: int = 12, seed: int = 0) -> np.ndarray:
    """Speech-like: 1 s of a 220 Hz tone (rms 0.1), 0.5 s of pause, repeated, over white noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    tone = 0.1 * np.sqrt(2) * np.sin(2 * np.pi * 220 * t)
    gate = ((t % 1.5) < 1.0).astype(np.float32)
    return (tone * gate + rng.normal(0, noise_rms, t.size)).astype(np.float32)


def test_snr_tracks_the_noise_level_and_is_ordered():
    clean, mid, noisy = (estimate_snr(_bursty(n), SR) for n in (0.001, 0.01, 0.05))
    assert clean > mid > noisy
    assert clean > 30          # true SNR ~40 dB
    assert 12 < mid < 28       # true SNR ~20 dB
    assert noisy < 12          # true SNR ~6 dB


def test_snr_of_a_stationary_signal_is_near_zero_so_it_gets_denoised():
    t = np.arange(SR * 3) / SR
    assert estimate_snr((0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), SR) < 3


def _words(text, start, step=0.4):
    return [{"w": w, "s": round(start + i * step, 2), "e": round(start + (i + 1) * step, 2)}
            for i, w in enumerate(text.split())]


def _seg(text, start):
    words = _words(text, start)
    return {"text": text, "start": start, "end": words[-1]["e"], "words": words}


def test_isolate_keeps_the_medical_stretch_padded_and_drops_small_talk():
    chatter = _seg("good morning how are you doing today", 0.0)
    medical = _seg("blood pressure is 130 over 80 and pulse 72", 20.0)
    spans = wm.isolate_medical({"segments": [chatter, medical]}, pad_s=1.0)
    assert len(spans) == 1
    assert spans[0]["start"] >= 18.9 and spans[0]["end"] <= medical["end"] + 1.0 + 1e-6
    assert "blood pressure" in spans[0]["whisper_text"]
    assert "good morning" not in spans[0]["whisper_text"]


def test_isolate_keeps_a_drug_order_whose_name_whisper_garbled():
    # "amoxicin" is in no formulary and has no dose, so the extractor finds nothing;
    # the medication cue ("started on") is what keeps the sentence.
    seg = _seg("she was started on amoxicin this morning for a throat infection", 5.0)
    assert len(wm.isolate_medical({"segments": [seg]})) == 1
    assert wm.isolate_medical({"segments": [_seg("the weather was lovely this morning", 5.0)]}) == []


def test_isolate_merges_neighbours_and_splits_long_slices():
    a = _seg("bp 120 over 80", 10.0)
    b = _seg("pulse 70", 12.5)
    assert len(wm.isolate_medical({"segments": [a, b]})) == 1
    long = {"text": "pulse 70", "start": 0.0, "end": 50.0,
            "words": [{"w": "pulse", "s": 0.0, "e": 0.4}, {"w": "70", "s": 48.0, "e": 48.4}]}
    spans = wm.isolate_medical({"segments": [long]}, pad_s=0.0, clip_s=20.0)
    assert len(spans) == 3 and all(s["end"] - s["start"] <= 20.0 + 1e-6 for s in spans)
    # a span just over the cap splits evenly, never leaving a sliver MedASR cannot chunk
    over = {"text": "pulse 70", "start": 0.0, "end": 20.5, "words": long["words"]}
    halves = wm.isolate_medical({"segments": [over]}, pad_s=0.0, clip_s=20.0)
    assert len(halves) == 2 and min(h["end"] - h["start"] for h in halves) > 10


def _wav(path: Path, seconds: float) -> None:
    pcm = np.zeros(int(seconds * SR), dtype="<i2")
    with wave.open(str(path), "wb") as h:
        h.setnchannels(1)
        h.setsampwidth(2)
        h.setframerate(SR)
        h.writeframes(pcm.tobytes())


def _fake_medasr(monkeypatch, texts):
    calls = []

    def fake(path, job_id="x"):
        calls.append(Path(path).name)
        with wave.open(str(path), "rb") as h:
            dur = h.getnframes() / h.getframerate()
        text = texts[len(calls) - 1]
        return {"text": text, "engine": "medasr:fake", "language": "en",
                "segments": [{"id": 0, "text": text, "start": 0.0, "end": round(dur, 2)}],
                "stt_provenance": {"is_mock": False}}

    monkeypatch.setattr(wm, "transcribe_medasr", fake)
    return calls


def test_clips_are_decoded_separately_and_stitched_in_time_order(tmp_path, monkeypatch):
    src = tmp_path / "c.wav"
    _wav(src, 30.0)
    calls = _fake_medasr(monkeypatch, ["bp 120 over 80", "pulse 72"])
    spans = [{"start": 2.0, "end": 6.0, "whisper_text": "w1"},
             {"start": 20.0, "end": 24.0, "whisper_text": "w2"}]
    res = wm.transcribe_clips(src, spans, tmp_path / "clips")
    assert calls == ["00.wav", "01.wav"]
    assert res["text"] == "bp 120 over 80 pulse 72"
    assert [s["start"] for s in res["segments"]] == [2.0, 20.0]      # original timeline
    assert [c["whisper_text"] for c in res["clips"]] == ["w1", "w2"]


def test_no_medical_clip_falls_back_to_the_whole_recording_and_says_so(tmp_path, monkeypatch):
    src = tmp_path / "c.wav"
    _wav(src, 3.0)
    _fake_medasr(monkeypatch, ["pulse 72"])
    res = wm.transcribe_clips(src, [], tmp_path / "clips")
    assert res["text"] == "pulse 72" and "fallback" in res["engine"] and res["clips"] == []


@pytest.mark.parametrize("snr,denoised", [(30.0, False), (6.0, True)])
def test_gate_denoises_only_below_threshold(tmp_path, monkeypatch, snr, denoised):
    backends = []

    def fake_clean(src, out, backend=None, config=None):
        backends.append(backend.value)
        _wav(out, 2.0)
        return {"snr_db": snr, "actual_backend": backend.value}

    monkeypatch.setattr(wm, "clean_audio", fake_clean)
    monkeypatch.setattr(wm, "transcribe_full", lambda *a, **k: {
        "text": "", "segments": [], "engine": "whisper:fake"})
    _fake_medasr(monkeypatch, ["pulse 72"])
    monkeypatch.setattr("base_pipeline.run.denoiser_config", lambda b: None)
    r = wm.run_variant(tmp_path / "in.wav", tmp_path / "arm", snr_threshold_db=15.0)
    assert r["gate"]["denoised"] is denoised
    assert backends == (["none", wm.DENOISER] if denoised else ["none"])
    for name in ("gate.json", "entities.json", "clips.json", "whisper_full.json", "report.md"):
        assert (tmp_path / "arm" / name).is_file()

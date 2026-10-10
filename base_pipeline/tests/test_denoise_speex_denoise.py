"""Real pinned SpeexDSP denoise-only behavior and configuration isolation."""
from __future__ import annotations

from pathlib import Path
import sys
import wave

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]

from base_pipeline.audio_clean import clean_audio
from base_pipeline.denoise import BackendId, BackendUnavailableError, EnhancementConfig, run_backend
from base_pipeline.denoise_backends import speex_denoise as adapter
from base_pipeline.paths import data_root

CORPUS = data_root() / "corpus/pilot-15-v1/noisy/sample-001.wav"


def config(**overrides: object) -> EnhancementConfig:
    fields = dict(backend=BackendId.SPEEX_DENOISE, variant_id=adapter.VARIANT,
                  model_path=None, model_sha256=None, parameters=adapter.PARAMETERS)
    fields.update(overrides)
    return EnhancementConfig(**fields)


def _corpus() -> np.ndarray:
    with wave.open(str(CORPUS), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768


def test_probe_reports_pinned_model_free_identity() -> None:
    probe = adapter.probe(config())
    assert probe.available is True
    assert probe.source_revision == "7a158783df74efe7c2d1c6ee8363c1e695c71226"
    assert probe.runtime_version == "1.2.1"
    assert probe.model_sha256 is None
    assert probe.native_sample_rate == 16000
    assert probe.stateful is True


def test_real_denoise_is_deterministic_and_reduces_noise() -> None:
    audio = _corpus()
    output = run_backend(audio, 16000, config())
    repeated = run_backend(audio, 16000, config())
    interleaved = run_backend(audio[::-1].copy(), 16000, config())

    assert output.actual_backend == BackendId.SPEEX_DENOISE
    assert output.audio.size == audio.size
    assert output.native_sample_rate == 16000 and output.delay_samples == 0
    assert np.array_equal(output.audio, repeated.audio)
    # A fresh state per call must not make reversed audio a fixed point of the map.
    assert not np.array_equal(output.audio, interleaved.audio)
    assert not np.array_equal(output.audio, audio)
    assert output.metadata["native_frame_samples"] == 160
    assert output.metadata["native_frames_processed"] == audio.size // 160
    assert output.metadata["noise_suppress_db"] < 0
    assert output.provenance["fallback_used"] is False
    assert output.provenance["denoise"] is True
    for disabled in ("agc", "vad", "dereverb", "residual_echo_suppression"):
        assert output.provenance[disabled] is False
    assert output.provenance["state_reset_per_call"] is True



def test_real_clean_audio_preserves_wav_geometry(tmp_path: Path) -> None:
    target = tmp_path / "enhanced.wav"
    meta = clean_audio(CORPUS, target, backend=BackendId.SPEEX_DENOISE, config=config())
    with wave.open(str(CORPUS), "rb") as source, wave.open(str(target), "rb") as enhanced:
        assert (enhanced.getframerate(), enhanced.getnchannels(), enhanced.getsampwidth()) == (16000, 1, 2)
        assert enhanced.getnframes() == source.getnframes()
    assert meta["actual_backend"] == "speex-denoise"
    assert meta["fallback_used"] is False
    assert meta["sample_count_drift"] == 0


def test_refuses_model_claims_parameter_drift_and_wrong_rate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exact requested backend"):
        adapter.probe(config(variant_id="1.2.1-denoise-only-and-more"))
    for drifted in (
        {**adapter.PARAMETERS, "denoise": False},
        {**adapter.PARAMETERS, "agc": True},
        {**adapter.PARAMETERS, "frame_size": 320},
        {**adapter.PARAMETERS, "residual_echo_suppression": True},
    ):
        with pytest.raises(ValueError, match="parameters"):
            adapter.probe(config(parameters=drifted))
    with pytest.raises(ValueError, match="model-free"):
        adapter.probe(config(model_sha256="0" * 64))
    with pytest.raises(ValueError, match="16 kHz"):
        adapter.enhance(np.ones(1600, dtype=np.float32), 48000, config())
    monkeypatch.setattr(adapter, "LIBRARY_PATH", tmp_path / "libspeexdsp.so")
    with pytest.raises(BackendUnavailableError, match="library is missing"):
        run_backend(np.ones(1600, dtype=np.float32), 16000, config())

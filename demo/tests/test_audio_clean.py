"""Observable product-cleaner behavior tests."""
from __future__ import annotations

import hashlib
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_DEMO = _ROOT / "demo"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_DEMO) not in sys.path:
    sys.path.insert(0, str(_DEMO))

from demo.audio_clean import clean_audio
from demo.denoise import (
    BackendContractError,
    BackendId,
    BackendProbe,
    BackendUnavailableError,
    EnhancementConfig,
    EnhancementOutput,
    run_backend,
)


def _tone(path: Path, seconds: float = 1.0, sample_rate: int = 16000) -> None:
    samples = (0.4 * np.sin(2 * np.pi * 440 * np.arange(round(seconds * sample_rate)) / sample_rate)).astype(np.float32)
    pcm = (samples * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def test_none_cleaner_is_deterministic_and_reports_measured_provenance(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    _tone(source)

    first_meta = clean_audio(source, first)
    second_meta = clean_audio(source, second)

    assert first_meta["requested_backend"] == BackendId.NONE.value
    assert first_meta["actual_backend"] == BackendId.NONE.value
    assert first_meta["variant_id"] == BackendId.NONE.value
    assert first_meta["fallback_used"] is False
    assert first_meta["input_sample_count"] == first_meta["output_sample_count"]
    assert first_meta["duration_drift_s"] == 0.0
    assert first_meta["output_sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    assert first.read_bytes() == second.read_bytes()
    assert first_meta["output_sha256"] == second_meta["output_sha256"]
    assert "rms_dbfs_before" in first_meta and "rms_dbfs_after" in first_meta
    assert "snr_before_db" not in first_meta
    assert "snr_after_db" not in first_meta
    assert "preprocess_gain_linear" in first_meta
    assert first_meta["none_reference_sha256"] == first_meta["output_sha256"]
    required_provenance = {
        "actual_backend",
        "backend",
        "device",
        "fallback_used",
        "model_sha256",
        "parameters",
        "provider",
        "requested_backend",
        "runtime_version",
        "source_revision",
        "variant_id",
    }
    assert required_provenance <= first_meta["provenance"].keys()
    assert first_meta["provenance"]["actual_backend"] == BackendId.NONE.value
    assert first_meta["provenance"]["fallback_used"] is False
    assert first_meta["native_sample_rate"] == 16_000
    assert first_meta["delay_samples"] == 0


def test_noisereduce_missing_is_a_hard_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    _tone(source)
    import demo.denoise_backends.noisereduce as adapter

    def unavailable():
        raise BackendUnavailableError("BACKEND_UNAVAILABLE: injected missing dependency")

    monkeypatch.setattr(adapter, "_package", unavailable)
    with pytest.raises(BackendUnavailableError, match="BACKEND_UNAVAILABLE"):
        clean_audio(source, tmp_path / "out.wav", backend=BackendId.NOISEREDUCE)


class _FakeAdapter:
    @staticmethod
    def probe(config: EnhancementConfig) -> BackendProbe:
        return BackendProbe(
            available=True,
            reason="test adapter",
            source_revision="test-revision",
            runtime_version="test-runtime",
            native_sample_rate=16000,
            delay_samples=0,
        )

    @staticmethod
    def enhance(audio: np.ndarray, sample_rate: int, config: EnhancementConfig) -> EnhancementOutput:
        return EnhancementOutput(
            audio=audio.copy(),
            actual_backend=BackendId.NONE,
            variant_id=config.variant_id,
            native_sample_rate=16000,
            delay_samples=0,
        )


@pytest.mark.parametrize(
    "audio",
    [
        np.empty(0, dtype=np.float32),
        np.zeros(160, dtype=np.float32),
        np.full(160, np.nan, dtype=np.float32),
        np.zeros((2, 160), dtype=np.float32),
        np.ones(159, dtype=np.float32),
    ],
)
def test_run_backend_fails_closed_on_invalid_adapter_output(monkeypatch: pytest.MonkeyPatch, audio: np.ndarray) -> None:
    import demo.denoise as contract

    adapter = _FakeAdapter()

    def invalid_enhance(_audio, _sample_rate, config):
        return EnhancementOutput(
            audio=audio,
            actual_backend=BackendId.NONE,
            variant_id=config.variant_id,
            native_sample_rate=16000,
            delay_samples=0,
        )

    monkeypatch.setattr(adapter, "enhance", staticmethod(invalid_enhance))
    monkeypatch.setattr(contract.importlib, "import_module", lambda _name: adapter)
    config = EnhancementConfig(backend=BackendId.NONE, variant_id="none")
    with pytest.raises(BackendContractError):
        run_backend(np.ones(160, dtype=np.float32), 16000, config)


def test_run_backend_rejects_fallback_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    import demo.denoise as contract

    class FallbackAdapter(_FakeAdapter):
        @staticmethod
        def enhance(audio, sample_rate, config):
            return EnhancementOutput(
                audio=audio.copy(),
                actual_backend=BackendId.NONE,
                variant_id=config.variant_id,
                native_sample_rate=16000,
                delay_samples=0,
                metadata={"fallback_used": True},
            )

    monkeypatch.setattr(contract.importlib, "import_module", lambda _name: FallbackAdapter())
    config = EnhancementConfig(backend=BackendId.NONE, variant_id="none")
    with pytest.raises(BackendContractError, match="fallback"):
        run_backend(np.ones(160, dtype=np.float32), 16000, config)

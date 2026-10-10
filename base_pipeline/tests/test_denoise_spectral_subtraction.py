"""Real pinned spectral subtraction: geometry, determinism, state isolation."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import wave

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]

from base_pipeline.audio_clean import clean_audio
from base_pipeline.denoise import BackendId, EnhancementConfig, run_backend
from base_pipeline.denoise_backends import spectral_subtraction as adapter
from base_pipeline.paths import data_root

CORPUS = data_root() / "corpus/pilot-15-v1/noisy/sample-001.wav"


needs_runtime = pytest.mark.skipif(
    importlib.util.find_spec("pyroomacoustics") is None,
    reason="pyroomacoustics is not installed here; the adapter reports BACKEND_UNAVAILABLE (install base_pipeline/denoise_backends/requirements-spectral-subtraction.txt)",
)


def config(**overrides: object) -> EnhancementConfig:
    fields = dict(backend=BackendId.SPECTRAL_SUBTRACTION, variant_id=adapter.VARIANT,
                  model_path=None, model_sha256=None, parameters=adapter.PARAMETERS)
    fields.update(overrides)
    return EnhancementConfig(**fields)


def _corpus() -> np.ndarray:
    with wave.open(str(CORPUS), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768


@needs_runtime
def test_probe_reports_the_pinned_model_free_identity() -> None:
    probe = adapter.probe(config())
    assert probe.available is True
    assert probe.source_revision == "f02b01dd6609709e2089aefa5d1e59c91d3a0601"
    assert probe.runtime_version == "0.10.1"
    assert probe.model_sha256 is None
    assert probe.native_sample_rate == 16000


@needs_runtime
def test_real_subtraction_preserves_geometry_and_is_deterministic() -> None:
    audio = _corpus()
    first = run_backend(audio, 16000, config())
    second = run_backend(audio, 16000, config())
    reversed_output = run_backend(audio[::-1].copy(), 16000, config()).audio

    assert first.actual_backend == BackendId.SPECTRAL_SUBTRACTION
    assert first.audio.size == audio.size
    assert first.native_sample_rate == 16000 and first.delay_samples == 0
    assert np.array_equal(first.audio, second.audio)
    # The lookback noise estimate is rebuilt per call, so a prior sample cannot
    # bias this one; a shared-state implementation would collapse these.
    assert not np.array_equal(first.audio, reversed_output)
    assert not np.array_equal(first.audio, audio)
    assert np.any(first.audio[-100:] != 0)
    assert first.metadata["algorithm"] == "apply_spectral_sub"
    assert first.metadata["config_sha256"] == adapter.CONFIG_SHA256
    assert first.provenance["fallback_used"] is False
    assert first.provenance["state_reset_per_call"] is True
    assert first.provenance["config_sha256"] == adapter.CONFIG_SHA256


@needs_runtime
def test_pinned_aggressiveness_is_measured_not_assumed() -> None:
    # The frozen row pins db_reduc=10, beta=3, alpha=1.2. Measured on a 400 Hz
    # tone at 0.2 amplitude under seeded broadband noise, that configuration
    # raises residual error against the reference rather than lowering it,
    # because the minimum-power noise estimate tracks the tone itself. This test
    # pins that measured direction so a future parameter change cannot silently
    # flip the arm's character; the corpus run is the authority on quality.
    rate = 16000
    axis = np.arange(rate * 2, dtype=np.float64) / rate
    tone = 0.2 * np.sin(2.0 * np.pi * 400.0 * axis)
    generator = np.random.default_rng(1729)
    noisy = (tone + generator.standard_normal(axis.size) * 0.02).astype(np.float32)
    output = run_backend(noisy, rate, config()).audio[:noisy.size].astype(np.float64)
    before = float(np.sqrt(np.mean((noisy - tone) ** 2)))
    after = float(np.sqrt(np.mean((output - tone) ** 2)))
    assert after > before

@needs_runtime
def test_real_clean_audio_preserves_wav_geometry(tmp_path: Path) -> None:
    target = tmp_path / "enhanced.wav"
    meta = clean_audio(CORPUS, target, backend=BackendId.SPECTRAL_SUBTRACTION, config=config())
    with wave.open(str(CORPUS), "rb") as source, wave.open(str(target), "rb") as enhanced:
        assert (enhanced.getframerate(), enhanced.getnchannels(), enhanced.getsampwidth()) == (16000, 1, 2)
        assert enhanced.getnframes() == source.getnframes()
    assert meta["actual_backend"] == "spectral-subtraction"
    assert meta["fallback_used"] is False
    assert meta["sample_count_drift"] == 0


@needs_runtime
def test_refuses_model_claims_parameter_drift_and_wrong_rate(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exact requested backend"):
        adapter.probe(config(variant_id="pyroomacoustics-0.10.1-ss-wiener"))
    for drifted in (
        {**adapter.PARAMETERS, "db_reduc": 25},
        {**adapter.PARAMETERS, "nfft": 1024},
        {**adapter.PARAMETERS, "beta": 30},
        {**adapter.PARAMETERS, "alpha": 1.0},
        {**adapter.PARAMETERS, "lookback": 6},
    ):
        with pytest.raises(ValueError, match="parameters"):
            adapter.probe(config(parameters=drifted))
    with pytest.raises(ValueError, match="model-free"):
        adapter.probe(config(model_sha256="0" * 64))
    with pytest.raises(ValueError, match="16 kHz"):
        adapter.enhance(np.ones(8000, dtype=np.float32), 8000, config())

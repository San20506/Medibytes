"""Real pinned MetricGAN+ inference: offline load, geometry, determinism, isolation."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import wave

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]

from base_pipeline.audio_clean import clean_audio
from base_pipeline.denoise import BackendId, BackendUnavailableError, EnhancementConfig, run_backend
from base_pipeline.denoise_backends import speechbrain_metricgan_plus as adapter
from base_pipeline.paths import data_root

CORPUS = data_root() / "corpus/pilot-15-v1/noisy/sample-001.wav"


needs_runtime = pytest.mark.skipif(
    importlib.util.find_spec("speechbrain") is None,
    reason="speechbrain is not installed here; the adapter reports BACKEND_UNAVAILABLE (install base_pipeline/denoise_backends/requirements-speechbrain-metricgan-plus.txt)",
)


def config(**overrides: object) -> EnhancementConfig:
    fields = dict(backend=BackendId.SPEECHBRAIN_METRICGAN_PLUS, variant_id=adapter.VARIANT,
                  model_path=adapter.CHECKPOINT, model_sha256=adapter.MODEL_SHA256,
                  parameters=adapter.PARAMETERS)
    fields.update(overrides)
    return EnhancementConfig(**fields)


def _corpus() -> np.ndarray:
    with wave.open(str(CORPUS), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768


@needs_runtime
def test_probe_reports_the_pinned_offline_identity() -> None:
    probe = adapter.probe(config())
    assert probe.available is True
    assert probe.source_revision == "89ead74d163463d30c62329a09cfdb4c54f5abc1"
    assert probe.runtime_version == "1.1.1"
    assert probe.model_sha256 == adapter.MODEL_SHA256
    assert probe.native_sample_rate == 16000
    assert probe.stateful is True


@needs_runtime
def test_real_model_is_deterministic_and_state_isolated() -> None:
    audio = _corpus()
    first = run_backend(audio, 16000, config())
    reversed_output = run_backend(audio[::-1].copy(), 16000, config()).audio
    second = run_backend(audio, 16000, config())

    assert first.actual_backend == BackendId.SPEECHBRAIN_METRICGAN_PLUS
    assert first.audio.size == audio.size
    assert first.native_sample_rate == 16000 and first.delay_samples == 0
    assert np.array_equal(first.audio, second.audio)
    assert not np.array_equal(first.audio, reversed_output)
    assert not np.array_equal(first.audio, audio)
    assert np.any(first.audio[-100:] != 0)
    assert first.provenance["fallback_used"] is False
    assert first.provenance["architecture"] == "SpectralMaskEnhancement"
    assert first.provenance["network_access"] == "disabled"
    assert first.provenance["state_reset_per_call"] is True
    assert first.provenance["device"] == "cpu"
    assert first.metadata["model_per_call"] == 1


@needs_runtime
def test_every_call_builds_a_fresh_model_from_the_pinned_files() -> None:
    first = run_backend(_corpus()[:16000], 16000, config())
    second = run_backend(_corpus()[:16000], 16000, config())
    # Same weights and same input must give the same mask; a pooled or cached
    # model would drift once a second sample has been seen.
    assert np.array_equal(first.audio, second.audio)
    assert first.provenance["model_sha256"] == adapter.MODEL_SHA256
    assert first.provenance["hyperparams_sha256"] == adapter.HYPERPARAMS_SHA256
    assert first.provenance["model_revision"] == adapter.MODEL_REVISION
    # Only this architecture, never a SepFormer or SEGAN substitute.
    assert "sepformer" not in str(first.provenance).casefold()
    assert "segan" not in str(first.provenance).casefold()


@needs_runtime
def test_real_clean_audio_preserves_wav_geometry(tmp_path: Path) -> None:
    target = tmp_path / "enhanced.wav"
    meta = clean_audio(CORPUS, target, backend=BackendId.SPEECHBRAIN_METRICGAN_PLUS, config=config())
    with wave.open(str(CORPUS), "rb") as source, wave.open(str(target), "rb") as enhanced:
        assert (enhanced.getframerate(), enhanced.getnchannels(), enhanced.getsampwidth()) == (16000, 1, 2)
        assert enhanced.getnframes() == source.getnframes()
    assert meta["actual_backend"] == "speechbrain-metricgan-plus"
    assert meta["fallback_used"] is False
    assert meta["model_sha256"] == adapter.MODEL_SHA256
    assert meta["sample_count_drift"] == 0


@needs_runtime
def test_refuses_substituted_checkpoint_variant_and_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="exact requested backend"):
        adapter.probe(config(variant_id="metricgan-plus-other"))
    for drifted in (
        {**adapter.PARAMETERS, "device": "cuda"},
        {**adapter.PARAMETERS, "sample_rate": 8000},
        {**adapter.PARAMETERS, "model": "speechbrain/sepformer-wsj02mix"},
    ):
        with pytest.raises(ValueError, match="parameters"):
            adapter.probe(config(parameters=drifted))
    with pytest.raises(ValueError, match="non-symlink"):
        adapter.probe(config(model_path=tmp_path / "enhance_model.ckpt"))
    with pytest.raises(ValueError, match="null or the pinned digest"):
        adapter.probe(config(model_sha256="0" * 64))
    with pytest.raises(ValueError, match="16 kHz"):
        adapter.enhance(np.ones(1600, dtype=np.float32), 8000, config())
    monkeypatch.setattr(adapter, "HYPERPARAMS_SHA256", "0" * 64)
    with pytest.raises(BackendUnavailableError, match="hyperparams.yaml digest"):
        run_backend(np.ones(1600, dtype=np.float32), 16000, config())

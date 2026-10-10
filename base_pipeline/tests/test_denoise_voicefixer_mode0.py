"""Real pinned VoiceFixer mode-0 restoration and fail-closed checkpoint handling."""
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
from base_pipeline.denoise_backends import voicefixer_mode0 as adapter
from base_pipeline.paths import data_root

CORPUS = data_root() / "corpus/pilot-15-v1/noisy/sample-001.wav"
SHORT = CORPUS


needs_runtime = pytest.mark.skipif(
    importlib.util.find_spec("voicefixer") is None,
    reason="voicefixer is not installed here; the adapter reports BACKEND_UNAVAILABLE (install base_pipeline/denoise_backends/requirements-voicefixer-mode0.txt)",
)


def config(**overrides: object) -> EnhancementConfig:
    fields = dict(backend=BackendId.VOICEFIXER_MODE0, variant_id=adapter.VARIANT,
                  model_path=adapter.MODEL_DIR, model_sha256=None, parameters=adapter.PARAMETERS)
    fields.update(overrides)
    return EnhancementConfig(**fields)


def _corpus() -> np.ndarray:
    with wave.open(str(CORPUS), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768


@needs_runtime
def test_probe_reports_the_pinned_restoration_identity() -> None:
    probe = adapter.probe(config())
    assert probe.available is True
    assert probe.source_revision == "zenodo-5600188"
    assert probe.runtime_version == "0.1.3"
    assert probe.model_sha256 == adapter.ANALYSIS_SHA256
    assert probe.native_sample_rate == 44100


@needs_runtime
def test_real_restoration_preserves_geometry_and_is_deterministic() -> None:
    audio = _corpus()
    first = run_backend(audio, 16000, config())
    second = run_backend(audio, 16000, config())

    assert first.actual_backend == BackendId.VOICEFIXER_MODE0
    # The adapter is a 44.1 kHz native backend; the shared cleaner resamples back.
    assert first.native_sample_rate == 44100
    assert first.delay_samples == 0
    assert first.audio.size == int(round(audio.size * 44100 / 16000))
    assert first.metadata["native_output_samples"] == first.metadata["native_input_samples"]
    assert first.metadata["execution_class"] == "generative_restoration"
    assert first.metadata["analysis_ckpt_md5"] == adapter.ANALYSIS_MD5
    assert np.array_equal(first.audio, second.audio)
    assert not np.array_equal(first.audio, np.zeros_like(first.audio))
    assert first.provenance["fallback_used"] is False
    assert first.provenance["mode"] == 0
    assert first.provenance["cuda"] is False
    assert first.provenance["device"] == "cpu"


@needs_runtime
def test_real_clean_audio_round_trips_back_to_16_khz(tmp_path: Path) -> None:
    target = tmp_path / "restored.wav"
    meta = clean_audio(SHORT, target, backend=BackendId.VOICEFIXER_MODE0, config=config())
    with wave.open(str(SHORT), "rb") as source, wave.open(str(target), "rb") as restored:
        assert (restored.getframerate(), restored.getnchannels(), restored.getsampwidth()) == (16000, 1, 2)
        assert restored.getnframes() == source.getnframes()
    assert meta["actual_backend"] == "voicefixer-mode0"
    assert meta["fallback_used"] is False
    assert meta["sample_count_drift"] == 0


def test_truncated_analysis_checkpoint_is_refused_not_loaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # VoiceFixer loads the analysis module with strict=False, so a short file
    # would otherwise load silently. The adapter must fail closed instead.
    truncated = tmp_path / "vf.ckpt"
    truncated.write_bytes(adapter.ANALYSIS_SOURCE.read_bytes()[:1_000_000])
    monkeypatch.setattr(adapter, "ANALYSIS_SOURCE", truncated)
    with pytest.raises(BackendUnavailableError, match="bytes, expected"):
        run_backend(np.ones(16000, dtype=np.float32) * 0.1, 16000, config())


@needs_runtime
def test_refuses_substituted_variant_mode_and_device(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exact requested backend"):
        adapter.probe(config(variant_id="0.1.3-mode1"))
    for drifted in (
        {**adapter.PARAMETERS, "mode": 1},
        {**adapter.PARAMETERS, "mode": 2},
        {**adapter.PARAMETERS, "cuda": True},
        {**adapter.PARAMETERS, "execution_class": "denoising"},
    ):
        with pytest.raises(ValueError, match="parameters"):
            adapter.probe(config(parameters=drifted))
    with pytest.raises(ValueError, match="non-symlink"):
        adapter.probe(config(model_path=tmp_path / "voicefixer"))
    with pytest.raises(ValueError, match="null or the pinned analysis digest"):
        adapter.probe(config(model_sha256="0" * 64))
    with pytest.raises(ValueError, match="16 kHz"):
        adapter.enhance(np.ones(1600, dtype=np.float32), 44100, config())

"""Pinned DPDFNet2 model identity, state isolation, and fail-closed behavior."""
from __future__ import annotations

from pathlib import Path
import wave

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]

from base_pipeline.audio_clean import clean_audio
from base_pipeline.denoise import BackendId, BackendUnavailableError, EnhancementConfig, run_backend
from base_pipeline.denoise_backends import dpdfnet2_onnx as adapter
from base_pipeline.paths import data_root
from arm_support import require_arm

arm_available = require_arm(BackendId.DPDFNET2_ONNX)

MODEL = data_root() / adapter.MODEL_RELATIVE
CORPUS = data_root() / "corpus/pilot-15-v1/noisy/sample-001.wav"


def config(**overrides: object) -> EnhancementConfig:
    values = dict(backend=BackendId.DPDFNET2_ONNX, variant_id=adapter.VARIANT,
                  model_path=MODEL, model_sha256=adapter.MODEL_SHA256,
                  parameters=adapter.PARAMETERS)
    values.update(overrides)
    return EnhancementConfig(**values)


def test_real_model_resets_for_independent_samples() -> None:
    with wave.open(str(CORPUS), "rb") as source:
        assert (source.getframerate(), source.getnchannels(), source.getsampwidth()) == (16000, 1, 2)
        audio = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768
    first = run_backend(audio, 16000, config())
    run_backend(audio[::-1].copy(), 16000, config())
    second = run_backend(audio, 16000, config())
    assert first.actual_backend == BackendId.DPDFNET2_ONNX
    assert first.native_sample_rate == 16000 and first.delay_samples == 0
    assert first.audio.shape == audio.shape
    assert np.array_equal(first.audio, second.audio)
    assert not np.array_equal(first.audio, audio)
    assert first.provenance["model_sha256"] == adapter.MODEL_SHA256
    assert first.provenance["provider"] == "CPUExecutionProvider"
    assert first.provenance["fallback_used"] is False


def test_real_cleaner_preserves_geometry_and_backend(tmp_path: Path) -> None:
    target = tmp_path / "cleaned.wav"
    metadata = clean_audio(CORPUS, target, backend=BackendId.DPDFNET2_ONNX, config=config())
    with wave.open(str(CORPUS), "rb") as source, wave.open(str(target), "rb") as output:
        assert output.getnframes() == source.getnframes()
        assert (output.getframerate(), output.getnchannels(), output.getsampwidth()) == (16000, 1, 2)
    assert metadata["actual_backend"] == "dpdfnet2-onnx"
    assert metadata["fallback_used"] is False
    assert metadata["model_sha256"] == adapter.MODEL_SHA256
    assert metadata["sample_count_drift"] == 0


def test_rejects_model_substitution_and_parameter_drift(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert adapter.probe(config()).available
    with pytest.raises(ValueError, match="exact local"):
        adapter.probe(config(model_path=tmp_path / "dpdfnet2.onnx"))
    with pytest.raises(ValueError, match="parameters"):
        adapter.probe(config(parameters={**adapter.PARAMETERS, "attn_limit_db": 12}))
    with pytest.raises(ValueError, match="SHA-256"):
        adapter.probe(config(model_sha256="0" * 64))
    with pytest.raises(ValueError, match="16 kHz"):
        adapter.enhance(np.ones(1600, dtype=np.float32), 48000, config())
    monkeypatch.setattr(adapter, "MODEL_SHA256", "0" * 64)
    with pytest.raises(BackendUnavailableError, match="digest mismatch"):
        run_backend(np.ones(1600, dtype=np.float32), 16000,
                    config(model_sha256="0" * 64))

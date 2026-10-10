"""Real pinned GTCRN inference, tail preservation, and state isolation."""
from __future__ import annotations

from pathlib import Path
import sys
import wave

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]

from base_pipeline.audio_clean import clean_audio
from base_pipeline.denoise import BackendId, BackendUnavailableError, EnhancementConfig, run_backend
from base_pipeline.denoise_backends import sherpa_gtcrn_simple as adapter
from base_pipeline.paths import data_root

CORPUS = data_root() / "corpus/pilot-15-v1/noisy/sample-001.wav"


def config(**overrides: object) -> EnhancementConfig:
    fields = dict(backend=BackendId.SHERPA_GTCRN_SIMPLE, variant_id=adapter.VARIANT,
                  model_path=adapter.MODEL_PATH, model_sha256=None,
                  parameters=adapter.PARAMETERS)
    fields.update(overrides)
    return EnhancementConfig(**fields)


def test_real_model_retains_partial_final_hop_and_resets_state() -> None:
    with wave.open(str(CORPUS), "rb") as wav:
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
    audio = audio[:16007]  # deliberately not a multiple of the native hop
    first = run_backend(audio, 16000, config())
    run_backend(audio[::-1].copy(), 16000, config())
    second = run_backend(audio, 16000, config())
    assert np.array_equal(first.audio, second.audio)
    assert first.actual_backend == BackendId.SHERPA_GTCRN_SIMPLE
    assert first.audio.size == audio.size
    assert first.native_sample_rate == 16000 and first.delay_samples == 0
    assert first.metadata["native_padding_samples"] == (-audio.size) % adapter.FRAME_HOP
    assert first.metadata["native_output_samples"] == first.metadata["native_input_samples"]
    assert np.any(first.audio[-100:] != 0)
    assert not np.array_equal(first.audio, audio)
    assert first.provenance["model_sha256"] == adapter.MODEL_SHA256
    assert first.provenance["runtime_git_sha"]
    assert first.provenance["provider"] == "cpu"
    assert first.provenance["fallback_used"] is False


def test_real_clean_audio_preserves_wav_geometry(tmp_path: Path) -> None:
    output = tmp_path / "enhanced.wav"
    meta = clean_audio(CORPUS, output, backend=BackendId.SHERPA_GTCRN_SIMPLE, config=config())
    with wave.open(str(CORPUS), "rb") as source, wave.open(str(output), "rb") as enhanced:
        assert (enhanced.getframerate(), enhanced.getnchannels(), enhanced.getsampwidth()) == (16000, 1, 2)
        assert enhanced.getnframes() == source.getnframes()
    assert meta["actual_backend"] == "sherpa-gtcrn-simple"
    assert meta["model_sha256"] == adapter.MODEL_SHA256
    assert meta["fallback_used"] is False
    assert meta["sample_count_drift"] == 0


def test_refuses_changed_backend_model_and_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert adapter.probe(config()).available
    with pytest.raises(ValueError, match="exact requested backend"):
        adapter.probe(config(variant_id="dpdfnet2-onnx"))
    with pytest.raises(ValueError, match="exact local"):
        adapter.probe(config(model_path=tmp_path / "gtcrn_simple.onnx"))
    with pytest.raises(ValueError, match="parameters"):
        adapter.probe(config(parameters={**adapter.PARAMETERS, "num_threads": 2}))
    with pytest.raises(ValueError, match="16 kHz"):
        adapter.enhance(np.ones(1024, dtype=np.float32), 48000, config())
    monkeypatch.setattr(adapter, "MODEL_SHA256", "0" * 64)
    with pytest.raises(BackendUnavailableError, match="digest mismatch"):
        run_backend(np.ones(1024, dtype=np.float32), 16000, config())

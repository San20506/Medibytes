"""Observable RNNoise method behavior against the pinned native runtime."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import wave

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]

from base_pipeline.audio_clean import clean_audio
from base_pipeline.denoise import BackendId, EnhancementConfig, run_backend
pytest.importorskip("scipy", reason="the rnnoise adapter resamples with scipy (see README, Denoiser arms)")
from base_pipeline.denoise_backends import rnnoise as adapter
from base_pipeline.paths import data_root
from arm_support import require_arm

arm_available = require_arm(BackendId.RNNOISE)


DATA_ROOT = data_root()
MODEL_PATH = DATA_ROOT / "models/rnnoise/v0.2-0b50c45/rnnoise_data-0b50c45.tar.gz"
CORPUS_PATH = DATA_ROOT / "corpus/pilot-15-v1/noisy/sample-001.wav"
VARIANT = "v0.2-0b50c45"
PARAMETERS = {"native_sample_rate": 48_000, "state_per_sample": True}
FROZEN_CONFIG = json.loads(
    (_ROOT / "eval/config/methods/rnnoise.json").read_text(encoding="utf-8")
)


def _config() -> EnhancementConfig:
    return EnhancementConfig(
        backend=BackendId.RNNOISE,
        variant_id=VARIANT,
        model_path=MODEL_PATH,
        model_sha256=None,
        parameters=PARAMETERS,
    )


def _speech_fixture(path: Path) -> None:
    if CORPUS_PATH.is_file():
        with wave.open(str(CORPUS_PATH), "rb") as source:
            assert source.getframerate() == 16_000
            assert source.getnchannels() == 1
            rate = source.getframerate()
            samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
        pcm = samples
    else:
        rate = 16_000
        t = np.arange(rate * 2, dtype=np.float64) / rate
        envelope = 0.25 * (1.0 - np.cos(2.0 * np.pi * np.arange(t.size) / max(1, t.size - 1)))
        pcm = np.rint(
            5_000.0
            * envelope
            * (
                np.sin(2.0 * np.pi * 180.0 * t)
                + 0.35 * np.sin(2.0 * np.pi * 720.0 * t)
                + 0.12 * np.sin(2.0 * np.pi * 1_900.0 * t)
            )
        ).astype(np.int16)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(pcm.tobytes())


def test_probe_and_real_adapter_are_exact_and_deterministic() -> None:
    config = _config()
    probe = adapter.probe(config)
    assert probe.available is True
    assert probe.source_revision == "v0.2"
    assert probe.runtime_version == "0.2"
    assert probe.model_sha256 == adapter._MODEL_SHA256
    assert probe.native_sample_rate == 48_000
    assert probe.delay_samples == 0
    assert probe.stateful is True

    rate = 16_000
    t = np.arange(rate, dtype=np.float64) / rate
    audio = (0.2 * np.sin(2.0 * np.pi * 240.0 * t)).astype(np.float32)
    first = run_backend(audio, rate, config)
    second = run_backend(audio, rate, config)
    assert first.actual_backend == BackendId.RNNOISE
    assert first.variant_id == VARIANT
    assert first.native_sample_rate == 48_000
    assert first.delay_samples == 0
    assert first.audio.shape == (3 * audio.size,)
    assert first.provenance["fallback_used"] is False
    assert first.provenance["source_revision"] == "v0.2"
    assert first.provenance["model_sha256"] == adapter._MODEL_SHA256
    assert first.metadata["frame_size_samples"] == 480
    assert first.metadata["state_created_per_call"] == 1
    assert hashlib.sha256(first.audio.tobytes()).digest() == hashlib.sha256(second.audio.tobytes()).digest()


def test_adapter_matches_the_frozen_shared_method_config() -> None:
    assert FROZEN_CONFIG["backend"] == "rnnoise"
    assert FROZEN_CONFIG["variant_id"] == VARIANT
    assert FROZEN_CONFIG["source_revision"] == "v0.2"
    assert FROZEN_CONFIG["model_sha256"] is None
    assert FROZEN_CONFIG["parameters"] == PARAMETERS

    output = run_backend(
        (0.2 * np.sin(2.0 * np.pi * 240.0 * np.arange(16_000) / 16_000)).astype(np.float32),
        16_000,
        _config(),
    )
    assert dict(output.provenance["parameters"]) == FROZEN_CONFIG["parameters"]
    assert output.provenance["source_revision"] == FROZEN_CONFIG["source_revision"]
    assert output.provenance["runtime_version"] == FROZEN_CONFIG["package_version"]
    assert output.provenance["model_sha256"] == adapter._MODEL_SHA256
    for key in (
        "model_archive_sha256",
        "generated_rnn_data_sha256",
        "native_wrapper_sha256",
        "source_commit",
        "frame_size_samples",
    ):
        assert key not in output.provenance["parameters"]
        assert output.provenance[key] == adapter._PINNED_BUILD[key]
        assert output.metadata[key] == adapter._PINNED_BUILD[key]


def test_clean_audio_preserves_rate_length_and_hash(tmp_path: Path) -> None:
    source = tmp_path / "speech.wav"
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    _speech_fixture(source)

    first_metadata = clean_audio(source, first, backend=BackendId.RNNOISE, config=_config())
    second_metadata = clean_audio(source, second, backend=BackendId.RNNOISE, config=_config())

    assert first_metadata["requested_backend"] == BackendId.RNNOISE.value
    assert first_metadata["actual_backend"] == BackendId.RNNOISE.value
    assert first_metadata["variant_id"] == VARIANT
    assert first_metadata["fallback_used"] is False
    assert first_metadata["sample_rate"] == 16_000
    assert first_metadata["channels"] == 1
    assert first_metadata["native_sample_rate"] == 48_000
    assert first_metadata["delay_samples"] == 0
    assert first_metadata["input_sample_count"] == first_metadata["output_sample_count"]
    assert first_metadata["sample_count_drift"] == 0
    assert first_metadata["duration_drift_s"] == 0.0
    assert first_metadata["processing_time_s"] > 0.0
    assert first.read_bytes() == second.read_bytes()
    assert first_metadata["output_sha256"] == second_metadata["output_sha256"]
    with wave.open(str(first), "rb") as output:
        assert output.getframerate() == 16_000
        assert output.getnchannels() == 1
        assert output.getsampwidth() == 2
        assert output.getnframes() == first_metadata["output_sample_count"]


def test_adapter_rejects_configuration_or_rate_substitution() -> None:
    with pytest.raises(ValueError, match="variant"):
        adapter.probe(
            EnhancementConfig(
                backend=BackendId.RNNOISE,
                variant_id="substitution",
                model_path=MODEL_PATH,
                model_sha256=adapter._MODEL_SHA256,
            )
        )
    for drifted in (
        {**PARAMETERS, "frame_size_samples": 480},
        {"native_sample_rate": 48_000},
        {**PARAMETERS, "state_per_sample": False},
    ):
        with pytest.raises(ValueError, match="parameters"):
            adapter.probe(
                EnhancementConfig(
                    backend=BackendId.RNNOISE,
                    variant_id=VARIANT,
                    model_path=MODEL_PATH,
                    model_sha256=None,
                    parameters=drifted,
                )
            )
    with pytest.raises(ValueError, match="16 kHz"):
        adapter.enhance(
            np.ones(480, dtype=np.float32),
            48_000,
            _config(),
        )

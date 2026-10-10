"""Observable DeepFilterNet3 method behavior against the pinned model and environment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import wave

import numpy as np
import pytest


_ROOT = Path(__file__).resolve().parents[2]

from base_pipeline.audio_clean import clean_audio
from base_pipeline.denoise import (
    BackendId,
    BackendUnavailableError,
    EnhancementConfig,
    run_backend,
)
from base_pipeline.denoise_backends import deepfilternet3 as adapter
from base_pipeline.paths import data_root
from arm_support import require_arm

arm_available = require_arm(BackendId.DEEPFILTERNET3)


DATA_ROOT = data_root()
CORPUS_PATH = DATA_ROOT / "corpus/pilot-15-v1/noisy/sample-001.wav"
MODEL_PATH = DATA_ROOT / "models/deepfilternet/v0.5.6/DeepFilterNet3.zip"
VARIANT = "v0.5.6-dfn3"
PARAMETERS = {"architecture": "DeepFilterNet3", "device": "cpu", "state_per_sample": True}
FROZEN_CONFIG = json.loads(
    (_ROOT / "eval/config/methods/deepfilternet3.json").read_text(encoding="utf-8")
)
ENV_CONFIG = json.loads(
    (_ROOT / "base_pipeline/denoise_backends/deepfilternet3.env.json").read_text(encoding="utf-8")
)


def _config(**overrides) -> EnhancementConfig:
    values = {
        "backend": BackendId.DEEPFILTERNET3,
        "variant_id": VARIANT,
        "model_path": MODEL_PATH,
        "model_sha256": None,
        "parameters": PARAMETERS,
    }
    values.update(overrides)
    return EnhancementConfig(**values)


def _tone(seconds: float = 1.0, rate: int = 16_000) -> np.ndarray:
    axis = np.arange(int(rate * seconds), dtype=np.float64) / rate
    return (0.2 * np.sin(2.0 * np.pi * 240.0 * axis)).astype(np.float32)


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
        axis = np.arange(rate * 3, dtype=np.float64) / rate
        envelope = 0.25 * (1.0 - np.cos(2.0 * np.pi * np.arange(axis.size) / max(1, axis.size - 1)))
        pcm = np.rint(
            5_000.0
            * envelope
            * (
                np.sin(2.0 * np.pi * 180.0 * axis)
                + 0.35 * np.sin(2.0 * np.pi * 720.0 * axis)
                + 0.12 * np.sin(2.0 * np.pi * 1_900.0 * axis)
            )
        ).astype(np.int16)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(pcm.tobytes())


def test_probe_and_real_adapter_are_exact_and_deterministic() -> None:
    probe = adapter.probe(_config())
    assert probe.available is True
    assert probe.source_revision == "v0.5.6"
    assert probe.runtime_version == "0.5.6"
    assert probe.model_sha256 == ENV_CONFIG["model"]["archive_sha256"]
    assert probe.native_sample_rate == 48_000
    assert probe.delay_samples == 0
    assert probe.stateful is True

    audio = _tone()
    first = run_backend(audio, 16_000, _config())
    second = run_backend(audio, 16_000, _config())
    assert first.actual_backend == BackendId.DEEPFILTERNET3
    assert first.variant_id == VARIANT
    assert first.native_sample_rate == 48_000
    assert first.audio.shape == (3 * audio.size,)
    assert first.provenance["fallback_used"] is False
    assert first.provenance["source_revision"] == "v0.5.6"
    assert first.provenance["device"] == "cpu"
    assert first.provenance["model_sha256"] == ENV_CONFIG["model"]["archive_sha256"]
    # The pinned native geometry is reported instead of being silently absorbed.
    assert first.metadata["native_fft_size"] == 960
    assert first.metadata["native_hop_size"] == 480
    assert first.metadata["algorithm_delay_samples_native"] == 480
    assert first.metadata["native_padding_samples"] == 960
    assert first.metadata["native_input_samples"] == 3 * audio.size
    assert first.metadata["native_output_samples"] == 3 * audio.size
    assert first.metadata["state_created_per_call"] == 1
    assert first.provenance["torch_threads"] == ENV_CONFIG["environment"]["torch_threads"]
    assert (
        hashlib.sha256(first.audio.tobytes()).digest()
        == hashlib.sha256(second.audio.tobytes()).digest()
    )


def test_adapter_matches_the_frozen_shared_method_config() -> None:
    assert FROZEN_CONFIG["backend"] == "deepfilternet3"
    assert FROZEN_CONFIG["variant_id"] == VARIANT
    assert FROZEN_CONFIG["source_revision"] == "v0.5.6"
    assert FROZEN_CONFIG["package_version"] == "0.5.6"
    assert FROZEN_CONFIG["parameters"] == PARAMETERS
    assert ENV_CONFIG["variant_id"] == FROZEN_CONFIG["variant_id"]
    assert ENV_CONFIG["model"]["archive_relative"] == FROZEN_CONFIG["model_path"]
    assert ENV_CONFIG["source_tag_commit"] == "978576aa8400552a4ce9730838c635aa30db5e61"

    output = run_backend(_tone(), 16_000, _config())
    assert dict(output.provenance["parameters"]) == FROZEN_CONFIG["parameters"]
    assert output.provenance["source_revision"] == FROZEN_CONFIG["source_revision"]
    assert output.provenance["runtime_version"] == FROZEN_CONFIG["package_version"]
    assert output.provenance["architecture"] == "DeepFilterNet3"
    # The arm is DeepFilterNet3 and must never be labelled as a later architecture.
    assert "dfn4" not in json.dumps(dict(output.provenance)).casefold()
    assert output.provenance["model_sha256"] == ENV_CONFIG["model"]["archive_sha256"]
    for key in (
        "model_archive_sha256",
        "model_config_sha256",
        "model_checkpoint_sha256",
        "source_tag_commit",
        "torch_version",
        "method_python_version",
    ):
        assert key not in output.provenance["parameters"]
        assert key in output.provenance
        assert output.provenance[key] == output.metadata[key]


def test_clean_audio_preserves_rate_length_and_hash(tmp_path: Path) -> None:
    source = tmp_path / "speech.wav"
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    _speech_fixture(source)

    first_metadata = clean_audio(source, first, backend=BackendId.DEEPFILTERNET3, config=_config())
    second_metadata = clean_audio(source, second, backend=BackendId.DEEPFILTERNET3, config=_config())

    assert first_metadata["requested_backend"] == BackendId.DEEPFILTERNET3.value
    assert first_metadata["actual_backend"] == BackendId.DEEPFILTERNET3.value
    assert first_metadata["variant_id"] == VARIANT
    assert first_metadata["fallback_used"] is False
    assert first_metadata["sample_rate"] == 16_000
    assert first_metadata["channels"] == 1
    assert first_metadata["native_sample_rate"] == 48_000
    assert first_metadata["delay_samples"] == 0
    assert first_metadata["input_sample_count"] == first_metadata["output_sample_count"]
    assert first_metadata["sample_count_drift"] == 0
    assert first_metadata["duration_drift_s"] == 0.0
    assert first_metadata["model_sha256"] == ENV_CONFIG["model"]["archive_sha256"]
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
        adapter.probe(_config(variant_id="substitution"))
    for drifted in (
        {**PARAMETERS, "architecture": "DeepFilterNet4"},
        {**PARAMETERS, "device": "cuda"},
        {**PARAMETERS, "state_per_sample": False},
        {"architecture": "DeepFilterNet3", "device": "cpu"},
    ):
        with pytest.raises(ValueError, match="parameters"):
            adapter.probe(_config(parameters=drifted))
    with pytest.raises(ValueError, match="model_path"):
        adapter.probe(_config(model_path=DATA_ROOT / "models/deepfilternet/v0.5.6/DeepFilterNet2.zip"))
    with pytest.raises(ValueError, match="model_sha256"):
        adapter.probe(_config(model_sha256="0" * 64))
    with pytest.raises(ValueError, match="16 kHz"):
        adapter.enhance(_tone(seconds=0.1), 48_000, _config())


def test_missing_or_altered_staged_model_fails_closed(tmp_path: Path, monkeypatch) -> None:
    def _probe_with(config: dict) -> object:
        path = tmp_path / "deepfilternet3.env.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        monkeypatch.setattr(adapter, "_ENV_CONFIG_PATH", path)
        return adapter.probe(_config())

    # A method configuration that widens the pinned artifact is refused outright.
    widened = json.loads(json.dumps(ENV_CONFIG))
    widened["model"]["archive_sha256"] = "1" * 64
    widened_probe = _probe_with(widened)
    assert widened_probe.available is False
    assert widened_probe.reason == "BACKEND_UNAVAILABLE: method environment model pins drift"

    # A staged archive that no longer matches the pinned digest is refused too.
    tampered_root = tmp_path / "data"
    (tampered_root / "models/deepfilternet/v0.5.6").mkdir(parents=True)
    tampered = tampered_root / "models/deepfilternet/v0.5.6/DeepFilterNet3.zip"
    original = MODEL_PATH.read_bytes()
    tampered.write_bytes(bytes([original[0] ^ 0xFF]) + original[1:])
    relabeled = json.loads(json.dumps(ENV_CONFIG))
    relabeled["data_root"] = str(tampered_root)
    tampered_probe = _probe_with(relabeled)
    assert tampered_probe.available is False
    assert "DeepFilterNet3 archive SHA-256" in tampered_probe.reason

    # An unstaged artifact is a retained prerequisite failure, never a fallback.
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    unstaged = json.loads(json.dumps(ENV_CONFIG))
    unstaged["data_root"] = str(empty_root)
    unstaged_probe = _probe_with(unstaged)
    assert unstaged_probe.available is False
    assert "is not staged" in unstaged_probe.reason
    with pytest.raises(BackendUnavailableError, match="BACKEND_UNAVAILABLE"):
        run_backend(_tone(seconds=0.2), 16_000, _config())

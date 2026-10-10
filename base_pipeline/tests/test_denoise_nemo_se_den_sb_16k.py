"""Real pinned NeMo Schroedinger-bridge denoising: geometry, determinism, isolation."""
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
from base_pipeline.denoise_backends import nemo_se_den_sb_16k as adapter
from base_pipeline.paths import data_root

CORPUS = data_root() / "corpus/pilot-15-v1/noisy/sample-001.wav"


needs_runtime = pytest.mark.skipif(
    importlib.util.find_spec("nemo") is None,
    reason="nemo-toolkit is not installed here; the adapter reports BACKEND_UNAVAILABLE (install base_pipeline/denoise_backends/requirements-nemo-se-den-sb-16k.txt)",
)


def config(**overrides: object) -> EnhancementConfig:
    fields = dict(backend=BackendId.NEMO_SE_DEN_SB_16K, variant_id=adapter.VARIANT,
                  model_path=adapter.MODEL_PATH, model_sha256=adapter.MODEL_SHA256,
                  parameters=adapter.PARAMETERS)
    fields.update(overrides)
    return EnhancementConfig(**fields)


def _corpus() -> np.ndarray:
    with wave.open(str(CORPUS), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768


@needs_runtime
def test_probe_reports_the_pinned_research_only_identity() -> None:
    probe = adapter.probe(config())
    assert probe.available is True
    assert probe.source_revision == "1d4ee423806d461f9146ae982f9da8eb32495ae7"
    assert probe.runtime_version == "2.7.3"
    assert probe.model_sha256 == adapter.MODEL_SHA256
    assert probe.native_sample_rate == 16000
    assert probe.stateful is True


@needs_runtime
def test_real_model_is_deterministic_and_state_isolated() -> None:
    audio = _corpus()
    first = run_backend(audio, 16000, config())
    run_backend(audio[::-1].copy(), 16000, config())
    second = run_backend(audio, 16000, config())

    assert first.actual_backend == BackendId.NEMO_SE_DEN_SB_16K
    assert first.audio.size == audio.size
    assert first.native_sample_rate == 16000 and first.delay_samples == 0
    # The sampler is ODE but the checkpoint still carries an RNG surface; prove
    # repeats are bitwise identical and that an interleaved call cannot leak.
    assert np.array_equal(first.audio, second.audio)
    reversed_output = run_backend(audio[::-1].copy(), 16000, config()).audio
    assert not np.array_equal(first.audio, reversed_output)
    assert np.any(first.audio[-100:] != 0)
    assert first.provenance["fallback_used"] is False
    assert first.provenance["sampler_process"] == "ode"
    assert first.provenance["sampler_num_steps"] == 10
    assert first.provenance["seed"] == 1729
    assert first.provenance["model_license"] == "CC-BY-NC-SA-4.0 research-only"
    assert first.provenance["state_reset_per_call"] is True
    assert first.provenance["device"] in {"cpu", "cuda:0"}
    assert first.provenance["cuda_version"] != "unavailable" or first.provenance["device"] == "cpu"


@needs_runtime
def test_loss_target_compatibility_shim_is_exactly_two_remaps() -> None:
    output = run_backend(_corpus()[:16000], 16000, config())
    assert output.metadata["loss_target_remaps"] == 2
    assert output.metadata["embedded_config_sha256"] == adapter.EMBEDDED_CONFIG_SHA256
    assert output.metadata["model_revision"] == "fe9725b1ae90f7f6b11d9b95c5f9ccec736a211f"
    # Weights come from the pinned archive, so the checkpoint digest is echoed exactly.
    assert output.provenance["model_sha256"] == adapter.MODEL_SHA256


@needs_runtime
def test_real_clean_audio_preserves_wav_geometry(tmp_path: Path) -> None:
    output = tmp_path / "enhanced.wav"
    meta = clean_audio(CORPUS, output, backend=BackendId.NEMO_SE_DEN_SB_16K, config=config())
    with wave.open(str(CORPUS), "rb") as source, wave.open(str(output), "rb") as enhanced:
        assert (enhanced.getframerate(), enhanced.getnchannels(), enhanced.getsampwidth()) == (16000, 1, 2)
        assert enhanced.getnframes() == source.getnframes()
    assert meta["actual_backend"] == "nemo-se-den-sb-16k"
    assert meta["fallback_used"] is False
    assert meta["model_sha256"] == adapter.MODEL_SHA256
    assert meta["sample_count_drift"] == 0


@needs_runtime
def test_refuses_substituted_model_variant_and_parameters(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exact requested backend"):
        adapter.probe(config(variant_id="official-checkpoint-audit"))
    with pytest.raises(ValueError, match="parameters"):
        adapter.probe(config(parameters={**adapter.PARAMETERS, "num_steps": 50}))
    with pytest.raises(ValueError, match="non-symlink"):
        adapter.probe(config(model_path=tmp_path / "se_den_sb_16k_small.nemo"))
    with pytest.raises(ValueError, match="pinned checkpoint SHA-256"):
        adapter.probe(config(model_sha256="0" * 64))
    with pytest.raises(ValueError, match="16 kHz"):
        adapter.enhance(np.ones(1024, dtype=np.float32), 48000, config())
    monkeypatch.setattr(adapter, "MODEL_SHA256", "0" * 64)
    with pytest.raises(BackendUnavailableError, match="digest mismatch"):
        run_backend(np.ones(1600, dtype=np.float32), 16000, config(model_sha256="0" * 64))

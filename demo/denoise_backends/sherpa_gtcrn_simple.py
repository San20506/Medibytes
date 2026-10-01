"""Single-backend Sherpa-ONNX GTCRN inference on a pinned 16 kHz model."""
from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from demo.denoise import (
    BackendId, BackendProbe, BackendUnavailableError, EnhancementConfig,
    EnhancementOutput,
)

BACKEND = BackendId.SHERPA_GTCRN_SIMPLE
VARIANT = "gtcrn-simple-1.13.8"
SOURCE_REVISION = "362ddf2c077e3c04759c93b7a47212a5c9108ca7"
MODEL_SHA256 = "e77603ac0c23dac3227dd2d7135b3a585cbee2679048aecfa886657d3ae1b534"
MODEL_PATH = Path("/home/sandy/.local/share/medibytes-eval/models/sherpa-onnx/gtcrn_simple.onnx")
PARAMETERS = {"sample_rate": 16000, "provider": "cpu", "num_threads": 1}
FRAME_HOP = 256  # 16 ms native GTCRN hop; offline API returns complete hops only.


def _checked_model(config: EnhancementConfig) -> Path:
    if config.backend != BACKEND or config.variant_id != VARIANT:
        raise ValueError("GTCRN requires the exact requested backend and variant")
    if dict(config.parameters) != PARAMETERS:
        raise ValueError("GTCRN parameters differ from the frozen method config")
    if config.model_path != MODEL_PATH or MODEL_PATH.is_symlink():
        raise ValueError("GTCRN requires the exact local non-symlink model path")
    if config.model_sha256 not in (None, MODEL_SHA256):
        raise ValueError("GTCRN model_sha256 must be null or the pinned digest")
    digest = hashlib.sha256()
    try:
        with MODEL_PATH.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise BackendUnavailableError("BACKEND_UNAVAILABLE: pinned GTCRN model is missing") from error
    if digest.hexdigest() != MODEL_SHA256:
        raise BackendUnavailableError("BACKEND_UNAVAILABLE: pinned GTCRN model digest mismatch")
    return MODEL_PATH


def _runtime():
    try:
        import sherpa_onnx
        if version("sherpa-onnx") != "1.13.8" or version("sherpa-onnx-bin") != "1.13.8":
            raise BackendUnavailableError("BACKEND_UNAVAILABLE: sherpa-onnx and sherpa-onnx-bin must both be 1.13.8")
        if sherpa_onnx.__version__ != "1.13.8":
            raise BackendUnavailableError("BACKEND_UNAVAILABLE: Sherpa runtime version mismatch")
        return sherpa_onnx
    except (ImportError, PackageNotFoundError) as error:
        raise BackendUnavailableError("BACKEND_UNAVAILABLE: Sherpa-ONNX 1.13.8 CPU runtime missing") from error


def _denoiser(sherpa, model: Path):
    config = sherpa.OfflineSpeechDenoiserConfig(
        model=sherpa.OfflineSpeechDenoiserModelConfig(
            gtcrn=sherpa.OfflineSpeechDenoiserGtcrnModelConfig(model=str(model)),
            num_threads=1, provider="cpu",
        ),
    )
    denoiser = sherpa.OfflineSpeechDenoiser(config)
    if denoiser.sample_rate != 16000:
        raise BackendUnavailableError("BACKEND_UNAVAILABLE: GTCRN model is not 16 kHz")
    return denoiser


def probe(config: EnhancementConfig) -> BackendProbe:
    model = _checked_model(config)
    sherpa = _runtime()
    try:
        _denoiser(sherpa, model)
    except (RuntimeError, ValueError, OSError) as error:
        return BackendProbe(
            available=False, reason=f"BACKEND_UNAVAILABLE: {error}",
            source_revision=SOURCE_REVISION, runtime_version="1.13.8",
            model_sha256=MODEL_SHA256, native_sample_rate=16000, stateful=True,
        )
    return BackendProbe(
        available=True, reason="pinned GTCRN simple model and Sherpa CPU runtime verified",
        source_revision=SOURCE_REVISION, runtime_version="1.13.8",
        model_sha256=MODEL_SHA256, native_sample_rate=16000, stateful=True,
    )


def enhance(audio: np.ndarray, sample_rate: int, config: EnhancementConfig) -> EnhancementOutput:
    model = _checked_model(config)
    sherpa = _runtime()
    if sample_rate != 16000:
        raise ValueError("GTCRN adapter requires 16 kHz audio")
    source = np.asarray(audio, dtype=np.float32)
    if source.ndim != 1 or source.size == 0 or not np.isfinite(source).all():
        raise ValueError("GTCRN requires finite nonempty mono input")

    # The native offline API emits whole 256-sample hops. Pad only the final
    # partial hop so no original tail is discarded; trim only our known zeros.
    padding = (-source.size) % FRAME_HOP
    native_input = np.pad(source, (0, padding)) if padding else source
    denoiser = _denoiser(sherpa, model)  # a fresh state per independent sample
    result = denoiser.run(native_input, sample_rate)
    native_output = np.asarray(result.samples, dtype=np.float32)
    if result.sample_rate != 16000 or native_output.ndim != 1 or native_output.size != native_input.size:
        raise ValueError("GTCRN native output differs from padded input geometry")
    output = np.ascontiguousarray(native_output[:source.size])
    if not np.isfinite(output).all() or not np.any(output):
        raise ValueError("GTCRN returned nonfinite or silent audio")
    provenance = {
        "requested_backend": BACKEND.value, "actual_backend": BACKEND.value,
        "backend": BACKEND.value, "variant_id": VARIANT, "fallback_used": False,
        "source_revision": SOURCE_REVISION, "runtime_version": sherpa.__version__,
        "runtime_git_sha": str(sherpa.git_sha1), "runtime_build_date": str(sherpa.git_date),
        "onnxruntime_version": str(sherpa.onnxruntime_version),
        "model_sha256": MODEL_SHA256, "device": "cpu", "provider": "cpu",
        "num_threads": 1, "parameters": dict(config.parameters),
    }
    return EnhancementOutput(
        audio=output, actual_backend=BACKEND, variant_id=VARIANT,
        native_sample_rate=16000, delay_samples=0,
        metadata={"native_hop_samples": FRAME_HOP, "native_padding_samples": padding,
                  "native_input_samples": int(native_input.size),
                  "native_output_samples": int(native_output.size), "state_per_call": True},
        provenance=provenance,
    )

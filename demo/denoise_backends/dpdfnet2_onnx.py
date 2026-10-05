"""Pinned Ceva-IP DPDFNet2 CPU ONNX arm; never resolves or downloads a model implicitly."""
from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from demo.denoise import (
    BackendId,
    BackendProbe,
    BackendUnavailableError,
    EnhancementConfig,
    EnhancementOutput,
)

BACKEND = BackendId.DPDFNET2_ONNX
VARIANT = "v0.6.0"
SOURCE_REVISION = "de503b39ddfb16023b9d599b05ca872877506047"
MODEL_REVISION = "dd6818d00f50c836fed43a6243ebe49116de5964"
MODEL_SHA256 = "4f0ee28935b4a32abecc717d745416976565834d839601acf43031094b4dc94c"
MODEL_RELATIVE = "models/dpdfnet/dpdf8c40d95cd/dpPDFNet/dpdfnet2.onnx"
PARAMETERS = {"sample_rate": 16000, "reset_state_per_condition": True, "attn_limit_db": "package_default"}


def _model(config: EnhancementConfig) -> Path:
    if config.backend != BACKEND or config.variant_id != VARIANT:
        raise ValueError("DPDFNet2 adapter requires its exact backend and v0.6.0 variant")
    if dict(config.parameters) != PARAMETERS:
        raise ValueError("DPDFNet2 parameters differ from the frozen method config")
    if config.model_sha256 != MODEL_SHA256 or config.model_path is None:
        raise ValueError("DPDFNet2 requires the pinned model SHA-256 and path")
    expected = Path("/home/sandy/.local/share/medibytes-eval") / MODEL_RELATIVE
    if config.model_path != expected or config.model_path.is_symlink():
        raise ValueError("DPDFNet2 requires the exact local, non-symlink model path")
    try:
        with expected.open("rb") as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise BackendUnavailableError("BACKEND_UNAVAILABLE: pinned DPDFNet2 ONNX file is missing") from error
    if digest.hexdigest() != MODEL_SHA256:
        raise BackendUnavailableError("BACKEND_UNAVAILABLE: pinned DPDFNet2 ONNX digest mismatch")
    return expected


def _runtime_version() -> str:
    try:
        actual = version("dpdfnet")
        if actual != "0.6.0":
            raise BackendUnavailableError(f"BACKEND_UNAVAILABLE: dpdfnet==0.6.0 required, got {actual}")
        return actual
    except PackageNotFoundError as error:
        raise BackendUnavailableError("BACKEND_UNAVAILABLE: dpdfnet==0.6.0 is not installed") from error


def probe(config: EnhancementConfig) -> BackendProbe:
    model = _model(config)
    runtime = _runtime_version()
    try:
        from dpdfnet.onnx_backend import build_runtime_model

        session = build_runtime_model(model).session
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("DPDFNet2 ONNX did not select only CPUExecutionProvider")
    except (ImportError, RuntimeError, ValueError) as error:
        return BackendProbe(
            available=False, reason=f"BACKEND_UNAVAILABLE: {error}",
            source_revision=SOURCE_REVISION, runtime_version=runtime,
            model_sha256=MODEL_SHA256, native_sample_rate=16000, stateful=True,
        )
    return BackendProbe(
        available=True, reason="pinned local Ceva-IP DPDFNet2 ONNX CPU model verified",
        source_revision=SOURCE_REVISION, runtime_version=runtime,
        model_sha256=MODEL_SHA256, native_sample_rate=16000, stateful=True,
    )


def enhance(audio: np.ndarray, sample_rate: int, config: EnhancementConfig) -> EnhancementOutput:
    model = _model(config)
    runtime = _runtime_version()
    if sample_rate != 16000:
        raise ValueError("DPDFNet2 requires mono 16 kHz input")
    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("DPDFNet2 input must be nonempty, finite mono audio")
    from dpdfnet import enhance as dpdfnet_enhance

    # The package's offline path initializes a new ONNX session and RNN state for
    # each call, pads/flushed the final window, and fits output to input length.
    # Explicit onnx_path prevents its automatic latest-revision download/cache path.
    result = dpdfnet_enhance(
        audio, sample_rate=sample_rate, model="dpdfnet2", onnx_path=model,
        attn_limit_db=None,
    )
    output = np.asarray(result, dtype=np.float32)
    if output.ndim != 1 or output.size != audio.size or not np.isfinite(output).all():
        raise ValueError("DPDFNet2 returned invalid audio length, channels or samples")
    provenance = {
        "requested_backend": BACKEND.value, "actual_backend": BACKEND.value,
        "backend": BACKEND.value, "variant_id": VARIANT, "fallback_used": False,
        "model_sha256": MODEL_SHA256, "source_revision": SOURCE_REVISION,
        "model_revision": MODEL_REVISION, "runtime_version": runtime,
        "onnxruntime_version": version("onnxruntime"),
        "provider": "CPUExecutionProvider", "device": "cpu",
        "parameters": dict(config.parameters),
    }
    return EnhancementOutput(
        audio=output, actual_backend=BACKEND, variant_id=VARIANT,
        native_sample_rate=16000, delay_samples=0,
        metadata={"state_per_call": True, "offline_final_window_flushed": True,
                  "attn_limit_db": "package_default_None", "model_revision": MODEL_REVISION},
        provenance=provenance,
    )

"""Pinned pyroomacoustics 0.10.1 spectral subtraction, one deterministic path."""
from __future__ import annotations

import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from ..denoise import (
    BackendId,
    BackendProbe,
    BackendUnavailableError,
    EnhancementConfig,
    EnhancementOutput,
)

BACKEND = BackendId.SPECTRAL_SUBTRACTION
VARIANT = "pyroomacoustics-0.10.1-ss"
SOURCE_REVISION = "f02b01dd6609709e2089aefa5d1e59c91d3a0601"
NATIVE_RATE = 16_000
PARAMETERS = {"nfft": 512, "db_reduc": 10, "lookback": 12, "beta": 3, "alpha": 1.2, "sample_rate": 16000}
ALGORITHM = "apply_spectral_sub"
CONFIG_SHA256 = hashlib.sha256(
    json.dumps({"algorithm": ALGORITHM, "parameters": PARAMETERS}, sort_keys=True).encode("utf-8")
).hexdigest()


def _checked_inputs(config: EnhancementConfig) -> None:
    if config.backend != BACKEND or config.variant_id != VARIANT:
        raise ValueError("Spectral subtraction requires the exact requested backend and variant")
    if dict(config.parameters) != PARAMETERS:
        raise ValueError("Spectral subtraction parameters differ from the frozen method config")
    if config.model_path is not None or config.model_sha256 is not None:
        raise ValueError("Spectral subtraction is model-free and must not carry a model path or digest")


def _runtime_version() -> str:
    try:
        actual = version("pyroomacoustics")
    except PackageNotFoundError as error:
        raise BackendUnavailableError("BACKEND_UNAVAILABLE: pyroomacoustics is not installed") from error
    if actual != "0.10.1":
        raise BackendUnavailableError(f"BACKEND_UNAVAILABLE: pyroomacoustics==0.10.1 required, got {actual}")
    return actual


def probe(config: EnhancementConfig) -> BackendProbe:
    _checked_inputs(config)
    runtime = _runtime_version()
    try:
        from pyroomacoustics.denoise.spectral_subtraction import apply_spectral_sub
    except ImportError as error:
        return BackendProbe(
            available=False,
            reason=f"BACKEND_UNAVAILABLE: {error}",
            source_revision=SOURCE_REVISION,
            runtime_version=runtime,
            model_sha256=None,
            native_sample_rate=NATIVE_RATE,
            stateful=True,
        )
    probe_signal = (0.05 * np.sin(2.0 * np.pi * 220.0 * np.arange(NATIVE_RATE) / NATIVE_RATE)).astype(np.float32)
    result = np.asarray(apply_spectral_sub(probe_signal, **{
        key: value for key, value in PARAMETERS.items() if key != "sample_rate"
    }))
    if result.shape != probe_signal.shape or not np.isfinite(result).all():
        return BackendProbe(
            available=False,
            reason="BACKEND_UNAVAILABLE: spectral subtraction changed geometry or produced non-finite audio",
            source_revision=SOURCE_REVISION,
            runtime_version=runtime,
            model_sha256=None,
            native_sample_rate=NATIVE_RATE,
            stateful=True,
        )
    return BackendProbe(
        available=True,
        reason="pinned pyroomacoustics 0.10.1 spectral subtraction verified with the frozen parameters",
        source_revision=SOURCE_REVISION,
        runtime_version=runtime,
        model_sha256=None,
        native_sample_rate=NATIVE_RATE,
        stateful=True,
    )


def enhance(audio: np.ndarray, sample_rate: int, config: EnhancementConfig) -> EnhancementOutput:
    _checked_inputs(config)
    runtime = _runtime_version()
    if sample_rate != NATIVE_RATE:
        raise ValueError("Spectral subtraction requires mono 16 kHz input")
    source = np.asarray(audio, dtype=np.float32)
    if source.ndim != 1 or source.size == 0 or not np.isfinite(source).all():
        raise ValueError("Spectral subtraction requires finite nonempty mono input")

    from pyroomacoustics.denoise.spectral_subtraction import apply_spectral_sub

    # apply_spectral_sub builds its own zero-initialised noise estimate and STFT
    # per call, so nothing carries across independent samples.
    result = apply_spectral_sub(source, nfft=512, db_reduc=10, lookback=12, beta=3, alpha=1.2)
    output = np.ascontiguousarray(np.asarray(result, dtype=np.float32).reshape(-1))
    if output.size != source.size:
        raise ValueError(f"spectral subtraction returned {output.size} samples for {source.size}")
    if not np.isfinite(output).all() or not np.any(output):
        raise ValueError("Spectral subtraction returned nonfinite or silent audio")
    return EnhancementOutput(
        audio=output,
        actual_backend=BACKEND,
        variant_id=VARIANT,
        native_sample_rate=NATIVE_RATE,
        delay_samples=0,
        metadata={
            "state_per_call": True,
            "algorithm": ALGORITHM,
            "path": "one-shot apply_spectral_sub with upstream asymmetric full-length Hann STFT",
            "config_sha256": CONFIG_SHA256,
        },
        provenance={
            "requested_backend": BACKEND.value,
            "actual_backend": BACKEND.value,
            "backend": BACKEND.value,
            "variant_id": VARIANT,
            "fallback_used": False,
            "source_revision": SOURCE_REVISION,
            "runtime_version": runtime,
            "device": "cpu",
            "provider": "pyroomacoustics/numpy",
            "model_sha256": None,
            "config_sha256": CONFIG_SHA256,
            "algorithm": ALGORITHM,
            "state_reset_per_call": True,
            "parameters": dict(config.parameters),
        },
    )

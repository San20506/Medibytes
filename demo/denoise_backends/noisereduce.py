"""Strict optional noisereduce control.

This adapter intentionally has no fallback path.  The product environment may
omit the dependency; callers receive an explicit unavailable result.
"""
from __future__ import annotations

import importlib
import importlib.metadata

import numpy as np

from demo.denoise import (
    BackendId,
    BackendProbe,
    BackendUnavailableError,
    EnhancementConfig,
    EnhancementOutput,
)


_REQUIRED_VERSION = "3.0.2"
BACKEND = BackendId.NOISEREDUCE


def _package() -> tuple[object, str]:
    try:
        module = importlib.import_module("noisereduce")
        version = importlib.metadata.version("noisereduce")
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise BackendUnavailableError(
            "BACKEND_UNAVAILABLE: noisereduce==3.0.2 is not installed"
        ) from exc
    if version != _REQUIRED_VERSION:
        raise BackendUnavailableError(
            f"BACKEND_UNAVAILABLE: noisereduce=={_REQUIRED_VERSION} required, found {version}"
        )
    return module, version


def probe(config: EnhancementConfig) -> BackendProbe:
    try:
        _, version = _package()
    except BackendUnavailableError as exc:
        return BackendProbe(
            available=False,
            reason=str(exc),
            source_revision=None,
            runtime_version=None,
            model_sha256=None,
            native_sample_rate=16000,
            delay_samples=0,
            stateful=True,
        )
    return BackendProbe(
        available=True,
        reason="pinned control dependency",
        runtime_version=version,
        model_sha256=None,
        source_revision=version,
        delay_samples=0,
        stateful=True,
    )


def enhance(audio: np.ndarray, sample_rate: int, config: EnhancementConfig) -> EnhancementOutput:
    module, version = _package()
    if sample_rate != 16000:
        raise ValueError("noisereduce backend requires the shared 16 kHz boundary")
    parameters = dict(config.parameters)
    # Keep the adapter's public config JSON-shaped, but do not allow an
    # unrecognised option to silently alter the pinned control.
    allowed = {"prop_decrease", "n_grad_freq", "n_grad_time", "win_size", "hop_length", "time_constant_s"}
    unknown = set(parameters) - allowed
    if unknown:
        raise ValueError(f"unsupported noisereduce parameters: {sorted(unknown)}")
    result = module.reduce_noise(y=np.asarray(audio, dtype=np.float32), sr=sample_rate, **parameters)
    return EnhancementOutput(
        audio=np.asarray(result, dtype=np.float32),
        actual_backend=BackendId.NOISEREDUCE,
        variant_id=config.variant_id,
        native_sample_rate=16000,
        delay_samples=0,
        metadata={"parameters": parameters},
        provenance={
            "requested_backend": BackendId.NOISEREDUCE.value,
            "actual_backend": BackendId.NOISEREDUCE.value,
            "backend": BackendId.NOISEREDUCE.value,
            "variant_id": config.variant_id,
            "model_sha256": None,
            "package": "noisereduce",
            "package_version": version,
            "provider": "noisereduce",
            "runtime_version": version,
            "source_revision": version,
            "device": "cpu",
            "parameters": parameters,
            "fallback_used": False,
        },
    )

"""Pinned Xiph RNNoise v0.2 adapter with an explicit 48 kHz native boundary."""
from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
import subprocess

import numpy as np
from scipy.signal import resample_poly

from ..denoise import (
    BackendId,
    BackendProbe,
    BackendUnavailableError,
    EnhancementConfig,
    EnhancementOutput,
)
from ..paths import data_root


BACKEND = BackendId.RNNOISE
_VARIANT = "v0.2-0b50c45"
_MODEL_VERSION = "0b50c45"
_SOURCE_REVISION = "v0.2"
_SOURCE_COMMIT = "904a876dce1f9ab8860c0a5000ed151f9f6eef58"
_RUNTIME_VERSION = "0.2"
_MODEL_SHA256 = "4ac81c5c0884ec4bd5907026aaae16209b7b76cd9d7f71af582094a2f98f4b43"
_RNN_DATA_SHA256 = "522b6a64fded05bf85e58c06206eafe57ce7d94f3af58c725b17628b481d7890"
# The compiled wrapper as build_rnnoise.sh produces it: -march=x86-64-v2 (never -march=native), gcc 16.2.1.
# Two independent from-scratch builds gave this digest. A different compiler gives a different one and
# the arm refuses to run until it is re-pinned. The earlier -march=native build (e066ae51…) differs
# from this one by 1 LSB (1/32768) on the pilot clips.
_WRAPPER_SHA256 = "72a55064afbf13ac597421b1e26cce4fa02bfc45571fc989b5515de705478ede"
_NATIVE_RATE = 48_000
_COMMON_RATE = 16_000
_FRAME_SIZE = 480
_DATA_ROOT = data_root()
_MODEL_PATH = _DATA_ROOT / "models/rnnoise/v0.2-0b50c45/rnnoise_data-0b50c45.tar.gz"
_SOURCE_DIR = _DATA_ROOT / "methods/rnnoise/v0.2/source"
_RNN_DATA_PATH = _SOURCE_DIR / "src/rnnoise_data.c"
_LIBRARY_PATH = _DATA_ROOT / "methods/rnnoise/v0.2/lib/librnnoise_medibytes.so"
_EXPECTED_PARAMETERS = {
    "native_sample_rate": _NATIVE_RATE,
    "state_per_sample": True,
}
_PINNED_BUILD = {
    "source_commit": _SOURCE_COMMIT,
    "frame_size_samples": _FRAME_SIZE,
    "model_archive_sha256": _MODEL_SHA256,
    "generated_rnn_data_sha256": _RNN_DATA_SHA256,
    "native_wrapper_sha256": _WRAPPER_SHA256,
}


class _NativeLibrary:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._library = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
        frame_size = self._library.medibytes_rnnoise_frame_size
        frame_size.argtypes = []
        frame_size.restype = ctypes.c_int
        process = self._library.medibytes_rnnoise_process_pcm16
        process.argtypes = [
            ctypes.POINTER(ctypes.c_int16),
            ctypes.POINTER(ctypes.c_int16),
            ctypes.c_size_t,
        ]
        process.restype = ctypes.c_int
        actual_frame_size = int(frame_size())
        if actual_frame_size != _FRAME_SIZE:
            raise BackendUnavailableError(
                f"BACKEND_UNAVAILABLE: RNNoise frame size {actual_frame_size} != {_FRAME_SIZE}"
            )
        self._frame_size = actual_frame_size
        self._process = process

    def process_pcm16(self, samples: np.ndarray) -> np.ndarray:
        if samples.dtype != np.dtype("<i2") or samples.ndim != 1 or samples.size == 0:
            raise ValueError("native RNNoise input must be a non-empty one-dimensional int16 array")
        output = np.empty_like(samples)
        result = self._process(
            samples.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            samples.size,
        )
        if result != 0:
            raise RuntimeError(f"RNNoise native processing failed with code {result}")
        return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_file(path: Path, expected_sha256: str, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise BackendUnavailableError(f"BACKEND_UNAVAILABLE: {label} is unavailable at {path}") from error
    if resolved.is_symlink() or not resolved.is_file():
        raise BackendUnavailableError(f"BACKEND_UNAVAILABLE: {label} must be a regular non-symlink file")
    actual = _sha256(resolved)
    if actual != expected_sha256:
        raise BackendUnavailableError(
            f"BACKEND_UNAVAILABLE: {label} SHA-256 {actual} != pinned {expected_sha256}"
        )
    return resolved


def _checked_text(path: Path, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise BackendUnavailableError(
            f"BACKEND_UNAVAILABLE: {label} is unavailable at {path}"
        ) from error




def _validate_config(config: EnhancementConfig) -> dict[str, object]:
    if config.backend != BACKEND:
        raise ValueError(f"RNNoise adapter cannot serve backend {config.backend.value}")
    if config.variant_id != _VARIANT:
        raise ValueError(f"RNNoise variant must be {_VARIANT!r}, got {config.variant_id!r}")
    parameters = dict(config.parameters)
    if parameters != _EXPECTED_PARAMETERS:
        raise ValueError(
            f"RNNoise parameters must be exactly {_EXPECTED_PARAMETERS}, got {parameters}"
        )
    if config.model_path is None:
        raise BackendUnavailableError("BACKEND_UNAVAILABLE: RNNoise model_path is required")
    if config.model_sha256 not in {None, _MODEL_SHA256}:
        raise ValueError(f"RNNoise model_sha256 must be null or pinned to {_MODEL_SHA256}")
    try:
        configured_model = config.model_path.resolve(strict=True)
    except OSError as error:
        raise BackendUnavailableError(
            f"BACKEND_UNAVAILABLE: RNNoise model_path is unavailable at {config.model_path}"
        ) from error
    if configured_model != _MODEL_PATH.resolve(strict=True):
        raise ValueError(f"RNNoise model_path must resolve to {_MODEL_PATH}")
    return parameters


def _runtime_identity() -> tuple[Path, Path, Path]:
    model = _checked_file(_MODEL_PATH, _MODEL_SHA256, "RNNoise model archive")
    generated = _checked_file(_RNN_DATA_PATH, _RNN_DATA_SHA256, "generated RNNoise rnn_data.c")
    version = _checked_text(_SOURCE_DIR / "model_version", "RNNoise model_version")
    if version != _MODEL_VERSION:
        raise BackendUnavailableError(
            f"BACKEND_UNAVAILABLE: RNNoise model_version {version!r} differs from {_MODEL_VERSION}"
        )
    library = _checked_file(_LIBRARY_PATH, _WRAPPER_SHA256, "RNNoise native wrapper library")
    revision = subprocess.run(
        ["git", "-C", str(_SOURCE_DIR), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != _SOURCE_COMMIT:
        raise BackendUnavailableError(
            f"BACKEND_UNAVAILABLE: RNNoise source commit {revision} != pinned {_SOURCE_COMMIT}"
        )
    return model, generated, library


def probe(config: EnhancementConfig) -> BackendProbe:
    try:
        _validate_config(config)
        _runtime_identity()
        return BackendProbe(
            available=True,
            reason="pinned Xiph RNNoise v0.2 runtime and model verified",
            source_revision=_SOURCE_REVISION,
            runtime_version=_RUNTIME_VERSION,
            model_sha256=_MODEL_SHA256,
            native_sample_rate=_NATIVE_RATE,
            delay_samples=0,
            stateful=True,
        )
    except (BackendUnavailableError, OSError, subprocess.CalledProcessError) as error:
        return BackendProbe(
            available=False,
            reason=str(error),
            source_revision=_SOURCE_REVISION,
            runtime_version=_RUNTIME_VERSION,
            model_sha256=_MODEL_SHA256,
            native_sample_rate=_NATIVE_RATE,
            delay_samples=0,
            stateful=True,
        )


def _pcm16(audio: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 32767.0 / 32768.0)
    return np.rint(clipped * 32768.0).astype("<i2", copy=False)


def enhance(audio: np.ndarray, sample_rate: int, config: EnhancementConfig) -> EnhancementOutput:
    parameters = _validate_config(config)
    if sample_rate != _COMMON_RATE:
        raise ValueError("RNNoise adapter requires the shared 16 kHz input boundary")
    _model, _generated, library_path = _runtime_identity()
    source = np.asarray(audio, dtype=np.float32)
    if source.ndim != 1 or source.size == 0 or not np.isfinite(source).all():
        raise ValueError("RNNoise input must be non-empty, finite, and mono")

    native_input = resample_poly(source, up=3, down=1).astype(np.float32, copy=False)
    expected_native = int(round(source.size * _NATIVE_RATE / sample_rate))
    if native_input.size != expected_native:
        raise RuntimeError(
            f"16 kHz to 48 kHz resampler length mismatch: {native_input.size} != {expected_native}"
        )
    pcm_input = _pcm16(native_input)
    native = _NativeLibrary(library_path).process_pcm16(pcm_input)
    if native.size != expected_native:
        raise RuntimeError("RNNoise native output length differs from resampled input")
    native_float = native.astype(np.float32) / np.float32(32768.0)
    if not np.isfinite(native_float).all() or not np.any(native_float):
        raise ValueError("RNNoise returned empty, all-zero, or non-finite native audio")

    return EnhancementOutput(
        audio=np.ascontiguousarray(native_float, dtype=np.float32),
        actual_backend=BACKEND,
        variant_id=config.variant_id,
        native_sample_rate=_NATIVE_RATE,
        delay_samples=0,
        metadata={
            "parameters": parameters,
            **_PINNED_BUILD,
            "common_sample_rate": _COMMON_RATE,
            "native_input_samples": expected_native,
            "native_output_samples": int(native.size),
            "pcm_format": "signed_pcm16",
            "resampler": "scipy.signal.resample_poly(up=3,down=1)",
            "model_version": _MODEL_VERSION,
            "state_created_per_call": 1,
            "flush_strategy": "zero-pad final frame; destroy state before return",
        },
        provenance={
            "requested_backend": BACKEND.value,
            "actual_backend": BACKEND.value,
            "backend": BACKEND.value,
            "variant_id": config.variant_id,
            "model_sha256": _MODEL_SHA256,
            "package": "xiph-rnnoise",
            "package_version": _RUNTIME_VERSION,
            "provider": "Xiph.Org",
            "runtime_version": _RUNTIME_VERSION,
            "source_revision": _SOURCE_REVISION,
            "device": "cpu",
            "parameters": parameters,
            "fallback_used": False,
            **_PINNED_BUILD,
        },
    )


__all__ = ["enhance", "probe"]

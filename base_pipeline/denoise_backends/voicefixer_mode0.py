"""Pinned VoiceFixer 0.1.3 mode-0 CPU restoration, single backend only.

VoiceFixer 0.1.3 hard-codes its two checkpoint locations under
``~/.cache/voicefixer`` and loads the analysis weights with ``strict=False``.
That combination is dangerous here: a truncated analysis download still loads
without error. This adapter therefore pins both Zenodo 5600188 files by MD5 and
SHA-256, verifies the bytes at the paths VoiceFixer will actually read, and
refuses to run if they differ.
"""
from __future__ import annotations

import hashlib
import shutil
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
from ..paths import data_root

BACKEND = BackendId.VOICEFIXER_MODE0
VARIANT = "0.1.3-mode0"
SOURCE_REVISION = "zenodo-5600188"
NATIVE_RATE = 44_100
COMMON_RATE = 16_000
PARAMETERS = {"mode": 0, "cuda": False, "execution_class": "generative_restoration"}
MODEL_DIR = data_root() / "models/voicefixer/0.1.3"
ANALYSIS_SOURCE = MODEL_DIR / "vf.ckpt"
ANALYSIS_MD5 = "1f479dabb9fa5724de5e20f2164cbc7a"
ANALYSIS_SHA256 = "748411b70089cadf34a6c11054f95f3a454e614af562c23b13a82f6cb413109f"
ANALYSIS_BYTES = 489_307_071
VOCODER_SOURCE = MODEL_DIR / "model.ckpt-1490000_trimed.pt"
VOCODER_MD5 = "ac6de16561dd6042daf8bb3a806dc818"
VOCODER_SHA256 = "9410d0b528c10a251ae947bd299d1939b0b3247df680c81c4164e94f5d87dc45"
CACHE_ROOT = Path.home() / ".cache" / "voicefixer"
ANALYSIS_CACHE = CACHE_ROOT / "analysis_module" / "checkpoints" / "vf.ckpt"
VOCODER_CACHE = CACHE_ROOT / "synthesis_module" / "44100" / "model.ckpt-1490000_trimed.pt"


def _digests(path: Path) -> tuple[str, str, int]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
            size += len(chunk)
    return md5.hexdigest(), sha256.hexdigest(), size


def _verify_file(path: Path, md5: str, sha256: str, size: int, label: str) -> None:
    if not path.is_file():
        raise BackendUnavailableError(f"BACKEND_UNAVAILABLE: pinned {label} is missing at {path}")
    actual_md5, actual_sha256, actual_size = _digests(path)
    if actual_size != size:
        raise BackendUnavailableError(
            f"BACKEND_UNAVAILABLE: {label} is {actual_size} bytes, expected {size}"
        )
    if actual_md5 != md5 or actual_sha256 != sha256:
        raise BackendUnavailableError(
            f"BACKEND_UNAVAILABLE: {label} digest {actual_md5}/{actual_sha256} != pinned {md5}/{sha256}"
        )


def _checked_inputs(config: EnhancementConfig) -> None:
    if config.backend != BACKEND or config.variant_id != VARIANT:
        raise ValueError("VoiceFixer requires the exact requested backend and variant")
    if dict(config.parameters) != PARAMETERS:
        raise ValueError("VoiceFixer parameters differ from the frozen method config")
    if config.model_path != MODEL_DIR or MODEL_DIR.is_symlink():
        raise ValueError("VoiceFixer requires the exact local non-symlink model directory")
    if config.model_sha256 not in (None, ANALYSIS_SHA256):
        raise ValueError("VoiceFixer model_sha256 must be null or the pinned analysis digest")
    _verify_file(ANALYSIS_SOURCE, ANALYSIS_MD5, ANALYSIS_SHA256, ANALYSIS_BYTES, "vf.ckpt")
    _verify_file(VOCODER_SOURCE, VOCODER_MD5, VOCODER_SHA256, 135_613_039, "model.ckpt-1490000_trimed.pt")
    _stage_cache(ANALYSIS_SOURCE, ANALYSIS_CACHE, ANALYSIS_MD5, ANALYSIS_SHA256, ANALYSIS_BYTES)
    _stage_cache(VOCODER_SOURCE, VOCODER_CACHE, VOCODER_MD5, VOCODER_SHA256, 135_613_039)


def _stage_cache(source: Path, target: Path, md5: str, sha256: str, size: int) -> None:
    """Make the path VoiceFixer will read byte-identical to the pinned file."""
    if target.is_file():
        actual_md5, actual_sha256, actual_size = _digests(target)
        if actual_size == size and actual_md5 == md5 and actual_sha256 == sha256:
            return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    _verify_file(target, md5, sha256, size, f"staged cache copy {target.name}")


def _runtime_version() -> str:
    try:
        actual = version("voicefixer")
    except PackageNotFoundError as error:
        raise BackendUnavailableError("BACKEND_UNAVAILABLE: voicefixer is not installed") from error
    if actual != "0.1.3":
        raise BackendUnavailableError(f"BACKEND_UNAVAILABLE: voicefixer==0.1.3 required, got {actual}")
    return actual


def probe(config: EnhancementConfig) -> BackendProbe:
    _checked_inputs(config)
    runtime = _runtime_version()
    return BackendProbe(
        available=True,
        reason="pinned Zenodo 5600188 analysis and vocoder checkpoints verified at the paths VoiceFixer reads",
        source_revision=SOURCE_REVISION,
        runtime_version=runtime,
        model_sha256=ANALYSIS_SHA256,
        native_sample_rate=NATIVE_RATE,
        delay_samples=0,
        stateful=True,
    )


def enhance(audio: np.ndarray, sample_rate: int, config: EnhancementConfig) -> EnhancementOutput:
    _checked_inputs(config)
    runtime = _runtime_version()
    if sample_rate != COMMON_RATE:
        raise ValueError("VoiceFixer adapter requires mono 16 kHz input")
    source = np.asarray(audio, dtype=np.float32)
    if source.ndim != 1 or source.size == 0 or not np.isfinite(source).all():
        raise ValueError("VoiceFixer requires finite nonempty mono input")

    import librosa
    import torch

    native_input = librosa.resample(source, orig_sr=COMMON_RATE, target_sr=NATIVE_RATE).astype(np.float32)
    expected_native = int(round(source.size * NATIVE_RATE / COMMON_RATE))
    if native_input.size != expected_native:
        raise RuntimeError(
            f"16 kHz to 44.1 kHz resampler length mismatch: {native_input.size} != {expected_native}"
        )

    from voicefixer import VoiceFixer

    # restore_inmem returns one tensor per processed segment; each segment is
    # center-trimmed by upstream to its own input length, so the joined result
    # is expected to be exactly expected_native long.
    fixer = VoiceFixer()
    segments = fixer.restore_inmem(native_input, cuda=False, mode=0)
    parts = segments if isinstance(segments, (list, tuple)) else [segments]
    normalized = []
    for part in parts:
        if hasattr(part, "detach"):
            normalized.append(np.asarray(part.detach().to("cpu").numpy(), dtype=np.float32))
        else:
            normalized.append(np.asarray(part, dtype=np.float32))
    native_output = np.concatenate(normalized, axis=-1).reshape(-1)
    if native_output.size != expected_native:
        raise ValueError(
            f"VoiceFixer returned {native_output.size} native samples, expected {expected_native}"
        )
    if not np.isfinite(native_output).all() or not np.any(native_output):
        raise ValueError("VoiceFixer returned nonfinite or silent audio")

    return EnhancementOutput(
        audio=np.ascontiguousarray(native_output, dtype=np.float32),
        actual_backend=BACKEND,
        variant_id=VARIANT,
        native_sample_rate=NATIVE_RATE,
        delay_samples=0,
        metadata={
            "state_per_call": True,
            "execution_class": "generative_restoration",
            "native_input_samples": expected_native,
            "native_output_samples": int(native_output.size),
            "segments_processed": len(segments) if isinstance(segments, (list, tuple)) else 1,
            "segment_seconds": 30,
            "analysis_ckpt_md5": ANALYSIS_MD5,
            "vocoder_ckpt_md5": VOCODER_MD5,
        },
        provenance={
            "requested_backend": BACKEND.value,
            "actual_backend": BACKEND.value,
            "backend": BACKEND.value,
            "variant_id": VARIANT,
            "fallback_used": False,
            "source_revision": SOURCE_REVISION,
            "runtime_version": runtime,
            "torch_version": torch.__version__,
            "device": "cpu",
            "provider": "torch",
            "model_sha256": ANALYSIS_SHA256,
            "model_license": "MIT package code; checkpoints are separately licensed assets",
            "execution_class": "generative_restoration",
            "mode": 0,
            "cuda": False,
            "state_reset_per_call": True,
            "parameters": dict(config.parameters),
        },
    )

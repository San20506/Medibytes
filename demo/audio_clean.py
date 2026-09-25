"""Stage 0 receive and Stage 1 single-backend cleaning for the product demo."""
from __future__ import annotations

import contextlib
import hashlib
import math
from pathlib import Path
import struct
import sys
import time
import wave

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from demo.denoise import BackendId, EnhancementConfig, JSONValue, run_backend

TARGET_SR = 16000
MAX_BYTES = 100 * 1024 * 1024
MAX_SECONDS = 30 * 60


def _wav_info(path: Path):
    """Return WAV geometry, or ``None`` for non-WAV input."""
    try:
        with contextlib.closing(wave.open(str(path), "rb")) as wav:
            frames = wav.getnframes()
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            duration = frames / float(sample_rate) if sample_rate else 0.0
            return frames, sample_rate, channels, duration
    except (OSError, wave.Error):
        return None


def _soundfile_info(path: Path):
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return info.frames, info.samplerate, info.channels, info.duration
    except Exception:
        return None


def check_audio(path: Path) -> dict[str, JSONValue]:
    """Validate the receive-stage size and duration guards."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"404 NOT_FOUND: {path}")
    size = path.stat().st_size
    if size > MAX_BYTES:
        raise ValueError(f"413 TOO_BIG: {size} bytes > 100MB")
    if size == 0:
        raise ValueError("422 NO_AUDIO: empty file")

    info = _wav_info(path) or _soundfile_info(path)
    if info is None:
        try:
            import imageio_ffmpeg  # noqa: F401
        except Exception as exc:
            raise ValueError(
                "422 DECODE_FAIL: need wav or pip install imageio-ffmpeg "
                f"({exc})"
            ) from exc
        return {"size": size, "duration": None, "note": "non-wav, will convert via ffmpeg"}

    _, _, _, duration = info
    if duration > MAX_SECONDS:
        raise ValueError(f"413 TOO_LONG: {duration:.1f}s > 30min")
    if duration < 0.6:
        raise ValueError("422 NO_AUDIO: <0.6s")
    return {"size": size, "duration": duration}


def _resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Duration-preserving endpoint resampling used at the shared boundary."""
    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float32).copy()
    if audio.size == 0:
        return np.empty(0, dtype=np.float32)
    target_length = max(1, int(round(audio.size * target_rate / source_rate)))
    if audio.size == 1:
        return np.full(target_length, float(audio[0]), dtype=np.float32)
    source_positions = np.arange(audio.size, dtype=np.float64)
    target_positions = np.linspace(0.0, audio.size - 1, target_length, dtype=np.float64)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def _decode_pcm(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if sample_width == 3:
        octets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = octets[:, 0].astype(np.int32)
        values |= octets[:, 1].astype(np.int32) << 8
        values |= octets[:, 2].astype(np.int32) << 16
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float32) / float(2**23)
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32)
    raise ValueError(f"422 DECODE_FAIL: unsupported WAV sample width {sample_width}")


def _load_mono_float(path: Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Decode any supported input to mono float32 at the shared rate."""
    path = Path(path)
    info = _wav_info(path)
    if info is not None:
        _, source_rate, channels, _ = info
        with contextlib.closing(wave.open(str(path), "rb")) as wav:
            raw = wav.readframes(wav.getnframes())
            sample_width = wav.getsampwidth()
        samples = _decode_pcm(raw, sample_width)
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        if sample_width == 1:
            audio = (samples - 128.0) / 128.0
        else:
            divisor = float(2 ** (8 * sample_width - 1))
            audio = samples / divisor
        return _resample_linear(audio, source_rate, target_sr), target_sr

    try:
        import imageio_ffmpeg
        import subprocess
        import tempfile

        executable = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            subprocess.run(
                [executable, "-y", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(target_sr), str(temporary_path)],
                check=True,
                capture_output=True,
            )
            return _load_mono_float(temporary_path, target_sr)
        finally:
            temporary_path.unlink(missing_ok=True)
    except Exception as exc:
        raise ValueError(
            "422 DECODE_FAIL: need wav or pip install imageio-ffmpeg "
            f"({exc})"
        ) from exc


def _save_wav(path: Path, audio: np.ndarray, sample_rate: int = TARGET_SR) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm = (values * 32767.0).astype(np.int16)
    with contextlib.closing(wave.open(str(path), "wb")) as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def _rms_vad(audio: np.ndarray, sample_rate: int, frame_ms: int = 30, threshold: float = 0.02):
    """Return a diagnostic RMS speech mask without trimming the timeline."""
    if audio.size == 0:
        return np.zeros(0, dtype=bool), 0.0
    frame_length = max(1, int(sample_rate * frame_ms / 1000))
    frame_count = audio.size // frame_length
    if frame_count == 0:
        return np.ones(audio.size, dtype=bool), 1.0
    frames = audio[: frame_count * frame_length].reshape(frame_count, frame_length)
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
    speech_frames = frame_rms >= threshold
    if not speech_frames.any():
        keep = max(1, frame_count // 5)
        speech_frames[np.argsort(frame_rms)[-keep:]] = True
    mask = np.repeat(speech_frames, frame_length)
    remainder = audio.size - mask.size
    if remainder:
        mask = np.concatenate((mask, np.full(remainder, speech_frames[-1], dtype=bool)))
    return mask, float(speech_frames.mean())


def _dbfs(rms: float) -> float:
    return float(20.0 * math.log10(max(float(rms), 1e-12)))


def _array_sha256(audio: np.ndarray) -> str:
    canonical = np.asarray(audio, dtype="<f4", order="C")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _pcm16_sha256(audio: np.ndarray) -> str:
    pcm = (np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0) * 32767.0).astype(np.int16)
    return hashlib.sha256(pcm.tobytes()).hexdigest()


def _canonical_wav_sha256(audio: np.ndarray, sample_rate: int = TARGET_SR) -> str:
    """Hash the exact 44-byte-header PCM16 file written by ``_save_wav``."""
    pcm = (np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0) * 32767.0).astype(np.int16)
    data = pcm.tobytes()
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", len(data))
    )
    return hashlib.sha256(header + data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()




def clean_audio(
    in_path: Path,
    out_path: Path,
    backend: BackendId = BackendId.NONE,
    config: EnhancementConfig | None = None,
) -> dict[str, JSONValue]:
    """Decode, standardize, run one backend, validate, and save PCM16 audio.

    ``rms_dbfs_before`` is measured on the exact standardized signal handed to
    the selected backend.  The returned hashes and gains are observations, not
    nominal quality estimates.
    """
    started = time.perf_counter()
    in_path = Path(in_path)
    out_path = Path(out_path)
    requested_backend = backend if isinstance(backend, BackendId) else BackendId(backend)
    if config is None:
        enhancement_config = EnhancementConfig(
            backend=requested_backend,
            variant_id=requested_backend.value,
        )
    else:
        if not isinstance(config, EnhancementConfig):
            raise TypeError("config must be an EnhancementConfig or None")
        if config.backend != requested_backend:
            raise ValueError(
                f"backend mismatch: requested {requested_backend.value}, config {config.backend.value}"
            )
        enhancement_config = config

    input_info = check_audio(in_path)
    input_hash = _file_sha256(in_path)
    raw_audio, sample_rate = _load_mono_float(in_path)
    if raw_audio.ndim != 1 or raw_audio.size == 0:
        raise ValueError("422 NO_AUDIO: decoder returned empty or non-mono audio")
    if not np.isfinite(raw_audio).all():
        raise ValueError("422 DECODE_FAIL: decoder returned non-finite samples")
    raw_peak = float(np.max(np.abs(raw_audio)))
    raw_rms = float(np.sqrt(np.mean(np.square(raw_audio))))
    if raw_peak < 0.005 or raw_rms <= 0.0:
        raise ValueError("422 NO_AUDIO: digital silence (peak<0.005) - saves GPU, blocks thank-you loop")

    # Standardize before enhancement.  The second scale is explicit so the
    # metadata can reconstruct the exact signal used as the evaluation reference.
    rms_gain = 0.1 / raw_rms
    rms_scaled_peak = raw_peak * rms_gain
    peak_scale = min(1.0, 0.98 / rms_scaled_peak) if rms_scaled_peak > 0 else 1.0
    total_gain = rms_gain * peak_scale
    prepared = np.asarray(raw_audio * total_gain, dtype=np.float32)
    prepared_peak = float(np.max(np.abs(prepared)))
    prepared_rms = float(np.sqrt(np.mean(np.square(prepared))))
    if not np.isfinite(prepared).all() or prepared.size == 0:
        raise ValueError("422 DECODE_FAIL: preprocessing produced invalid audio")
    if prepared_peak < 0.005:
        raise ValueError("422 NO_AUDIO: digital silence after preprocessing")

    _, vad_ratio = _rms_vad(prepared, sample_rate)
    if vad_ratio < 0.05 or prepared.size < int(sample_rate * 0.6):
        raise ValueError("422 NO_AUDIO: >95% silent or <0.6s speech (saves GPU 4-8x)")

    enhanced_native = run_backend(prepared, sample_rate, enhancement_config)
    native_audio = np.asarray(enhanced_native.audio, dtype=np.float32)
    delay = enhanced_native.delay_samples
    if delay:
        if delay >= native_audio.size:
            raise ValueError("422 ENHANCE_FAIL: declared delay exceeds output")
        native_audio = native_audio[delay:]
    expected_native_length = int(round(prepared.size * enhanced_native.native_sample_rate / sample_rate))
    if native_audio.size != expected_native_length:
        raise ValueError(
            "422 ENHANCE_FAIL: uncompensated output length "
            f"({native_audio.size} != {expected_native_length})"
        )
    enhanced = _resample_linear(native_audio, enhanced_native.native_sample_rate, sample_rate)
    if enhanced.size != prepared.size:
        raise ValueError(
            "422 ENHANCE_FAIL: output length drift after rate conversion "
            f"({enhanced.size} != {prepared.size})"
        )
    if not np.isfinite(enhanced).all() or enhanced.size == 0 or not np.any(enhanced != 0):
        raise ValueError("422 ENHANCE_FAIL: backend returned empty, all-zero, or non-finite audio")

    output_peak = float(np.max(np.abs(enhanced)))
    if output_peak > 1.0:
        raise ValueError("422 ENHANCE_FAIL: backend output exceeds PCM16 full scale")
    output_clipping_count = int(np.count_nonzero(np.abs(enhanced) >= 1.0))
    _save_wav(out_path, enhanced, sample_rate)
    output_hash = _file_sha256(out_path)
    output_duration = enhanced.size / float(sample_rate)
    input_duration = prepared.size / float(sample_rate)
    metadata = dict(enhanced_native.metadata)
    adapter_provenance = dict(enhanced_native.provenance)
    model_sha256 = (
        enhancement_config.model_sha256
        or adapter_provenance.get("model_sha256")
        or metadata.get("model_sha256")
    )
    runtime_version = adapter_provenance.get("runtime_version") or metadata.get("runtime_version")
    source_revision = adapter_provenance.get("source_revision") or metadata.get("source_revision")
    provider = str(
        adapter_provenance.get("provider")
        or adapter_provenance.get("package")
        or requested_backend.value
    )
    device = str(adapter_provenance.get("device") or "cpu")
    provenance: dict[str, JSONValue] = {
        **adapter_provenance,
        "requested_backend": requested_backend.value,
        "actual_backend": enhanced_native.actual_backend.value,
        "backend": requested_backend.value,
        "variant_id": enhanced_native.variant_id,
        "model_sha256": model_sha256,
        "runtime_version": runtime_version,
        "source_revision": source_revision,
        "device": device,
        "provider": provider,
        "parameters": dict(enhancement_config.parameters),
        "fallback_used": False,
    }
    result: dict[str, JSONValue] = {
        "input": input_info,
        "input_sha256": input_hash,
        "output_sha256": output_hash,
        "sample_rate": sample_rate,
        "channels": 1,
        "input_sample_count": int(prepared.size),
        "output_sample_count": int(enhanced.size),
        "sample_count_drift": int(enhanced.size - prepared.size),
        "sample_count_before": int(prepared.size),
        "sample_count_after": int(enhanced.size),
        "duration_s": round(output_duration, 6),
        "input_duration_s": round(input_duration, 6),
        "output_duration_s": round(output_duration, 6),
        "duration_drift_s": round(output_duration - input_duration, 9),
        "vad_ratio": round(vad_ratio, 3),
        "rms_dbfs_before": round(_dbfs(prepared_rms), 3),
        "rms_dbfs_after": round(_dbfs(float(np.sqrt(np.mean(np.square(enhanced))))), 3),
        "preprocess_gain_linear": round(float(rms_gain), 9),
        "preprocess_rms_gain_linear": round(float(rms_gain), 9),
        "preprocess_peak_scale": round(float(peak_scale), 9),
        "preprocess_total_gain_linear": round(float(total_gain), 9),
        "preprocess_peak_before": round(raw_peak, 9),
        "preprocess_peak_after": round(prepared_peak, 9),
        "output_peak": round(output_peak, 9),
        "output_clipping_count": output_clipping_count,
        "requested_backend": requested_backend.value,
        "actual_backend": enhanced_native.actual_backend.value,
        "variant_id": enhanced_native.variant_id,
        "model_sha256": model_sha256,
        "runtime_version": runtime_version,
        "source_revision": source_revision,
        "native_sample_rate": enhanced_native.native_sample_rate,
        "delay_samples": delay,
        "fallback_used": False,
        "preprocess_reference_sha256": _array_sha256(prepared),
        "none_reference_sha256": _canonical_wav_sha256(prepared, sample_rate),
        "enhancement_metadata": metadata,
        "provenance": provenance,

        "diarization": "off (default per report)",
        "processing_time_s": round(time.perf_counter() - started, 6),
    }
    return result


__all__ = ["BackendId", "EnhancementConfig", "check_audio", "clean_audio"]

"""Pinned FFmpeg decoding and canonical source conversion."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import numpy as np

from eval.corpus.audio import (
    SAMPLE_RATE,
    TARGET_PEAK_DBFS,
    AudioValidationError,
    clipping_count,
    float_to_pcm16,
    pcm16_peak,
    validate_pcm16,
    write_pcm16,
)
from eval.corpus.manifest import file_sha256

FFMPEG_PROVIDER = "imageio-ffmpeg==0.6.0"


@dataclass(frozen=True)
class Decoder:
    executable: Path
    sha256: str

    @property
    def provider(self) -> str:
        return FFMPEG_PROVIDER


def bundled_decoder() -> Decoder:
    executable = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve(strict=True)
    return Decoder(executable=executable, sha256=file_sha256(executable))


def _decode_command(ffmpeg: Path, input_name: str) -> list[str]:
    return [
        str(ffmpeg),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        input_name,
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        "-f",
        "s16le",
        "pipe:1",
    ]


def decode_bytes(audio_bytes: bytes, decoder: Decoder) -> np.ndarray:
    if not audio_bytes:
        raise AudioValidationError("empty source audio bytes")
    command = _decode_command(decoder.executable, "pipe:0")
    process = subprocess.run(
        command,
        input=audio_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        message = process.stderr.decode("utf-8", errors="replace").strip()
        raise AudioValidationError(f"FFmpeg source decode failed: {message}")
    if len(process.stdout) % 2:
        raise AudioValidationError("FFmpeg returned a partial PCM16 sample")
    samples = np.frombuffer(process.stdout, dtype="<i2").copy()
    validate_pcm16(samples, SAMPLE_RATE)
    return samples


def decode_file(path: Path, decoder: Decoder) -> np.ndarray:
    command = _decode_command(decoder.executable, str(path))
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        message = process.stderr.decode("utf-8", errors="replace").strip()
        raise AudioValidationError(f"FFmpeg source decode failed for {path}: {message}")
    if len(process.stdout) % 2:
        raise AudioValidationError(f"FFmpeg returned a partial PCM16 sample for {path}")
    samples = np.frombuffer(process.stdout, dtype="<i2").copy()
    validate_pcm16(samples, SAMPLE_RATE)
    return samples


def decode_segment(
    path: Path,
    start_sample: int,
    end_sample: int,
    decoder: Decoder,
) -> np.ndarray:
    if start_sample < 0 or end_sample <= start_sample:
        raise ValueError("MUCS segment sample bounds are invalid")
    recording = decode_file(path, decoder)
    if end_sample > recording.size:
        raise AudioValidationError(
            f"MUCS segment exceeds decoded recording {path}: {end_sample}>{recording.size}"
        )
    segment = recording[start_sample:end_sample]
    validate_pcm16(segment, SAMPLE_RATE)
    return segment


def canonicalize_clean(
    decoded: np.ndarray,
    output_path: Path,
) -> dict[str, object]:
    """Peak-normalize decoded mono PCM16 and write the canonical WAV."""

    validate_pcm16(decoded, SAMPLE_RATE)
    source_peak = pcm16_peak(decoded)
    if source_peak <= 0:
        raise AudioValidationError("source peak must be positive")
    target_peak = 10 ** (TARGET_PEAK_DBFS / 20)
    normalized = decoded.astype(np.float64) / 32768.0 * (target_peak / source_peak)
    pcm = float_to_pcm16(normalized)
    measured_peak = pcm16_peak(pcm)
    measured_dbfs = 20.0 * np.log10(measured_peak)
    if not -3.01 <= measured_dbfs <= -2.99:
        raise AudioValidationError(
            f"PCM16 normalization missed -3 dBFS target: {measured_dbfs:.6f}"
        )
    if clipping_count(pcm):
        raise AudioValidationError("normalized clean audio is clipped")
    write_pcm16(output_path, pcm, SAMPLE_RATE)
    return {
        "sample_count": int(pcm.size),
        "duration_s": pcm.size / SAMPLE_RATE,
        "peak": measured_peak,
        "peak_dbfs": measured_dbfs,
        "source_peak": source_peak,
        "normalization_gain": target_peak / source_peak,
    }

"""Deterministic PCM16 and paired-noise primitives for the pilot corpus."""

from __future__ import annotations

import hashlib
import math
import struct
from pathlib import Path
from typing import Final

import numpy as np

SAMPLE_RATE: Final = 16_000
CHANNELS: Final = 1
BITS_PER_SAMPLE: Final = 16
CODEC: Final = "pcm_s16le"
WAV_HEADER_BYTES: Final = 44
TARGET_PEAK_DBFS: Final = -3.0
TARGET_SNR_DB: Final = 5.0
NOISE_TYPES: Final = (
    "white",
    "pink_voss16",
    "hum50",
    "am_babble_proxy",
    "echo",
)
VOSS16_COEFFICIENTS: Final = (
    -0.99879179,
    -0.96516981,
    -0.90522398,
    -0.82278677,
    -0.71656495,
    -0.58682000,
    -0.43040030,
    -0.25065828,
    -0.04513266,
    0.17055027,
    0.39316334,
    0.62192068,
    0.85025024,
    1.00000000,
    1.17848345,
    1.32699684,
)
HUM_FREQUENCIES: Final = (50.0, 100.0, 150.0)
HUM_AMPLITUDES: Final = (0.60, 0.25, 0.10)
HUM_PHASES: Final = (0.0, 0.0, 0.0)
HUM_WHITE_AMPLITUDE: Final = 0.05
BABBLE_FREQUENCIES: Final = (3.1, 4.7, 6.3, 8.1)


class AudioValidationError(ValueError):
    """Raised when source or generated audio violates the corpus contract."""


def derive_seed(sample_id: str, noise_type: str) -> int:
    """Derive the approved unsigned 64-bit PCG64 seed."""

    if not sample_id:
        raise ValueError("sample_id must not be empty")
    if noise_type not in NOISE_TYPES:
        raise ValueError(f"unsupported noise type: {noise_type!r}")
    digest = hashlib.sha256(
        f"medibytes-corpus-v1|{sample_id}|{noise_type}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def seed_hex(sample_id: str, noise_type: str) -> str:
    return f"{derive_seed(sample_id, noise_type):016x}"


def pink_voss16(
    sample_count: int,
    rng: np.random.Generator,
    coefficients: tuple[float, ...] = VOSS16_COEFFICIENTS,
) -> np.ndarray:
    """Generate the pinned 16-row Voss-McCartney stream."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if len(coefficients) != 16:
        raise ValueError("pink_voss16 requires exactly 16 coefficients")
    white = rng.standard_normal(sample_count)
    rows = np.zeros((16, sample_count), dtype=np.float64)
    for row_index, coefficient in enumerate(coefficients):
        rows[row_index, 1:] = (
            coefficient * rows[row_index, :-1] + white[1:]
        )
    signal = np.sum(rows, axis=0) / math.sqrt(
        sum(coefficient * coefficient for coefficient in coefficients)
    )
    signal -= np.mean(signal)
    standard_deviation = float(np.std(signal))
    if not math.isfinite(standard_deviation) or standard_deviation <= 0:
        raise AudioValidationError("Voss-McCartney output is not finite")
    signal /= standard_deviation
    return signal


def generate_raw_noise(
    noise_type: str,
    sample_id: str,
    sample_count: int,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[np.ndarray, dict[str, object]]:
    """Generate the pinned pre-SNR-scaled condition stream."""

    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"corpus noise requires {SAMPLE_RATE} Hz")
    seed = derive_seed(sample_id, noise_type)
    rng = np.random.Generator(np.random.PCG64(seed))
    if noise_type == "white":
        noise = rng.standard_normal(sample_count)
        metadata: dict[str, object] = {"distribution": "standard_normal"}
    elif noise_type == "pink_voss16":
        noise = pink_voss16(sample_count, rng)
        metadata = {
            "distribution": "voss_mccartney_16",
            "coefficients": list(VOSS16_COEFFICIENTS),
        }
    elif noise_type == "hum50":
        time = np.arange(sample_count, dtype=np.float64) / sample_rate
        hum = np.zeros(sample_count, dtype=np.float64)
        for frequency, amplitude, phase in zip(
            HUM_FREQUENCIES, HUM_AMPLITUDES, HUM_PHASES, strict=True
        ):
            hum += amplitude * np.cos(2 * np.pi * frequency * time + phase)
        noise = hum + HUM_WHITE_AMPLITUDE * rng.standard_normal(sample_count)
        metadata = {
            "frequencies_hz": list(HUM_FREQUENCIES),
            "amplitudes": list(HUM_AMPLITUDES),
            "phases_rad": list(HUM_PHASES),
            "white_noise_amplitude": HUM_WHITE_AMPLITUDE,
        }
    elif noise_type == "am_babble_proxy":
        child_states = np.random.SeedSequence(seed).generate_state(
            4, dtype=np.uint64
        )
        stream_seeds = [int(value) for value in child_states]
        time = np.arange(sample_count, dtype=np.float64) / sample_rate
        mixed = np.zeros(sample_count, dtype=np.float64)
        for frequency, stream_seed in zip(
            BABBLE_FREQUENCIES, stream_seeds, strict=True
        ):
            stream = pink_voss16(
                sample_count, np.random.Generator(np.random.PCG64(stream_seed))
            )
            modulator = 0.5 + 0.5 * np.sin(2 * np.pi * frequency * time)
            mixed += stream * modulator
        noise = mixed / 4.0
        noise -= np.mean(noise)
        noise /= np.std(noise)
        metadata = {
            "modulation_frequencies_hz": list(BABBLE_FREQUENCIES),
            "stream_seed_hex": [f"{value:016x}" for value in stream_seeds],
            "modulation": "0.5+0.5*sin(2*pi*f*t)",
        }
    elif noise_type == "echo":
        noise = rng.standard_normal(sample_count)
        metadata = {"additive_component": "standard_normal"}
    else:
        raise ValueError(f"unsupported noise type: {noise_type!r}")
    if noise.shape != (sample_count,) or not np.isfinite(noise).all():
        raise AudioValidationError(f"nonfinite or malformed {noise_type} noise")
    return noise, metadata


def active_mask(
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    window_ms: float = 20.0,
    hop_ms: float = 10.0,
    threshold_db_below_max: float = 40.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-sample activity weights and the exact 20 ms/10 ms RMS frames."""

    signal = np.asarray(samples, dtype=np.float64)
    if signal.ndim != 1 or signal.size == 0 or not np.isfinite(signal).all():
        raise AudioValidationError("active-mask input must be nonempty finite mono audio")
    window = round(sample_rate * window_ms / 1000)
    hop = round(sample_rate * hop_ms / 1000)
    if signal.size < window:
        raise AudioValidationError("active-mask input is shorter than one 20 ms frame")
    frames = np.lib.stride_tricks.sliding_window_view(signal, window)[::hop]
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
    threshold = float(np.max(frame_rms)) * 10 ** (-threshold_db_below_max / 20)
    active_frames = frame_rms > threshold
    weights = np.zeros(signal.size, dtype=np.float64)
    counts = np.zeros(signal.size, dtype=np.int16)
    for frame_index, is_active in enumerate(active_frames):
        if not is_active:
            continue
        start = frame_index * hop
        stop = start + window
        weights[start:stop] += 1.0
        counts[start:stop] += 1
    mask = np.divide(
        weights,
        counts,
        out=np.zeros_like(weights),
        where=counts > 0,
    )
    return mask, frame_rms


def _masked_snr_db(
    reference: np.ndarray, estimate: np.ndarray, mask: np.ndarray
) -> float:
    error = estimate - reference
    numerator = float(np.dot(reference[mask], reference[mask]))
    denominator = float(np.dot(error[mask], error[mask]))
    if numerator <= 0:
        raise AudioValidationError("active reference energy is zero")
    if denominator <= 0:
        return math.inf
    return 10.0 * math.log10(numerator / denominator)


def mix_at_target_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    mask: np.ndarray,
    *,
    target_snr_db: float = TARGET_SNR_DB,
) -> tuple[np.ndarray, dict[str, float]]:
    """Scale additive noise to exact active-mask SNR before quantization."""

    reference = np.asarray(clean, dtype=np.float64)
    raw_noise = np.asarray(noise, dtype=np.float64)
    weights = np.asarray(mask, dtype=np.float64)
    if reference.ndim != 1 or raw_noise.shape != reference.shape or weights.shape != reference.shape:
        raise AudioValidationError("clean, noise, and active mask must be equal-length mono vectors")
    if not np.isfinite(reference).all() or not np.isfinite(raw_noise).all():
        raise AudioValidationError("clean and noise must be finite")
    active = weights > 0
    reference_energy = float(np.dot(reference[active], reference[active]))
    noise_energy = float(np.dot(raw_noise[active], raw_noise[active]))
    if reference_energy <= 0 or noise_energy <= 0:
        raise AudioValidationError("active clean and noise energy must be positive")
    target_ratio = 10 ** (target_snr_db / 10)
    scale = math.sqrt(reference_energy / (noise_energy * target_ratio))
    mixture = reference + scale * raw_noise
    measured = _masked_snr_db(reference, mixture, active)
    return mixture, {
        "target_snr_db": float(target_snr_db),
        "measured_snr_db": float(measured),
        "active_reference_rms": math.sqrt(
            reference_energy / float(np.count_nonzero(active))
        ),
        "noise_scale": scale,
    }


def float_to_pcm16(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise AudioValidationError("audio must be a nonempty finite mono vector")
    if np.max(np.abs(values)) > 1.0:
        raise AudioValidationError("audio exceeds signed PCM16 full scale")
    return np.rint(values * 32768.0).astype(np.int16)


def pcm16_peak(samples: np.ndarray) -> float:
    values = np.asarray(samples)
    if values.ndim != 1 or values.size == 0:
        raise AudioValidationError("audio must be a nonempty mono vector")
    return float(np.max(np.abs(values.astype(np.float64))) / 32768.0)


def clipping_count(samples: np.ndarray) -> int:
    values = np.asarray(samples, dtype=np.int16)
    return int(np.count_nonzero((values == 32767) | (values == -32768)))


def validate_pcm16(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    values = np.asarray(samples)
    if values.dtype != np.int16 or values.ndim != 1 or values.size == 0:
        raise AudioValidationError("audio must be nonempty mono signed PCM16")
    if sample_rate != SAMPLE_RATE:
        raise AudioValidationError(f"audio must use {SAMPLE_RATE} Hz")
    if not np.any(values):
        raise AudioValidationError("all-zero audio is forbidden")
    if clipping_count(values):
        raise AudioValidationError("clipped audio is forbidden")


def write_pcm16(
    path: Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE
) -> None:
    values = np.asarray(samples)
    validate_pcm16(values, sample_rate)
    data = values.astype("<i2", copy=False).tobytes()
    byte_rate = sample_rate * CHANNELS * BITS_PER_SAMPLE // 8
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(data),
        b"WAVE",
        b"fmt ",
        16,
        1,
        CHANNELS,
        sample_rate,
        byte_rate,
        CHANNELS * BITS_PER_SAMPLE // 8,
        BITS_PER_SAMPLE,
        b"data",
        len(data),
    )
    if len(header) != WAV_HEADER_BYTES:
        raise AssertionError("canonical RIFF header must be exactly 44 bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + data)


def read_pcm16(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) < WAV_HEADER_BYTES:
        raise AudioValidationError(f"truncated WAV: {path}")
    fields = struct.unpack_from("<4sI4s4sIHHIIHH4sI", raw, 0)
    (
        riff,
        riff_size,
        wave,
        fmt_chunk,
        fmt_size,
        audio_format,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        data_chunk,
        data_size,
    ) = fields
    if (
        riff != b"RIFF"
        or riff_size != len(raw) - 8
        or wave != b"WAVE"
        or fmt_chunk != b"fmt "
        or fmt_size != 16
        or audio_format != 1
        or channels != CHANNELS
        or sample_rate != SAMPLE_RATE
        or byte_rate != SAMPLE_RATE * 2
        or block_align != 2
        or bits_per_sample != BITS_PER_SAMPLE
        or data_chunk != b"data"
        or data_size != len(raw) - WAV_HEADER_BYTES
        or data_size % 2
    ):
        raise AudioValidationError(f"noncanonical WAV header: {path}")
    samples = np.frombuffer(raw, dtype="<i2", offset=WAV_HEADER_BYTES).copy()
    validate_pcm16(samples, sample_rate)
    return samples


def apply_echo(
    clean: np.ndarray,
    sample_id: str,
    *,
    sample_rate: int = SAMPLE_RATE,
    direct_delay_ms: float = 0.0,
    rt60_s: float = 0.4,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply the fixed deterministic direct-plus-decaying-noise RIR."""

    signal = np.asarray(clean, dtype=np.float64)
    if signal.ndim != 1 or signal.size == 0 or not np.isfinite(signal).all():
        raise AudioValidationError("echo input must be nonempty finite mono audio")
    if sample_rate != SAMPLE_RATE or direct_delay_ms != 0.0 or rt60_s != 0.4:
        raise AudioValidationError("echo parameters do not match the pinned corpus policy")
    rir_length = round(rt60_s * sample_rate) + 1
    rng = np.random.Generator(np.random.PCG64(derive_seed(sample_id, "echo")))
    tail_time = np.arange(1, rir_length, dtype=np.float64) / sample_rate
    rir = np.empty(rir_length, dtype=np.float64)
    rir[0] = 1.0
    rir[1:] = rng.standard_normal(rir_length - 1) * np.power(
        10.0, -3.0 * tail_time / rt60_s
    )
    fft_length = 1 << (signal.size + rir_length - 2).bit_length()
    convolved = np.fft.irfft(
        np.fft.rfft(signal, fft_length) * np.fft.rfft(rir, fft_length),
        fft_length,
    )[: signal.size]
    if convolved.size < signal.size:
        convolved = np.pad(convolved, (0, signal.size - convolved.size))
    if not np.isfinite(convolved).all():
        raise AudioValidationError("echo convolution produced nonfinite audio")
    return convolved, {
        "direct_delay_ms": direct_delay_ms,
        "rt60_s": rt60_s,
        "rir_length_samples": rir_length,
        "rir_leading_energy": float(rir[0] * rir[0]),
        "rir_sha256": hashlib.sha256(
            np.asarray(rir, dtype="<f8").tobytes()
        ).hexdigest(),
    }


def mix_quantized_at_target_snr(
    reference: np.ndarray,
    raw_noise: np.ndarray,
    mask: np.ndarray,
    *,
    target_snr_db: float = TARGET_SNR_DB,
) -> tuple[np.ndarray, dict[str, float]]:
    """Choose a deterministic scale whose PCM16 result meets active-mask SNR."""

    clean = np.asarray(reference, dtype=np.float64)
    noise = np.asarray(raw_noise, dtype=np.float64)
    weights = np.asarray(mask, dtype=np.float64) > 0
    if clean.ndim != 1 or noise.shape != clean.shape or weights.shape != clean.shape:
        raise AudioValidationError("quantized mixing inputs must have equal mono geometry")
    reference_energy = float(np.dot(clean[weights], clean[weights]))
    noise_energy = float(np.dot(noise[weights], noise[weights]))
    if reference_energy <= 0 or noise_energy <= 0:
        raise AudioValidationError("active clean and noise energy must be positive")
    nominal = math.sqrt(
        reference_energy / (noise_energy * 10 ** (target_snr_db / 10))
    )

    def candidate(scale: float) -> tuple[np.ndarray, float] | None:
        try:
            pcm = float_to_pcm16(clean + scale * noise)
        except AudioValidationError:
            return None
        return pcm, _masked_snr_db(
            clean, pcm.astype(np.float64) / 32768.0, weights
        )

    low = 0.0
    high = nominal
    for _ in range(24):
        result = candidate(high)
        if result is None or result[1] <= target_snr_db:
            break
        high *= 1.5
    else:
        raise AudioValidationError("could not bracket quantized target SNR")
    best: tuple[float, float, np.ndarray, float] | None = None
    for _ in range(64):
        scale = (low + high) / 2
        result = candidate(scale)
        if result is None:
            high = scale
            continue
        pcm, measured = result
        error = abs(measured - target_snr_db)
        if best is None or error < best[0]:
            best = (error, scale, pcm, measured)
        if measured > target_snr_db:
            low = scale
        else:
            high = scale
        if error <= 1e-10:
            break
    if best is None:
        raise AudioValidationError("quantized SNR search produced no nonclipped candidate")
    error, best_scale, pcm, measured = best
    if error > 0.01:
        raise AudioValidationError(f"quantized SNR search failed: error={error}")
    validate_pcm16(pcm)
    return pcm, {
        "target_snr_db": float(target_snr_db),
        "measured_snr_db": float(measured),
        "noise_scale": float(best_scale),
        "active_reference_rms": math.sqrt(
            reference_energy / float(np.count_nonzero(weights))
        ),
    }

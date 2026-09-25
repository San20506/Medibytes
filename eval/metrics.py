"""Explicit paired-audio, transcript, and non-clinical scoring formulas."""

from __future__ import annotations

import math
import unicodedata
from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
from typing import Callable, Mapping, Sequence

import numpy as np

TARGET_SAMPLE_RATE = 16_000
LSD_FRAME_SIZE = 512
LSD_HOP_SIZE = 256
LSD_EPSILON = 1e-10
NOT_APPLICABLE_REASON = "non_clinical_source_reference_corpus"
ENTITY_GROUPS = (
    "drugs",
    "symptoms",
    "vitals",
    "allergies",
    "negations",
    "diagnosis",
    "followup",
)
CLINICAL_ENTITY_GROUPS = (
    "drugs",
    "symptoms",
    "vitals",
    "allergies",
    "diagnosis",
    "followup",
)


class MetricInputError(ValueError):
    """A metric received incompatible, empty, or non-finite audio/text."""


@dataclass(frozen=True)
class LevenshteinCounts:
    substitutions: int
    deletions: int
    insertions: int

    @property
    def edits(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def _mono_audio(audio: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(audio)
    if array.ndim != 1:
        raise MetricInputError(f"{label} must be mono one-dimensional audio")
    if array.size == 0:
        raise MetricInputError(f"{label} must not be empty")
    values = array.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise MetricInputError(f"{label} contains non-finite samples")
    return values


def read_pcm16_mono(path, expected_sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Read canonical signed PCM16 mono WAV without accepting implicit conversion."""

    import wave

    try:
        with wave.open(str(path), "rb") as source:
            channels = source.getnchannels()
            width = source.getsampwidth()
            sample_rate = source.getframerate()
            count = source.getnframes()
            raw = source.readframes(count)
    except (OSError, EOFError, wave.Error) as error:
        raise MetricInputError(f"cannot decode PCM16 WAV {path.name}: {error}") from error
    if channels != 1:
        raise MetricInputError("audio channel count must be exactly one")
    if width != 2:
        raise MetricInputError("audio sample width must be signed PCM16")
    if sample_rate != expected_sample_rate:
        raise MetricInputError(
            f"audio sample rate must be {expected_sample_rate}, got {sample_rate}"
        )
    if count <= 0 or len(raw) != count * 2:
        raise MetricInputError("audio is empty or truncated")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    return _mono_audio(samples, path.name)


def standardize_reference(
    clean: np.ndarray,
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
    preprocess_total_gain_linear: float | None = None,
) -> np.ndarray:
    """Apply the measured RMS gain and peak scale product exactly once."""

    if sample_rate <= 0:
        raise MetricInputError("sample_rate must be positive")
    if preprocess_total_gain_linear is None:
        raise MetricInputError("preprocess_total_gain_linear is required")
    total_gain = float(preprocess_total_gain_linear)
    if not math.isfinite(total_gain) or total_gain <= 0:
        raise MetricInputError("preprocess_total_gain_linear must be positive and finite")
    raw = np.asarray(clean, dtype=np.float32)
    if raw.ndim != 1 or raw.size == 0 or not np.isfinite(raw).all():
        raise MetricInputError("clean reference must be nonempty mono finite audio")
    reference = raw * np.float32(total_gain)
    return _mono_audio(reference, "standardized reference")


def snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    """10*log10(sum(reference^2)/sum((estimate-reference)^2))."""

    clean = _mono_audio(reference, "SNR reference")
    enhanced = _mono_audio(estimate, "SNR estimate")
    if clean.shape != enhanced.shape:
        raise MetricInputError("SNR inputs must have identical sample counts")
    signal_energy = float(np.dot(clean, clean))
    error = enhanced - clean
    error_energy = float(np.dot(error, error))
    if signal_energy <= 0:
        raise MetricInputError("SNR reference energy must be positive")
    if error_energy == 0:
        return math.inf
    return 10.0 * math.log10(signal_energy / error_energy)


def delta_snr_db(
    reference: np.ndarray, noisy_input: np.ndarray, enhanced: np.ndarray
) -> float:
    """Enhanced SNR minus noisy-input SNR, both against the same aligned clean path."""

    return snr_db(reference, enhanced) - snr_db(reference, noisy_input)


def si_sdr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-invariant SDR using zero-mean target projection."""

    clean = _mono_audio(reference, "SI-SDR reference") - float(np.mean(reference))
    enhanced = _mono_audio(estimate, "SI-SDR estimate") - float(np.mean(estimate))
    if clean.shape != enhanced.shape:
        raise MetricInputError("SI-SDR inputs must have identical sample counts")
    clean_energy = float(np.dot(clean, clean))
    if clean_energy <= 0:
        raise MetricInputError("SI-SDR reference energy must be positive")
    projection = (float(np.dot(enhanced, clean)) / clean_energy) * clean
    target_energy = float(np.dot(projection, projection))
    if target_energy <= 0:
        raise MetricInputError("SI-SDR projected target energy must be positive")
    residual = enhanced - projection
    residual_energy = float(np.dot(residual, residual))
    if residual_energy == 0:
        return math.inf
    return 10.0 * math.log10(target_energy / residual_energy)


def log_spectral_distance(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    frame_size: int = LSD_FRAME_SIZE,
    hop_size: int = LSD_HOP_SIZE,
    epsilon: float = LSD_EPSILON,
) -> float:
    """Mean RMS log-spectral distance with 20 ms Hann, 50% overlap, nfft=512."""

    clean = _mono_audio(reference, "LSD reference")
    enhanced = _mono_audio(estimate, "LSD estimate")
    if clean.shape != enhanced.shape:
        raise MetricInputError("LSD inputs must have identical sample counts")
    if clean.size < frame_size:
        raise MetricInputError("LSD input is shorter than one 20 ms frame")
    if frame_size != 512 or hop_size != 256:
        raise MetricInputError("the frozen pilot requires nfft=512 and 50% overlap")
    if epsilon != 1e-10:
        raise MetricInputError("the frozen pilot requires LSD epsilon=1e-10")
    window = np.hanning(frame_size).astype(np.float64)
    starts = np.arange(0, clean.size - frame_size + 1, hop_size)
    clean_frames = np.lib.stride_tricks.sliding_window_view(clean, frame_size)[starts]
    enhanced_frames = np.lib.stride_tricks.sliding_window_view(enhanced, frame_size)[starts]
    clean_power = np.abs(np.fft.rfft(clean_frames * window, n=frame_size)) ** 2
    enhanced_power = np.abs(np.fft.rfft(enhanced_frames * window, n=frame_size)) ** 2
    clean_db = 10.0 * np.log10(clean_power + epsilon)
    enhanced_db = 10.0 * np.log10(enhanced_power + epsilon)
    return float(np.mean(np.sqrt(np.mean((clean_db - enhanced_db) ** 2, axis=1))))


def _package_version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def _optional_audio_metric(
    name: str,
    function: Callable[..., float],
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int,
    distribution: str,
) -> dict[str, object]:
    if sample_rate != TARGET_SAMPLE_RATE:
        return {
            "status": "not_applicable",
            "reason": "wideband_metric_requires_16_kHz_audio",
            "package_version": _package_version(distribution),
        }
    if reference.shape != estimate.shape:
        return {
            "status": "not_applicable",
            "reason": "sample_count_mismatch",
            "package_version": _package_version(distribution),
        }
    try:
        value = float(function(reference, estimate, sample_rate))
    except (ImportError, ModuleNotFoundError) as error:
        return {
            "status": "not_applicable",
            "reason": f"metric_package_unavailable:{type(error).__name__}",
            "package_version": None,
        }
    except (ValueError, RuntimeError) as error:
        return {
            "status": "not_applicable",
            "reason": f"metric_incompatible:{type(error).__name__}",
            "package_version": _package_version(distribution),
        }
    if not math.isfinite(value):
        return {
            "status": "not_applicable",
            "reason": "metric_was_not_finite",
            "package_version": _package_version(distribution),
        }
    return {
        "status": "observed",
        "value": value,
        "package_version": _package_version(distribution),
    }


def pesq_mos_lqo(reference: np.ndarray, estimate: np.ndarray, sample_rate: int) -> dict[str, object]:
    try:
        from pesq import pesq as run_pesq
    except (ImportError, ModuleNotFoundError):
        return {
            "status": "not_applicable",
            "reason": "metric_package_unavailable:ImportError",
            "package_version": None,
        }
    return _optional_audio_metric(
        "PESQ", lambda ref, est, rate: run_pesq(rate, ref, est, "wb"), reference, estimate, sample_rate, "pesq"
    )


def stoi(reference: np.ndarray, estimate: np.ndarray, sample_rate: int) -> dict[str, object]:
    try:
        from pystoi import stoi as run_stoi
    except (ImportError, ModuleNotFoundError):
        return {
            "status": "not_applicable",
            "reason": "metric_package_unavailable:ImportError",
            "package_version": None,
        }
    return _optional_audio_metric(
        "STOI", lambda ref, est, rate: run_stoi(ref, est, rate, extended=False), reference, estimate, sample_rate, "pystoi"
    )


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def audio_quality_metrics(
    reference: np.ndarray,
    noisy_input: np.ndarray,
    enhanced: np.ndarray,
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> dict[str, object]:
    """Compute every frozen paired-audio metric for one aligned condition."""

    clean = _mono_audio(reference, "audio reference")
    noisy = _mono_audio(noisy_input, "audio input")
    output = _mono_audio(enhanced, "enhanced audio")
    if clean.shape != noisy.shape or clean.shape != output.shape:
        raise MetricInputError("all paired audio metrics require identical sample counts")
    differences = np.abs(np.diff(output)) if output.size > 1 else np.array([0.0])
    rms = float(np.sqrt(np.mean(output * output)))
    clipping_threshold = 32767.0 / 32768.0
    clipping_count = int(np.count_nonzero(np.abs(output) >= clipping_threshold))
    delta = delta_snr_db(clean, noisy, output)
    si_sdr = si_sdr_db(clean, output)
    return {
        "snr_db": _finite_or_none(snr_db(clean, output)),
        "input_snr_db": _finite_or_none(snr_db(clean, noisy)),
        "delta_snr_db": _finite_or_none(delta),
        "si_sdr_db": _finite_or_none(si_sdr),
        "pesq_mos_lqo": pesq_mos_lqo(clean, output, sample_rate),
        "stoi": stoi(clean, output, sample_rate),
        "lsd": log_spectral_distance(clean, output),
        "clipping_ratio": clipping_count / output.size,
        "clipping_count": clipping_count,
        "dc_offset": float(np.mean(output)),
        "max_adjacent_discontinuity": float(np.max(differences)),
        "p99_99_adjacent_discontinuity": float(np.quantile(differences, 0.9999)),
        "rms_dbfs": _finite_or_none(20.0 * math.log10(rms)) if rms > 0 else None,
        "reference_sample_count": int(clean.size),
        "enhanced_sample_count": int(output.size),
        "sample_count_drift": int(output.size - clean.size),
        "duration_drift_s": float((output.size - clean.size) / sample_rate),
    }


def evaluation_tokens(text: str) -> list[str]:
    """NFC + casefold tokenization retaining letters/marks/numbers only."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFC", text).casefold()
    tokens: list[str] = []
    current: list[str] = []
    for character in normalized:
        category = unicodedata.category(character)
        if category[0] in {"L", "M", "N"}:
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def levenshtein_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> LevenshteinCounts:
    """Deterministic edit counts; ties prefer substitution, deletion, insertion."""

    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    distance = np.zeros((rows, columns), dtype=np.int32)
    operations = np.empty((rows, columns), dtype=np.int8)
    distance[:, 0] = np.arange(rows)
    distance[0, :] = np.arange(columns)
    operations[:, 0] = 1  # deletion
    operations[0, 1:] = 2  # insertion
    for row in range(1, rows):
        for column in range(1, columns):
            if reference[row - 1] == hypothesis[column - 1]:
                distance[row, column] = distance[row - 1, column - 1]
                operations[row, column] = 0
                continue
            candidates = (
                (int(distance[row - 1, column - 1]) + 1, 0),
                (int(distance[row - 1, column]) + 1, 1),
                (int(distance[row, column - 1]) + 1, 2),
            )
            distance[row, column], operations[row, column] = min(candidates, key=lambda item: item[0])
    substitutions = deletions = insertions = 0
    row, column = len(reference), len(hypothesis)
    while row or column:
        operation = int(operations[row, column])
        if row and column and operation == 0:
            if reference[row - 1] != hypothesis[column - 1]:
                substitutions += 1
            row -= 1
            column -= 1
        elif row and (not column or operation == 1):
            deletions += 1
            row -= 1
        else:
            insertions += 1
            column -= 1
    return LevenshteinCounts(substitutions, deletions, insertions)


def _wer(reference: Sequence[str], hypothesis: Sequence[str]) -> dict[str, object]:
    counts = levenshtein_counts(reference, hypothesis)
    denominator = len(reference)
    return {
        "reference_tokens": denominator,
        "hypothesis_tokens": len(hypothesis),
        "numerator": counts.edits,
        "denominator": denominator,
        "wer": counts.edits / denominator if denominator else None,
        **counts.as_dict(),
    }


def frozen_normalize(text: str) -> str:
    from demo.stt_extract import normalize_text

    return str(normalize_text(text)["normalized_en"])


def transcript_metrics(reference: str, hypothesis: str) -> dict[str, object]:
    """Score raw Unicode text and the exact frozen demo normalization separately."""

    reference_tokens = evaluation_tokens(reference)
    hypothesis_tokens = evaluation_tokens(hypothesis)
    raw = _wer(reference_tokens, hypothesis_tokens)
    normalized_reference = frozen_normalize(reference)
    normalized_hypothesis = frozen_normalize(hypothesis)
    normalized = _wer(
        evaluation_tokens(normalized_reference), evaluation_tokens(normalized_hypothesis)
    )
    return {
        "status": "observed",
        "raw": {
            **raw,
            "exact_match": reference_tokens == hypothesis_tokens,
        },
        "normalized": {
            **normalized,
            "exact_match": normalized_reference == normalized_hypothesis,
        },
    }


def strict_empty_gold_entities(entities: Mapping[str, object]) -> dict[str, object]:
    """Count clinical false-positive fields against deliberately empty gold."""

    counts: dict[str, int] = {}
    for group in ENTITY_GROUPS:
        value = entities.get(group, [])
        if isinstance(value, list):
            counts[group] = len(value)
        elif isinstance(value, Mapping):
            counts[group] = int(bool(value))
        else:
            raise MetricInputError(f"entity group {group} must be a list or object")
    false_positive_fields = sum(counts[group] for group in CLINICAL_ENTITY_GROUPS)
    return {
        "entity_counts": counts,
        "false_positive_fields": false_positive_fields,
        "gold_status": "empty_nonclinical",
    }


def clinical_metrics(entities: Mapping[str, object]) -> dict[str, object]:
    """Return explicit not-applicable clinical metrics, never synthetic zeroes."""

    not_applicable = {"status": "not_applicable", "reason": NOT_APPLICABLE_REASON}
    return {
        "medical_wer": dict(not_applicable),
        "medication_dose_accuracy": dict(not_applicable),
        "clinical_ner_f1": dict(not_applicable),
        "field_accuracy": dict(not_applicable),
        "critical_error_rate": dict(not_applicable),
        **strict_empty_gold_entities(entities),
    }


def unavailable_text_metrics(reason: str) -> dict[str, object]:
    return {"status": "not_available", "reason": reason}


def unavailable_audio_metrics(reason: str) -> dict[str, object]:
    return {"status": "not_available", "reason": reason}


__all__ = [
    "CLINICAL_ENTITY_GROUPS",
    "ENTITY_GROUPS",
    "LSD_EPSILON",
    "LSD_FRAME_SIZE",
    "LSD_HOP_SIZE",
    "LevenshteinCounts",
    "MetricInputError",
    "TARGET_SAMPLE_RATE",
    "audio_quality_metrics",
    "clinical_metrics",
    "delta_snr_db",
    "evaluation_tokens",
    "frozen_normalize",
    "levenshtein_counts",
    "log_spectral_distance",
    "pesq_mos_lqo",
    "read_pcm16_mono",
    "si_sdr_db",
    "snr_db",
    "standardize_reference",
    "stoi",
    "strict_empty_gold_entities",
    "transcript_metrics",
    "unavailable_audio_metrics",
    "unavailable_text_metrics",
]

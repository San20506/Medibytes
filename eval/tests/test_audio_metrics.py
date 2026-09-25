from __future__ import annotations

import math

import numpy as np
import pytest

from eval.metrics import (
    MetricInputError,
    audio_quality_metrics,
    delta_snr_db,
    log_spectral_distance,
    si_sdr_db,
    snr_db,
    standardize_reference,
)


def test_snr_and_delta_use_frozen_energy_formula() -> None:
    reference = np.array([1.0, 0.0, -1.0])
    noisy = np.array([1.0, 0.25, -1.0])
    enhanced = np.array([1.0, 0.5, -1.0])

    expected = 10.0 * math.log10(2.0 / 0.25)
    assert snr_db(reference, enhanced) == pytest.approx(expected)
    assert delta_snr_db(reference, noisy, enhanced) == pytest.approx(
        expected - 10.0 * math.log10(2.0 / 0.0625)
    )




def test_peak_limited_reference_applies_total_gain_once() -> None:
    clean = np.array([1.0, -1.0], dtype=np.float32)
    reference = standardize_reference(
        clean,
        preprocess_total_gain_linear=10.0 * 0.1,
    )
    assert reference == pytest.approx(clean)
def test_si_sdr_uses_zero_mean_target_projection() -> None:
    reference = np.array([1.0, -1.0, 0.0])
    estimate = np.array([1.0, 0.0, 0.0])

    # Zero-mean projection is 0.5*[1, -1, 0], leaving [1/6, 1/6, -1/3].
    assert si_sdr_db(reference, estimate) == pytest.approx(10.0 * math.log10(3.0))


def test_identical_signals_have_zero_lsd_and_infinite_snr() -> None:
    signal = np.sin(2.0 * np.pi * 220.0 * np.arange(16_000) / 16_000)
    assert log_spectral_distance(signal, signal) == pytest.approx(0.0, abs=1e-12)
    assert math.isinf(snr_db(signal, signal))


def test_audio_quality_reports_diagnostics_and_exact_geometry() -> None:
    reference = np.linspace(-0.5, 0.5, 16_000, dtype=np.float64)
    noisy = reference + 0.01
    enhanced = reference + 0.002
    metrics = audio_quality_metrics(reference, noisy, enhanced)

    assert metrics["reference_sample_count"] == 16_000
    assert metrics["enhanced_sample_count"] == 16_000
    assert metrics["sample_count_drift"] == 0
    assert metrics["duration_drift_s"] == 0.0
    assert metrics["dc_offset"] == pytest.approx(0.002)
    assert metrics["clipping_count"] == 0
    assert metrics["clipping_ratio"] == 0.0
    assert metrics["max_adjacent_discontinuity"] > 0
    assert metrics["p99_99_adjacent_discontinuity"] > 0
    assert metrics["lsd"] >= 0
    assert "pesq_mos_lqo" in metrics
    assert "pesq" not in metrics


def test_paired_audio_metrics_reject_length_mismatch() -> None:
    with pytest.raises(MetricInputError):
        audio_quality_metrics(np.ones(10), np.ones(9), np.ones(10))

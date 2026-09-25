from __future__ import annotations

import hashlib
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from eval.corpus.audio import (
    SAMPLE_RATE,
    active_mask,
    apply_echo,
    derive_seed,
    generate_raw_noise,
    mix_at_target_snr,
    pink_voss16,
    read_pcm16,
    validate_pcm16,
    write_pcm16,
)
from eval.corpus.manifest import NOISE_TYPES, noise_type_for_index


def test_seed_is_first_eight_sha256_bytes_little_endian() -> None:
    expected = int.from_bytes(
        hashlib.sha256(b"medibytes-corpus-v1|sample-001|white").digest()[:8],
        "little",
    )

    assert derive_seed("sample-001", "white") == expected


def test_noise_assignment_repeats_each_condition_three_times() -> None:
    assigned = [noise_type_for_index(index) for index in range(15)]

    assert assigned == [
        "white",
        "white",
        "white",
        "pink_voss16",
        "pink_voss16",
        "pink_voss16",
        "hum50",
        "hum50",
        "hum50",
        "am_babble_proxy",
        "am_babble_proxy",
        "am_babble_proxy",
        "echo",
        "echo",
        "echo",
    ]
    assert {name: assigned.count(name) for name in NOISE_TYPES} == {
        name: 3 for name in NOISE_TYPES
    }


def test_active_mask_uses_20ms_rms_and_10ms_hop() -> None:
    samples = np.concatenate(
        (
            np.zeros(SAMPLE_RATE, dtype=np.float64),
            np.full(SAMPLE_RATE, 0.25, dtype=np.float64),
        )
    )

    mask, frame_rms = active_mask(samples)

    assert frame_rms.shape == (199,)
    assert frame_rms[:7].max() == pytest.approx(0.0)
    assert frame_rms[100:190].min() == pytest.approx(0.25)
    assert mask.shape == samples.shape
    assert not mask[: SAMPLE_RATE - 320].any()
    assert mask[SAMPLE_RATE + 160 :].mean() > 0.99


def test_white_noise_and_active_mask_snr_are_exact_and_deterministic() -> None:
    clean = np.sin(2 * np.pi * 220 * np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE)
    clean = clean / np.max(np.abs(clean))
    first, first_meta = generate_raw_noise("white", "sample-001", clean.size)
    second, second_meta = generate_raw_noise("white", "sample-001", clean.size)
    mask, _ = active_mask(clean)
    noisy, mix_meta = mix_at_target_snr(clean, first, mask, target_snr_db=5.0)

    assert np.array_equal(first, second)
    assert first_meta == second_meta
    assert mix_meta["measured_snr_db"] == pytest.approx(5.0, abs=1e-10)
    assert mix_meta["active_reference_rms"] > 0
    assert np.isfinite(noisy).all()
    assert np.max(np.abs(noisy)) > 0


def test_voss_stream_uses_pinned_coefficients() -> None:
    rng = np.random.Generator(np.random.PCG64(derive_seed("sample-002", "pink_voss16")))
    first = pink_voss16(SAMPLE_RATE, rng)

    assert first.shape == (SAMPLE_RATE,)
    assert first.mean() == pytest.approx(0.0, abs=1e-12)
    assert first.std() == pytest.approx(1.0, abs=1e-12)
    assert math.isfinite(float(np.dot(first, first)))


def test_am_babble_uses_four_independent_voss_streams() -> None:
    noise, metadata = generate_raw_noise(
        "am_babble_proxy", "sample-004", SAMPLE_RATE * 2
    )

    assert noise.shape == (SAMPLE_RATE * 2,)
    assert metadata["modulation_frequencies_hz"] == [3.1, 4.7, 6.3, 8.1]
    assert len(set(metadata["stream_seed_hex"])) == 4
    assert np.isfinite(noise).all()


def test_echo_has_zero_direct_delay_unit_leading_energy_and_preserves_length() -> None:
    clean = np.zeros(SAMPLE_RATE, dtype=np.float64)
    clean[0] = 1.0
    reference, metadata = apply_echo(clean, "sample-005")

    assert reference.shape == clean.shape
    assert reference[0] == pytest.approx(1.0)
    assert metadata["rt60_s"] == 0.4
    assert metadata["direct_delay_ms"] == 0.0
    assert metadata["rir_leading_energy"] == pytest.approx(1.0)


def test_pcm16_writer_has_canonical_44_byte_header(tmp_path: Path) -> None:
    samples = np.array([0, 1, -1, 32766, -32767], dtype=np.int16)
    path = tmp_path / "sample.wav"

    write_pcm16(path, samples, SAMPLE_RATE)
    raw = path.read_bytes()
    decoded = read_pcm16(path)

    assert raw[:4] == b"RIFF"
    assert raw[8:12] == b"WAVE"
    assert raw[12:16] == b"fmt "
    assert raw[36:40] == b"data"
    assert len(raw) == 44 + samples.nbytes
    assert struct.unpack_from("<I", raw, 24)[0] == SAMPLE_RATE
    assert np.array_equal(decoded, samples)


@pytest.mark.parametrize(
    "samples",
    [
        np.array([], dtype=np.int16),
        np.zeros(4, dtype=np.int16),
        np.array([32767, -32768, 0, 1], dtype=np.int16),
    ],
)
def test_pcm16_validation_rejects_empty_zero_and_clipped_audio(
    samples: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        validate_pcm16(samples, SAMPLE_RATE)

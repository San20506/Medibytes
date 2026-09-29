"""Behavioural tests for the noise-matrix builder's mixing and guards."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from en_pilot import build_matrix
from en_pilot.build_matrix import MatrixError, derive_seed, fit_noise, quantize
from eval.corpus.audio import active_mask, write_pcm16


def _speech(sample_count: int = 16_000) -> np.ndarray:
    time = np.arange(sample_count) / 16_000.0
    envelope = 0.5 + 0.5 * np.sin(2.0 * math.pi * 3.0 * time)
    return 0.5 * envelope * np.sin(2.0 * math.pi * 220.0 * time)


def _noise(sample_count: int = 40_000, seed: int = 7) -> np.ndarray:
    return np.random.Generator(np.random.PCG64(seed)).standard_normal(sample_count) * 0.05


@pytest.mark.parametrize("target_snr_db", [0.0, 5.0])
def test_mixing_reproduces_the_target_snr_under_the_frozen_mask(target_snr_db):
    clean = _speech()
    mask, _ = active_mask(clean)
    noise = fit_noise(_noise(), clean.size, derive_seed("sample-001", "demand_office_01", target_snr_db))[0]
    reference, mixture, metadata = build_matrix.mix(clean, noise, mask, target_snr_db)
    assert metadata["measured_snr_db"] == pytest.approx(target_snr_db, abs=1e-6)
    quantize(
        mixture,
        condition_id="sample-001_demand_office_01_%ddb" % int(target_snr_db),
        snr_db=target_snr_db,
        measured_snr_db=metadata["measured_snr_db"],
    )
    # The pre-gain, when one was needed, is applied to speech and noise together
    # and therefore leaves the achieved SNR exactly on target.
    assert metadata["pre_mix_gain"] <= 1.0
    assert float(np.max(np.abs(mixture))) <= build_matrix.config.CLIPPING_CEILING
    assert reference.size == mixture.size


def test_a_clipping_mixture_raises_instead_of_being_written():
    full_scale = np.ones(4_096, dtype=np.float64)
    with pytest.raises(MatrixError, match="clips"):
        quantize(
            full_scale,
            condition_id="sample-001_demand_office_01_0db",
            snr_db=0.0,
            measured_snr_db=0.0,
        )


def test_a_mixture_outside_the_snr_tolerance_raises(tmp_path):
    clean = _speech()
    with pytest.raises(MatrixError, match="against a"):
        quantize(
            clean,
            condition_id="sample-001_demand_office_01_0db",
            snr_db=0.0,
            measured_snr_db=build_matrix.config.SNR_TOLERANCE_DB + 0.01,
        )


def test_a_mixture_cancelled_by_pre_gain_stays_non_clipping():
    # Real 0 dB mixtures peak near 2.0 before the joint gain; the ceiling check
    # must move the peak, not the SNR.
    clean = _speech()
    mask, _ = active_mask(clean)
    loud = _noise(clean.size, seed=11) * 4.0
    _, mixture, metadata = build_matrix.mix(clean, loud, mask, 0.0)
    assert metadata["pre_mix_gain"] < 1.0
    assert float(np.max(np.abs(mixture))) <= build_matrix.config.CLIPPING_CEILING
    assert metadata["measured_snr_db"] == pytest.approx(0.0, abs=1e-6)


def test_derive_seed_is_stable_and_varies_with_every_argument():
    base = derive_seed("sample-001", "demand_office_01", 0.0)
    assert base == derive_seed("sample-001", "demand_office_01", 0.0)
    assert base != derive_seed("sample-002", "demand_office_01", 0.0)
    assert base != derive_seed("sample-001", "demand_office_02", 0.0)
    assert base != derive_seed("sample-001", "demand_office_01", 5.0)
    assert 0 <= base < 2**64


def test_derive_seed_does_not_collide_with_the_matrix_v2_namespace():
    # Same construction as evidence/matrix-v2/build_noise_matrix.py:75-87 under a
    # different namespace; a shared namespace would silently reuse mixtures.
    def matrix_v2_seed(sample_id: str, noise_id: str, snr_db: float) -> int:
        digest = hashlib.sha256(
            f"medibytes-noise-matrix-v1|{sample_id}|{noise_id}|{snr_db}".encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:8], "little", signed=False)

    assert derive_seed("sample-001", "demand_office_01", 0.0) != matrix_v2_seed(
        "sample-001", "demand_office_01", 0.0
    )


def test_fit_noise_tiles_short_recordings_and_crops_long_ones():
    long_noise = _noise(40_000)
    cropped, offset = fit_noise(long_noise, 10_000, 5)
    assert cropped.size == 10_000
    assert 0 <= offset <= long_noise.size - 10_000
    assert np.array_equal(cropped, long_noise[offset : offset + 10_000])

    short_noise = _noise(3_000)
    tiled, rotation = fit_noise(short_noise, 10_000, 5)
    assert tiled.size == 10_000
    assert 0 <= rotation < short_noise.size
    assert np.array_equal(tiled[: 3_000 - rotation], short_noise[rotation:])
    assert fit_noise(short_noise, 10_000, 5)[0].tolist() == tiled.tolist()


def test_noise_digest_mismatch_is_a_hard_failure(tmp_path):
    bank_dir = tmp_path / "demand-16k"
    bank_dir.mkdir()
    write_pcm16(bank_dir / "demand_office_01.wav", (np.ones(1_000) * 0.1 * 32768).astype(np.int16))
    manifest = tmp_path / "bank.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "noise_id": "demand_office_01",
                "category": "office",
                "sha256": "0" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(MatrixError, match="digest mismatch"):
        build_matrix.load_noise_bank(bank_dir, manifest)


def test_an_unlicensed_noise_recording_is_refused(tmp_path):
    bank_dir = tmp_path / "demand-16k"
    bank_dir.mkdir()
    write_pcm16(bank_dir / "demand_office_01.wav", (np.ones(1_000) * 0.1 * 32768).astype(np.int16))
    write_pcm16(bank_dir / "rogue_01.wav", (np.ones(1_000) * 0.1 * 32768).astype(np.int16))
    manifest = tmp_path / "bank.jsonl"
    digest = hashlib.sha256((bank_dir / "demand_office_01.wav").read_bytes()).hexdigest()
    manifest.write_text(
        json.dumps({"noise_id": "demand_office_01", "category": "office", "sha256": digest})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(MatrixError, match="unlicensed"):
        build_matrix.load_noise_bank(bank_dir, manifest)


def test_build_refuses_to_overwrite_an_existing_matrix(tmp_path):
    out_root = tmp_path / "matrix"
    out_root.mkdir()
    clean_root = tmp_path / "clean"
    clean_root.mkdir()
    write_pcm16(clean_root / "sample-001.wav", (_speech() * 32768).astype(np.int16))
    bank = {
        "demand_office_01": {
            "path": clean_root / "sample-001.wav",
            "sha256": "0" * 64,
            "category": "office",
        }
    }
    with pytest.raises(MatrixError, match="already exists"):
        build_matrix.build(
            clean_root, out_root, noise_ids=["demand_office_01"], snr_db_values=[0.0], bank=bank
        )

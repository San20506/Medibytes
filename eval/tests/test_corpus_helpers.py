from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from eval.corpus.audio import (
    SAMPLE_RATE,
    active_mask,
    apply_echo,
    generate_raw_noise,
    float_to_pcm16,
    mix_at_target_snr,
    mix_quantized_at_target_snr,
    pcm16_peak,
    seed_hex,
    write_pcm16,
)
from eval.corpus.manifest import (
    assert_rebuild_identity,
    file_sha256,
    validate_complete_manifest,
    verify_checksum_ledger,
)
from eval.corpus.paths import resolve_data_root
from eval.corpus.selection import (
    privacy_reasons,
    select_fleurs_rows,
    select_mucs_segments,
)


def test_data_root_guard_rejects_missing_empty_and_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MEDIBYTES_EVAL_DATA_ROOT", raising=False)
    with pytest.raises(ValueError, match="MEDIBYTES_EVAL_DATA_ROOT"):
        resolve_data_root()

    monkeypatch.setenv("MEDIBYTES_EVAL_DATA_ROOT", "")
    with pytest.raises(ValueError, match="empty"):
        resolve_data_root()

    monkeypatch.setenv("MEDIBYTES_EVAL_DATA_ROOT", "relative/eval-data")
    with pytest.raises(ValueError, match="absolute"):
        resolve_data_root()

    monkeypatch.setenv("MEDIBYTES_EVAL_DATA_ROOT", str(tmp_path))
    assert resolve_data_root() == tmp_path.resolve()


def test_privacy_filter_rejects_each_forbidden_value() -> None:
    assert privacy_reasons("hello") == ()
    assert "digit" in privacy_reasons("call 9876543210")
    assert "url" in privacy_reasons("visit https://example.invalid")
    assert "email" in privacy_reasons("write to person@example.invalid")
    assert "phone" in privacy_reasons("phone +91 98765 43210")
    assert "privacy_token" in privacy_reasons("my Aadhaar number is private")
    assert "privacy_token" in privacy_reasons("my password is private")


def test_fleurs_selection_uses_numeric_id_and_admission_filters() -> None:
    rows = [
        {"id": "10", "transcription": "ten"},
        {"id": "2", "transcription": "two"},
        {"id": "1", "transcription": "one"},
        {"id": "3", "transcription": "call 12345"},
        {"id": "4", "transcription": "four"},
        {"id": "5", "transcription": "five"},
        {"id": "11", "transcription": "eleven"},
    ]
    durations = {1: 4.0, 2: 13.0, 4: 5.0, 5: 3.0, 10: 6.0, 11: 12.0}

    selected = select_fleurs_rows(
        rows, lambda row: durations.get(int(row["id"]))
    )

    assert [int(item.row["id"]) for item in selected] == [1, 4, 5, 10]


def test_fleurs_duplicate_ids_preserve_published_row_order() -> None:
    rows = [
        {"id": 7, "path": "first.wav", "transcription": "first"},
        {"id": 7, "path": "second.wav", "transcription": "second"},
        {"id": 8, "path": "third.wav", "transcription": "third"},
    ]

    selected = select_fleurs_rows(rows, lambda _row: 4.0, limit=3)

    assert [item.row["path"] for item in selected] == [
        "first.wav",
        "second.wav",
        "third.wav",
    ]


def test_mucs_selection_takes_first_segment_from_first_three_recordings() -> None:
    segments = [
        {"recording_id": "rec-b", "start_sample": 0, "end_sample": 80000, "transcript": "b"},
        {"recording_id": "rec-a", "start_sample": 16000, "end_sample": 96000, "transcript": "alpha two"},
        {"recording_id": "rec-a", "start_sample": 0, "end_sample": 80000, "transcript": "alpha one"},
        {"recording_id": "rec-c", "start_sample": 0, "end_sample": 80000, "transcript": "c"},
        {"recording_id": "rec-d", "start_sample": 0, "end_sample": 80000, "transcript": "d"},
    ]

    selected = select_mucs_segments(
        segments,
        lambda row: (row["end_sample"] - row["start_sample"]) / SAMPLE_RATE,
    )

    assert [(row.recording_id, row.start_sample) for row in selected] == [
        ("rec-a", 0),
        ("rec-b", 0),
        ("rec-c", 0),
    ]


def _audio_record(
    data_root: Path,
    path: Path,
    samples: np.ndarray,
    *,
    peak: float,
    noise: dict[str, object] | None = None,
) -> dict[str, object]:
    common: dict[str, object] = {
        "path": path.as_posix(),
        "sha256": file_sha256(data_root / path),
        "bytes": (data_root / path).stat().st_size,
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "codec": "pcm_s16le",
        "wav_header_bytes": 44,
        "sample_count": int(samples.size),
        "duration_s": samples.size / SAMPLE_RATE,
        "peak": peak,
        "peak_dbfs": 20 * math.log10(peak),
    }
    if noise is None:
        return common
    return common | noise




def _complete_fixture(data_root: Path) -> list[dict[str, object]]:
    buckets = ["en-US-proxy"] * 4 + ["hi"] * 4 + ["ta"] * 4 + ["hi-en-code-mix"] * 3
    noises = (
        ["white"] * 3
        + ["pink_voss16"] * 3
        + ["hum50"] * 3
        + ["am_babble_proxy"] * 3
        + ["echo"] * 3
    )
    records: list[dict[str, object]] = []
    corpus_prefix = Path("corpus") / "pilot-15-v1"
    for index, (bucket, noise_type) in enumerate(zip(buckets, noises, strict=True), 1):
        sample_id = f"sample-{index:03d}"
        clean_float = np.full(SAMPLE_RATE * 3, 0.0072, dtype=np.float64)
        burst_time = np.arange(320, dtype=np.float64) / SAMPLE_RATE
        clean_float[:320] = 0.7079 * np.sin(2 * np.pi * 220 * burst_time)
        clean_float *= 10 ** (-3.0 / 20.0) / np.max(np.abs(clean_float))
        clean_samples = float_to_pcm16(clean_float)
        clean_quantized = clean_samples.astype(np.float64) / 32768.0
        mask, _ = active_mask(clean_quantized)
        raw_noise, parameters = generate_raw_noise(noise_type, sample_id, clean_samples.size)
        post_mix_gain = 1.0
        if noise_type == "echo":
            reference, echo_metadata = apply_echo(clean_quantized, sample_id)
            parameters |= echo_metadata
            unscaled_mix, _ = mix_at_target_snr(
                reference, raw_noise, mask, target_snr_db=5.0
            )
            peak = float(np.max(np.abs(unscaled_mix)))
            post_mix_gain = min(1.0, 10 ** (-1.0 / 20.0) / peak)
            reference *= post_mix_gain
            raw_noise *= post_mix_gain
        else:
            reference = clean_quantized
        noisy_samples, mix_metadata = mix_quantized_at_target_snr(
            reference, raw_noise, mask, target_snr_db=5.0
        )
        clean_path = corpus_prefix / "clean" / f"{sample_id}.wav"
        noisy_path = corpus_prefix / "noisy" / f"{sample_id}.wav"
        write_pcm16(data_root / clean_path, clean_samples, SAMPLE_RATE)
        write_pcm16(data_root / noisy_path, noisy_samples, SAMPLE_RATE)
        clean_peak = pcm16_peak(clean_samples)
        noisy_peak = pcm16_peak(noisy_samples)
        source_digest = hashlib.sha256(f"source-{index}".encode()).hexdigest()
        transcript = f"source transcript {index}"
        record: dict[str, object] = {
            "schema_version": "1.0.0",
            "record_status": "complete",
            "corpus_id": "pilot-15-v1",
            "sample_id": sample_id,
            "group_id": f"source-group-{index}",
            "split_role": "smoke",
            "tuning_allowed": False,
            "bucket": bucket,
            "language_profile": bucket,
            "code_mix": bucket == "hi-en-code-mix",
            "locale_observed": "en_us" if index <= 4 else "hi_in" if index <= 8 else "ta_in" if index <= 12 else "hi-en",
            "source": {
                "dataset": (
                    "Google FLEURS" if index <= 12 else "MUCS 2021 Hindi-English test"
                ),
                "release": "fixture-v1",
                "revision": None,
                "split": "validation" if index <= 12 else "test",
                "digest": source_digest,
                "source_id": str(index),
                "member_path": f"audio/{index}.wav",
                "client_id": None,
                "recording_id": None,
                "speaker_id": None,
                "row": {"id": index},
                "transcript": transcript,
                "transcript_sha256": hashlib.sha256(transcript.encode()).hexdigest(),
                "license_spdx": "CC-BY-4.0" if index <= 12 else "CC-BY-SA-4.0",
                "attribution": "fixture",
            },
            "privacy_decision": {
                "eligible": True,
                "reasons": [],
                "policy_sha256": "0" * 64,
            },
            "clean": _audio_record(
                data_root, clean_path, clean_samples, peak=clean_peak
            ),
            "noisy": _audio_record(
                data_root,
                noisy_path,
                noisy_samples,
                peak=noisy_peak,
                noise={
                    "noise_type": noise_type,
                    "seed": seed_hex(sample_id, noise_type),
                    "target_snr_db": mix_metadata["target_snr_db"],
                    "measured_snr_db": mix_metadata["measured_snr_db"],
                    "active_rms": mix_metadata["active_reference_rms"],
                    "post_mix_gain": post_mix_gain,
                    "clipping_count": 0,
                    "parameters": parameters | {
                        "noise_scale": mix_metadata["noise_scale"]
                    },
                },
            ),
            "build": {
                "tool": "eval.corpus.build_corpus",
                "tool_sha256": "1" * 64,
                "decoder": "imageio-ffmpeg==0.6.0",
                "decoder_sha256": "2" * 64,
            },
            "reference_status": "source_reference",
        }
        records.append(record)
    return records


def test_complete_manifest_admission_checks_counts_geometry_and_hashes(
    tmp_path: Path,
) -> None:
    records = _complete_fixture(tmp_path)

    summary = validate_complete_manifest(records, tmp_path)

    assert summary == {
        "groups": 15,
        "files": 30,
        "buckets": {"en-US-proxy": 4, "hi": 4, "ta": 4, "hi-en-code-mix": 3},
        "noises": {
            "white": 3,
            "pink_voss16": 3,
            "hum50": 3,
            "am_babble_proxy": 3,
            "echo": 3,
        },
    }


def test_complete_manifest_rejects_tampered_file(tmp_path: Path) -> None:
    records = _complete_fixture(tmp_path)
    target = tmp_path / records[0]["clean"]["path"]
    target.write_bytes(target.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="SHA-256"):
        validate_complete_manifest(records, tmp_path)


def test_complete_manifest_rejects_tampered_snr_metadata(tmp_path: Path) -> None:
    records = _complete_fixture(tmp_path)
    records[0]["noisy"]["measured_snr_db"] = 5.1

    with pytest.raises(ValueError, match="does not match decoded audio"):
        validate_complete_manifest(records, tmp_path)


def test_rebuild_identity_compares_hashes_and_metadata_not_output_root(
    tmp_path: Path,
) -> None:
    records = _complete_fixture(tmp_path)
    rebuilt = json.loads(json.dumps(records))
    for record in rebuilt:
        record["clean"]["path"] = record["clean"]["path"].replace("pilot-15-v1", "rebuild")
        record["noisy"]["path"] = record["noisy"]["path"].replace("pilot-15-v1", "rebuild")

    assert_rebuild_identity(records, rebuilt)
    rebuilt[3]["noisy"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="rebuild identity"):
        assert_rebuild_identity(records, rebuilt)


def test_checksum_ledger_detects_changed_source(tmp_path: Path) -> None:
    relative = Path("sources") / "source.bin"
    source = tmp_path / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(b"first")
    ledger = tmp_path / "source-checksums.sha256"
    ledger.write_text(f"{hashlib.sha256(b'first').hexdigest()}  {relative.as_posix()}\n")

    verify_checksum_ledger(ledger, tmp_path, {relative.as_posix(): source})
    source.write_bytes(b"second")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_checksum_ledger(ledger, tmp_path, {relative.as_posix(): source})

"""Literal-source corpus construction and deterministic paired rendering."""

from __future__ import annotations

import copy
import math
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np

from eval.corpus.audio import (
    SAMPLE_RATE,
    active_mask,
    apply_echo,
    clipping_count,
    read_pcm16,
    generate_raw_noise,
    mix_at_target_snr,
    mix_quantized_at_target_snr,
    pcm16_peak,
    seed_hex,
    write_pcm16,
)
from eval.corpus.manifest import file_sha256, noise_type_for_index
from eval.corpus.media import Decoder, canonicalize_clean, decode_bytes, decode_segment
from eval.corpus.sources import FleursParquetReader

ECHO_HEADROOM_CEILING = 10 ** (-1.0 / 20.0)


class SourceDecoder:
    """Decode only literal source rows named by the resolved manifest."""

    def __init__(self, data_root: Path, decoder: Decoder):
        self.data_root = data_root
        self.decoder = decoder
        self._fleurs: dict[Path, FleursParquetReader] = {}

    def _fleurs_reader(self, path: Path) -> FleursParquetReader:
        if path not in self._fleurs:
            self._fleurs[path] = FleursParquetReader(path)
        return self._fleurs[path]

    def _extract_mucs_member(self, archive: Path, member_path: str) -> Path:
        safe = PurePosixPath(member_path)
        if safe.is_absolute() or ".." in safe.parts or not safe.parts:
            raise ValueError(f"unsafe MUCS archive member: {member_path!r}")
        cache = (
            self.data_root
            / "cache"
            / "corpus"
            / "mucs"
            / file_sha256(archive)
            / safe
        )
        if cache.is_file():
            return cache
        with tarfile.open(archive, mode="r:gz") as bundle:
            member = bundle.getmember(member_path)
            if not member.isfile():
                raise ValueError(f"MUCS member is not a regular file: {member_path}")
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError(f"cannot extract MUCS member: {member_path}")
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_name(cache.name + ".part")
            with temporary.open("wb") as output:
                shutil.copyfileobj(source, output)
            if temporary.stat().st_size != member.size:
                temporary.unlink(missing_ok=True)
                raise ValueError(f"MUCS member size mismatch: {member_path}")
            temporary.replace(cache)
        return cache

    def decode(self, record: Mapping[str, Any]) -> np.ndarray:
        source = record["source"]
        if source["dataset"] == "Google FLEURS":
            relative = str(source["member_path"]).split("#", 1)[0]
            parquet_path = self.data_root / relative
            if not parquet_path.is_file():
                raise ValueError(f"FLEURS parquet is missing: {relative}")
            if file_sha256(parquet_path) != source["digest"]:
                raise ValueError(f"FLEURS source digest changed: {relative}")
            source_id = str(source["source_id"])
            source_row_id, source_audio_id = source_id.split("@", 1)
            reader = self._fleurs_reader(parquet_path)
            matches = [
                row
                for row in reader.rows
                if int(row["id"]) == int(source_row_id)
                and PurePosixPath(str(row.get("path"))).stem == source_audio_id
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"FLEURS source identity must resolve exactly once: {source_id}"
                )
            row = matches[0]
            samples = decode_bytes(reader.audio_bytes(row), self.decoder)
            if samples.size != int(row["num_samples"]):
                raise ValueError(f"FLEURS duration mismatch for source ID {source_id}")
            return samples
        if source["dataset"] == "MUCS 2021 Hindi-English test":
            archive = self.data_root / "sources/mucs/Hindi-English_test.tar.gz"
            if not archive.is_file() or file_sha256(archive) != source["digest"]:
                raise ValueError("MUCS archive is missing or its digest changed")
            member = str(source["member_path"])
            recording_path = self._extract_mucs_member(archive, member)
            row = source["row"]
            return decode_segment(
                recording_path,
                int(row["start_sample"]),
                int(row["end_sample"]),
                self.decoder,
            )
        raise ValueError(f"unsupported source dataset: {source['dataset']}")


def _audio_metadata(
    data_root: Path,
    path: Path,
    samples: np.ndarray,
) -> dict[str, object]:
    peak = pcm16_peak(samples)
    return {
        "path": path.resolve(strict=False).relative_to(data_root).as_posix(),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "codec": "pcm_s16le",
        "wav_header_bytes": 44,
        "sample_count": int(samples.size),
        "duration_s": samples.size / SAMPLE_RATE,
        "peak": peak,
        "peak_dbfs": 20.0 * math.log10(peak),
    }


def render_noise(
    clean: np.ndarray,
    sample_id: str,
    noise_type: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Render one pinned condition and return PCM16 plus measured metadata."""

    clean_float = clean.astype(np.float64) / 32768.0
    mask, _ = active_mask(clean_float)
    raw_noise, parameters = generate_raw_noise(noise_type, sample_id, clean.size)
    post_mix_gain = 1.0
    if noise_type == "echo":
        reference, echo_metadata = apply_echo(clean_float, sample_id)
        parameters |= echo_metadata
        unscaled_mix, _ = mix_at_target_snr(
            reference, raw_noise, mask, target_snr_db=5.0
        )
        peak = float(np.max(np.abs(unscaled_mix)))
        if peak > ECHO_HEADROOM_CEILING:
            post_mix_gain = ECHO_HEADROOM_CEILING / peak
            reference *= post_mix_gain
            raw_noise *= post_mix_gain
    else:
        reference = clean_float
        unscaled_mix, _ = mix_at_target_snr(
            reference, raw_noise, mask, target_snr_db=5.0
        )
        if float(np.max(np.abs(unscaled_mix))) > 1.0:
            raise ValueError(
                f"additive condition would clip for {sample_id}/{noise_type}"
            )
    noisy, mix_metadata = mix_quantized_at_target_snr(
        reference, raw_noise, mask, target_snr_db=5.0
    )
    if clipping_count(noisy):
        raise ValueError(f"generated condition is clipped: {sample_id}/{noise_type}")
    return noisy, {
        "noise_type": noise_type,
        "seed": seed_hex(sample_id, noise_type),
        "target_snr_db": mix_metadata["target_snr_db"],
        "measured_snr_db": mix_metadata["measured_snr_db"],
        "active_rms": mix_metadata["active_reference_rms"],
        "post_mix_gain": post_mix_gain,
        "clipping_count": 0,
        "parameters": parameters | {"noise_scale": mix_metadata["noise_scale"]},
    }


def build_corpus_records(
    records: list[Mapping[str, Any]],
    *,
    data_root: Path,
    output_root: Path,
    decoder: Decoder,
    build_tool_sha256: str,
) -> list[dict[str, Any]]:
    """Build exactly one clean/noisy WAV per literal resolved source record."""

    if len(records) != 15:
        raise ValueError(f"build requires exactly 15 resolved sources, got {len(records)}")
    output_root = output_root.resolve(strict=False)
    try:
        output_root.relative_to(data_root)
    except ValueError as error:
        raise ValueError("corpus output must stay under MEDIBYTES_EVAL_DATA_ROOT") from error
    if output_root.exists():
        raise FileExistsError(f"corpus output is immutable and already exists: {output_root}")
    source_decoder = SourceDecoder(data_root, decoder)
    complete: list[dict[str, Any]] = []
    for index, resolved in enumerate(records):
        if resolved.get("record_status") not in {"resolved", "complete"}:
            raise ValueError(f"source record {index} is not resolved")
        sample_id = str(resolved["sample_id"])
        expected_noise = noise_type_for_index(index)
        source_samples = source_decoder.decode(resolved)
        clean_path = output_root / "clean" / f"{sample_id}.wav"
        clean_info = canonicalize_clean(source_samples, clean_path)
        clean = read_pcm16(clean_path)
        noisy, noise_info = render_noise(clean, sample_id, expected_noise)
        if noisy.size != clean.size:
            raise ValueError(f"clean/noisy sample count mismatch for {sample_id}")
        noisy_path = output_root / "noisy" / f"{sample_id}.wav"
        write_pcm16(noisy_path, noisy, SAMPLE_RATE)
        clean_metadata = _audio_metadata(data_root, clean_path, clean)
        if not math.isclose(
            float(clean_metadata["peak_dbfs"]),
            float(clean_info["peak_dbfs"]),
            abs_tol=1e-12,
        ):
            raise AssertionError("clean metadata normalization drift")
        noisy_metadata = _audio_metadata(data_root, noisy_path, noisy)
        record = copy.deepcopy(dict(resolved))
        record.update(
            {
                "record_status": "complete",
                "clean": clean_metadata,
                "noisy": noisy_metadata | noise_info,
                "build": {
                    "tool": "eval.corpus.build_corpus",
                    "tool_sha256": build_tool_sha256,
                    "decoder": decoder.provider,
                    "decoder_sha256": decoder.sha256,
                },
                "reference_status": "source_reference",
            }
        )
        complete.append(record)
    return complete

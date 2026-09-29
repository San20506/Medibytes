"""Build the 1,440-condition real-DEMAND-noise matrix on frozen audio primitives.

`evidence/matrix-v2/build_noise_matrix.py` is the shape this follows, with three
deliberate departures: it imports the frozen corpus primitives from this
repository instead of a hardcoded sibling worktree (that file `SystemExit`s when
`/home/sandy/Projects/Medibytes-worktrees/eval-compare` is absent), it namespaces
its seeds away from `matrix-v2` so the two studies never share a mixture, and it
verifies every noise recording against `evidence/noise-bank/demand-bank-manifest.jsonl`
before use instead of recording the digest it happened to read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from en_pilot import config
from eval.corpus.audio import (
    SAMPLE_RATE,
    active_mask,
    clipping_count,
    float_to_pcm16,
    mix_at_target_snr,
    read_pcm16,
    write_pcm16,
)
from eval.corpus.manifest import file_sha256
from eval.corpus.paths import resolve_data_root


class MatrixError(RuntimeError):
    """The requested matrix cannot be built as specified."""


# --------------------------------------------------------------------- seeds
def derive_seed(sample_id: str, noise_id: str, snr_db: float) -> int:
    """Deterministic unsigned 64-bit seed, namespaced away from every other study."""

    payload = f"{config.SEED_NAMESPACE}|{sample_id}|{noise_id}|{snr_db}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


# --------------------------------------------------------------------- noise
def load_noise_bank(
    bank_dir: Path, manifest_path: Path
) -> dict[str, dict[str, Any]]:
    """Index the real noise recordings and bind each to its licensed digest."""

    if not bank_dir.is_dir():
        raise MatrixError(f"noise bank folder does not exist: {bank_dir}")
    if not manifest_path.is_file():
        raise MatrixError(f"noise bank manifest does not exist: {manifest_path}")
    licensed = {
        json.loads(line)["noise_id"]: json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    on_disk = {path.stem: path for path in sorted(bank_dir.glob("*.wav"))}
    if not on_disk:
        raise MatrixError(f"no .wav files in {bank_dir}")
    missing = sorted(set(licensed) - set(on_disk))
    if missing:
        raise MatrixError(f"noise recordings absent for licensed ids: {missing}")
    unlisted = sorted(set(on_disk) - set(licensed))
    if unlisted:
        raise MatrixError(f"noise recordings present but unlicensed: {unlisted}")
    bank: dict[str, dict[str, Any]] = {}
    for noise_id, path in on_disk.items():
        digest = file_sha256(path)
        if digest != licensed[noise_id]["sha256"]:
            raise MatrixError(
                f"noise digest mismatch for {noise_id}: expected "
                f"{licensed[noise_id]['sha256']}, read {digest}"
            )
        bank[noise_id] = {
            "path": path,
            "sha256": digest,
            "category": licensed[noise_id]["category"],
        }
    return bank


def fit_noise(noise: np.ndarray, sample_count: int, seed: int) -> tuple[np.ndarray, int]:
    """Deterministically tile or crop a recording to the target length.

    The offset actually consumed is returned so a recording that gets looped is
    auditable rather than implicit. Same algorithm as
    `evidence/matrix-v2/build_noise_matrix.py:165-183`.
    """

    if noise.ndim != 1 or noise.size == 0 or not np.isfinite(noise).all():
        raise MatrixError("noise recording must be nonempty finite mono")
    if float(np.max(np.abs(noise))) <= 0:
        raise MatrixError("noise recording is silent")
    if noise.size >= sample_count:
        generator = np.random.Generator(np.random.PCG64(seed))
        offset = int(generator.integers(0, noise.size - sample_count + 1))
        return noise[offset : offset + sample_count], offset
    repeats = int(math.ceil(sample_count / noise.size))
    generator = np.random.Generator(np.random.PCG64(seed))
    rotation = int(generator.integers(0, noise.size))
    tiled = np.concatenate([noise[rotation:], noise[:rotation]] * repeats)
    return tiled[:sample_count], rotation


# --------------------------------------------------------------------- mixing
def full_snr_db(reference: np.ndarray, mixture: np.ndarray) -> float:
    error = mixture - reference
    numerator = float(np.dot(reference, reference))
    denominator = float(np.dot(error, error))
    if numerator <= 0:
        raise MatrixError("clean energy is zero")
    if denominator <= 0:
        return math.inf
    return 10.0 * math.log10(numerator / denominator)


def mix(
    clean: np.ndarray, noise: np.ndarray, mask: np.ndarray, snr_db: float
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Scale to the target SNR without clipping, recording both SNR definitions.

    A joint pre-gain on speech and noise leaves the SNR ratio unchanged, so
    headroom can be bought without moving the difficulty curve. A post-mix
    rescale would be wrong: it renormalises the noise and silently moves the
    achieved SNR off target.
    """

    mixture, metadata = mix_at_target_snr(clean, noise, mask, target_snr_db=snr_db)
    peak = float(np.max(np.abs(mixture)))
    pre_gain = 1.0
    if peak > config.CLIPPING_CEILING:
        pre_gain = (config.CLIPPING_CEILING * 0.999) / peak
        clean = clean * pre_gain
        noise = noise * pre_gain
        mask = mask * pre_gain
        mixture, metadata = mix_at_target_snr(
            clean, noise, mask, target_snr_db=snr_db
        )
    peak = float(np.max(np.abs(mixture)))
    if peak > config.CLIPPING_CEILING:
        raise MatrixError(
            f"mixture still clips after pre-gain: peak {peak:.4f}; recorded rather "
            f"than rescaled, because a post-mix rescale moves the achieved SNR"
        )
    metadata["measured_full_snr_db"] = full_snr_db(clean, mixture)
    metadata["peak"] = peak
    metadata["pre_mix_gain"] = pre_gain
    return clean, mixture, metadata


def quantize(
    mixture: np.ndarray, *, condition_id: str, snr_db: float, measured_snr_db: float
) -> np.ndarray:
    """Quantise and refuse to write anything that clips or misses its SNR target."""

    pcm = float_to_pcm16(mixture)
    clips = clipping_count(pcm)
    if clips:
        raise MatrixError(f"quantised mixture clips ({clips} samples): {condition_id}")
    if abs(measured_snr_db - snr_db) > config.SNR_TOLERANCE_DB:
        raise MatrixError(
            f"{condition_id}: achieved {measured_snr_db:.3f} dB against a "
            f"{snr_db:.1f} dB target, outside {config.SNR_TOLERANCE_DB} dB"
        )
    return pcm


# --------------------------------------------------------------------- driver
def build(
    clean_root: Path,
    out_root: Path,
    *,
    noise_ids: Sequence[str],
    snr_db_values: Sequence[float],
    bank: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the full matrix and return one provenance record per mixture."""

    if not clean_root.is_dir():
        raise MatrixError(f"clean root does not exist: {clean_root}")
    originals = {path.stem: path for path in sorted(clean_root.glob("*.wav"))}
    if not originals:
        raise MatrixError(f"no clean bases found in {clean_root}")
    missing = [noise_id for noise_id in noise_ids if noise_id not in bank]
    if missing:
        raise MatrixError(f"unknown noise ids: {missing}")

    before = {name: file_sha256(path) for name, path in originals.items()}
    staging = out_root.with_name(out_root.name + ".building")
    if out_root.exists():
        raise MatrixError(f"matrix output already exists: {out_root}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    records: list[dict[str, Any]] = []
    try:
        for sample_id, source_path in originals.items():
            clean = read_pcm16(source_path).astype(np.float64) / 32768.0
            mask, _ = active_mask(clean, SAMPLE_RATE)
            clean_rms = float(np.sqrt(np.mean(np.square(clean))))
            active = mask > 0
            active_rms = float(np.sqrt(np.mean(np.square(clean[active]))))

            for noise_id in noise_ids:
                entry = bank[noise_id]
                raw = read_pcm16(entry["path"]).astype(np.float64) / 32768.0
                for snr_db in snr_db_values:
                    seed = derive_seed(sample_id, noise_id, snr_db)
                    noise, offset = fit_noise(raw, clean.size, seed)
                    noise_rms = float(np.sqrt(np.mean(np.square(noise))))
                    reference, mixture, mix_meta = mix(clean, noise, mask, snr_db)
                    condition_id = f"{sample_id}_{noise_id}_{int(snr_db)}db"
                    pcm = quantize(
                        mixture,
                        condition_id=condition_id,
                        snr_db=snr_db,
                        measured_snr_db=mix_meta["measured_snr_db"],
                    )
                    target = staging / f"{condition_id}.wav"
                    write_pcm16(target, pcm, SAMPLE_RATE)
                    records.append(
                        {
                            "condition_id": condition_id,
                            "speech": sample_id,
                            "speech_path": str(source_path),
                            "speech_sha256": before[sample_id],
                            "bucket": config.BUCKET,
                            "noise": noise_id,
                            "category": entry["category"],
                            "snr_db": float(snr_db),
                            "noise_offset": int(offset),
                            "noise_fit": "crop" if raw.size >= clean.size else "tile",
                            "noise_source": "real_recording",
                            "noise_source_path": str(entry["path"]),
                            "noise_source_sha256": entry["sha256"],
                            "pre_mix_gain": float(mix_meta["pre_mix_gain"]),
                            "speech_rms": clean_rms,
                            "speech_active_rms": active_rms,
                            "speech_reference_rms": float(
                                np.sqrt(np.mean(np.square(reference)))
                            ),
                            "noise_rms_before_scaling": noise_rms,
                            "noise_scale": float(mix_meta["noise_scale"]),
                            "seed": f"{seed:016x}",
                            "measured_snr_db": float(mix_meta["measured_snr_db"]),
                            "measured_full_snr_db": float(
                                mix_meta["measured_full_snr_db"]
                            ),
                            "peak": float(mix_meta["peak"]),
                            "clipping_count": int(clipping_count(pcm)),
                            "sample_count": int(pcm.size),
                            "sample_rate": SAMPLE_RATE,
                            "output_path": str(out_root / f"{condition_id}.wav"),
                            "output_sha256": file_sha256(target),
                        }
                    )
        _check_originals(before, originals)
        _write_manifest(staging / "noise-matrix.jsonl", records)
        staging.replace(out_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return records


def _check_originals(
    before: dict[str, str], originals: dict[str, Path]
) -> None:
    changed = [name for name in before if before[name] != file_sha256(originals[name])]
    if changed:
        raise MatrixError(f"ORIGINALS MODIFIED: {changed}")


def _write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="en_pilot.build_matrix",
        description="Build the real-DEMAND-noise matrix for the English pilot.",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get(config.ENVIRONMENT_NAME, config.DATA_ROOT_DEFAULT),
    )
    parser.add_argument("--noise-bank", default=None)
    parser.add_argument("--clean-root", default=None)
    parser.add_argument("--out-root", default=None)
    parser.add_argument(
        "--snr", type=float, action="append", default=None, dest="snr_values"
    )
    arguments = parser.parse_args(argv)

    data_root = resolve_data_root(arguments.data_root)
    bank_dir = Path(arguments.noise_bank) if arguments.noise_bank else (
        data_root / config.NOISE_BANK_DIR
    )
    manifest_path = data_root / config.NOISE_BANK_MANIFEST
    clean_root = Path(arguments.clean_root) if arguments.clean_root else (
        data_root / config.CLEAN_DIR
    )
    out_root = Path(arguments.out_root) if arguments.out_root else (
        data_root / config.MATRIX_DIR
    )
    snr_values = tuple(
        arguments.snr_values
        if arguments.snr_values
        else config.SNR_TARGETS_DB
    )

    bank = load_noise_bank(bank_dir, manifest_path)
    records = build(
        clean_root,
        out_root,
        noise_ids=sorted(bank),
        snr_db_values=snr_values,
        bank=bank,
    )
    print(
        json.dumps(
            {
                "mixtures": len(records),
                "bases": len({record["speech"] for record in records}),
                "noises": len(bank),
                "snr_targets_db": list(snr_values),
                "originals_unchanged": True,
                "manifest": str(out_root / "noise-matrix.jsonl"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

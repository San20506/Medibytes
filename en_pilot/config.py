"""Pinned constants for the English-only denoising WER pilot.

Literals only: every study-wide decision that must not drift at runtime lives
here so `evidence/en-pilot/README.md` and `results/comparison.json` can quote the
exact values that produced a decision.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT_DEFAULT = "/home/sandy/.local/share/medibytes-eval"
ENVIRONMENT_NAME = "MEDIBYTES_EVAL_DATA_ROOT"

CORPUS_ID = "en-40-v1"
CLEAN_DIR = "corpus/en-40-v1/clean"
MATRIX_DIR = "matrix-en-40"
OUT_DIR = "en-pilot-run"
BASE_COUNT = 40
NOISE_BANK_DIR = "noise-bank/demand-16k"
SNR_TARGETS_DB = (0.0, 5.0)
SAMPLE_RATE = 16_000
TARGET_PEAK_DBFS = -3.0
MIN_DURATION_S = 3.0
MAX_DURATION_S = 12.0
SEED_NAMESPACE = "medibytes-en-pilot-v1"
SNR_TOLERANCE_DB = 0.5
CLIPPING_CEILING = 32767.0 / 32768.0
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 1729
HOLM_FAMILY_COUNT = 2
WER_DELTA_CI_LOW_MIN = -0.02
WER_DELTA_CI_HIGH_MAX = 0.0
PRIMARY_DECODER = "small"
CONTINUITY_DECODERS = ("base",)
BACKENDS = ("none", "dpdfnet2-onnx", "sherpa-gtcrn-simple")
CANDIDATE_BACKENDS = ("dpdfnet2-onnx", "sherpa-gtcrn-simple")
BASELINE_BACKEND = "none"

# One venv and one repository per backend. A shared interpreter would cache the
# first repository's `demo` package for every backend after it, so the driver
# launches `en_pilot.enhance` once per backend under that backend's own venv.
# Same values as `evidence/matrix-v2/matrix_eval.py:45-50`.
BACKEND_REPOS = {
    "sherpa-gtcrn-simple": "/home/sandy/Projects/Medibytes-worktrees/eval-sherpa-onnx",
    "dpdfnet2-onnx": "/home/sandy/Projects/Medibytes-worktrees/eval-dpdfnet",
}
# `none` is the identity control and needs no repository of its own; the compare
# worktree supplies `demo/denoise_backends/none.py`.
BASELINE_REPO = "/home/sandy/Projects/Medibytes-worktrees/eval-compare"

FLEURS_PARQUET_RELATIVE = "sources/fleurs/en_us/validation/0000.parquet"
FLEURS_REPO = "google/fleurs"
FLEURS_REVISION = "168de341b3db6859a9bac1c50a2ef5e3b47647e0"
FLEURS_SPLIT = "validation"
FLEURS_LICENSE = "CC-BY-4.0"
BUCKET = "en-US-proxy"
LANGUAGE_PROFILE = "en-US-proxy"

NOISE_BANK_MANIFEST = "noise-bank/demand-bank-manifest.jsonl"
METHOD_CONFIG_ROOT = REPO_ROOT / "eval" / "config" / "methods"
STT_DECODE_KWARGS = {
    "beam_size": 1,
    "word_timestamps": True,
    "temperature": 0,
}

DECISIONS = ("adopt", "reject", "baseline")


def repo_for_backend(backend: str) -> str:
    """Return the single repository root whose `demo` package a backend may use."""

    if backend == BASELINE_BACKEND:
        return BASELINE_REPO
    return BACKEND_REPOS[backend]


__all__ = [
    "BASELINE_BACKEND",
    "BASELINE_REPO",
    "BACKEND_REPOS",
    "BACKENDS",
    "BOOTSTRAP_ITERATIONS",
    "BOOTSTRAP_SEED",
    "BUCKET",
    "CANDIDATE_BACKENDS",
    "CLIPPING_CEILING",
    "CONTINUITY_DECODERS",
    "CORPUS_ID",
    "DATA_ROOT_DEFAULT",
    "DECISIONS",
    "ENVIRONMENT_NAME",
    "FLEURS_LICENSE",
    "FLEURS_PARQUET_RELATIVE",
    "FLEURS_REPO",
    "FLEURS_REVISION",
    "FLEURS_SPLIT",
    "HOLM_FAMILY_COUNT",
    "LANGUAGE_PROFILE",
    "MATRIX_DIR",
    "MAX_DURATION_S",
    "METHOD_CONFIG_ROOT",
    "MIN_DURATION_S",
    "NOISE_BANK_DIR",
    "NOISE_BANK_MANIFEST",
    "OUT_DIR",
    "PRIMARY_DECODER",
    "REPO_ROOT",
    "SAMPLE_RATE",
    "SEED_NAMESPACE",
    "SNR_TARGETS_DB",
    "SNR_TOLERANCE_DB",
    "STT_DECODE_KWARGS",
    "TARGET_PEAK_DBFS",
    "WER_DELTA_CI_HIGH_MAX",
    "WER_DELTA_CI_LOW_MIN",
    "repo_for_backend",
]

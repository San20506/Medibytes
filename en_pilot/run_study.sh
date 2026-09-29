#!/usr/bin/env bash
# Run the English-only denoising WER pilot end to end, from the repository root.
#
# One process per backend, each launched with that backend's own venv python, so
# `sys.modules` can never cache a second repository's `demo` package for a
# backend that is not the one running. Transcription and scoring stay in the
# driver venv, which is the only environment with faster-whisper and pyarrow.
set -euo pipefail

cd "$(dirname "$0")/.."

DRIVER_PYTHON="${DRIVER_PYTHON:-.venv/bin/python}"
DATA_ROOT="${MEDIBYTES_EVAL_DATA_ROOT:-/home/sandy/.local/share/medibytes-eval}"
#
# `build_corpus` and `build_matrix` fail closed if their output already exists,
# so a re-run needs `rm -rf $DATA_ROOT/corpus/en-40-v1 $DATA_ROOT/matrix-en-40`.
# The expensive phase, transcription, is resume-safe and skips clips already
# recorded in `transcripts-<decoder>.jsonl`.
WORKTREES="${WORKTREES:-/home/sandy/Projects/Medibytes-worktrees}"
ENHANCE_WORKERS="${ENHANCE_WORKERS:-4}"
# `en_pilot.transcribe` verifies the CTranslate2 runtime before committing to a
# device and falls back to cpu/int8 when it cannot load. On this 16-core host the
# measured CPU optimum is 2 streams x 8 threads (RTF 0.32 for `base`, 0.93 for
# `small`); on a working CUDA runtime, raise this toward 4-6.
TRANSCRIBE_STREAMS="${TRANSCRIBE_STREAMS:-2}"

phase() { printf '\n=== %s ===\n' "$1"; }

phase "build corpus (40 English bases)"
"$DRIVER_PYTHON" -m en_pilot.build_corpus --data-root "$DATA_ROOT"

phase "build noise matrix (40 bases x 18 DEMAND noises x 2 SNRs)"
"$DRIVER_PYTHON" -m en_pilot.build_matrix --data-root "$DATA_ROOT"

# One repo per PROCESS, never per thread. The `none` control and each candidate
# run in the venv that owns their adapter, in this order.
enhance_with() {
    local backend="$1" repo="$2"
    printf '\n=== enhance %s (%s) ===\n' "$backend" "$repo"
    PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
        "$repo/.venv/bin/python" -m en_pilot.enhance \
        --backend "$backend" \
        --data-root "$DATA_ROOT" \
        --workers "$ENHANCE_WORKERS" \
        --backend-repo-root "$repo"
}

enhance_with none "$WORKTREES/eval-compare"
enhance_with dpdfnet2-onnx "$WORKTREES/eval-dpdfnet"
enhance_with sherpa-gtcrn-simple "$WORKTREES/eval-sherpa-onnx"

phase "transcribe (primary tier: small)"
"$DRIVER_PYTHON" -m en_pilot.transcribe \
    --decoder small --data-root "$DATA_ROOT" --streams "$TRANSCRIBE_STREAMS"

phase "transcribe (continuity tier: base)"
"$DRIVER_PYTHON" -m en_pilot.transcribe \
    --decoder base --data-root "$DATA_ROOT" --streams "$TRANSCRIBE_STREAMS"

phase "score and decide"
"$DRIVER_PYTHON" -m en_pilot.score --data-root "$DATA_ROOT"

phase "decision table"
cat "$DATA_ROOT/en-pilot-run/results/decision.csv"

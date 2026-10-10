#!/usr/bin/env bash
# Run the English-only denoising WER pilot end to end, from the repository root.
#
# One process per backend. Transcription and scoring need faster-whisper and pyarrow, which
# only the driver venv has; set ENHANCE_PYTHON to run enhancement under another interpreter.
set -euo pipefail

cd "$(dirname "$0")/../.."

DRIVER_PYTHON="${DRIVER_PYTHON:-.venv/bin/python}"
DATA_ROOT="${MEDIBYTES_EVAL_DATA_ROOT:-$HOME/.local/share/medibytes-eval}"
#
# `build_corpus` and `build_matrix` fail closed if their output already exists,
# so a re-run needs `rm -rf $DATA_ROOT/corpus/en-40-v1 $DATA_ROOT/matrix-en-40`.
# The expensive phase, transcription, is resume-safe and skips clips already
# recorded in `transcripts-<decoder>.jsonl`.
ENHANCE_PYTHON="${ENHANCE_PYTHON:-$DRIVER_PYTHON}"
ENHANCE_WORKERS="${ENHANCE_WORKERS:-4}"
# `eval.en_pilot.transcribe` verifies the CTranslate2 runtime before committing to a
# device and falls back to cpu/int8 when it cannot load. On this 16-core host the
# measured CPU optimum is 2 streams x 8 threads (RTF 0.32 for `base`, 0.93 for
# `small`); on a working CUDA runtime, raise this toward 4-6.
TRANSCRIBE_STREAMS="${TRANSCRIBE_STREAMS:-2}"

phase() { printf '\n=== %s ===\n' "$1"; }

phase "build corpus (40 English bases)"
"$DRIVER_PYTHON" -m eval.en_pilot.build_corpus --data-root "$DATA_ROOT"

phase "build noise matrix (40 bases x 18 DEMAND noises x 2 SNRs)"
"$DRIVER_PYTHON" -m eval.en_pilot.build_matrix --data-root "$DATA_ROOT"

# The `none` control and each candidate, one process each.
enhance_with() {
    printf '\n=== enhance %s ===\n' "$1"
    "$ENHANCE_PYTHON" -m eval.en_pilot.enhance \
        --backend "$1" \
        --data-root "$DATA_ROOT" \
        --workers "$ENHANCE_WORKERS"
}

enhance_with none
enhance_with dpdfnet2-onnx
enhance_with sherpa-gtcrn-simple

phase "transcribe (primary tier: small)"
"$DRIVER_PYTHON" -m eval.en_pilot.transcribe \
    --decoder small --data-root "$DATA_ROOT" --streams "$TRANSCRIBE_STREAMS"

phase "transcribe (continuity tier: base)"
"$DRIVER_PYTHON" -m eval.en_pilot.transcribe \
    --decoder base --data-root "$DATA_ROOT" --streams "$TRANSCRIBE_STREAMS"

phase "score and decide"
"$DRIVER_PYTHON" -m eval.en_pilot.score --data-root "$DATA_ROOT"

phase "decision table"
cat "$DATA_ROOT/en-pilot-run/results/decision.csv"

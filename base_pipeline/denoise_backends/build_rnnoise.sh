#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${MEDIBYTES_EVAL_DATA_ROOT:-$HOME/.local/share/medibytes-eval}"
# Pinned ISA baseline, never -march=native: the source and model digests below prove the inputs,
# not the compiled library, and a native build silently ties the arm to the machine that built it.
# The library dispatches to AVX2/FMA at run time (--enable-x86-rtcd), so the baseline costs little.
MARCH="-march=x86-64-v2"
METHOD_ROOT="$DATA_ROOT/methods/rnnoise/v0.2"
SOURCE_DIR="$METHOD_ROOT/source"
MODEL_DIR="$DATA_ROOT/models/rnnoise/v0.2-0b50c45"
MODEL_ARCHIVE="$MODEL_DIR/rnnoise_data-0b50c45.tar.gz"
WRAPPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NATIVE_OUTPUT="$METHOD_ROOT/lib/librnnoise_medibytes.so"
EXPECTED_COMMIT="904a876dce1f9ab8860c0a5000ed151f9f6eef58"
EXPECTED_MODEL="0b50c45"
EXPECTED_ARCHIVE_SHA256="4ac81c5c0884ec4bd5907026aaae16209b7b76cd9d7f71af582094a2f98f4b43"
EXPECTED_RNN_DATA_SHA256="522b6a64fded05bf85e58c06206eafe57ce7d94f3af58c725b17628b481d7890"

require_file() {
    test -f "$1" || { printf 'missing required file: %s\n' "$1" >&2; exit 1; }
}

require_file "$SOURCE_DIR/model_version"
require_file "$SOURCE_DIR/include/rnnoise.h"
require_file "$MODEL_ARCHIVE"
test "$(tr -d '[:space:]' < "$SOURCE_DIR/model_version")" = "$EXPECTED_MODEL"
test "$(git -C "$SOURCE_DIR" rev-parse HEAD)" = "$EXPECTED_COMMIT"
echo "$EXPECTED_ARCHIVE_SHA256  $MODEL_ARCHIVE" | sha256sum --check --status

if ! test -f "$SOURCE_DIR/src/rnnoise_data.c"; then
    cp "$MODEL_ARCHIVE" "$SOURCE_DIR/rnnoise_data-$EXPECTED_MODEL.tar.gz"
    (cd "$SOURCE_DIR" && ./autogen.sh)
fi
echo "$EXPECTED_RNN_DATA_SHA256  $SOURCE_DIR/src/rnnoise_data.c" | sha256sum --check --status

if ! test -f "$SOURCE_DIR/Makefile"; then
    (cd "$SOURCE_DIR" && ./configure --disable-examples --disable-doc --enable-x86-rtcd CFLAGS="-O3 $MARCH")
fi
if ! test -f "$SOURCE_DIR/.libs/librnnoise.a"; then
    (cd "$SOURCE_DIR" && make -j2)
fi

mkdir -p "$(dirname "$NATIVE_OUTPUT")"
gcc -O3 "$MARCH" -fPIC -shared \
    -I"$SOURCE_DIR/include" \
    "$WRAPPER_DIR/rnnoise_native.c" \
    "$SOURCE_DIR/.libs/librnnoise.a" \
    -Wl,--no-undefined \
    -lm \
    -o "$NATIVE_OUTPUT"

printf '%s\n' "$NATIVE_OUTPUT"

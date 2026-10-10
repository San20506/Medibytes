#!/usr/bin/env bash
# Stage the pinned DeepFilterNet3 artifacts and build the method-local environment.
#
# The pinned upstream stack is deliberately kept out of the shared evaluation
# environment: deepfilternet==0.5.6 requires numpy<2 and deepfilterlib==0.5.6
# publishes no CPython 3.12 wheel.  This script reproduces both the staged model
# and the method environment from literal pins only.
set -euo pipefail

BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$BACKEND_DIR/../.." && pwd)"
ENV_CONFIG="$BACKEND_DIR/deepfilternet3.env.json"
LOCK="$BACKEND_DIR/requirements-deepfilternet3.lock"
STAMP_NAME=".medibytes-deepfilternet3-stamp.json"

read -r VARIANT ARCHIVE_RELATIVE ARCHIVE_SHA256 EXTRACTED_RELATIVE CONFIG_SHA256 \
    CHECKPOINT_NAME CHECKPOINT_SHA256 SOURCE_URL \
    ENV_RELATIVE PYTHON_EXECUTABLE PYTHON_VERSION TORCH_THREADS \
    TORCH_INDEX <<<"$(python3 - "$ENV_CONFIG" <<'PY'
import json
import sys

config = json.loads(open(sys.argv[1], encoding="utf-8").read())
model = config["model"]
env = config["environment"]
print(
    config["variant_id"],
    model["archive_relative"],
    model["archive_sha256"],
    model["extracted_relative"],
    model["config_sha256"],
    model["checkpoint_name"],
    model["checkpoint_sha256"],
    model["source_url"],
    env["relative_path"],
    env["python_executable"],
    env["python_version"],
    env["torch_threads"],
    env["torch_index"],
)
PY
)"

DATA_ROOT="${MEDIBYTES_EVAL_DATA_ROOT:-$HOME/.local/share/medibytes-eval}"
ARCHIVE="$DATA_ROOT/$ARCHIVE_RELATIVE"
MODEL_DIR="$DATA_ROOT/$EXTRACTED_RELATIVE"
ENV_ROOT="$REPO_ROOT/$ENV_RELATIVE"
INTERPRETER="$ENV_ROOT/$PYTHON_EXECUTABLE"

require_file() {
    test -f "$1" || { printf 'missing required file: %s\n' "$1" >&2; exit 1; }
}

require_file "$LOCK"

# 1. Stage the pinned model archive from its literal upstream URL.
mkdir -p "$(dirname "$ARCHIVE")"
if ! test -f "$ARCHIVE"; then
    curl --fail --location --retry 3 --output "$ARCHIVE" "$SOURCE_URL"
fi
echo "$ARCHIVE_SHA256  $ARCHIVE" | sha256sum --check --status

# 2. Extract and verify the configuration and checkpoint digests.
if ! test -f "$MODEL_DIR/config.ini"; then
    rm -rf "$MODEL_DIR"
    mkdir -p "$MODEL_DIR"
    (cd "$MODEL_DIR" && python3 -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(".")' "$ARCHIVE")
    extracted="$MODEL_DIR/DeepFilterNet3"
    if test -d "$extracted" && test "$extracted" != "$MODEL_DIR"; then
        mv "$extracted/config.ini" "$MODEL_DIR/config.ini"
        mkdir -p "$MODEL_DIR/checkpoints"
        mv "$extracted/checkpoints/$CHECKPOINT_NAME" "$MODEL_DIR/checkpoints/$CHECKPOINT_NAME"
        rm -rf "$extracted"
    fi
fi
echo "$CONFIG_SHA256  $MODEL_DIR/config.ini" | sha256sum --check --status
echo "$CHECKPOINT_SHA256  $MODEL_DIR/checkpoints/$CHECKPOINT_NAME" | sha256sum --check --status

# 3. Build the method-local environment from the hashed lock.
if ! test -x "$INTERPRETER"; then
    rm -rf "$ENV_ROOT"
    (cd "$REPO_ROOT" && uv venv --python "$PYTHON_VERSION" "$ENV_RELATIVE")
fi
VIRTUAL_ENV="$ENV_ROOT" uv pip install \
    --require-hashes \
    --index-url https://pypi.org/simple \
    --extra-index-url "$TORCH_INDEX" \
    --index-strategy unsafe-best-match \
    --python "$INTERPRETER" \
    --requirement "$LOCK"

# 4. Verify the resolved runtime before stamping the environment.
"$INTERPRETER" - "$ENV_CONFIG" <<'PY'
import importlib.metadata
import json
import platform
import sys

config = json.loads(open(sys.argv[1], encoding="utf-8").read())
env = config["environment"]
if not platform.python_version().startswith(env["python_version"]):
    raise SystemExit(
        f"method environment requires Python {env['python_version']}, found {platform.python_version()}"
    )
for name, expected in sorted(env["packages"].items()):
    actual = importlib.metadata.version(name)
    if actual != expected:
        raise SystemExit(f"pinned {name}=={expected} required, environment has {actual}")
from df.enhance import enhance, init_df  # noqa: F401  import check
from libdf import DF  # noqa: F401  import check

print(f"method environment verified: deepfilternet {importlib.metadata.version('deepfilternet')}")
PY

# 5. Record the exact lock this environment was built from.
LOCK_SHA256="$(sha256sum "$LOCK" | cut -d' ' -f1)"
python3 - "$ENV_ROOT/$STAMP_NAME" <<PY
import json
import sys

stamp_path = sys.argv[1]
stamp = {
    "schema_version": "1.0.0",
    "variant_id": "$VARIANT",
    "lock": "base_pipeline/denoise_backends/requirements-deepfilternet3.lock",
    "lock_sha256": "$LOCK_SHA256",
    "python_version": "$PYTHON_VERSION",
    "torch_threads": $TORCH_THREADS,
}
with open(stamp_path, "w", encoding="utf-8") as handle:
    json.dump(stamp, handle, indent=2, sort_keys=True)
print(stamp_path)
PY

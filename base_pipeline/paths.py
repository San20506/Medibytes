"""Where machine-local data lives: models, corpora and native libraries built outside the repo."""
import os
from pathlib import Path

ENV = "MEDIBYTES_EVAL_DATA_ROOT"


def data_root():
    """`$MEDIBYTES_EVAL_DATA_ROOT` if set, else `~/.local/share/medibytes-eval`."""
    raw = os.environ.get(ENV, "").strip()
    return Path(raw).expanduser() if raw else Path.home() / ".local" / "share" / "medibytes-eval"

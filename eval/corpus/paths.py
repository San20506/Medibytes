"""Canonical data-root validation for corpus commands."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATA_ROOT = Path("/home/sandy/.local/share/medibytes-eval")
ENVIRONMENT_NAME = "MEDIBYTES_EVAL_DATA_ROOT"


def resolve_data_root(value: str | os.PathLike[str] | None = None) -> Path:
    """Return an absolute, nonempty evaluation data root or fail closed."""

    raw = os.environ.get(ENVIRONMENT_NAME) if value is None else os.fspath(value)
    if raw is None or not raw.strip():
        raise ValueError(
            f"{ENVIRONMENT_NAME} is required and must be a nonempty absolute path"
        )
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{ENVIRONMENT_NAME} must be absolute: {raw!r}")
    resolved = path.resolve(strict=False)
    if resolved == Path(resolved.anchor):
        raise ValueError(f"{ENVIRONMENT_NAME} must not be the filesystem root")
    return resolved

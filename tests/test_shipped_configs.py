"""Each shipped adapter config must agree with the frozen method config the eval harness runs.

`run.py`, the server and the app select an arm from `base_pipeline/denoise_backends/<id>.config.json`;
the harness selects it from `eval/config/methods/<id>.json`. Two sources of the same identity can drift,
and an arm whose variant or parameters drift is refused by its own adapter, so pin them to each other.
"""
import json
from pathlib import Path

import pytest

from base_pipeline.denoise import BackendId
from base_pipeline.run import denoiser_config

ROOT = Path(__file__).resolve().parent.parent
SHIPPED = sorted((ROOT / "base_pipeline" / "denoise_backends").glob("*.config.json"))


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.name)
def test_shipped_config_matches_the_frozen_method_config(path):
    shipped = _load(path)
    method = _load(ROOT / "eval" / "config" / "methods" / f"{shipped['backend']}.json")
    for field in ("variant_id", "parameters"):
        assert shipped[field] == method[field], field
    assert shipped.get("model_path") == method.get("model_path")
    # The method config may leave the digest null for an adapter that pins it; it may never disagree.
    if method["model_sha256"] is not None:
        assert shipped["model_sha256"] == method["model_sha256"]
    if "timeout_s" in shipped:
        assert shipped["timeout_s"] == method["timeout_s"]


def test_every_arm_with_an_adapter_can_be_selected_from_the_cli_config():
    """No ported arm may be reachable only through the eval harness."""
    adapters = {p.stem for p in (ROOT / "base_pipeline" / "denoise_backends").glob("*.py")} - {"__init__"}
    for backend in BackendId:
        if backend.value.replace("-", "_") not in adapters or backend in (BackendId.NONE, BackendId.NOISEREDUCE):
            continue
        config = denoiser_config(backend)
        method = _load(ROOT / "eval" / "config" / "methods" / f"{backend.value}.json")
        assert config.variant_id == method["variant_id"], backend.value

"""The pilot's backends live in base_pipeline, and its provenance hash covers them."""
import importlib

import pytest

from base_pipeline.denoise import BackendId
from eval.en_pilot import config, enhance, score


def _module(backend):
    return backend.replace("-", "_")


@pytest.mark.parametrize("backend", config.BACKENDS)
def test_every_pilot_backend_is_registered_and_has_an_in_tree_adapter(backend):
    assert backend in {b.value for b in BackendId}
    importlib.import_module(f"base_pipeline.denoise_backends.{_module(backend)}")


def test_candidates_are_the_pilot_backends_minus_the_baseline():
    assert set(config.CANDIDATE_BACKENDS) == set(config.BACKENDS) - {config.BASELINE_BACKEND}


def test_enhance_refuses_a_backend_the_study_does_not_run():
    with pytest.raises(enhance.EnhanceError, match="rnnoise"):
        enhance.check_backend("rnnoise")
    enhance.check_backend("none")


def test_code_state_hashes_each_adapter_and_every_downstream_module():
    state = score._code_state()
    for backend in config.BACKENDS:
        assert f"{backend}:base_pipeline/denoise_backends/{_module(backend)}.py" in state
        assert f"{backend}:eval/config/methods/{backend}.json" in state
    # Extraction was split across three modules; the hash must follow it.
    for module in ("asr", "extract", "stt_extract"):
        assert f"base_pipeline/{module}.py" in state

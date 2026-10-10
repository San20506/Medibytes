"""Skip an arm's tests where the arm cannot run, instead of failing them.

An arm needs a runtime and artifacts (a package, a model, a built library) that live outside the
repository, so a fresh clone cannot have them. "Can run here" is `run.arm_status`: the adapter's own
probe under its shipped config. On a machine that does have an arm, set MEDIBYTES_REQUIRE_ARMS to a
comma list of backend ids (or `all`) and an unavailable arm fails its tests, so a broken port cannot
hide behind a skip.
"""
import functools
import os

import pytest

from base_pipeline.run import arm_status


@functools.lru_cache(maxsize=None)
def _status(backend):
    return arm_status(backend)


def _required():
    return {item.strip() for item in os.environ.get("MEDIBYTES_REQUIRE_ARMS", "").split(",") if item.strip()}


def require_arm(backend):
    """An autouse fixture for a test module: `arm_available = require_arm(BackendId.X)`."""

    @pytest.fixture(autouse=True, name="arm_available")
    def arm_available():
        reason = _status(backend)
        if reason is None:
            return
        if "all" in _required() or backend.value in _required():
            pytest.fail(f"{backend.value} is required (MEDIBYTES_REQUIRE_ARMS) but unavailable: {reason}")
        pytest.skip(f"{backend.value} cannot run here: {reason}")

    return arm_available

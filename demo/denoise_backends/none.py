"""The honest no-op enhancement control."""
from __future__ import annotations

import numpy as np

from demo.denoise import BackendId, BackendProbe, EnhancementConfig, EnhancementOutput


BACKEND = BackendId.NONE


def probe(config: EnhancementConfig) -> BackendProbe:
    return BackendProbe(
        available=True,
        reason="baseline identity control",
        source_revision="b0",
        runtime_version="stdlib",
        model_sha256=None,
        native_sample_rate=16000,
        delay_samples=0,
        stateful=False,
    )


def enhance(audio: np.ndarray, sample_rate: int, config: EnhancementConfig) -> EnhancementOutput:
    if sample_rate != 16000:
        raise ValueError("none backend requires the shared 16 kHz boundary")
    return EnhancementOutput(
        audio=np.asarray(audio, dtype=np.float32).copy(),
        actual_backend=BackendId.NONE,
        variant_id=config.variant_id,
        native_sample_rate=16000,
        delay_samples=0,
        metadata={"changed_audio": False, "parameters": dict(config.parameters)},
        provenance={
            "requested_backend": BackendId.NONE.value,
            "actual_backend": BackendId.NONE.value,
            "backend": BackendId.NONE.value,
            "variant_id": config.variant_id,
            "model_sha256": None,
            "runtime_version": "stdlib",
            "source_revision": "b0",
            "device": "cpu",
            "provider": "MediBytes shared baseline",
            "parameters": dict(config.parameters),
            "fallback_used": False,
        },

    )

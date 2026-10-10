"""Pinned SpeechBrain 1.1.1 MetricGAN+ voicebank model, single backend only."""
from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from ..denoise import (
    BackendId,
    BackendProbe,
    BackendUnavailableError,
    EnhancementConfig,
    EnhancementOutput,
)
from ..paths import data_root

BACKEND = BackendId.SPEECHBRAIN_METRICGAN_PLUS
VARIANT = "metricgan-plus-voicebank-1.1.1"
SOURCE_REVISION = "89ead74d163463d30c62329a09cfdb4c54f5abc1"
MODEL_REVISION = "a196ce26b3bdace6fa1d819017584bdbcce462a8"
MODEL_SHA256 = "147bfb866bac8264603546e035bf283370e716ed2f4b7412d308d2bcee88304f"
HYPERPARAMS_SHA256 = "7bcc0835243c27f78907de0df27bc5f33d533a675787ab2eff34188404ea225a"
MODEL_DIR = data_root() / "models/speechbrain/metricgan-plus-voicebank"
CHECKPOINT = MODEL_DIR / "enhance_model.ckpt"
HYPERPARAMS = MODEL_DIR / "hyperparams.yaml"
NATIVE_RATE = 16_000
PARAMETERS = {"sample_rate": 16000, "device": "cpu", "model": "speechbrain/metricgan-plus-voicebank"}


def _digest(path: Path) -> str:
    accumulator = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            accumulator.update(chunk)
    return accumulator.hexdigest()


def _checked_inputs(config: EnhancementConfig) -> None:
    if config.backend != BACKEND or config.variant_id != VARIANT:
        raise ValueError("MetricGAN+ requires the exact requested backend and variant")
    if dict(config.parameters) != PARAMETERS:
        raise ValueError("MetricGAN+ parameters differ from the frozen method config")
    if config.model_path != CHECKPOINT or CHECKPOINT.is_symlink():
        raise ValueError("MetricGAN+ requires the exact local non-symlink checkpoint path")
    # The frozen catalog leaves model_sha256 null for this row, so the adapter's
    # own pinned digest is the binding one; a supplied value must still match it.
    if config.model_sha256 not in (None, MODEL_SHA256):
        raise ValueError("MetricGAN+ model_sha256 must be null or the pinned digest")
    for path, expected, label in (
        (CHECKPOINT, MODEL_SHA256, "enhance_model.ckpt"),
        (HYPERPARAMS, HYPERPARAMS_SHA256, "hyperparams.yaml"),
    ):
        if not path.is_file():
            raise BackendUnavailableError(f"BACKEND_UNAVAILABLE: pinned {label} is missing")
        actual = _digest(path)
        if actual != expected:
            raise BackendUnavailableError(
                f"BACKEND_UNAVAILABLE: pinned {label} digest {actual} != pinned {expected}"
            )


def _runtime_version() -> str:
    try:
        actual = version("speechbrain")
    except PackageNotFoundError as error:
        raise BackendUnavailableError("BACKEND_UNAVAILABLE: speechbrain is not installed") from error
    if actual != "1.1.1":
        raise BackendUnavailableError(f"BACKEND_UNAVAILABLE: speechbrain==1.1.1 required, got {actual}")
    return actual


def _load_model():
    """Build a fresh SpectralMaskEnhancement from the staged, pinned local files."""
    from speechbrain.inference.enhancement import SpectralMaskEnhancement
    from speechbrain.utils.fetching import FetchConfig

    model = SpectralMaskEnhancement.from_hparams(
        source=str(MODEL_DIR),
        savedir=None,
        run_opts={"device": "cpu"},
        fetch_config=FetchConfig(allow_network=False),
    )
    if not isinstance(model, SpectralMaskEnhancement):
        raise BackendUnavailableError("BACKEND_INVARIANT: checkpoint did not yield SpectralMaskEnhancement")
    if getattr(model, "sample_rate", NATIVE_RATE) != NATIVE_RATE:
        raise BackendUnavailableError("BACKEND_UNAVAILABLE: checkpoint is not 16 kHz")
    return model


def probe(config: EnhancementConfig) -> BackendProbe:
    _checked_inputs(config)
    runtime = _runtime_version()
    try:
        _load_model()
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        return BackendProbe(
            available=False,
            reason=f"BACKEND_UNAVAILABLE: {error}",
            source_revision=SOURCE_REVISION,
            runtime_version=runtime,
            model_sha256=MODEL_SHA256,
            native_sample_rate=NATIVE_RATE,
            stateful=True,
        )
    return BackendProbe(
        available=True,
        reason="pinned MetricGAN+ voicebank checkpoint and hyperparams verified offline",
        source_revision=SOURCE_REVISION,
        runtime_version=runtime,
        model_sha256=MODEL_SHA256,
        native_sample_rate=NATIVE_RATE,
        stateful=True,
    )


def enhance(audio: np.ndarray, sample_rate: int, config: EnhancementConfig) -> EnhancementOutput:
    _checked_inputs(config)
    runtime = _runtime_version()
    if sample_rate != NATIVE_RATE:
        raise ValueError("MetricGAN+ requires mono 16 kHz input")
    source = np.asarray(audio, dtype=np.float32)
    if source.ndim != 1 or source.size == 0 or not np.isfinite(source).all():
        raise ValueError("MetricGAN+ requires finite nonempty mono input")

    import torch

    # A new model per sample: no SepFormer or SEGAN pooling, no shared state.
    model = _load_model()
    signal = torch.as_tensor(source, dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        enhanced = model.enhance_batch(signal, lengths=torch.tensor([1.0]))
    tensor = enhanced[0]
    if isinstance(tensor, (tuple, list)):
        tensor = tensor[0]
    output = tensor.detach().to("cpu").numpy().astype(np.float32, copy=False).reshape(-1)
    if output.size != source.size:
        raise ValueError(
            f"MetricGAN+ returned {output.size} samples for {source.size} input samples"
        )
    if not np.isfinite(output).all() or not np.any(output):
        raise ValueError("MetricGAN+ returned nonfinite or silent audio")

    provenance = {
        "requested_backend": BACKEND.value,
        "actual_backend": BACKEND.value,
        "backend": BACKEND.value,
        "variant_id": VARIANT,
        "fallback_used": False,
        "source_revision": SOURCE_REVISION,
        "runtime_version": runtime,
        "torch_version": torch.__version__,
        "device": "cpu",
        "provider": "torch",
        "model_sha256": MODEL_SHA256,
        "model_revision": MODEL_REVISION,
        "hyperparams_sha256": HYPERPARAMS_SHA256,
        "license": "Apache-2.0",
        "architecture": "SpectralMaskEnhancement",
        "network_access": "disabled",
        "state_reset_per_call": True,
        "parameters": dict(config.parameters),
    }
    return EnhancementOutput(
        audio=np.ascontiguousarray(output, dtype=np.float32),
        actual_backend=BACKEND,
        variant_id=VARIANT,
        native_sample_rate=NATIVE_RATE,
        delay_samples=0,
        metadata={"state_per_call": True, "model_per_call": 1, "model_revision": MODEL_REVISION},
        provenance=provenance,
    )

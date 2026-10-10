"""Pinned NVIDIA NeMo Schroedinger-bridge SE denoising model, single backend only.

The published checkpoint's embedded configuration targets its two training
losses at ``nemo.collections.audio.losses.MSELoss`` / ``.MAELoss``.  NeMo
2.7.3 moved both classes to the ``nemo.collections.audio.losses.audio``
module, so an unmodified restore fails before any weights load.  This adapter
rebuilds the embedded configuration in memory and rewrites exactly those two
target strings; the archive is never modified, weights load with
``strict=True``, and both the embedded-config digest and the checkpoint
digest are recorded in provenance.  This is a load-compatibility shim, not a
model substitution: encoder, estimator, noise schedule, sampler and every
weight tensor come from the pinned checkpoint.
"""
from __future__ import annotations

import hashlib
import io
import tarfile
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

BACKEND = BackendId.NEMO_SE_DEN_SB_16K
VARIANT = "se-den-sb-16k-small-2.7.3"
SOURCE_REVISION = "1d4ee423806d461f9146ae982f9da8eb32495ae7"
MODEL_REVISION = "fe9725b1ae90f7f6b11d9b95c5f9ccec736a211f"
MODEL_SHA256 = "9de7f36d08df4109402e68e41d78e781c8da6a3d7691a674c5231b840151ff09"
EMBEDDED_CONFIG_SHA256 = "a0a84565756dee2254fa05b5f2ca6c1726d8c9872ea550817a0554f829094317"
MODEL_PATH = data_root() / "models/nemo/se_den_sb_16k_small.nemo"
PARAMETERS = {"sample_rate": 16000, "seed": 1729, "num_steps": 10, "state_per_sample": True}
SAMPLER_PROCESS = "ode"
NATIVE_RATE = 16_000
_LOSS_TARGETS = {
    "nemo.collections.audio.losses.MSELoss": "nemo.collections.audio.losses.audio.MSELoss",
    "nemo.collections.audio.losses.MAELoss": "nemo.collections.audio.losses.audio.MAELoss",
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _checked_model(config: EnhancementConfig) -> Path:
    if config.backend != BACKEND or config.variant_id != VARIANT:
        raise ValueError("NeMo requires the exact requested backend and variant")
    if dict(config.parameters) != PARAMETERS:
        raise ValueError("NeMo parameters differ from the frozen method config")
    if config.model_path != MODEL_PATH or MODEL_PATH.is_symlink():
        raise ValueError("NeMo requires the exact local non-symlink checkpoint path")
    if config.model_sha256 != MODEL_SHA256:
        raise ValueError("NeMo requires the pinned checkpoint SHA-256")
    digest = hashlib.sha256()
    try:
        with MODEL_PATH.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise BackendUnavailableError("BACKEND_UNAVAILABLE: pinned NeMo checkpoint is missing") from error
    if digest.hexdigest() != MODEL_SHA256:
        raise BackendUnavailableError("BACKEND_UNAVAILABLE: pinned NeMo checkpoint digest mismatch")
    return MODEL_PATH


def _embedded_config() -> bytes:
    try:
        with tarfile.open(MODEL_PATH) as archive:
            member = archive.extractfile("./model_config.yaml")
            if member is None:
                raise BackendUnavailableError("BACKEND_UNAVAILABLE: checkpoint has no model_config.yaml")
            payload = member.read()
    except tarfile.TarError as error:
        raise BackendUnavailableError("BACKEND_UNAVAILABLE: checkpoint is not a readable archive") from error
    if _sha256_bytes(payload) != EMBEDDED_CONFIG_SHA256:
        raise BackendUnavailableError("BACKEND_UNAVAILABLE: embedded config digest mismatch")
    return payload


def _runtime_versions() -> dict[str, str]:
    try:
        toolkit = version("nemo-toolkit")
    except PackageNotFoundError as error:
        raise BackendUnavailableError("BACKEND_UNAVAILABLE: nemo-toolkit is not installed") from error
    if toolkit != "2.7.3":
        raise BackendUnavailableError(f"BACKEND_UNAVAILABLE: nemo-toolkit==2.7.3 required, got {toolkit}")
    return {"nemo_toolkit": toolkit, "torch": version("torch")}


def _embedded_config() -> str:
    try:
        with tarfile.open(MODEL_PATH) as archive:
            member = archive.extractfile("./model_config.yaml")
            if member is None:
                raise BackendUnavailableError("BACKEND_UNAVAILABLE: checkpoint has no model_config.yaml")
            payload = member.read()
    except tarfile.TarError as error:
        raise BackendUnavailableError("BACKEND_UNAVAILABLE: checkpoint is not a readable archive") from error
    if _sha256_bytes(payload) != EMBEDDED_CONFIG_SHA256:
        raise BackendUnavailableError("BACKEND_UNAVAILABLE: embedded config digest mismatch")
    return payload.decode("utf-8")


def _load_model():
    """Restore the pinned checkpoint with the two loss targets remapped."""
    import torch
    from omegaconf import OmegaConf

    from nemo.collections.audio.models.enhancement import SchroedingerBridgeAudioToAudioModel

    config = OmegaConf.create(_embedded_config())
    remapped = _rewrite_loss_targets(config)
    if remapped != len(_LOSS_TARGETS):
        raise BackendUnavailableError("BACKEND_UNAVAILABLE: no compatible loss targets found to remap")
    model = SchroedingerBridgeAudioToAudioModel.restore_from(
        restore_path=str(MODEL_PATH),
        override_config_path=config,
        map_location="cpu",
        strict=True,
    )
    if int(model.sample_rate) != NATIVE_RATE:
        raise BackendUnavailableError("BACKEND_UNAVAILABLE: checkpoint is not 16 kHz")
    if type(model.sampler).__name__ != "SBSampler":
        raise BackendUnavailableError("BACKEND_UNAVAILABLE: checkpoint is not a Schroedinger-bridge sampler")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    model.sampler.process = SAMPLER_PROCESS
    model.sampler.num_steps = int(PARAMETERS["num_steps"])
    if model.sampler.process != SAMPLER_PROCESS or int(model.sampler.num_steps) != PARAMETERS["num_steps"]:
        raise BackendUnavailableError("BACKEND_UNAVAILABLE: sampler did not accept the pinned ODE configuration")
    return model, device, remapped


def _rewrite_loss_targets(node) -> int:
    import omegaconf

    remapped = 0
    if isinstance(node, omegaconf.DictConfig):
        target = node.get("_target_")
        if isinstance(target, str) and target in _LOSS_TARGETS:
            node["_target_"] = _LOSS_TARGETS[target]
            remapped += 1
        for child in node.values():
            remapped += _rewrite_loss_targets(child)
    elif isinstance(node, (list, tuple)):
        for child in node:
            remapped += _rewrite_loss_targets(child)
    return remapped


def probe(config: EnhancementConfig) -> BackendProbe:
    _checked_model(config)
    versions = _runtime_versions()
    try:
        import torch

        device_kind = "cuda" if torch.cuda.is_available() else "cpu"
    except (ImportError, RuntimeError) as error:
        return BackendProbe(
            available=False,
            reason=f"BACKEND_UNAVAILABLE: {error}",
            source_revision=SOURCE_REVISION,
            runtime_version=versions["nemo_toolkit"],
            model_sha256=MODEL_SHA256,
            native_sample_rate=NATIVE_RATE,
            stateful=True,
        )
    return BackendProbe(
        available=True,
        reason="pinned research-only NeMo checkpoint and ODE sampler configuration verified",
        source_revision=SOURCE_REVISION,
        runtime_version=versions["nemo_toolkit"],
        model_sha256=MODEL_SHA256,
        native_sample_rate=NATIVE_RATE,
        stateful=True,
    )


def enhance(audio: np.ndarray, sample_rate: int, config: EnhancementConfig) -> EnhancementOutput:
    _checked_model(config)
    versions = _runtime_versions()
    if sample_rate != NATIVE_RATE:
        raise ValueError("NeMo adapter requires mono 16 kHz input")
    source = np.asarray(audio, dtype=np.float32)
    if source.ndim != 1 or source.size == 0 or not np.isfinite(source).all():
        raise ValueError("NeMo requires finite nonempty mono input")

    import torch

    # One fresh restore per sample: the sampler carries no cross-sample state.
    model, device, remapped = _load_model()
    torch.manual_seed(int(PARAMETERS["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(PARAMETERS["seed"]))

    signal = torch.as_tensor(source, dtype=torch.float32, device=device).reshape(1, 1, -1)
    length = torch.tensor([signal.shape[-1]], device=device)
    with torch.inference_mode():
        result, _ = model(input_signal=signal, input_length=length)
    output = result.detach().squeeze().to("cpu").numpy().astype(np.float32, copy=False)
    if output.ndim != 1 or output.size != source.size or not np.isfinite(output).all():
        raise ValueError("NeMo returned audio with a different length, rank or non-finite samples")

    provenance = {
        "requested_backend": BACKEND.value,
        "actual_backend": BACKEND.value,
        "backend": BACKEND.value,
        "variant_id": VARIANT,
        "fallback_used": False,
        "source_revision": SOURCE_REVISION,
        "runtime_version": versions["nemo_toolkit"],
        "torch_version": versions["torch"],
        "cuda_version": torch.version.cuda or "unavailable",
        "cudnn_version": torch.backends.cudnn.version() or "unavailable",
        "device": str(device),
        "provider": "torch",
        "model_sha256": MODEL_SHA256,
        "model_revision": MODEL_REVISION,
        "model_license": "CC-BY-NC-SA-4.0 research-only",
        "sampler_process": SAMPLER_PROCESS,
        "sampler_num_steps": int(PARAMETERS["num_steps"]),
        "seed": int(PARAMETERS["seed"]),
        "state_reset_per_call": True,
        "parameters": dict(config.parameters),
    }
    return EnhancementOutput(
        audio=np.ascontiguousarray(output, dtype=np.float32),
        actual_backend=BACKEND,
        variant_id=VARIANT,
        native_sample_rate=NATIVE_RATE,
        delay_samples=0,
        metadata={
            "state_per_call": True,
            "loss_target_remaps": remapped,
            "loss_target_remap_note": (
                "checkpoint config targets remapped in memory to the NeMo 2.7.3 "
                "module path; archive unmodified and weights loaded with strict=True"
            ),
            "embedded_config_sha256": EMBEDDED_CONFIG_SHA256,
            "model_revision": MODEL_REVISION,
        },
        provenance=provenance,
    )

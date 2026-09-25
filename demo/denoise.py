"""Shared single-backend enhancement contract.

The product demo intentionally keeps this module dependency-free apart from
NumPy.  Method adapters live beside it and are imported by their explicit
backend id; an unavailable or invalid adapter is always a hard failure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import importlib
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias, Union
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np


JSONScalar: TypeAlias = Union[None, bool, int, float, str]
JSONValue: TypeAlias = Union[
    JSONScalar,
    list["JSONValue"],
    dict[str, "JSONValue"],
]


class BackendId(StrEnum):
    """The complete, explicit catalog of shared enhancement identities."""

    NONE = "none"
    NOISEREDUCE = "noisereduce"
    RNNOISE = "rnnoise"
    DEEPFILTERNET3 = "deepfilternet3"
    DPDFNET2_ONNX = "dpdfnet2-onnx"
    SHERPA_GTCRN_SIMPLE = "sherpa-gtcrn-simple"
    PERCEPNET = "percepnet"
    NEMO_SE_DEN_SB_16K = "nemo-se-den-sb-16k"
    SPEECHT5 = "speecht5"
    SPEEX_DENOISE = "speex-denoise"
    SPEECHBRAIN_METRICGAN_PLUS = "speechbrain-metricgan-plus"
    VOICEFIXER_MODE0 = "voicefixer-mode0"
    CMSIS_DSP = "cmsis-dsp"
    SPECTRAL_SUBTRACTION = "spectral-subtraction"


@dataclass(frozen=True)
class EnhancementConfig:
    """Immutable request for exactly one enhancement backend."""

    backend: BackendId
    variant_id: str = "default"
    model_path: Path | None = None
    model_sha256: str | None = None
    parameters: Mapping[str, JSONValue] = field(default_factory=dict)
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        backend = self.backend if isinstance(self.backend, BackendId) else BackendId(self.backend)
        object.__setattr__(self, "backend", backend)
        if not isinstance(self.variant_id, str) or not self.variant_id.strip():
            raise ValueError("variant_id must be a non-empty string")
        if self.model_path is not None and not isinstance(self.model_path, Path):
            object.__setattr__(self, "model_path", Path(self.model_path))
        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        # Keep the top-level request immutable without rejecting ordinary JSON
        # mappings supplied by callers.
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        if not math.isfinite(float(self.timeout_s)) or self.timeout_s <= 0:
            raise ValueError("timeout_s must be a positive finite number")


@dataclass(frozen=True)
class BackendProbe:
    """Adapter availability and immutable execution metadata."""

    available: bool
    reason: str = ""
    source_revision: str | None = None
    runtime_version: str | None = None
    model_sha256: str | None = None
    native_sample_rate: int = 16000
    delay_samples: int = 0
    stateful: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.native_sample_rate, int) or self.native_sample_rate <= 0:
            raise ValueError("native_sample_rate must be a positive integer")
        if not isinstance(self.delay_samples, int) or self.delay_samples < 0:
            raise ValueError("delay_samples must be a non-negative integer")


@dataclass(frozen=True)
class EnhancementOutput:
    """One adapter's native-rate result before shared delay/rate reconciliation."""

    audio: np.ndarray
    actual_backend: BackendId
    variant_id: str
    native_sample_rate: int
    delay_samples: int
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
    provenance: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        backend = self.actual_backend if isinstance(self.actual_backend, BackendId) else BackendId(self.actual_backend)
        object.__setattr__(self, "actual_backend", backend)
        if not isinstance(self.variant_id, str) or not self.variant_id:
            raise ValueError("variant_id must be a non-empty string")
        if not isinstance(self.native_sample_rate, int) or self.native_sample_rate <= 0:
            raise ValueError("native_sample_rate must be a positive integer")
        if not isinstance(self.delay_samples, int) or self.delay_samples < 0:
            raise ValueError("delay_samples must be a non-negative integer")
        if not isinstance(self.metadata, Mapping) or not isinstance(self.provenance, Mapping):
            raise TypeError("metadata and provenance must be mappings")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


class BackendError(ValueError):
    """Base class for fail-closed backend errors."""


class BackendUnavailableError(BackendError):
    """Raised when the requested adapter cannot be used."""


class BackendContractError(BackendError):
    """Raised when an adapter violates the shared output contract."""


def _module_name(backend: BackendId) -> str:
    return backend.value.replace("-", "_")


def _check_audio_array(audio: Any, label: str) -> np.ndarray:
    if not isinstance(audio, np.ndarray):
        raise BackendContractError(f"{label} audio must be a numpy.ndarray")
    if audio.ndim != 1:
        raise BackendContractError(f"{label} audio must be mono (one dimension)")
    if audio.size == 0:
        raise BackendContractError(f"{label} audio is empty")
    if not np.issubdtype(audio.dtype, np.number) or np.issubdtype(audio.dtype, np.complexfloating):
        raise BackendContractError(f"{label} audio must have a real numeric dtype")
    if not bool(np.isfinite(audio).all()):
        raise BackendContractError(f"{label} audio contains non-finite samples")
    if not bool(np.any(audio != 0)):
        raise BackendContractError(f"{label} audio is all-zero")
    return audio


def _check_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BackendContractError(f"{label} must be a positive integer")
    return value


def _mapping_values(*values: Mapping[str, Any]) -> None:
    """Reject explicit fallback/composition markers at any metadata depth."""
    composition_keys = {
        "backends",
        "actual_backends",
        "backend_chain",
        "composed_backends",
        "multiple_backends",
        "composition",
        "composed",
        "cascade",
        "cascades",
        "layer",
        "layers",
        "layered",
        "sequential",
    }
    fallback_keys = {
        "fallback",
        "fallbacks",
        "used_fallback",
        "fallback_backend",
        "fallback_used",
    }

    def inspect(mapping: Mapping[str, Any]) -> None:
        for raw_key, value in mapping.items():
            key = str(raw_key).casefold()
            if key in fallback_keys and value not in (None, False, "", [], {}, ()):
                raise BackendContractError(f"adapter reported fallback via {raw_key}")
            if key in composition_keys and value not in (None, False, "", [], {}, ()):
                raise BackendContractError("adapter reported a backend composition")
            if key in {"actual_backend", "requested_backend"} and value is not None:
                if not isinstance(value, (str, BackendId)):
                    raise BackendContractError(f"adapter metadata {raw_key} is invalid")
            if isinstance(value, Mapping):
                inspect(value)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    if isinstance(child, Mapping):
                        inspect(child)

    for value in values:
        inspect(value)


def _validate_probe(probe: Any, config: EnhancementConfig) -> BackendProbe:
    if not isinstance(probe, BackendProbe):
        raise BackendContractError("adapter probe must return BackendProbe")
    if type(probe.available) is not bool:
        raise BackendContractError("probe.available must be a boolean")
    if not probe.available:
        reason = probe.reason or "adapter reported unavailable"
        if not reason.startswith("BACKEND_UNAVAILABLE:"):
            reason = f"BACKEND_UNAVAILABLE: {reason}"
        raise BackendUnavailableError(reason)
    if not isinstance(probe.reason, str) or not probe.reason:
        raise BackendContractError("available adapter probe must include a reason")
    if not isinstance(probe.source_revision, str) or not probe.source_revision:
        raise BackendContractError("available adapter probe must include source_revision")
    if not isinstance(probe.runtime_version, str) or not probe.runtime_version:
        raise BackendContractError("available adapter probe must include runtime_version")
    if type(probe.stateful) is not bool:
        raise BackendContractError("probe.stateful must be a boolean")
    native_rate = _check_positive_int(probe.native_sample_rate, "probe.native_sample_rate")
    if not isinstance(probe.delay_samples, int) or probe.delay_samples < 0:
        raise BackendContractError("probe.delay_samples must be non-negative")
    if config.model_sha256 is not None and probe.model_sha256 != config.model_sha256:
        raise BackendContractError("configured model hash differs from adapter probe")
    if probe.model_sha256 is not None and re.fullmatch(r"[0-9a-f]{64}", probe.model_sha256) is None:
        raise BackendContractError("probe model_sha256 must be null or a SHA-256")
    if native_rate <= 0:  # pragma: no cover - guarded above
        raise BackendContractError("invalid native sample rate")
    return probe


def _validate_output_provenance(
    provenance: Mapping[str, Any], config: EnhancementConfig, backend: BackendId
) -> None:
    required = {
        "actual_backend",
        "backend",
        "device",
        "fallback_used",
        "model_sha256",
        "parameters",
        "provider",
        "requested_backend",
        "runtime_version",
        "source_revision",
        "variant_id",
    }
    missing = required - provenance.keys()
    if missing:
        raise BackendContractError(f"adapter provenance is missing {sorted(missing)}")
    expected = {
        "requested_backend": backend.value,
        "actual_backend": backend.value,
        "backend": backend.value,
        "variant_id": config.variant_id,
        "fallback_used": False,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise BackendContractError(f"adapter provenance {key} mismatch")
    for key in ("device", "provider", "runtime_version", "source_revision"):
        if not isinstance(provenance.get(key), str) or not provenance[key]:
            raise BackendContractError(f"adapter provenance {key} must be a nonempty string")
    if not isinstance(provenance.get("parameters"), Mapping):
        raise BackendContractError("adapter provenance parameters must be a mapping")
    if dict(provenance["parameters"]) != dict(config.parameters):
        raise BackendContractError("adapter provenance parameters differ from EnhancementConfig")
    model_sha256 = provenance.get("model_sha256")
    if model_sha256 is not None and (
        not isinstance(model_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", model_sha256) is None
    ):
        raise BackendContractError("adapter provenance model_sha256 must be null or a SHA-256")
    if config.model_sha256 is not None and model_sha256 != config.model_sha256:
        raise BackendContractError("adapter provenance model hash differs from EnhancementConfig")


def _adapter_callable(module: Any, name: str) -> Any:
    function = getattr(module, name, None)
    if not callable(function):
        raise BackendContractError(f"adapter does not define callable {name}()")
    return function


def run_backend(audio: np.ndarray, sample_rate: int, config: EnhancementConfig) -> EnhancementOutput:
    """Run exactly one explicitly requested backend and validate its result.

    Adapter exceptions intentionally propagate.  In particular, an adapter
    failure is never converted into a gain-only result or another backend.
    """
    if not isinstance(config, EnhancementConfig):
        raise TypeError("config must be an EnhancementConfig")
    backend = config.backend if isinstance(config.backend, BackendId) else BackendId(config.backend)
    rate = _check_positive_int(sample_rate, "sample_rate")
    _check_audio_array(audio, "input")
    module_path = f"demo.denoise_backends.{_module_name(backend)}"
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        # A missing method module or one of its declared dependencies is a
        # terminal prerequisite failure; never select another implementation.
        raise BackendUnavailableError(
            f"BACKEND_UNAVAILABLE: adapter module {module_path} could not be imported ({exc})"
        ) from exc

    probe = _adapter_callable(module, "probe")(config)
    probe = _validate_probe(probe, config)
    output = _adapter_callable(module, "enhance")(audio, rate, config)
    if not isinstance(output, EnhancementOutput):
        raise BackendContractError("adapter enhance must return EnhancementOutput")
    _check_audio_array(output.audio, "enhanced")

    expected_backend = backend
    if output.actual_backend != expected_backend:
        raise BackendContractError(
            f"backend mismatch: requested {expected_backend.value}, got {output.actual_backend.value}"
        )
    if output.variant_id != config.variant_id:
        raise BackendContractError(
            f"variant mismatch: requested {config.variant_id!r}, got {output.variant_id!r}"
        )
    if output.native_sample_rate != probe.native_sample_rate:
        raise BackendContractError(
            f"native rate mismatch: probe {probe.native_sample_rate}, output {output.native_sample_rate}"
        )
    if output.delay_samples != probe.delay_samples:
        raise BackendContractError(
            f"delay mismatch: probe {probe.delay_samples}, output {output.delay_samples}"
        )
    _mapping_values(output.metadata, output.provenance)
    _validate_output_provenance(output.provenance, config, expected_backend)
    for key in ("source_revision", "runtime_version", "model_sha256"):
        probe_value = getattr(probe, key)
        if probe_value is not None and output.provenance.get(key) != probe_value:
            raise BackendContractError(f"adapter provenance {key} differs from probe")

    native_input_length = int(round(audio.size * probe.native_sample_rate / rate))
    if probe.delay_samples == 0:
        allowed_lengths = {native_input_length}
    else:
        # A delayed adapter must expose its declared leading padding.  Exact
        # native input length is not silently accepted for a delayed method.
        allowed_lengths = {native_input_length + probe.delay_samples}
    if output.audio.size not in allowed_lengths:
        expected = ", ".join(str(length) for length in sorted(allowed_lengths))
        raise BackendContractError(
            f"uncompensated output length: got {output.audio.size}, expected {expected}"
        )
    return output


__all__ = [
    "BackendId",
    "EnhancementConfig",
    "BackendProbe",
    "EnhancementOutput",
    "BackendError",
    "BackendUnavailableError",
    "BackendContractError",
    "JSONValue",
    "run_backend",
]

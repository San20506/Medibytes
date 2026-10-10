"""Pinned DeepFilterNet3 v0.5.6 adapter with an explicit 48 kHz native boundary.

The pinned upstream stack cannot live in the shared evaluation environment:
``deepfilternet==0.5.6`` requires ``numpy<2`` and ``deepfilterlib==0.5.6`` publishes
no CPython 3.12 wheel.  This adapter therefore owns a method-local environment
(``build_deepfilternet3.sh`` plus ``deepfilternet3.env.json``) and runs exactly one
``init_df()``/``enhance()`` call per sample inside it, then reconciles the result
with the shared 16 kHz boundary.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from .. import paths
from ..denoise import (
    BackendId,
    BackendProbe,
    BackendUnavailableError,
    EnhancementConfig,
    EnhancementOutput,
)


BACKEND = BackendId.DEEPFILTERNET3
_COMMON_RATE = 16_000
_NATIVE_RATE = 48_000
_VARIANT = "v0.5.6-dfn3"
_SOURCE_URL = "https://github.com/Rikorose/DeepFilterNet"
_SOURCE_REVISION = "v0.5.6"
_SOURCE_TAG_COMMIT = "978576aa8400552a4ce9730838c635aa30db5e61"
_RUNTIME_VERSION = "0.5.6"
_PROVIDER = "Rikorose/DeepFilterNet"
_ARCHITECTURE = "DeepFilterNet3"
_EXPECTED_PARAMETERS = {
    "architecture": _ARCHITECTURE,
    "device": "cpu",
    "state_per_sample": True,
}
# The staged DeepFilterNet3 digests are pinned here so the method configuration
# file can never widen the artifact this adapter is willing to execute.
_PINNED_MODEL = {
    "archive_relative": "models/deepfilternet/v0.5.6/DeepFilterNet3.zip",
    "archive_sha256": "49c52edc8947ae1f9bf50d81530beaf3a2c3245aeaf34b6f31ff535cd22284d2",
    "extracted_relative": "models/deepfilternet/v0.5.6/DeepFilterNet3",
    "config_sha256": "415eb925d44990d938fb739f514aa3662c1ec0ea836cff044fa1291b82cb4290",
    "checkpoint_name": "model_120.ckpt.best",
    "checkpoint_sha256": "23b92884f63ccf54bb026014604625ab231657b6480df65db4095c4c171e6003",
}
_STAMP_NAME = ".medibytes-deepfilternet3-stamp.json"
_RESAMPLER = "scipy.signal.resample_poly(up=3,down=1)"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_METHOD_DIR = Path(__file__).resolve().parent
_ENV_CONFIG_PATH = _METHOD_DIR / "deepfilternet3.env.json"
_WORKER_PATH = _METHOD_DIR / "deepfilter3_worker.py"
_LOCK_PATH = _METHOD_DIR / "requirements-deepfilternet3.lock"


def _method_environment() -> dict[str, Any]:
    """Load the method-local environment configuration and hold it to the pins."""

    try:
        raw = json.loads(_ENV_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BackendUnavailableError(
            f"BACKEND_UNAVAILABLE: method environment config is unreadable ({error})"
        ) from error
    if raw.get("backend") != BACKEND.value or raw.get("variant_id") != _VARIANT:
        raise BackendUnavailableError("BACKEND_UNAVAILABLE: method environment config identity drift")
    if raw.get("source_revision") != _SOURCE_REVISION:
        raise BackendUnavailableError("BACKEND_UNAVAILABLE: method environment source revision drift")
    model = raw.get("model")
    if not isinstance(model, dict) or any(
        model.get(key) != value for key, value in _PINNED_MODEL.items()
    ):
        raise BackendUnavailableError("BACKEND_UNAVAILABLE: method environment model pins drift")
    environment = raw.get("environment")
    if not isinstance(environment, dict) or environment.get("lock") != (
        "base_pipeline/denoise_backends/requirements-deepfilternet3.lock"
    ):
        raise BackendUnavailableError("BACKEND_UNAVAILABLE: method environment lock reference drift")
    # The data root is machine-local, so it is not pinned in the JSON; a config that names one wins.
    raw.setdefault("data_root", str(paths.data_root()))
    return raw


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_file(path: Path, expected_sha256: str, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise BackendUnavailableError(
            f"BACKEND_UNAVAILABLE: {label} is not staged at {path} ({error})"
        ) from error
    if not resolved.is_file():
        raise BackendUnavailableError(f"BACKEND_UNAVAILABLE: {label} is not a regular file")
    actual = _sha256(resolved)
    if actual != expected_sha256:
        raise BackendUnavailableError(
            f"BACKEND_UNAVAILABLE: {label} SHA-256 {actual} differs from the pinned {expected_sha256}"
        )
    return resolved


def _staged_model(environment: dict[str, Any]) -> tuple[Path, dict[str, str]]:
    """Verify the staged archive, model config, and checkpoint digests."""

    data_root = Path(environment["data_root"])
    _checked_file(
        data_root / _PINNED_MODEL["archive_relative"],
        _PINNED_MODEL["archive_sha256"],
        "DeepFilterNet3 archive",
    )
    _checked_file(
        data_root / _PINNED_MODEL["extracted_relative"] / "config.ini",
        _PINNED_MODEL["config_sha256"],
        "DeepFilterNet3 model config",
    )
    _checked_file(
        data_root
        / _PINNED_MODEL["extracted_relative"]
        / "checkpoints"
        / _PINNED_MODEL["checkpoint_name"],
        _PINNED_MODEL["checkpoint_sha256"],
        "DeepFilterNet3 checkpoint",
    )
    digests = {
        "model_archive_sha256": _PINNED_MODEL["archive_sha256"],
        "model_config_sha256": _PINNED_MODEL["config_sha256"],
        "model_checkpoint_sha256": _PINNED_MODEL["checkpoint_sha256"],
    }
    return data_root / _PINNED_MODEL["extracted_relative"], digests


def _method_python(environment: dict[str, Any]) -> Path:
    """Return the method interpreter after verifying the build stamp and lock."""

    spec = environment["environment"]
    root = _REPOSITORY_ROOT / spec["relative_path"]
    stamp_path = root / _STAMP_NAME
    try:
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BackendUnavailableError(
            f"BACKEND_UNAVAILABLE: method environment stamp is missing at {stamp_path}; "
            f"run base_pipeline/denoise_backends/build_deepfilternet3.sh ({error})"
        ) from error
    if stamp.get("lock") != spec["lock"]:
        raise BackendUnavailableError("BACKEND_UNAVAILABLE: method environment lock reference drift")
    if stamp.get("variant_id") != _VARIANT or stamp.get("lock_sha256") != _sha256(_LOCK_PATH):
        raise BackendUnavailableError(
            "BACKEND_UNAVAILABLE: the method environment was not built from this dependency lock"
        )
    if not _WORKER_PATH.is_file():
        raise BackendUnavailableError(f"BACKEND_UNAVAILABLE: method worker is missing at {_WORKER_PATH}")
    interpreter = root / spec["python_executable"]
    if not interpreter.is_file():
        raise BackendUnavailableError(
            f"BACKEND_UNAVAILABLE: method interpreter is missing at {interpreter}"
        )
    return interpreter


def _validate_config(config: EnhancementConfig) -> dict[str, Any]:
    if config.backend != BACKEND:
        raise ValueError(f"deepfilternet3 adapter cannot serve backend {config.backend!r}")
    if config.variant_id != _VARIANT:
        raise ValueError(
            f"deepfilternet3 variant must be the frozen {_VARIANT!r}, got {config.variant_id!r}"
        )
    parameters = dict(config.parameters)
    if parameters != _EXPECTED_PARAMETERS:
        raise ValueError(
            f"deepfilternet3 parameters must be the frozen {_EXPECTED_PARAMETERS}, got {parameters}"
        )
    return parameters


def _verified_runtime(config: EnhancementConfig) -> dict[str, Any]:
    environment = _method_environment()
    model_dir, digests = _staged_model(environment)
    interpreter = _method_python(environment)
    staged_archive = Path(environment["data_root"]) / _PINNED_MODEL["archive_relative"]
    if config.model_path is not None and Path(config.model_path).resolve() != staged_archive:
        raise ValueError(f"deepfilternet3 model_path must be the staged archive {staged_archive}")
    if config.model_sha256 is not None and config.model_sha256 != digests["model_archive_sha256"]:
        raise ValueError("deepfilternet3 model_sha256 differs from the pinned DeepFilterNet3 archive")
    return {
        "environment": environment,
        "model_dir": model_dir,
        "interpreter": interpreter,
        "digests": digests,
    }


def probe(config: EnhancementConfig) -> BackendProbe:
    _validate_config(config)
    try:
        _verified_runtime(config)
    except (BackendUnavailableError, OSError) as error:
        return BackendProbe(
            available=False,
            reason=str(error),
            source_revision=_SOURCE_REVISION,
            runtime_version=_RUNTIME_VERSION,
            model_sha256=_PINNED_MODEL["archive_sha256"],
            native_sample_rate=_NATIVE_RATE,
            delay_samples=0,
            stateful=True,
        )
    return BackendProbe(
        available=True,
        reason="pinned DeepFilterNet v0.5.6 DeepFilterNet3 model and method environment verified",
        source_revision=_SOURCE_REVISION,
        runtime_version=_RUNTIME_VERSION,
        model_sha256=_PINNED_MODEL["archive_sha256"],
        native_sample_rate=_NATIVE_RATE,
        delay_samples=0,
        stateful=True,
    )


def _run_worker(
    runtime: dict[str, Any], native_input: np.ndarray, timeout_s: float
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run exactly one pinned DeepFilterNet3 call in the method environment."""

    spec = runtime["environment"]["environment"]
    worker_environment = dict(os.environ)
    for inherited in ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH"):
        worker_environment.pop(inherited, None)
    worker_environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": str(spec["torch_threads"]),
            "MKL_NUM_THREADS": str(spec["torch_threads"]),
        }
    )
    command = [
        str(runtime["interpreter"]),
        str(_WORKER_PATH),
        "--model-dir",
        str(runtime["model_dir"]),
        "--threads",
        str(spec["torch_threads"]),
    ]
    with tempfile.TemporaryDirectory(prefix="medibytes-dfn3-") as temporary:
        work = Path(temporary)
        input_path = work / "native-input.npy"
        output_path = work / "native-output.npy"
        report_path = work / "runtime-report.json"
        np.save(input_path, native_input, allow_pickle=False)
        command += [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=str(work),
                env=worker_environment,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"DeepFilterNet3 worker exceeded its {timeout_s:.1f}s budget") from error
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()[-2000:]
            raise RuntimeError(
                f"DeepFilterNet3 worker failed with exit {completed.returncode}: {detail}"
            )
        if not report_path.is_file() or not output_path.is_file():
            raise RuntimeError("DeepFilterNet3 worker produced no output or runtime report")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        output = np.load(output_path, allow_pickle=False)
    if output.dtype != np.float32 or output.ndim != 1:
        raise RuntimeError("DeepFilterNet3 worker output must be mono float32")
    return output, report


def _verify_report(report: dict[str, Any], environment: dict[str, Any], expected_samples: int) -> None:
    required = {
        "python_version",
        "torch_version",
        "config_allow_defaults",
        "cuda_available",
        "native_sample_rate",
        "fft_size",
        "post_filter",
        "algorithm_delay_samples",
        "padding_samples",
        "input_samples",
        "output_samples",
        "package_versions",
    }
    missing = required - report.keys()
    if missing:
        raise RuntimeError(f"DeepFilterNet3 runtime report is missing {sorted(missing)}")
    spec = environment["environment"]
    if report["device"] != "cpu" or report["cuda_available"] is not False:
        raise RuntimeError(f"DeepFilterNet3 must run on CPU, report says {report['device']!r}")
    if report["native_sample_rate"] != _NATIVE_RATE:
        raise RuntimeError(
            f"DeepFilterNet3 reported native rate {report['native_sample_rate']} != {_NATIVE_RATE}"
        )
    for name, expected in spec["packages"].items():
        actual = report["package_versions"].get(name)
        if actual != expected:
            raise RuntimeError(f"pinned {name}=={expected} required, worker reports {actual!r}")
    if not str(report["python_version"]).startswith(spec["python_version"]):
        raise RuntimeError(
            f"method environment requires Python {spec['python_version']}, "
            f"worker reports {report['python_version']}"
        )
    if int(report["torch_threads"]) != int(spec["torch_threads"]):
        raise RuntimeError(
            f"method environment pins {spec['torch_threads']} torch threads, "
            f"worker reports {report['torch_threads']}"
        )
    if int(report["input_samples"]) != expected_samples or int(report["output_samples"]) != expected_samples:
        raise RuntimeError("DeepFilterNet3 worker reported a sample-count change")
    if int(report["algorithm_delay_samples"]) != int(report["fft_size"]) - int(report["hop_size"]):
        raise RuntimeError("DeepFilterNet3 reported an inconsistent algorithmic delay")
    if int(report["padding_samples"]) != int(report["fft_size"]):
        raise RuntimeError("DeepFilterNet3 reported inconsistent padding for the compensated delay")


def enhance(audio: np.ndarray, sample_rate: int, config: EnhancementConfig) -> EnhancementOutput:
    parameters = _validate_config(config)
    if sample_rate != _COMMON_RATE:
        raise ValueError("DeepFilterNet3 adapter requires the shared 16 kHz input boundary")
    runtime = _verified_runtime(config)
    source = np.asarray(audio, dtype=np.float32)
    if source.ndim != 1 or source.size == 0 or not np.isfinite(source).all():
        raise ValueError("DeepFilterNet3 input must be non-empty, finite, and mono")

    native_input = np.asarray(resample_poly(source, up=3, down=1), dtype=np.float32)
    expected_native = int(round(source.size * _NATIVE_RATE / sample_rate))
    if native_input.size != expected_native:
        raise RuntimeError(
            f"16 kHz to 48 kHz resampler length mismatch: {native_input.size} != {expected_native}"
        )
    native, report = _run_worker(runtime, np.ascontiguousarray(native_input), float(config.timeout_s))
    _verify_report(report, runtime["environment"], expected_native)
    if native.size != expected_native:
        raise RuntimeError(
            f"DeepFilterNet3 native output length differs from its input: {native.size} != {expected_native}"
        )
    if not np.isfinite(native).all() or not np.any(native):
        raise ValueError("DeepFilterNet3 returned empty, all-zero, or non-finite native audio")

    digests = runtime["digests"]
    environment = runtime["environment"]
    pinned = {
        "architecture": _ARCHITECTURE,
        "source_url": _SOURCE_URL,
        "source_tag": _SOURCE_REVISION,
        "source_tag_commit": _SOURCE_TAG_COMMIT,
        "package": "deepfilternet",
        "package_version": _RUNTIME_VERSION,
        "torch_version": report["torch_version"],
        "method_python_version": report["python_version"],
        "config_allow_defaults": bool(report["config_allow_defaults"]),
        "post_filter": bool(report["post_filter"]),
        **digests,
        "native_fft_size": int(report["fft_size"]),
        "native_hop_size": int(report["hop_size"]),
        "algorithm_delay_samples_native": int(report["algorithm_delay_samples"]),
        "native_padding_samples": int(report["padding_samples"]),
        "torch_threads": int(report["torch_threads"]),
    }
    return EnhancementOutput(
        audio=np.ascontiguousarray(native, dtype=np.float32),
        actual_backend=BACKEND,
        variant_id=config.variant_id,
        native_sample_rate=_NATIVE_RATE,
        delay_samples=0,
        metadata={
            "parameters": parameters,
            "common_sample_rate": _COMMON_RATE,
            "native_input_samples": expected_native,
            "native_output_samples": int(native.size),
            "resampler": _RESAMPLER,
            "model_suffix": report["model_suffix"],
            "nb_df": int(report["nb_df"]),
            "state_created_per_call": 1,
            "state_reset": "fresh init_df() per sample in a new process",
            "device": "cpu",
            "delay_reconciled_in_adapter": True,
            "delay_compensation": "upstream pad=True: n_fft trailing zeros, crop [n_fft-hop, T+n_fft-hop]",
            "model_load_s": float(report["model_load_s"]),
            "enhance_s": float(report["enhance_s"]),
            "method_environment": environment["environment"]["relative_path"],
            "method_environment_deviation": environment["environment"]["deviation"],
            **pinned,
        },
        provenance={
            "requested_backend": BACKEND.value,
            "actual_backend": BACKEND.value,
            "backend": BACKEND.value,
            "variant_id": config.variant_id,
            "model_sha256": digests["model_archive_sha256"],
            "provider": _PROVIDER,
            "runtime_version": _RUNTIME_VERSION,
            "source_revision": _SOURCE_REVISION,
            "device": "cpu",
            "parameters": parameters,
            "fallback_used": False,
            "license": "MIT",
            **pinned,
        },
    )


__all__ = ["enhance", "probe"]

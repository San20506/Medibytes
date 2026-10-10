"""Pinned SpeexDSP 1.2.1 preprocessor, denoise only, driven through ctypes."""
from __future__ import annotations

import ctypes
import hashlib
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

BACKEND = BackendId.SPEEX_DENOISE
VARIANT = "1.2.1-denoise-only"
SOURCE_REVISION = "7a158783df74efe7c2d1c6ee8363c1e695c71226"
LIBRARY_SHA256 = "a388aded3a0e08dc5310082539afa907028639ad2c57a1b29274f041d48907ed"
METHOD_ROOT = data_root() / "methods/speexdsp"
SOURCE_DIR = METHOD_ROOT / f"speexdsp-{SOURCE_REVISION}"
LIBRARY_PATH = SOURCE_DIR / "libspeexdsp/.libs/libspeexdsp.so.1.5.2"
NATIVE_RATE = 16_000
PARAMETERS = {
    "frame_size": 160,
    "sample_rate": 16000,
    "denoise": True,
    "agc": False,
    "vad": False,
    "dereverb": False,
    "residual_echo_suppression": False,
}
FRAME_SIZE = PARAMETERS["frame_size"]

# speex_preprocess_ctl request codes from the pinned public header.
_SET_DENOISE = 0
_SET_AGC = 2
_SET_VAD = 4
_SET_DEREVERB = 8
_SET_NOISE_SUPPRESS = 18
_GET_NOISE_SUPPRESS = 19
_SET_ECHO_SUPPRESS = 20
_SET_ECHO_SUPPRESS_ACTIVE = 22
_SET_ECHO_STATE = 24


class _Library:
    def __init__(self, path: Path) -> None:
        library = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
        state_init = library.speex_preprocess_state_init
        state_init.argtypes = [ctypes.c_int, ctypes.c_int]
        state_init.restype = ctypes.c_void_p
        control = library.speex_preprocess_ctl
        control.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        control.restype = ctypes.c_int
        run = library.speex_preprocess_run
        run.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int16)]
        run.restype = ctypes.c_int
        destroy = library.speex_preprocess_state_destroy
        destroy.argtypes = [ctypes.c_void_p]
        destroy.restype = None
        self._state_init = state_init
        self._control = control
        self._run = run
        self._destroy = destroy

    def set_int(self, state: int, request: int, value: int) -> None:
        payload = ctypes.c_int32(value)
        if self._control(state, request, ctypes.byref(payload)) != 0:
            raise ValueError(f"speex_preprocess_ctl request {request} failed")

    def get_int(self, state: int, request: int) -> int:
        payload = ctypes.c_int32(0)
        if self._control(state, request, ctypes.byref(payload)) != 0:
            raise ValueError(f"speex_preprocess_ctl request {request} failed")
        return int(payload.value)

    def disable_echo_state(self, state: int) -> None:
        # SET_ECHO_STATE is the one control that takes the state pointer itself,
        # not the address of a value, so a null pointer is the disabled form.
        if self._control(state, _SET_ECHO_STATE, ctypes.c_void_p(None)) != 0:
            raise ValueError("speex_preprocess_ctl request 24 failed")

    def new_state(self, frame_size: int, rate: int) -> int:
        state = self._state_init(frame_size, rate)
        if not state:
            raise RuntimeError("speex_preprocess_state_init returned NULL")
        return state

    def run_frame(self, state: int, frame: np.ndarray) -> None:
        self._run(state, frame.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)))

    def destroy(self, state: int) -> None:
        self._destroy(state)


def _checked_inputs(config: EnhancementConfig) -> Path:
    if config.backend != BACKEND or config.variant_id != VARIANT:
        raise ValueError("SpeexDSP requires the exact requested backend and variant")
    if dict(config.parameters) != PARAMETERS:
        raise ValueError("SpeexDSP parameters differ from the frozen method config")
    if config.model_path is not None or config.model_sha256 is not None:
        raise ValueError("SpeexDSP is model-free and must not carry a model path or digest")
    digest = hashlib.sha256()
    try:
        with LIBRARY_PATH.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise BackendUnavailableError("BACKEND_UNAVAILABLE: pinned SpeexDSP library is missing") from error
    if digest.hexdigest() != LIBRARY_SHA256:
        raise BackendUnavailableError("BACKEND_UNAVAILABLE: pinned SpeexDSP library digest mismatch")
    if not (SOURCE_DIR / "configure.ac").is_file():
        raise BackendUnavailableError("BACKEND_UNAVAILABLE: pinned SpeexDSP source tree is missing")
    return LIBRARY_PATH


def _configure(library: _Library, state: int) -> int:
    library.set_int(state, _SET_DENOISE, 1)
    library.set_int(state, _SET_AGC, 0)
    library.set_int(state, _SET_VAD, 0)
    library.set_int(state, _SET_DEREVERB, 0)
    library.set_int(state, _SET_ECHO_SUPPRESS, 0)
    library.set_int(state, _SET_ECHO_SUPPRESS_ACTIVE, 0)
    library.disable_echo_state(state)
    return library.get_int(state, _GET_NOISE_SUPPRESS)


def probe(config: EnhancementConfig) -> BackendProbe:
    _checked_inputs(config)
    try:
        library = _Library(LIBRARY_PATH)
        state = library.new_state(FRAME_SIZE, NATIVE_RATE)
        try:
            suppression = _configure(library, state)
        finally:
            library.destroy(state)
    except (OSError, ValueError, RuntimeError) as error:
        return BackendProbe(
            available=False,
            reason=f"BACKEND_UNAVAILABLE: {error}",
            source_revision=SOURCE_REVISION,
            runtime_version="1.2.1",
            model_sha256=None,
            native_sample_rate=NATIVE_RATE,
            stateful=True,
        )
    if suppression >= 0:
        raise BackendUnavailableError("BACKEND_UNAVAILABLE: expected a negative noise suppression default")
    return BackendProbe(
        available=True,
        reason=f"pinned SpeexDSP 1.2.1 denoise-only preprocessor verified (default {suppression} dB)",
        source_revision=SOURCE_REVISION,
        runtime_version="1.2.1",
        model_sha256=None,
        native_sample_rate=NATIVE_RATE,
        stateful=True,
    )


def enhance(audio: np.ndarray, sample_rate: int, config: EnhancementConfig) -> EnhancementOutput:
    _checked_inputs(config)
    if sample_rate != NATIVE_RATE:
        raise ValueError("SpeexDSP adapter requires mono 16 kHz input")
    source = np.asarray(audio, dtype=np.float32)
    if source.ndim != 1 or source.size == 0 or not np.isfinite(source).all():
        raise ValueError("SpeexDSP requires finite nonempty mono input")

    library = _Library(LIBRARY_PATH)
    # One fresh preprocessor state per independent sample.
    state = library.new_state(FRAME_SIZE, NATIVE_RATE)
    try:
        suppression = _configure(library, state)
        padding = (-source.size) % FRAME_SIZE
        padded = np.pad(source, (0, padding)) if padding else source
        clipped = np.clip(padded, -1.0, 32767.0 / 32768.0)
        pcm = np.rint(clipped * 32768.0).astype("<i2", copy=False)
        frames = pcm.size // FRAME_SIZE
        for index in range(frames):
            library.run_frame(state, pcm[index * FRAME_SIZE:(index + 1) * FRAME_SIZE])
    finally:
        library.destroy(state)

    output = np.ascontiguousarray(pcm.astype(np.float32) / 32768.0)
    output = output[:source.size]
    if not np.isfinite(output).all() or not np.any(output):
        raise ValueError("SpeexDSP returned nonfinite or silent audio")
    return EnhancementOutput(
        audio=output,
        actual_backend=BACKEND,
        variant_id=VARIANT,
        native_sample_rate=NATIVE_RATE,
        delay_samples=0,
        metadata={
            "state_per_call": True,
            "native_frame_samples": FRAME_SIZE,
            "native_padding_samples": padding,
            "native_frames_processed": frames,
            "noise_suppress_db": suppression,
            "noise_suppress_policy": "upstream default, queried not overridden",
        },
        provenance={
            "requested_backend": BACKEND.value,
            "actual_backend": BACKEND.value,
            "backend": BACKEND.value,
            "variant_id": VARIANT,
            "fallback_used": False,
            "source_revision": SOURCE_REVISION,
            "runtime_version": "1.2.1",
            "device": "cpu",
            "provider": "ctypes/libspeexdsp",
            "model_sha256": None,
            "library_sha256": LIBRARY_SHA256,
            "denoise": True,
            "agc": False,
            "vad": False,
            "dereverb": False,
            "residual_echo_suppression": False,
            "state_reset_per_call": True,
            "parameters": dict(config.parameters),
        },
    )

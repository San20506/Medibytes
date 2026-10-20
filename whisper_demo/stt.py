"""Speech to text with Whisper large-v3 only (faster-whisper / CTranslate2).

There is no mock and no other decoder here: a failure raises, and the job is reported as failed.

Device policy: CUDA float16 when CTranslate2 sees a GPU *and* a real warm-up decode succeeds,
otherwise CPU int8. The warm-up matters: model construction succeeds on a machine whose
cuBLAS cannot be loaded, and the error only appears on the first `transcribe()`.
"""
import ctypes
import glob
import importlib.util
import os
import re
import threading
from importlib import metadata as importlib_metadata

MODEL_NAME = "large-v3"
LANGUAGES = ("en", "hi", "ta")  # forced-language choices; None means auto-detect
NO_SPEECH_LIMIT = 0.6

# beam search, deterministic, with faster-whisper's own silence / repetition guards.
DECODE_KWARGS = {"beam_size": 5, "temperature": 0.0, "no_speech_threshold": NO_SPEECH_LIMIT,
                 "compression_ratio_threshold": 2.4, "word_timestamps": True}

_CUDA_LIBS = ("cublas/lib/libcublasLt.so.12", "cublas/lib/libcublas.so.12", "cudnn/lib/libcudnn*.so.9")
_state = {"model": None, "device": None, "compute_type": None, "fallback_reason": None,
          "cuda_libs": []}
_lock = threading.Lock()


def _preload_cuda_libs():
    """Load the pip-installed cuBLAS 12 / cuDNN 9 so CTranslate2 finds them. Silent if absent."""
    loaded = []
    try:
        spec = importlib.util.find_spec("nvidia")
    except (ImportError, ValueError):
        spec = None
    if spec is None or not spec.submodule_search_locations:
        return loaded
    base = list(spec.submodule_search_locations)[0]
    for pattern in _CUDA_LIBS:
        for lib in sorted(glob.glob(os.path.join(base, pattern))):
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                loaded.append(os.path.basename(lib))
            except OSError:
                pass
    return loaded


def _cuda_device_count():
    try:
        import ctranslate2
        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def _warm_up(model):
    import numpy as np
    segments, _ = model.transcribe(np.zeros(16000, dtype=np.float32), beam_size=1)
    list(segments)  # decoding is lazy; consume it so a compute-time failure surfaces here


def load():
    """The shared model, loaded and warmed once per process. Raises if no device works."""
    with _lock:
        if _state["model"] is not None:
            return _state
        from faster_whisper import WhisperModel

        _state["cuda_libs"] = _preload_cuda_libs()
        if _cuda_device_count() > 0:
            try:
                model = WhisperModel(MODEL_NAME, device="cuda", compute_type="float16")
                _warm_up(model)
                _state.update(model=model, device="cuda", compute_type="float16",
                              fallback_reason=None)
                return _state
            except Exception as exc:
                _state["fallback_reason"] = f"{type(exc).__name__}: {exc}"[:300]
        else:
            _state["fallback_reason"] = "no CUDA device visible to CTranslate2"
        model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
        _warm_up(model)
        _state.update(model=model, device="cpu", compute_type="int8")
        return _state


def status():
    """What `/api/health` reports; never loads the model."""
    return {"loaded": _state["model"] is not None, "model": MODEL_NAME,
            "device": _state["device"], "compute_type": _state["compute_type"],
            "fallback_reason": _state["fallback_reason"]}


def _runtime_version():
    try:
        return importlib_metadata.version("faster-whisper")
    except importlib_metadata.PackageNotFoundError:
        return None


def segment_confidence(avg_logprob, no_speech_prob, text):
    """Same mapping as base_pipeline/asr.py, with the guard flagged on silence-like segments."""
    if re.search(r"(thank you\s*){3,}", text, re.I) or (no_speech_prob or 0.0) > NO_SPEECH_LIMIT:
        return 0.2
    if avg_logprob is None:
        return 0.88
    return round(float(avg_logprob) * -1 + 0.7, 2)


def transcribe(audio, language=None, job_id="job"):
    """Transcribe a float32 mono 16 kHz array. Passing the array keeps PyAV out of the loop."""
    if language is not None and language not in LANGUAGES:
        raise ValueError(f"unknown language {language!r}; valid: auto, {', '.join(LANGUAGES)}")
    state = load()
    kwargs = dict(DECODE_KWARGS)
    if language:
        kwargs["language"] = language
    segments, info = state["model"].transcribe(audio, **kwargs)
    segs, full = [], []
    for i, s in enumerate(segments):
        text = s.text.strip()
        full.append(text)
        segs.append({
            "id": i, "text": text, "start": round(s.start, 2), "end": round(s.end, 2),
            "lang": getattr(info, "language", None),
            "confidence": segment_confidence(getattr(s, "avg_logprob", None),
                                             getattr(s, "no_speech_prob", None), text),
            "no_speech_prob": round(float(getattr(s, "no_speech_prob", 0.0) or 0.0), 3),
            "words": [{"w": w.word, "s": round(w.start, 2), "e": round(w.end, 2)}
                      for w in (s.words or [])]})
    provenance = {
        "job_id": job_id, "provider": "faster-whisper", "requested_model": MODEL_NAME,
        "actual_model": MODEL_NAME, "device": state["device"],
        "compute_type": state["compute_type"], "fallback_reason": state["fallback_reason"],
        "beam_size": kwargs["beam_size"], "temperature": kwargs["temperature"],
        "decode_kwargs": kwargs, "forced_language": language,
        "detected_language": getattr(info, "language", None),
        "language_probability": round(float(getattr(info, "language_probability", 0.0) or 0.0), 3),
        "word_timestamps": True, "runtime_version": _runtime_version(), "is_mock": False}
    return {"text": " ".join(full), "segments": segs,
            "engine": f"faster-whisper:{MODEL_NAME}-{state['device']}-{state['compute_type']}",
            "language": getattr(info, "language", None), "stt_provenance": provenance}

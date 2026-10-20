"""Stage 2 - speech to text: MedASR (default), faster-whisper, and the labelled mock.

`transcribe` never falls back silently: every fallback is a mock result that says why.
"""
import contextlib
import os
import re
import wave
from importlib import metadata as importlib_metadata

from . import decode_presets
# The mock splits sentences exactly as the extractor does.
from .extract import SENT_SPLIT


# ---- Mock transcripts so investor demo runs with zero model downloads ----
MOCK_TEXTS = {
    "demo-001": "Patient ko fever hai, bukhar 101 degree, give paracetamol 500 mg BID, do time, 3 din tak. No allergy.",
    "demo-002": "Khansi hai, cough for 5 days, BP 130 by 80, give azithromycin 500 mg once daily 3 days. No penicillin allergy, patient denies chest pain.",
    "sample1": "Patient ko fever hai, bukhar 101 degree, give paracetamol 500 mg BID, do time, 3 din tak. No allergy.",
    "sample2": "Khansi hai, cough for 5 days, BP 130 by 80, give azithromycin 500 mg once daily 3 days. No penicillin allergy, patient denies chest pain.",
}


def _mock_segments(text, lang="mix"):
    """Fake word timestamps evenly spaced - enough for click-word demo."""
    words, t, segs, seg_id = text.split(), 0.2, [], 0
    for sent in SENT_SPLIT.split(text):
        sw = sent.split()
        if not sw:
            continue
        dur = max(0.6, 0.32 * len(sw))
        wlist = [{"w": w, "s": round(t + i * 0.32, 2), "e": round(t + (i + 1) * 0.32, 2)}
                 for i, w in enumerate(sw)]
        segs.append({"id": seg_id, "text": sent, "start": round(t, 2),
                     "end": round(t + dur, 2), "lang": lang,
                     "confidence": 0.9, "words": wlist})
        t += dur + 0.3
        seg_id += 1
    return segs

def _faster_whisper_runtime_version(module):
    version = getattr(module, "__version__", None)
    if version:
        return str(version)
    try:
        return importlib_metadata.version("faster-whisper")
    except importlib_metadata.PackageNotFoundError:
        return None


def _mock_result(clean_wav, job_id, model, exc, reason):
    """The explicit, labelled mock fallback shared by every real STT backend."""
    key = job_id if job_id in MOCK_TEXTS else None
    if key is None:
        base = os.path.splitext(os.path.basename(clean_wav))[0].lower()
        for k in MOCK_TEXTS:
            if k in base or base in k:
                key = k
                break
        key = key or "demo-001"
    text = MOCK_TEXTS[key]
    return {"text": text, "segments": _mock_segments(text),
            "engine": f"mock ({reason}: {type(exc).__name__})",
            "language": "mix",
            "stt_provenance": _mock_stt_provenance(model, job_id)}


def _transformers_version():
    try:
        return importlib_metadata.version("transformers")
    except importlib_metadata.PackageNotFoundError:
        return None


def _mock_stt_provenance(requested_model, job_id):
    return {
        "job_id": job_id,
        "provider": "mock",
        "requested_model": requested_model,
        "actual_model": "mock",
        "device": None,
        "compute_type": None,
        "beam_size": None,
        "word_timestamps": True,
        "temperature": 0.0,
        "runtime_version": None,
        "model_hash": None,
        "model_snapshot": None,
        "is_mock": True,
    }


def _model_attribute(model, *names):
    for name in names:
        value = getattr(model, name, None)
        if value is not None and str(value):
            return str(value)
    return None




# ---- MedASR (google/medasr) STT backend ----------------------------------
# A CTC medical-dictation model, not a seq2seq decoder: it emits one unsegmented
# string with no timestamps and no language id, so the segment list below is a
# single span with evenly-spaced word times and `word_timestamps` is recorded as
# False.  Nothing else in the contract changes.
MEDASR_MODEL_ID = "google/medasr"

# MedASR is a dictation model: it does not emit punctuation characters, it emits
# the spoken command that produced them (`{period}`), the report section the
# dictation is in (`[FINDINGS]`), and its own CTC end token.  Left alone, every
# one of those becomes a spurious word downstream - `{period}` tokenises to
# "period" and `</s>` to "s" - so the markup is resolved here, once, before any
# consumer sees the text.  The punctuation map is what the model actually
# produced on this project's audio; an unrecognised `{directive}` is dropped
# rather than guessed at, and section headers are dropped because the demo's
# extractor reads prose, not report structure.
MEDASR_PUNCTUATION = {
    "period": ".",
    "full stop": ".",
    "comma": ",",
    "colon": ":",
    "semicolon": ";",
    "question mark": "?",
    "exclamation point": "!",
    "hyphen": "-",
    "dash": "-",
    "slash": "/",
    "apostrophe": "'",
    "open paren": "(",
    "close paren": ")",
    "open parenthesis": "(",
    "close parenthesis": ")",
    "open quote": '"',
    "close quote": '"',
    "new paragraph": "\n\n",
    "new line": "\n",
    "next line": "\n",
}
_MEDASR_DIRECTIVE = re.compile(r"\{\s*([^{}]*?)\s*\}")
_MEDASR_SECTION = re.compile(r"\[\s*[^\[\]]*\s*\]")
_MEDASR_SPECIAL = re.compile(r"</?s>|<unk>|<pad>|<epsilon>|<extra_id_\d+>")


def medasr_detokenize(text):
    """Resolve MedASR's dictation markup into ordinary punctuated prose."""
    out = _MEDASR_SPECIAL.sub(" ", str(text))
    out = _MEDASR_SECTION.sub(" ", out)
    out = _MEDASR_DIRECTIVE.sub(
        lambda m: MEDASR_PUNCTUATION.get(m.group(1).strip().lower(), " "), out
    )
    # Unmatched brackets survive a truncated or mis-decoded directive.
    out = re.sub(r"[\[\]{}]", " ", out)
    out = re.sub(r"\s+([.,:;?!])", r"\1", out)
    out = re.sub(r"[ \t]+", " ", out)
    return out.strip()
_MEDASR_PIPE = {}


def _medasr_pipeline(device=None):
    """Load `google/medasr` once per process and keep it for later calls."""
    import shutil

    import torch
    from transformers import pipeline as hf_pipeline

    if shutil.which("ffmpeg") is None:
        # transformers' ASR pipeline shells out to an `ffmpeg` binary to read
        # audio filenames. imageio-ffmpeg (already a hard dependency for
        # mp3/m4a input) ships a static Windows build, so expose it rather
        # than requiring a separate system install.
        try:
            import imageio_ffmpeg

            _ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
            if _ffmpeg_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        except Exception:
            pass

    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    if device not in _MEDASR_PIPE:
        _MEDASR_PIPE[device] = hf_pipeline(
            "automatic-speech-recognition", model=MEDASR_MODEL_ID, device=device
        )
    return _MEDASR_PIPE[device]


def _medasr_segments(text, duration_s):
    """One span over the whole clip; word times interpolated, never measured."""
    words = text.split()
    if not words:
        return []
    step = duration_s / len(words) if duration_s > 0 else 0.32
    return [{
        "id": 0,
        "text": text,
        "start": 0.0,
        "end": round(duration_s, 2),
        "lang": "en",
        "confidence": 0.9,
        "words": [{"w": w, "s": round(i * step, 2), "e": round((i + 1) * step, 2)}
                  for i, w in enumerate(words)],
    }]


def transcribe_medasr(clean_wav, job_id="demo-001", device=None):
    """Transcribe one 16 kHz mono WAV with MedASR, in the shared result shape."""
    import torch

    pipe = _medasr_pipeline(device)
    resolved = getattr(pipe, "device", None)
    device_name = str(resolved) if resolved is not None else "cpu"
    with contextlib.closing(wave.open(str(clean_wav), "rb")) as handle:
        duration_s = handle.getnframes() / float(handle.getframerate())
        _rate = handle.getframerate()
        _channels = handle.getnchannels()
        _width = handle.getsampwidth()
        _frames = handle.readframes(handle.getnframes())
    import numpy as _np

    _pcm = _np.frombuffer(_frames, dtype="<i2").astype(_np.float32) / 32768.0
    if _channels > 1:
        _pcm = _pcm.reshape(-1, _channels).mean(axis=1)
    # Pass raw audio, not a filename: transformers' ASR pipeline shells out to
    # a system `ffmpeg` binary for filenames (absent on Windows; the bundled
    # imageio-ffmpeg exe is differently named). clean_audio always writes
    # 16 kHz mono PCM16, which is exactly what this CTC model consumes.
    _target_rate = int(getattr(getattr(pipe, "feature_extractor", None), "sampling_rate", 16000) or 16000)
    raw_text = str(pipe({"array": _pcm, "sampling_rate": _rate},
                        chunk_length_s=20, stride_length_s=2)["text"])
    text = medasr_detokenize(raw_text)
    provenance = {
        "job_id": job_id,
        "provider": "transformers",
        "requested_model": "medasr",
        "actual_model": MEDASR_MODEL_ID,
        "device": device_name,
        "compute_type": str(getattr(pipe.model, "dtype", torch.float32)),
        "word_timestamps": False,
        "temperature": 0.0,
        "beam_size": None,
        "runtime_version": _transformers_version(),
        "model_hash": None,
        "model_snapshot": _model_attribute(pipe.model, "name_or_path"),
        "is_mock": False,
    }
    return {"text": text, "raw_text": raw_text,
            "segments": _medasr_segments(text, duration_s),
            "engine": f"medasr:{MEDASR_MODEL_ID}@{device_name}",
            "language": "en", "stt_provenance": provenance}


_WHISPER_MODELS = {}


def _whisper_model(WhisperModel, size, model):
    """One `WhisperModel` per (class, size) per process, with the device it settled on."""
    key = (WhisperModel, size)
    if key not in _WHISPER_MODELS:
        try:
            import torch as _torch
            _has_cuda = _torch.cuda.is_available()
        except Exception:
            _has_cuda = False
        if model in ("medium", "large-v3") and _has_cuda:
            try:
                _WHISPER_MODELS[key] = (WhisperModel(size, device="cuda", compute_type="float16"),
                                        "cuda", "float16")
            except Exception:
                # ctranslate2 build may want a libcublas older than the
                # system CUDA (e.g. .so.12 vs installed .so.13) - fall back.
                _WHISPER_MODELS[key] = (WhisperModel(size, device="cpu", compute_type="int8"),
                                        "cpu", "int8")
        else:
            _WHISPER_MODELS[key] = (WhisperModel(size, device="cpu", compute_type="int8"),
                                    "cpu", "int8")
    return _WHISPER_MODELS[key]


# faster-whisper model names as callers spell them -> the size handed to `WhisperModel`.
WHISPER_SIZES = {"tiny-int8": "tiny", "base-int8": "base", "small-int8": "small",
                 "medium": "medium", "large-v3": "large-v3"}
KNOWN_MODELS = ("medasr", "medasr-lm", "mock", *WHISPER_SIZES)


def transcribe(clean_wav, job_id="demo-001", model="medasr", strict: bool = False,
               decode=decode_presets.DEFAULT):
    """Transcribe with explicit mock fixtures or strict/fallback real STT.

    `decode` names a preset from `decode_presets` and applies to faster-whisper models only.
    An unknown `model` raises whatever `strict` says: it is a configuration error, not a runtime
    failure, and a fallback here would measure a different engine than the one requested.
    """
    if model not in KNOWN_MODELS:
        raise ValueError(f"unknown STT model {model!r}; valid models: {', '.join(KNOWN_MODELS)}")
    decode_presets.get(decode)  # unknown names fail before any audio is decoded
    if decode != decode_presets.DEFAULT and model in ("medasr", "medasr-lm"):
        raise ValueError(f"decode preset {decode!r} applies to faster-whisper models only, "
                         f"not {model!r}")
    if model == "mock":
        key = job_id if job_id in MOCK_TEXTS else "demo-001"
        return {"text": MOCK_TEXTS[key], "segments": _mock_segments(MOCK_TEXTS[key]),
                "engine": "mock (forced --model mock)", "language": "mix",
                "stt_provenance": _mock_stt_provenance(model, job_id)}
    if model == "medasr":
        try:
            return transcribe_medasr(clean_wav, job_id)
        except Exception as exc:
            if strict:
                raise
            # MedASR is the default decoder, so its failure path matters more
            # than the others': dropping straight to the mock would hand back
            # invented vitals and drugs that look like a real transcript, on
            # any machine without the weights. Degrade to a real decoder
            # instead, and only mock if that is unavailable too.
            medasr_error = exc
            model = "small-int8"
        else:
            medasr_error = None
    if model == "medasr-lm":
        try:
            from .medasr_lm import transcribe_medasr_lm
            return transcribe_medasr_lm(clean_wav, job_id)
        except Exception as exc:
            if strict:
                raise
            return _mock_result(clean_wav, job_id, model, exc, "no medasr-lm")
    try:
        import faster_whisper as faster_whisper_module
        from faster_whisper import WhisperModel
        size = WHISPER_SIZES[model]
        wm, _device, _compute = _whisper_model(WhisperModel, size, model)
        decode_kwargs = decode_presets.kwargs(decode)
        segments, info = wm.transcribe(clean_wav, **decode_kwargs)
        segs, full = [], []
        for i, s in enumerate(segments):
            full.append(s.text.strip())
            segs.append({"id": i, "text": s.text.strip(), "start": round(s.start, 2),
                         "end": round(s.end, 2), "lang": getattr(info, "language", "en"),
                         "confidence": round(float(getattr(s, "avg_logprob", -0.2)) * -1 + 0.7, 2) if hasattr(s, "avg_logprob") else 0.88,
                         "words": [{"w": w.word, "s": round(w.start, 2), "e": round(w.end, 2)}
                                   for w in (s.words or [])]})
            # hallucination guard: drop thank-you loops on silence
            if re.search(r"(thank you\s*){3,}", s.text, re.I):
                segs[-1]["confidence"] = 0.2
        text = " ".join(full)
        provenance = {
            "job_id": job_id,
            "provider": "faster-whisper",
            "requested_model": model,
            "actual_model": size,
            "device": _device,
            "compute_type": _compute,
            "word_timestamps": True,
            "temperature": decode_kwargs["temperature"],
            "beam_size": decode_kwargs["beam_size"],
            "decode_preset": decode,
            "decode_kwargs": decode_kwargs,
            "runtime_version": _faster_whisper_runtime_version(faster_whisper_module),
            "model_hash": _model_attribute(wm, "model_sha256", "model_hash"),
            "model_snapshot": _model_attribute(
                wm, "model_snapshot", "snapshot", "model_path"
            ),
            "is_mock": False,
        }
        engine = f"faster-whisper:{size}-{_device}-{_compute}"
        if locals().get("medasr_error") is not None:
            engine += f" (medasr unavailable: {type(medasr_error).__name__})"
        return {"text": text, "segments": segs,
                "engine": engine,
                "language": getattr(info, "language", "en"),
                "stt_provenance": provenance}
    except Exception as exc:
        if strict:
            raise
        return _mock_result(clean_wav, job_id, model, exc, "no faster-whisper")

"""Named decoding presets for the faster-whisper path.

Spec: openspec/changes/raw-asr-decoding-presets. A preset is data - the keyword arguments
of `WhisperModel.transcribe` - and each one adds exactly one change over the one before it,
so a difference between neighbours is attributable to that change. `greedy` is the
pre-existing call, unchanged, and stays the default.
"""
DEFAULT = "greedy"

GUARD = {"no_speech_threshold": 0.6, "compression_ratio_threshold": 2.4}
FALLBACK = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
PROMPT = "Medical dictation. Medication names, brand names and dosages."

_BEAM5 = {"beam_size": 5, "temperature": 0.0}
_BEAM5_EN = {**_BEAM5, "language": "en"}
_FALLBACK = {**_BEAM5_EN, "temperature": FALLBACK}

PRESETS = {
    "greedy": {"beam_size": 1, "temperature": 0.0},
    "beam5": _BEAM5,
    "beam5-en": _BEAM5_EN,
    "beam5-en-fallback": _FALLBACK,
    "beam5-en-fallback-prompt": {**_FALLBACK, "initial_prompt": PROMPT},
}


def get(name):
    """The preset's own settings; ValueError naming the valid presets if unknown."""
    if name not in PRESETS:
        raise ValueError(f"unknown decode preset {name!r}; valid presets: {', '.join(PRESETS)}")
    return PRESETS[name]


def kwargs(name):
    """Everything passed to `WhisperModel.transcribe`: the preset, the guard, word timestamps."""
    return {**get(name), **GUARD, "word_timestamps": True}

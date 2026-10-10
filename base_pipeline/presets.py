"""Named pipeline presets: one approach = one name, and a name is data.

A preset is the full stage selection for one `run.py` run: denoiser, decoder, decoding
preset, second pass, and which LLM steps run. `DEFAULTS` is the pipeline as it ships;
each preset lists only what it changes, so the difference between two presets is its
own dictionary. Decoding presets (`decode_presets.py`) and denoiser backends
(`denoise_backends/`) are the registries the `decode` and `denoiser` fields point into.

The names come from configurations that were measured (docs/PIPELINE_EVAL_REPORT.md Part E,
CHANGELOG.md); the comment on each says what evidence there is. A preset fixes the flags,
not the code: arms measured before a bug fix (e.g. the "guard") cannot be reproduced by name.
"""
from . import decode_presets

# Field -> value for the pipeline as it ships. Types follow `run.run_clip`:
#   second_pass     "auto" | a decoder name | None (skip)
#   use_llm         True = LLM-primary extraction and the discharge note
#   validate_terms  "auto" (only when a model is served) | True (always try) | False
#   context_repair  False | True
DEFAULTS = {
    "denoiser": "none",
    "model": "medasr",
    "decode": decode_presets.DEFAULT,
    "second_pass": "auto",
    "use_llm": False,
    "validate_terms": "auto",
    "context_repair": False,
    "variant": None,            # "whisper-medasr" replaces the whole chain (see whisper_medasr.py)
    "snr_threshold": 15.0,      # variant only
    "whisper_model": "small-int8",  # variant only
}

DEFAULT = "baseline"

PRESETS = {
    # What `python -m base_pipeline.run` does with no flags. The validator runs only if a
    # model server is up, so two machines can differ; `regex-only` removes that variable.
    "baseline": {},
    # Report arm A/A2: no LLM anywhere, so a run is reproducible on any machine.
    "regex-only": {"validate_terms": False},
    # `--no-second-pass`: the primary decoder alone (v0_nosp), to price the second pass.
    "no-second-pass": {"second_pass": None},
    # Report arm B2: the LLM term validator forced on; failures are recorded on each row.
    "validator": {"validate_terms": True},
    # Report arm C3: validator plus LLM context repair of RED rows.
    "context-repair": {"validate_terms": True, "context_repair": True},
    # LLM-primary extraction and the discharge note. Not measured: the report lists it as untested.
    "llm": {"use_llm": True},
    # SNR-gated denoise -> Whisper -> MedASR on the medical stretches -> extract -> verify.
    "whisper-medasr": {"variant": "whisper-medasr"},
}


def names():
    return list(PRESETS)


def resolve(name, overrides=None):
    """`DEFAULTS` + the preset + explicit overrides, each checked against the known fields."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; valid presets: {', '.join(PRESETS)}")
    unknown = sorted(set(overrides or ()) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown preset field(s) {', '.join(unknown)}; valid: {', '.join(DEFAULTS)}")
    return {**DEFAULTS, **PRESETS[name], **(overrides or {})}

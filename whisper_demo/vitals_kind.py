# Forked from base_pipeline/vitals_kind.py at c68d7a9; not kept in sync.
"""Vitals-span classifier shared by the extractor and the chart.

Lives in the pipeline because the LLM gap-fill gate in llm_extract must agree
exactly with the chart renderer about which vitals box a span fills.
"""
import re

# A percentage is an oxygen saturation only when a saturation cue is present.
# The bare word "percent" is not one: "100 percent compliance" and "50 percent
# of cases" are not readings, and 50 sits inside the plausible SpO2 range, so a
# mischarted peri-arrest value would reach the document unchallenged.
SPO2_CUE_RE = re.compile(r"sp[o0]2|\bo2\b|\bsats\b|saturat|room\s*air|rhoomair", re.I)
_PERCENT_RE = re.compile(r"\d\s*(?:%|percent)|percent|%", re.I)


def is_spo2_span(span, context=""):
    """True when a vitals span is an oxygen saturation reading.

    The vitals pattern captures a bare "95 percent", dropping the cue that made
    it a reading ("her saturation is maintaining around ninety-five percent"),
    so a cue-less percentage falls back to the sentence it came from. The
    fallback is clause-scoped and only applies to a percentage - a pulse or
    glucose span that merely shares a sentence with the word "saturation" is
    not promoted, which would wrongly mark the SpO2 slot as already filled.
    """
    span = str(span or "")
    if SPO2_CUE_RE.search(span):
        return True
    if not _PERCENT_RE.search(span):
        return False
    # A full stop between digits is a decimal point ("95.5 percent"), not a
    # clause break; splitting there would strand the percentage from its cue.
    for clause in re.split(r"[,;:]|\.(?!\d)", str(context or "")):
        if _PERCENT_RE.search(clause) and SPO2_CUE_RE.search(clause):
            return True
    return False


def vitals_box_kind(v):
    """Which vitals box a span belongs in: bp / temp / spo2 / None.

    Single source of truth for coords and for llm_extract's gap-fill gate; the
    two must agree or the LLM either overwrites a charted vital or refuses to
    fill an empty slot. An item carrying an explicit "kind" (LLM-built vitals,
    which have no cue text and no source sentence) is taken at its word.
    """
    kind = v.get("kind")
    if kind in ("bp", "temp", "spo2"):
        return kind
    t = str(v.get("text", ""))
    if re.search(r"(\d{2,3}(?:\.\d{1,2})?)\s*(by|/|over|of)\s*(\d{2,3}(?:\.\d{1,2})?)", t):
        return "bp"
    if re.search(r"degree|temp|fahrenheit|°", t, re.I):
        return "temp"
    if is_spo2_span(t, v.get("source_sentence", "")):
        return "spo2"
    return None

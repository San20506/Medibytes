"""Stages 2/3/4 - STT + Normalize + Extract. CPU-only demo with graceful fallbacks.

Report: Sec 4.4 (STT faster-whisper), 4.5 (normalize), 5 (NER ensemble).
Demo: faster-whisper tiny/base-int8 if installed else deterministic mock;
normalize via unicode-range LID + hardcoded Hinglish map; extract via regex
+ drug_list_mini.json + optional spaCy-sm + optional Ollama qwen2.5:0.5b tidy.
"""
import contextlib
import json
from importlib import metadata as importlib_metadata

import os
import re
import subprocess
import wave

_HERE = os.path.dirname(os.path.abspath(__file__))

# ---- Mock transcripts so investor demo runs with zero model downloads ----
MOCK_TEXTS = {
    "demo-001": "Patient ko fever hai, bukhar 101 degree, give paracetamol 500 mg BID, do time, 3 din tak. No allergy.",
    "demo-002": "Khansi hai, cough for 5 days, BP 130 by 80, give azithromycin 500 mg once daily 3 days. No penicillin allergy, patient denies chest pain.",
    "sample1": "Patient ko fever hai, bukhar 101 degree, give paracetamol 500 mg BID, do time, 3 din tak. No allergy.",
    "sample2": "Khansi hai, cough for 5 days, BP 130 by 80, give azithromycin 500 mg once daily 3 days. No penicillin allergy, patient denies chest pain.",
}

HINGLISH_MAP = [
    ("bukhar", "fever"), ("बुखार", "fever"), ("khansi", "cough"), ("खांसी", "cough"),
    ("do time", "twice daily"), ("din tak", "days"), ("din", "days"),
    ("BID", "twice daily"), ("OD", "once daily"), ("TDS", "thrice daily"),
    ("500 मिलीग्राम", "500 mg"), ("मिलीग्राम", "mg"), ("माइक्रोग्राम", "mcg"),
]

SYMPTOMS = ["fever", "cough", "chest pain", "headache", "vomiting", "diarrhea",
            "breathlessness", "sore throat", "cold", "pain", "bukhar", "khansi"]
# Numeral-tuned vitals: decimals, 'of' separator, long-form 'blood pressure',
# temp with F/C units, SpO2 with room-air mishear tolerance.
TEMP_NUM = r"\d{2,3}(?:\.\d{1,2})?\s*(?:degrees?(?:\s*(?:fahrenheit|celsius|F|C))?|°\s*[FC]?)"
# Dictation puts a few words between the cue and the value - a copula, a hedge,
# a repeated word where the speaker restarted. The existing `oxygen saturation`
# and `temp` rules already allow that with a bounded digit-free gap; every cue
# below uses the same idiom rather than enumerating `is|was|around|...`, so
# "pulse 92", "pulse rate is 88" and "heart rate was around 96" all match.
# A comma ends the cue's scope - "low sugar diet, review after 15 days" is not
# a glucose reading - and a number followed by a duration unit is an interval,
# not a measurement ("heart failure for 10 years").
_CUE_GAP = r"[^\d.,]{0,14}"
_NOT_DURATION = (r"(?!\s*(?:years?|yrs?|months?|weeks?|days?|hours?|hrs?"
                 r"|minutes?|mins?|seconds?|secs?)\b)")
_NUM = r"(?<!\d)\d{2,3}(?!\d)"
VITALS_PAT = re.compile(
    # BP's gap allows digits: a decoder routinely runs a stray number into the
    # copula ("blood pressure is19 160/90"). The pair itself is the signature,
    # so the lazy gap stops at the first real systolic/diastolic pair.
    r"((?:BP|B[\s-]*P|blood pressure)[^.]{0,24}?(?<!\d)\d{2,3}(?:\.\d{1,2})?\s*(by|/|over|of)\s*\d{2,3}(?:\.\d{1,2})?"
    r"|\b" + TEMP_NUM + r"\b"
    r"|SpO2\s*\d{2,3}(?:\.\d{1,2})?%?"
    r"|\b\d{2,3}(?:\.\d{1,2})?\s*(?:%|percent)\s*(?:on\s+(?:room\s*air|rhoomair))?"
    r"|oxygen saturation[^\d.]{0,10}\d{2,3}(?:\.\d{1,2})?\s*%?"
    r"|(?:pulse|heart)(?:\s*rate)?" + _CUE_GAP + _NUM + _NOT_DURATION +
    r"|(?:respiratory|resp(?:iration)?|breathing)\s*rate" + _CUE_GAP + r"(?<!\d)\d{1,3}(?!\d)" + _NOT_DURATION +
    r"|\b(?:blood\s+)?(?:glucose|sugar|BGL|RBS|CBG)[^\d.,]{0,40}(?<!\d)\d{2,3}(?:\.\d{1,2})?(?!\d)" + _NOT_DURATION +
    r"|(?<!\d)\d{1,3}(?:\.\d{1,2})?\s*(?:mg\s*/\s*d[lL]|milligrams?\s+per\s+decilit(?:re|er)"
    r"|mmol\s*/\s*[lL]|millimoles?\s+per\s+lit(?:re|er))"
    r"|temp[^\d.]{0,10}" + TEMP_NUM + r")",
    re.I,
)
# Physiological plausibility: generous bounds that only catch ASR garbage
# (pulse 940, BP 400/65, SpO2 400%) — never a real reading. A violation keeps
# the row (recall unchanged) but forces RED so it cannot chart at YELLOW.
VITALS_RANGES = {
    "spo2": (50.0, 100.0),
    "pulse": (20.0, 250.0),
    "temp_f": (90.0, 110.0),
    # The same reading in the other scale. Widening `temp_f` to cover Celsius
    # would have to reach down to 30, and 38 - a textbook fever in Celsius -
    # is also what a decoder produces when it drops a digit from a Fahrenheit
    # reading, so one range cannot both admit Celsius and still catch garbage.
    # The unit the span carries decides which range applies.
    "temp_c": (30.0, 45.0),
    "bp_sys": (50.0, 300.0),
    "bp_dia": (20.0, 200.0),
    "rr": (5.0, 80.0),
    "glucose": (20.0, 1500.0),
}


def _vital_span_kind(span):
    """Kind of a vitals span for the plausibility check (mirrors scorer cues)."""
    if re.search(r"\bBP\b|blood\s*pressure|\d+\s*(?:by|/|over|of)\s*\d+", span, re.I):
        return "bp"
    if re.search(r"spo2|sp02|o2\s*sat|oxygen\s*sat|saturation|%|percent", span, re.I):
        return "spo2"
    if re.search(r"temp|fahrenheit|celsius|degree|°", span, re.I):
        return "temp_f"
    if re.search(r"respiratory\s*rate|resp\b|respiration|breathing\s*rate", span, re.I):
        return "rr"
    if re.search(r"pulse|heart\s*rate|\bHR\b", span, re.I):
        return "pulse"
    if re.search(r"glucose|sugar|BGL|RBS|CBG|mg\s*/\s*d[lL]|mmol"
                 r"|milligrams?\s+per\s+decilit(?:re|er)"
                 r"|millimoles?\s+per\s+lit(?:re|er)", span, re.I):
        return "glucose"
    return None


# A Celsius marker attached to the number itself: "38 degrees Celsius",
# "38 °C", "38 C". The digit prefix is what keeps it from firing on a stray
# capital C elsewhere in the span, and `(?![A-Za-z])` stops a bare `C` from
# matching the first letter of `Celsius`'s neighbours or of `cardiac`.
_CELSIUS = re.compile(r"\d\s*(?:°\s*|degrees?\s*)?(?:celsius\b|centigrade\b|C(?![A-Za-z]))", re.I)


def _temp_scale(span):
    """`temp_c` when the span says Celsius, else `temp_f`.

    Unmarked stays Fahrenheit: that is the scale this project's gold data and
    its charts are written in, and treating an unmarked number as "whichever
    scale makes it plausible" would retire the check entirely.
    """
    return "temp_c" if _CELSIUS.search(span) else "temp_f"


def vital_span_plausible(span):
    """(ok, note) — False only when a number in the span is outside any human range."""
    kind = _vital_span_kind(span)
    if kind == "temp_f":
        kind = _temp_scale(span)
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", span)]
    if kind == "bp" and len(nums) >= 2:
        lo_sys, hi_sys = VITALS_RANGES["bp_sys"]
        lo_dia, hi_dia = VITALS_RANGES["bp_dia"]
        if not (lo_sys <= nums[0] <= hi_sys and lo_dia <= nums[1] <= hi_dia):
            return False, (f"physiologically implausible BP {nums[0]:g}/{nums[1]:g} "
                           "- human must confirm")
        return True, ""
    if kind in VITALS_RANGES and nums:
        lo, hi = VITALS_RANGES[kind]
        bad = [n for n in nums if not (lo <= n <= hi)]
        if bad:
            return False, (f"physiologically implausible {kind} "
                           f"{', '.join(f'{n:g}' for n in bad)} - human must confirm")
    return True, ""


def box_value_plausible(box_key, text):
    """Box-level check for coords: box_key is sys/dia/temp/spo2."""
    m = re.search(r"\d+(?:\.\d+)?", text or "")
    if not m:
        return True
    v = float(m.group(0))
    # The temp box keeps the unit it was cut from, so the same scale test the
    # sentence-level check uses applies here too.
    ranges = {"sys": VITALS_RANGES["bp_sys"], "dia": VITALS_RANGES["bp_dia"],
              "temp": VITALS_RANGES[_temp_scale(text or "")],
              "spo2": VITALS_RANGES["spo2"]}
    lo, hi = ranges.get(box_key, (float("-inf"), float("inf")))
    return lo <= v <= hi


NEG_PAT = re.compile(r"\b(no|not|denies|denied|denying|without|never|neither|nor|negative for|no known|don't|doesn't|aren't|isn't|wasn't|weren't|can't|cannot)\b", re.I)
DENIED_SYMPTOMS = ("chest pain", "breathlessness")
FREQ_PAT = (r"BID|OD|TDS|QID|once daily|twice daily|thrice daily|daily|B\.I\.D\.?"
            r"|twice a day|two times a day|thrice a day|three times a day"
            r"|every\s+\d+\s*(?:hours?|hrs?|minutes?|mins?)|q\d+h"
            r"|as needed|as\s+required|PRN|S\.?O\.?S\.?|stat|BD|TID")
DRUG_CTX = re.compile(
    r"(?P<name>[A-Za-z]+),?\s+(?:for\s+)?(?P<dose>\d+(?:\.\d+)?)\s*(?P<unit>mg|mcg|g|ml|U|units?|milligrams?|micrograms?|grams?|milliliters?)\b"
    r"(?:\s*(?P<freq>" + FREQ_PAT + r"))?"
    r"(?:[^.]{0,60}?(?P<dur>\d+)\s*(?P<durunit>days?|din\b|weeks?|months?|hours?|hrs?|minutes?|mins?))?",
    re.I,
)
# Split multi-drug sentences: "amox ... along with ibuprofen ..." -> two clauses
DRUG_SPLIT_PAT = re.compile(r"\s+(?:along with|alongwith|plus|and also|\+)\s+", re.I)
# Generic allergy mention: "allergy to <word>" (candidate, YELLOW; negated flag from NEG_PAT)
ALLERGY_CTX = re.compile(
    r"allerg(?:y|ies|ic)\s+(?:to|of|from)\s+(?:a\s+|an\s+|the\s+)?(?P<drug>[A-Za-z]+)", re.I)
# The other half of the grammar: the drug comes FIRST and the word "allergy"
# follows it - "document amoxicillin as a suspected drug allergy", "she has a
# penicillin allergy". Without this the only allergies the record can ever
# carry are negated ones, because `ALLERGY_CTX` is the sole producer and the
# block that runs it is entered only on a negated sentence.
ALLERGY_ASSERT = re.compile(
    r"\b(?P<drug>[A-Za-z][A-Za-z-]{2,})\s+"
    r"(?:as\s+(?:an?\s+|the\s+)?)?"
    r"(?:suspected\s+|possible\s+|probable\s+|confirmed\s+|known\s+|severe\s+)*"
    r"(?:drug\s+|medication\s+|food\s+)?allerg(?:y|ies)\b",
    re.I,
)
# Words that precede "<x> allergy" without being the substance.
ALLERGY_STOPWORDS = {"a", "an", "the", "any", "no", "this", "that", "his", "her",
                     "their", "my", "your", "its", "drug", "medication", "food",
                     "suspected", "possible", "probable", "confirmed", "known",
                     "severe", "previous", "other", "and", "or", "with", "of",
                     # quantifiers and bare nouns: "allergic to a lot of things"
                     "nothing", "anything", "something", "everything", "lot",
                     "lots", "many", "several", "few", "things", "stuff",
                     "medications", "medicines", "drugs", "foods", "antibiotics"}
# A contrast marker ends a negation's reach: "does not have asthma BUT has a
# penicillin allergy" asserts the allergy. A comma does not - "no history of
# asthma, diabetes, or penicillin allergy" denies all three.
_CONTRAST = re.compile(r"\b(?:but|however|although|though|except|whereas)\b|;", re.I)
UCUM = {"milligram": "mg", "milligrams": "mg", "microgram": "mcg", "micrograms": "mcg",
        "gram": "g", "grams": "g", "milliliter": "ml", "milliliters": "ml",
        "units": "U", "unit": "U"}
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
         "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
         "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
         "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
         "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_NUMW = sorted(list(_ONES) + list(_TENS) + ["hundred", "and"], key=len, reverse=True)
_NUMW_RUN = re.compile(r"\b(?:%s)(?:[ -](?:%s))*\b" % ("|".join(_NUMW), "|".join(_NUMW)), re.I)


def _parse_numwords(toks):
    had_and = "and" in toks
    toks = [t for t in toks if t != "and"]
    # Digit-by-digit dictation, which is how a strength or a BP is read aloud:
    # "six two five" is 625, not 6+2+5.  Summing a run of single digits turned
    # `amoxicillin six two five milligram` into a 13 mg row at GREEN - a 48x
    # underdose carrying the pipeline's highest confidence.  Ten through
    # nineteen are excluded by the `<= 9` test, so "one ten" -> 110 and
    # "two fifteen" -> 215 still reach the spoken-hundreds rule below.
    if len(toks) >= 2 and all(t in _ONES and _ONES[t] <= 9 for t in toks):
        if had_and:
            # "one and two" is a conjunction, not a number.  Returning 0 leaves
            # the original words in place, so the dose fails to parse and the
            # row goes RED rather than inventing a value either way.
            return 0
        return int("".join(str(_ONES[t]) for t in toks))
    # "one thirty" -> 130, and "one ten" / "two fifteen" -> 110 / 215: a spoken
    # three-digit number drops the word "hundred" whether the remainder is a
    # tens word or a teen.
    if (len(toks) >= 2 and toks[0] in _ONES and 1 <= _ONES[toks[0]] <= 9
            and (toks[1] in _TENS or (toks[1] in _ONES and _ONES[toks[1]] >= 10))):
        total, toks = _ONES[toks[0]] * 100, toks[1:]
    else:
        total = 0
    cur = 0
    for w in toks:
        if w == "hundred":
            cur = (cur or 1) * 100
        elif w in _TENS:
            cur += _TENS[w]
        elif w in _ONES:
            cur += _ONES[w]
    return total + cur


# A spoken decimal is two integer runs joined by "point": "ninety-eight point
# six" -> 98.6. Without this the integers convert and the separator survives as
# the word "point", so the temperature never reaches a numeric field.
_SPOKEN_POINT = re.compile(r"(?<![\d.])(\d{1,3})\s+point\s+(\d{1,2})\b(?!\s*\d)", re.I)


# "for two three days" is a range - two to three days - not the number 23.
# Indian-English dictation says it constantly. Summing gave 5 and concatenating
# gives 23; both are wrong, and 23 is the more dangerous of the two, so the
# words are left alone and the duration simply does not parse.
# Dose units are included: "two three milligram" is as likely 2-3 mg as it is
# 23 mg, and no case this fix needs to catch has two consecutive ascending
# digits before a unit - "six two five" is three tokens, and the "eight zero"
# of a diastolic is not consecutive.
_RANGE_TAIL = re.compile(r"^\s*(?:days?|din\b|weeks?|months?|hours?|hrs?|minutes?"
                         r"|mins?|times?|tablets?|tabs?|doses?"
                         r"|mg|mcg|g|ml|units?|milligrams?|micrograms?|grams?"
                         r"|milliliters?)\b", re.I)


def numwords_to_digits(text):
    """Dictation words -> digits: five-hundred milligram -> 500 milligram."""
    def rep(m):
        toks = re.split(r"[ -]", m.group(0).lower())
        toks = [t for t in toks if t]
        if not any(t in _ONES or t in _TENS or t == "hundred" for t in toks):
            return m.group(0)
        # Two consecutive digits followed by a unit of time or count is a
        # spoken range, not a two-digit number.
        if (len(toks) == 2 and all(t in _ONES and _ONES[t] <= 9 for t in toks)
                and _ONES[toks[1]] == _ONES[toks[0]] + 1
                and _RANGE_TAIL.match(text[m.end():])):
            return m.group(0)
        v = _parse_numwords(toks)
        return str(v) if v > 0 else m.group(0)
    return _SPOKEN_POINT.sub(r"\1.\2", _NUMW_RUN.sub(rep, text))


def _load_drug_list():
    for p in (os.path.join(_HERE, "assets", "drug_list_mini.json"),
              os.path.join(_HERE, "..", "demo", "assets", "drug_list_mini.json")):
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    return [{"name": "paracetamol"}, {"name": "azithromycin"}]


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
    import torch
    from transformers import pipeline as hf_pipeline

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
    raw_text = str(pipe(str(clean_wav), chunk_length_s=20, stride_length_s=2)["text"])
    text = medasr_detokenize(raw_text)
    with contextlib.closing(wave.open(str(clean_wav), "rb")) as handle:
        duration_s = handle.getnframes() / float(handle.getframerate())
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


def transcribe(clean_wav, job_id="demo-001", model="medasr", strict: bool = False):
    """Transcribe with explicit mock fixtures or strict/fallback real STT."""
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
            from medasr_lm import transcribe_medasr_lm
            return transcribe_medasr_lm(clean_wav, job_id)
        except Exception as exc:
            if strict:
                raise
            return _mock_result(clean_wav, job_id, model, exc, "no medasr-lm")
    try:
        import faster_whisper as faster_whisper_module
        from faster_whisper import WhisperModel
        size = {"tiny-int8": "tiny", "base-int8": "base", "small-int8": "small",
                "medium": "medium", "large-v3": "large-v3"}.get(model, "tiny")
        try:
            import torch as _torch
            _has_cuda = _torch.cuda.is_available()
        except Exception:
            _has_cuda = False
        if model in ("medium", "large-v3") and _has_cuda:
            try:
                wm = WhisperModel(size, device="cuda", compute_type="float16")
                _device, _compute = "cuda", "float16"
            except Exception:
                # ctranslate2 build may want a libcublas older than the
                # system CUDA (e.g. .so.12 vs installed .so.13) - fall back.
                wm = WhisperModel(size, device="cpu", compute_type="int8")
                _device, _compute = "cpu", "int8"
        else:
            wm = WhisperModel(size, device="cpu", compute_type="int8")
            _device, _compute = "cpu", "int8"
        segments, info = wm.transcribe(
            clean_wav, beam_size=1, word_timestamps=True, temperature=0.0,
            no_speech_threshold=0.6, compression_ratio_threshold=2.4,
        )
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
            "temperature": 0.0,
            "beam_size": 1,
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



def detect_lang_tag(text):
    hi = len(re.findall(r"[\u0900-\u097F]", text))
    ta = len(re.findall(r"[\u0B80-\u0BFF]", text))
    hinglish = bool(re.search(r"\b(ko|hai|din|tak|do time|bukhar|khansi)\b", text, re.I))
    scripts = sum([hi > 0, ta > 0, bool(re.search(r"[A-Za-z]", text))])
    if scripts > 1 or (hinglish and re.search(r"[A-Za-z]", text)):
        return "mix"
    if ta:
        return "ta-IN"
    if hi:
        return "hi-IN"
    return "en-IN"


def normalize_text(text):
    """Stage 3: Hinglish map + UCUM-ish unit spacing + mix tag."""
    normalizations = []
    out = text
    for src, dst in HINGLISH_MAP:
        # word-boundary replace stops OD matching inside blood/method
        pat = r"\b" + re.escape(src) + r"\b"
        if re.search(pat, out, flags=re.I):
            out = re.sub(pat, dst, out, flags=re.I)
            normalizations.append({"from": src, "to": dst})
    out = re.sub(r"(\d+)\s*(mg|mcg|ml|g)\b", r"\1 \2", out)
    out = re.sub(r"\bB[\s-]*P\b", "BP", out)  # B-P -> BP
    # Narrow phonetic: BPA + digits -> BP (ward STT mishear, e.g. "BPA 130 by AC")
    if re.search(r"\bBPA\s*\d", out, flags=re.I):
        out = re.sub(r"\bBPA(?=\s*\d)", "BP", out, flags=re.I)
        normalizations.append({"from": "BPA", "to": "BP"})
    # Narrow phonetic: "<sys> by AC" -> "<sys> by 80" (AC misheard for eighty)
    if re.search(r"\d\s*(?:by|/|over|of)\s*AC\b", out, flags=re.I):
        out = re.sub(r"(\d\s*(?:by|/|over|of)\s*)AC\b", r"\g<1>80", out, flags=re.I)
        normalizations.append({"from": "by AC", "to": "by 80"})
    # Room-air mishear: Rhoomair -> room air (SpO2 sentence only)
    if re.search(r"rhoomair", out, flags=re.I):
        out = re.sub(r"rhoomair", "room air", out, flags=re.I)
        normalizations.append({"from": "Rhoomair", "to": "room air"})
    before_spo2 = out
    out = re.sub(r"\bS[\s-]*P[\s-]*O[\s-]*(two|2)\b", "SpO2", out, flags=re.I)
    if out != before_spo2:
        normalizations.append({"from": "S-P-O-two", "to": "SpO2"})
    # q8h-style shorthand -> words before digitization
    if re.search(r"\bq(\d+)h\b", out, flags=re.I):
        out = re.sub(r"\bq(\d+)h\b", r"every \1 hours", out, flags=re.I)
        normalizations.append({"from": "qNh", "to": "every N hours"})
    # BD/TID shorthand (word-boundary safe)
    for _src, _dst in (("BD", "twice daily"), ("TID", "thrice daily")):
        if re.search(r"\b" + _src + r"\b", out):
            out = re.sub(r"\b" + _src + r"\b", _dst, out)
            normalizations.append({"from": _src, "to": _dst})
    out = numwords_to_digits(out)  # dictation words -> digits for extraction
    out = re.sub(r"(\d+(?:\.\d+)?)\s*(by|over)\s*(\d+(?:\.\d+)?)", r"\1/\3", out)  # BP 130 by/over 80
    # 'of' separator ONLY in BP context ( "... BP ... 122 of 78" / "blood pressure ... 122 of 78" )
    out = re.sub(
        r"((?:BP|blood pressure)[^.]{0,40}?\d+(?:\.\d+)?)\s+of\s+(\d+(?:\.\d+)?)",
        r"\1/\2", out, flags=re.I,
    )
    tag = detect_lang_tag(text)
    return {"normalized_en": out, "normalizations": normalizations, "lang_tag": tag}


def _negated(sentence):
    return bool(NEG_PAT.search(sentence))


def _negated_before(prefix):
    """Is the substance at the end of `prefix` denied by something before it?

    Sentence-level negation is the wrong test for an allergy. "Document X as a
    suspected allergy so that it is NOT given again" asserts; "she does NOT have
    asthma BUT has a penicillin allergy" asserts; "NO history of asthma,
    diabetes, or penicillin allergy" denies. So the scope runs from the last
    contrast marker to the substance, and commas do not break it.
    """
    markers = list(_CONTRAST.finditer(prefix))
    return _negated(prefix[markers[-1].end():] if markers else prefix)


# An imperative verb starts a new order, and that ends the previous clause's
# negation.  `_CONTRAST` deliberately lets commas through, because "no history
# of asthma, diabetes, or penicillin allergy" denies all three - but a denial
# followed by an order is the common ward sentence, and treating the boundary
# as transparent cancels a real prescription.  "Stop amoxicillin and start
# azithromycin" is the case that matters most: a drug switch, where getting
# this wrong drops the drug the patient is now on.
_RX_VERB = re.compile(
    r"\b(?:give|giving|given|gave|start|started|starting|begin|begun|continue"
    r"|continued|continuing|prescribe|prescribed|add|added|administer"
    r"|administered|take|takes|taking|switch(?:ed)?\s+to|change(?:d)?\s+to)\b",
    re.I)
# Words that may sit between a negation and the verb it governs. "do not give"
# and "not to give" are negated orders; "no fever give" is a denial followed by
# one, and the noun in between is what tells them apart.
_AUX_ONLY = re.compile(r"^[\s,]*(?:(?:do|does|did|to|be|been|being|is|are|was"
                       r"|were|should|shall|must|will|would|can|could|may"
                       r"|please|kindly)[\s,]+)*$", re.I)
# Stopping a drug without a negation particle. `NEG_PAT` has no "stop", so
# "stop metformin 500 mg" charted as an active GREEN order.
_DISCONTINUE = re.compile(
    r"\b(?:stop|stopped|stopping|avoid|avoided|withhold|withheld|hold|held"
    r"|discontinue|discontinued|omit|omitted|cease|ceased"
    r"|refus(?:e|ed|ing)|declin(?:e|ed|ing))\b", re.I)


def _order_scope(prefix):
    """`prefix` trimmed to the current order, i.e. after the last fresh verb.

    A verb only starts a *fresh* order when a negation is not governing it, so
    the scan skips any verb whose negation sits immediately before it with
    nothing but an auxiliary between.  That keeps "do not give X" inside the
    negation while letting "...no fever, give X" out of it.
    """
    cut = 0
    for m in _RX_VERB.finditer(prefix):
        # Either kind of stop word can govern the verb.  Checking only
        # `NEG_PAT` left the commonest phrasing of all wide open: in "stop
        # taking metformin 500 mg" the verb is `taking`, no negation particle
        # precedes it, so the scope reset past "stop" and the row charted as an
        # active GREEN order.
        before = prefix[:m.start()]
        stops = list(NEG_PAT.finditer(before)) + list(_DISCONTINUE.finditer(before))
        last = max((x.end() for x in stops), default=None)
        if last is None or not _AUX_ONLY.match(prefix[last:m.start()]):
            cut = m.end()
    return prefix[cut:]


def _drug_negated(prefix):
    """Is a drug at the end of `prefix` prohibited, stopped or refused?

    Two sources, both scoped the same way - to the current order, and then to
    the text after the last contrast marker.  Scoping them alike is the point:
    an earlier draft ran the discontinuation check over the whole prefix while
    the negation check respected contrast, so "stopped metformin but continue
    insulin 10 U" marked the insulin stopped.
    """
    scope = _order_scope(prefix)
    markers = list(_CONTRAST.finditer(scope))
    if markers:
        scope = scope[markers[-1].end():]
    return _negated(scope) or bool(_DISCONTINUE.search(scope))


def _prefer_dosed(drugs):
    """One row per drug, and the dosed row wins.

    A drug can be named twice - once bare ("continuing ceftriaxone") and once
    with its dose. Whichever is dictated first, the row carrying the dose is the
    one the record needs, so the bare row is dropped rather than left to shadow
    it.
    """
    best = {}
    for drug in drugs:
        name = drug["name"]
        if name not in best or (best[name].get("dose") is None and drug.get("dose") is not None):
            best[name] = drug
    return list(best.values())


# `DRUG_CTX`'s name group is a single token, so a two-word medicine reaches
# the matcher as its last word only: "clavulanic acid 125 mg" offers `acid`,
# and "folic acid 5 mg" offers `acid` and is dropped outright. Both are orders
# whose dose was dictated, and what the chart showed was either no row at all
# or a row saying "no dose dictated" - a statement about the recording that is
# simply false.
#
# The name is grown leftwards after the match rather than widened in the
# pattern. A greedy multi-token name group would make the common cases worse,
# offering "was 200 mg" and "and paracetamol 500 mg" as drugs called `was` and
# `and paracetamol`. So an extension is accepted only when the joined words
# are themselves a name the vocabulary resolves - the mini list, or an exact
# substance in the national pack, with the pack's brand and stopword gates
# applied. That is the same bar `_reference_only_row` uses downstream, so a
# longer match can never invent a drug: it can only recognise one that would
# equally have been accepted had it been dictated as a single word.
MAX_NAME_WORDS = 3
_WORD_BEFORE = re.compile(r"([A-Za-z]+)\s+$")
# The word immediately before a multi-word name, however it is joined -
# "amoxicillin and clavulanic acid", "amoxicillin clavulanic acid",
# "amoxicillin-clavulanic acid". When that word is itself a drug, the 625 mg
# that follows is the combination's total strength, not this component's dose.
_COMBINATION_JOIN = re.compile(
    r"([A-Za-z]+)\s*(?:[-/,+]\s*)?(?:(?:and|with|plus)\s+)?$", re.I)


def _name_resolves(name, drugs_known, alias_to_canonical):
    """True when `name` is a medicine this pipeline already knows by name."""
    if name in drugs_known or name in alias_to_canonical:
        return True
    try:
        import drug_reference
    except ImportError:
        return False
    return bool(drug_reference.lookup(
        name, block=drug_reference.STOPWORDS, brands=False))


def _extend_name(clause, start, end, drugs_known, alias_to_canonical):
    """(name, start, extended) - the longest resolvable name ending at `end`.

    Longest-first, so "sodium valproate" wins over "valproate" where both
    resolve. Falls back to the single token the pattern matched, leaving every
    one-word case exactly as it was.
    """
    starts, at = [], start
    while len(starts) < MAX_NAME_WORDS - 1:
        m = _WORD_BEFORE.search(clause, 0, at)
        if not m:
            break
        at = m.start(1)
        starts.append(at)
    for s in reversed(starts):
        cand = " ".join(clause[s:end].split()).lower()
        if _name_resolves(cand, drugs_known, alias_to_canonical):
            return cand, s, True
    return clause[start:end].lower(), start, False


def _combination_partner(clause, start, drugs_known, alias_to_canonical):
    """The drug this one is conjoined to, when a shared dose follows both."""
    m = _COMBINATION_JOIN.search(clause, 0, start)
    if not m:
        return None
    other = m.group(1).lower()
    return other if _name_resolves(other, drugs_known, alias_to_canonical) else None


def _drug_name_pattern(alias_to_canonical):
    """One word-boundary alternation over every known name and alias."""
    names = sorted(alias_to_canonical, key=len, reverse=True)
    return re.compile(r"\b(?:%s)\b" % "|".join(re.escape(n) for n in names), re.I)


_ALLERGY_NEAR = re.compile(r"allerg(?:y|ies|ic)", re.I)


def _allergy_spans(sentence):
    """(start, end) of every substance an allergy pattern actually captures.

    A proximity window was tried first and was too blunt: it dropped the insulin
    from "he takes insulin and has no known drug allergies". Only a word the
    allergy rules really claimed is withheld from the drug table.
    """
    return [(m.start("drug"), m.end("drug"))
            for pattern in (ALLERGY_CTX, ALLERGY_ASSERT)
            for m in pattern.finditer(sentence)]


def _overlaps(spans, start, end):
    return any(start < s_end and s_start < end for s_start, s_end in spans)

def _annotate_entity_span(entity, normalized_en):
    anchor = next(
        (entity.get(key).strip() for key in ("name", "text", "span")
         if isinstance(entity.get(key), str) and entity.get(key).strip()),
        None,
    )
    matches = list(re.finditer(re.escape(anchor), normalized_en, re.I)) if anchor else []
    if len(matches) == 1:
        entity.update(start_char=matches[0].start(), end_char=matches[0].end(),
                      span_status="exact")
    else:
        entity.update(start_char=-1, end_char=-1,
                      span_status="ambiguous" if matches else "not_found")
    return entity


def _annotate_entity_spans(entities, normalized_en):
    for group in ("drugs", "symptoms", "vitals", "allergies", "negations"):
        for entity in entities.get(group, []):
            if isinstance(entity, dict):
                _annotate_entity_span(entity, normalized_en)
    for group in ("diagnosis", "followup"):
        entity = entities.get(group)
        if isinstance(entity, dict) and entity:
            _annotate_entity_span(entity, normalized_en)
    return entities




def _fuzzy_candidates(token, drugs_known, alias_to_canonical, floor, limit=5, band=0.08):
    """Every canonical drug a misheard token could plausibly be, best first.

    The matcher used to keep only difflib's winner, which threw away exactly the
    information a validator needs: whether the runner-up was a near tie.
    Sound-alike pairs are the dangerous case, so candidates within ``band`` of
    the best score are kept even when they sit under ``floor`` - a tie that the
    validator must see rather than a match it should act on.  Ordering is
    ``(-score, name)`` so it does not depend on set iteration order, which the
    previous loop did.
    """
    import difflib
    vocab = sorted(set(drugs_known) | set(alias_to_canonical))
    scored = sorted(((difflib.SequenceMatcher(None, token, known).ratio(), known)
                     for known in vocab), key=lambda pair: (-pair[0], pair[1]))
    if not scored or scored[0][0] < floor:
        return []
    cutoff = scored[0][0] - band  # deliberately not clamped to `floor`
    out, seen = [], set()
    for score, known in scored:
        if score < cutoff:
            break
        name = alias_to_canonical.get(known, known)
        if name in seen:
            continue
        seen.add(name)
        out.append({"name": name, "score": round(score, 3), "matched_alias": known})
        if len(out) == limit:
            break
    return out


# How far a national-reference name must out-score the best curated mini-list
# candidate before it is allowed to lead the row.  A guard against cheap
# high scores in a 74k-name table, not a measured threshold: it sits above the
# 0.05 that made `oxacillin` outrank `amoxicillin` and below the 0.11 that
# makes `hydroxyzine` the clear answer over `levothyroxine`.
REFERENCE_MARGIN = 0.08


# "glucose 180 mg/dL" is a lab result, not a prescription, but `DRUG_CTX`'s
# `\b` after the unit matches happily before the slash and puts the analyte in
# the drug-name slot.  The mini list used to drop those rows by not recognising
# `glucose`; the national reference does recognise it, and would file four
# false medicines off one line of bloodwork.  A dose is an amount, a lab value
# is a concentration, and `per decilitre` or `per litre` is what tells them
# apart - `mg/ml` and `mg/kg` stay, because those really are how drugs are
# ordered.
_LAB_DENOM = re.compile(r"\s*(?:/|per\s+)\s*(?:d[lL]\b|L\b|lit(?:re|er)s?\b"
                        r"|decilit(?:re|er)s?\b|cu\s*mm\b|mm3\b)")


def _is_lab_concentration(text, match):
    """True when the matched `<word> <num> <unit>` is a lab value, not a dose."""
    return bool(_LAB_DENOM.match(text, match.end("unit")))


def _demote_implausible(candidates, dose, unit):
    """Re-rank candidates the pack says this dose cannot belong to.

    Demoted rather than deleted: the pack's strength table is a record of what
    is marketed, not a statement of every dose a clinician may write, so a
    candidate it cannot account for drops to the back of the queue instead of
    vanishing.  The ordering is what reaches the chart, so moving the
    implausible ones down is enough to stop them leading a row, while the
    validator still sees them.
    """
    try:
        import drug_reference
    except ImportError:
        return candidates, [], False
    ruled = []
    for c in candidates:
        ref = c.get("reference") or {}
        rec = {"units": ref.get("units", []), "mass_mg": ref.get("mass_mg")}
        if (drug_reference.unit_fits(rec, unit) is False
                or drug_reference.dose_conflicts(rec, dose, unit)):
            c["dose_implausible"] = True
            ruled.append(c["name"])
    all_bad = bool(ruled) and len(ruled) == len(candidates)
    if not ruled or all_bad:
        # Nothing to demote, or everything is implausible - reordering a
        # uniformly doubtful shortlist would only hide that fact.
        return candidates, ruled, all_bad
    candidates = sorted(candidates, key=lambda c: (bool(c.get("dose_implausible")),))
    return candidates, ruled, False


def _ref_block(rec):
    """The pack's facts about one drug, as they travel on a candidate row.

    `mass_mg` is the pair the dose check reads, so it has to survive the trip:
    without it every candidate looks like one the pack has no opinion on, and
    the check silently passes everything.
    """
    return {"kind": rec["kind"], "name": rec["name"],
            "substances": rec.get("substances", []),
            "units": rec.get("units", []),
            "mass_mg": rec.get("mass_mg"),
            "forms": rec.get("forms", []),
            "strengths": rec.get("strengths", [])[:6]}


def _reference_candidates(token, mini_candidates, drugs_known=frozenset()):
    """Merge the national drug reference into the mini-list's candidate set.

    The mini list is 33 medicines, so its top scorer is "the nearest of 33
    strings" - right for `asitromaisin`, wrong for `hydroxazine`, where the
    real drug is simply not in the list and `levothyroxine` is merely the
    closest thing that is.  The reference supplies the missing referent.

    The two matchers are not equally trustworthy and are not merged as equals.
    The mini list is curated: its aliases are mishearings someone observed and
    wrote down, and it is the vocabulary this project's gold data is written
    in.  The reference is raw string proximity over 74k names, where a
    near-perfect score is cheap - `moxacillin` scores 0.95 against `oxacillin`
    and 0.90 against the mini list's `amoxicillin`, and the higher number is
    the wrong antibiotic.  So a reference name takes the lead only when it
    beats the best mini-list candidate by `REFERENCE_MARGIN`, which
    `hydroxazine -> hydroxyzine` (0.91 against levothyroxine's 0.80) clears and
    `moxacillin -> oxacillin` does not.

    Losing that contest costs a candidate nothing but first place: it stays in
    the shortlist, so `term_validate` still sees both and can rule on which
    drug the sentence is about.  Scores are left as each matcher produced them
    so the tie rule still sees the real spread.
    """
    try:
        import drug_reference
    except ImportError:
        return mini_candidates
    if not drug_reference.available():
        return mini_candidates
    # Enrich the mini list's own candidates first. They are the ones most
    # likely to be wrong in the dangerous way - `levothyroxine` reached the
    # shortlist because it is the nearest of 33 strings - and without the
    # pack's units attached to them the unit check downstream has nothing to
    # test and silently passes everything.
    out = []
    for c in mini_candidates:
        rec = drug_reference.lookup(c["name"])
        out.append({**c, "source": "mini",
                    **({"reference": _ref_block(rec)} if rec else {})})
    # Keyed on substances as well as names so a bridged INN spelling cannot
    # arrive as a rival of the mini list's own entry for the same molecule -
    # two names for one drug would otherwise read as a tie.
    known = {c["name"].lower() for c in out}
    for c in out:
        known.update(x.lower() for x in (c.get("reference") or {}).get("substances", []))
    for rec in drug_reference.candidates(token):
        # A reference brand resolves to its generic where the mini list already
        # knows that generic, so `azee` does not arrive as a rival of
        # `azithromycin`.
        name = _chart_name(rec["name"], drugs_known)
        for sub in rec.get("substances", []):
            if _chart_name(sub, drugs_known) in known:
                name = _chart_name(sub, drugs_known)
                break
        if name.lower() in known:
            continue
        known.add(name.lower())
        out.append({"name": name, "score": rec["score"],
                    "matched_alias": rec["matched_name"], "source": "reference",
                    "reference": _ref_block(rec)})
    # Rank mini-list candidates first among themselves, then let a reference
    # name jump the queue only on a decisive margin.
    mini_names = {c["name"].lower() for c in mini_candidates}
    best_mini = max((c["score"] for c in out if c["name"].lower() in mini_names),
                    default=None)

    def rank(c):
        from_mini = c["name"].lower() in mini_names
        if from_mini or best_mini is None:
            promoted = from_mini
        else:
            promoted = c["score"] >= best_mini + REFERENCE_MARGIN
        return (0 if promoted else 1, -c["score"], c["name"])

    out.sort(key=rank)
    return out[:6]


def _chart_name(name, drugs_known):
    """The spelling this project charts a pack name under.

    Two authorities, in order.  The mini list wins outright: it is this
    project's own vocabulary and its gold data is written in it, so a pack
    name it already carries is charted exactly as it carries it - `furosemide`
    stays `furosemide` and is not rewritten to `frusemide` on the strength of
    a synonym table.  Only for a drug the mini list has never heard of does
    the INN bridge decide, which is what keeps a `Crocin` tablet from charting
    as `acetaminophen`.
    """
    n = str(name or "").strip().lower()
    if n in drugs_known:
        return n
    try:
        import drug_reference
    except ImportError:
        return n
    return drug_reference.inn_name(n)


def _reference_only_row(token, drugs_known=frozenset()):
    """An exact reference hit for a token the mini list has never heard of.

    The matcher used to drop these: no mini-list hit and no fuzzy match above
    the floor meant `continue`, and a real medicine dictated perfectly
    vanished from the chart because a 33-entry demo list did not contain it.
    A name the national pack lists exactly is a medicine, so the row is
    emitted - YELLOW rather than GREEN, because it reached the chart through
    a list this pipeline has not curated and nobody has reviewed the match.
    """
    try:
        import drug_reference
    except ImportError:
        return None
    # The exact-hit path gets the same gate as the fuzzy one. Without it a
    # word like `level` or `din` - both registered Indian brands - becomes a
    # medicine the moment a dose follows it.
    # Substances only. A brand name that happens to be an ordinary English
    # word must not become a medicine just because a dose follows it, and 161
    # of the English words in this project's own ASR output are brands.
    rec = drug_reference.lookup(token, block=drug_reference.STOPWORDS, brands=False)
    if not rec:
        return None
    # A brand resolves to its single generic; a multi-substance combination
    # keeps the brand name, since no one generic names it.
    subs = rec.get("substances", [])
    name = subs[0] if rec["kind"] == "brand" and len(subs) == 1 else rec["name"]
    # The pack is named in USAN and Indian charts are written in INN, so a
    # clinician who dictated `frusemide` must not read `furosemide` back off
    # the note. The spoken spelling is the one that goes on the chart; the
    # pack's own name is kept in the reference block for traceability.
    t = str(token).strip().lower()
    if t != name and drug_reference.load()["aliases"].get(t) == name:
        name = t
    return _chart_name(name, drugs_known), rec


def _split_clauses(sent):
    """Multi-drug clauses as `(text, offset)`, offset measured in `sent`.

    The offset is what lets a negation keep its reach across the split: "do not
    give amoxicillin along with ibuprofen" denies both drugs, and a clause
    examined on its own would have lost the "not" that governs it.
    """
    out, at = [], 0
    for m in DRUG_SPLIT_PAT.finditer(sent):
        out.append((sent[at:m.start()], at))
        at = m.end()
    out.append((sent[at:], at))
    return out


# A sentence that orders laboratory work is not prescribing. "Blood samples
# have been sent to check creatinine, urea, sodium and potassium levels" names
# four substances, three of which the national pack lists as real medicines -
# potassium and sodium are genuinely prescribed, just not here. Rather than
# block those names globally (which would lose a real potassium order), the
# sentence itself is read: a lab-request cue suppresses reference-only doseless
# rows within it. The curated mini list is exempt, because a name someone put
# in this project's own formulary is a deliberate choice.
_LAB_REQUEST = re.compile(
    r"\b(?:sent|send|sending|collected|check(?:ing|ed)?|request(?:ed)?|advised"
    r"|ordered)\b[^.]{0,80}?\b(?:sample|samples|test|tests|level|levels|profile"
    r"|culture|screening|panel|analysis|function)\b"
    r"|\b(?:sample|samples|test|tests|level|levels|profile|culture|screening"
    r"|panel)\b[^.]{0,40}?\b(?:sent|send|collected|requested|advised)\b", re.I)


_LAB_VALUE_AFTER = re.compile(
    r"\s*(?:is|was|of|at)?\s*\d+(?:\.\d+)?\s*"
    r"(?:mg|mcg|g|mmol|meq|units?)\s*(?:/|per\s+)\s*"
    r"(?:d[lL]\b|L\b|lit(?:re|er)s?\b|decilit(?:re|er)s?\b)", re.I)


def _reference_doseless(sent, taken, alias_to_canonical, drugs_known):
    """Doseless drug mentions the curated list has never heard of.

    `known_name_pat` is built from a 36-entry formulary, so a ward handover
    saying "nebulisation with salbutamol and ipratropium bromide" records only
    the half that list happens to contain.  The national pack carries 2,542
    substances, 628 of them multi-word, and consulting it here is what makes
    the doseless path generalise instead of being extended by hand every time
    an unseen clip names a drug the list lacks.

    Longest match wins, so "sodium valproate" is not recorded as "sodium".
    """
    try:
        import drug_reference
    except ImportError:
        return []
    if not drug_reference.available() or _LAB_REQUEST.search(sent):
        return []
    words = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[A-Za-z]+", sent)]
    out, consumed = [], set()
    for n in (3, 2, 1):
        for i in range(len(words) - n + 1):
            if any(j in consumed for j in range(i, i + n)):
                continue
            join = " ".join(w[0] for w in words[i:i + n]).lower()
            if join in alias_to_canonical or join in drugs_known or join in taken:
                continue
            rec = drug_reference.lookup(join, block=drug_reference.STOPWORDS,
                                        brands=False)
            if not rec:
                continue
            # "calcium 9 mg/dL" is a result, not an order. The dosed path
            # already refuses a per-volume concentration; the doseless path
            # needs the same test, because the analyte is a real medicine and
            # cannot simply be blocked by name.
            if _LAB_VALUE_AFTER.match(sent, words[i + n - 1][2]):
                continue
            consumed.update(range(i, i + n))
            out.append((drug_reference.inn_name(rec["name"]),
                        words[i][1], words[i + n - 1][2]))
    return out


def extract_entities(text, normalized_en, segments):
    """Stage 4 regex core. Returns demo entities_json."""
    drugs_known = {d["name"].lower() for d in _load_drug_list()}
    alias_to_canonical = {}
    for d in _load_drug_list():
        for a in [d["name"]] + d.get("aliases", []):
            alias_to_canonical[a.lower()] = d["name"].lower()
    known_name_pat = _drug_name_pattern(alias_to_canonical)

    drugs, symptoms, vitals, allergies, negations = [], [], [], [], []
    # cross-sentence join: "paracetamol,\n500 mg" -> "paracetamol 500 mg"
    # also "ibuprofen, for 100 mg" -> "ibuprofen for 100 mg" (second-drug clause)
    text = re.sub(r"([A-Za-z]+),\s+(?=(?:for\s+)?\d+\s*(?:mg|mcg|g|ml|U)\b)", r"\1 ", text)
    sents = SENT_SPLIT.split(text)
    # match on digitized copy, keep original wording as proof
    etext = numwords_to_digits(text)
    esents = SENT_SPLIT.split(etext)
    pairs = list(zip(esents, sents)) if len(esents) == len(sents) else [(s, s) for s in sents]
    sents = [p[0] for p in pairs]
    raws = [p[1] for p in pairs]
    diagnosis, followup = {}, {}

    for sent, raw in zip(sents, raws):
        neg = _negated(sent)
        proof_sent = raw.strip()
        allergy_spans = _allergy_spans(sent)
        # multi-drug split: "amox ... along with ibuprofen ..." -> separate clauses
        # so the second drug's dose/freq is not swallowed by the first row's duration window
        drug_clauses = _split_clauses(sent)
        for clause, clause_at in drug_clauses:
            for m in DRUG_CTX.finditer(clause):
                if _is_lab_concentration(clause, m):
                    continue
                rawm, name_at, extended = _extend_name(
                    clause, m.start("name"), m.end("name"),
                    drugs_known, alias_to_canonical)
                canon = alias_to_canonical.get(rawm)
                fuzzy_note, candidates = "", []
                ref_hit = None
                if canon is None and rawm not in drugs_known:
                    # An exact hit in the national reference is not a guess, so
                    # it short-circuits the sound-alike path entirely: no
                    # candidates, nothing for the validator to weigh.
                    exact = _reference_only_row(rawm, drugs_known)
                    if exact:
                        canon, ref_hit = exact
                        fuzzy_note = ("reference-only: in the national drug "
                                      "list, not in this pipeline's own list")
                    else:
                        # fuzzy sound-alike fallback: asitromaisin -> azithromycin (difflib stand-in)
                        floor = 0.55 if len(rawm) >= 8 else 0.6
                        candidates = _fuzzy_candidates(rawm, drugs_known, alias_to_canonical, floor)
                        candidates = _reference_candidates(rawm, candidates, drugs_known)
                        if candidates:
                            canon, score = candidates[0]["name"], candidates[0]["score"]
                            fuzzy_note = f"fuzzy {rawm}->{canon} ({score:.2f})"
                        else:
                            continue  # skip non-drug words; drug_list is the mini-RxNorm
                if _overlaps(allergy_spans, name_at, m.end("name")):
                    continue  # the allergy rules below own this mention
                dose = float(m.group("dose")) if m.group("dose") else None
                raw_unit = (m.group("unit") or "").strip()
                unit = UCUM.get(raw_unit.lower(), raw_unit) or None  # milligram->mg (UCUM)
                freq = (m.group("freq") or "").strip()
                dur = f"{m.group('dur')} {m.group('durunit')}" if m.group("dur") else ""
                if not dose or not unit:
                    conf, color = 0.70, "RED"  # dose/unit missing -> physician must fix
                else:
                    conf = 0.96
                    color = "GREEN"
                matcher_pick = canon
                dose_red = ""
                if candidates and dose and unit:
                    # The pack's dose check runs here, not only in the
                    # validator: `term_validate` needs Ollama, and a sentence
                    # whose dose is impossible for the matched drug must not
                    # depend on a model being pulled to say so.
                    candidates, ruled_out, all_bad = _demote_implausible(
                        candidates, dose, unit)
                    if all_bad:
                        # Every candidate is a drug this dose cannot belong to.
                        # Nothing here is safe to put on a chart.
                        dose_red = (f"dose {('%g' % dose)} {unit} implausible for "
                                    f"every candidate: {', '.join(ruled_out)}")
                    elif ruled_out and candidates[0]["name"] != canon:
                        # The pack overruled the string matcher. Two signals
                        # disagreeing about which drug was said is the
                        # sound-alike hazard itself, and `term_validate`
                        # sends that to a human rather than to a chart.
                        dose_red = (f"{canon} ruled out on dose; pack prefers "
                                    f"{candidates[0]['name']}")
                        canon = candidates[0]["name"]
                    elif ruled_out:
                        fuzzy_note = (f"{fuzzy_note}; dose implausible for "
                                      f"{', '.join(ruled_out)}")
                # "amoxicillin and clavulanic acid 625 mg" is one combination
                # tablet and 625 is its total strength, not this component's
                # dose. Before the multi-word name existed this sentence could
                # not produce a dosed row at all, so the guard is scoped to
                # the newly reachable case rather than rewriting how a dose
                # attaches to a conjoined single-word drug, which is older
                # behaviour nobody asked to change here.
                if extended and dose and not dose_red:
                    partner = _combination_partner(
                        clause, name_at, drugs_known, alias_to_canonical)
                    if partner:
                        dose_red = (f"combination order: {dose:g} {unit or ''} may be "
                                    f"the total for {partner} + {canon or rawm} "
                                    "- physician must confirm")
                if fuzzy_note:
                    conf = 0.88  # fuzzy match -> YELLOW, human must glance
                    color = "YELLOW"
                if dose_red:
                    fuzzy_note = f"{fuzzy_note}; {dose_red}".strip("; ")
                    conf, color = 0.70, "RED"
                # A prohibition is not a prescription. `neg` was computed for
                # this sentence and used only for symptoms, so "do not give
                # ibuprofen 400 mg" and "stop metformin 500 mg" both charted as
                # active GREEN orders. The doseless path already guards this
                # with `_negated_before`; the dosed path simply never did.
                # Scope is the same prefix test, so "she has no fever, give
                # paracetamol 500 mg" still prescribes.
                negated = _drug_negated(sent[:clause_at + name_at])
                if negated:
                    # The row leaves the active medication list and is rendered
                    # under NOT GIVEN instead, so a mis-scoped negation is
                    # visible as a wrong entry there rather than as an order
                    # that simply vanished. RED marks it for review either way.
                    conf, color = min(conf, 0.70), "RED"
                drugs.append({"name": canon or rawm, "dose": dose, "unit": unit,
                              "frequency": freq, "duration": dur,
                              "confidence": conf, "color": color,
                              "source_sentence": proof_sent, "negated": negated,
                              **({"note": fuzzy_note} if fuzzy_note else {}),
                              # the validation stage needs the word that was
                              # actually heard and every term it could have been,
                              # not just the one difflib ranked first
                              # the validation stage needs the word that was
                              # actually heard and every term it could have been,
                              # not just the one difflib ranked first. An exact
                              # reference hit has no candidates, so it is not a
                              # flagged row and the validator leaves it alone.
                              **({"raw_token": rawm, "candidates": candidates,
                                  # the string matcher's own pick, recorded
                                  # before the dose check reordered anything,
                                  # so the validator can still tell whether
                                  # the drug on this row is the one the
                                  # matcher proposed
                                  "matcher_pick": matcher_pick}
                                 if fuzzy_note and candidates else {}),
                              **({"reference": {"kind": ref_hit["kind"],
                                                "name": ref_hit["name"],
                                                "substances": ref_hit.get("substances", []),
                                                "units": ref_hit.get("units", []),
                                                "mass_mg": ref_hit.get("mass_mg")}}
                                 if ref_hit else {})})
        # Doseless drugs. `DRUG_CTX` requires name + number + unit, so a drug
        # dictated without one ("he is continuing ceftriaxone", "takes insulin
        # before meals") is invisible even with a perfect transcript. Those are
        # most of what a ward handover says, so a known drug-list name standing
        # on its own is recorded too - RED, because the missing dose is exactly
        # what the existing rule says a physician must fill in.
        for nm in known_name_pat.finditer(sent):
            canon = alias_to_canonical.get(nm.group(0).lower())
            if not canon or any(d["name"] == canon for d in drugs):
                continue
            if _overlaps(allergy_spans, nm.start(), nm.end()):
                continue
            if _negated_before(sent[:nm.start()]):
                continue  # "cannot take amoxicillin" is a prohibition, not an order
            drugs.append({"name": canon, "dose": None, "unit": None,
                          "frequency": "", "duration": "",
                          "confidence": 0.70, "color": "RED",
                          "source_sentence": proof_sent, "negated": False,
                          "note": "no dose dictated - physician must confirm"})
        for canon, cstart, cend in _reference_doseless(
                sent, {d["name"] for d in drugs}, alias_to_canonical, drugs_known):
            if any(d["name"] == canon for d in drugs):
                continue
            if _overlaps(allergy_spans, cstart, cend):
                continue
            if _drug_negated(sent[:cstart]):
                continue
            drugs.append({"name": canon, "dose": None, "unit": None,
                          "frequency": "", "duration": "",
                          "confidence": 0.70, "color": "RED",
                          "source_sentence": proof_sent, "negated": False,
                          "note": "no dose dictated - physician must confirm "
                                  "(national drug list, not this pipeline's own)"})
        low = sent.lower()
        matched = set()
        for s in SYMPTOMS:
            if s in low:
                # skip substring dup: "pain" inside already-matched "chest pain"
                if any(s in m and s != m for m in matched):
                    continue
                matched.add(s)
                is_neg = neg and s in DENIED_SYMPTOMS
                symptoms.append({"text": s, "confidence": 0.9 if not is_neg else 0.91,
                                 "color": "GREEN" if is_neg else "YELLOW",
                                 "source_sentence": proof_sent,
                                 "negated": is_neg,
                                 **({"note": "DENIED - excluded"} if is_neg else {})})
        # standalone allergy mention without dose, e.g. "No penicillin allergy"
        # legacy allowlist first, then generic "allergy to X" candidate (YELLOW, negated if NEG_PAT)
        # Both word orders - "allergy to X" and "X ... allergy" - and in both the
        # polarity is read from the text BEFORE the substance, never from the
        # sentence. The two are not the same: "document X as a suspected allergy
        # so that it is not given again" is an ASSERTION inside a sentence that
        # trips NEG_PAT, and recording that as a denial is the failure mode that
        # gets a patient re-dosed.
        #
        # `ALLERGY_CTX` keeps its open capture, so an unknown word after
        # "allergy to" is still surfaced for a human to check. `ALLERGY_ASSERT`
        # scans backwards from the word "allergy" and cannot, so it accepts only
        # known drug names - otherwise "patient denies drug allergies" records an
        # allergy to "denies".
        for pattern, known_only in ((ALLERGY_CTX, False), (ALLERGY_ASSERT, True)):
            for am in pattern.finditer(sent):
                cand = (am.group("drug") or "").lower()
                canon = alias_to_canonical.get(cand, cand)
                if not cand or cand in ALLERGY_STOPWORDS:
                    continue
                if known_only and cand not in alias_to_canonical:
                    continue
                if any(a.get("text") == canon for a in allergies):
                    continue
                if _negated_before(sent[:am.start("drug")]):
                    allergies.append({"text": canon, "negated": True, "confidence": 0.82,
                                      "color": "YELLOW", "source_sentence": proof_sent,
                                      "note": "NEGATED candidate - verify drug name"})
                    negations.append({"span": proof_sent[:60], "negated": True})
                else:
                    allergies.append({"text": canon, "negated": False, "confidence": 0.82,
                                      "color": "RED", "source_sentence": proof_sent,
                                      "note": "ASSERTED allergy - confirm before prescribing"})
        for vm in VITALS_PAT.finditer(sent):
            ok, note = vital_span_plausible(vm.group(0))
            if ok:
                vitals.append({"text": vm.group(0), "confidence": 0.87, "color": "YELLOW",
                               "source_sentence": proof_sent})
            else:
                vitals.append({"text": vm.group(0), "confidence": 0.70, "color": "RED",
                               "source_sentence": proof_sent, "note": note})
        if re.search(r"\bno allergy\b", sent, re.I):
            negations.append({"span": "No allergy", "negated": True,
                              "note": "rule stand-in for CAN-BERT"})

    # diagnosis + follow-up sentences (LLM-primary covers more; regex floor)
    m = re.search(r"diagnosis\s*[—–\-:]*\s*([^.]+)", text, re.I)
    if m and m.group(1).strip():
        diagnosis = {"text": m.group(1).strip(), "icd10": "", "confidence": 0.85,
                     "color": "YELLOW", "source_sentence": m.group(0).strip()}
    m = re.search(r"(review after[^.]+|follow[- ]?up[^.]+|come back[^.]+|in one week[^.]*|in \d+ (?:days?|weeks?)[^.]*follow[^.]*)", text, re.I)
    if m:
        followup = {"text": m.group(1).strip(), "confidence": 0.88, "color": "YELLOW",
                    "source_sentence": m.group(0).strip()}

    drugs = _prefer_dosed(drugs)

    # optional spaCy-sm boost (adds no new entities in demo, just confidence nudge)
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        _ = nlp(text[:500])
    except Exception:
        pass
    return _annotate_entity_spans(
        {"drugs": drugs, "symptoms": symptoms, "vitals": vitals,
         "allergies": allergies, "negations": negations,
         "diagnosis": diagnosis, "followup": followup},
        normalized_en,
    )


UNITS_OK = {"mg", "mcg", "g", "ml", "U"}


def _merge_tidy_drugs(regex_drugs, tidy_drugs):
    """Per-drug schema merge: tidy fills gaps, never overwrites present values."""
    by_name = {str(d.get("name", "")).lower(): dict(d) for d in regex_drugs if d.get("name")}
    dropped = 0
    for t in tidy_drugs if isinstance(tidy_drugs, list) else []:
        if not isinstance(t, dict) or not t.get("name"):
            dropped += 1
            continue
        name = str(t["name"]).lower()
        base = by_name.get(name, {"name": t["name"], "confidence": 0.85, "color": "YELLOW",
                                  "source_sentence": "", "negated": False})
        for k in ("dose", "unit", "frequency", "duration", "confidence"):
            if base.get(k) in (None, "", 0) and t.get(k) not in (None, "", 0):
                base[k] = t[k]
        try:
            if base.get("dose") is not None:
                base["dose"] = float(base["dose"])
        except (TypeError, ValueError):
            base["dose"] = None
        if base.get("unit") not in UNITS_OK:
            base["unit"] = None
        by_name[name] = base
    return list(by_name.values()), dropped


def ollama_tidy(entities, normalized_en, model="llama3.2:3b", timeout=60):
    """Optional LLM tidy via local Ollama. Per-drug schema merge; falls back to regex."""
    prompt = ("Output strict JSON only with keys drugs[{name,dose,unit,frequency,duration}],"
              "symptoms,vitals,allergies,diagnosis{followup}. Doses numeric + UCUM units mg/mcg/g/ml/U. "
              f"Normalize from: {normalized_en}. Draft: {json.dumps(entities)[:2000]}")
    try:
        out = subprocess.run(["ollama", "run", "--format", "json", model, prompt],
                             capture_output=True, encoding="utf-8", errors="ignore",
                             timeout=timeout)
        txt = (out.stdout or "").strip()
        m = re.search(r"\{.*\}", txt, re.S)
        if m:
            tidy = json.loads(m.group(0))
            if isinstance(tidy, dict) and "drugs" in tidy:
                # per-drug merge: keep regex symptoms/vitals/allergies if model drops them
                merged = dict(entities)
                drugs, dropped = _merge_tidy_drugs(entities.get("drugs", []), tidy.get("drugs"))
                merged["drugs"] = drugs
                for k in ("symptoms", "vitals", "allergies", "diagnosis", "followup"):
                    if tidy.get(k):
                        merged[k] = tidy[k]
                merged["tidy_engine"] = f"ollama:{model}"
                merged["tidy_dropped"] = dropped
                return merged
    except Exception as e:
        entities["tidy_engine"] = f"regex-only (ollama failed: {type(e).__name__})"
        return entities
    entities["tidy_engine"] = "regex-only (ollama skipped/failed)"
    return entities


def _run_term_validation(ent, validate_terms, ollama_model):
    """Stage 4b gate. `auto` runs only when the model is actually pulled.

    A fuzzy match that nobody validated and one a validator cleared must not
    look alike on the chart, so the skip reason is recorded on the rows rather
    than swallowed.
    """
    if validate_terms is False:
        return ent
    from term_validate import apply_validation, flagged_rows
    rows = flagged_rows(ent)
    if not rows:
        return ent
    if validate_terms == "auto":
        from llm_extract import ollama_available
        ok, why = ollama_available(ollama_model)
        if not ok:
            for row in rows:
                row["validation"] = {"status": "skipped", "reason": why}
                row["note"] = f"{row.get('note', '')} | validator skipped".strip(" |")
            ent["term_validator"] = f"unavailable ({why})"
            return ent
    return apply_validation(ent, model=ollama_model)


def run_stt_extract(
    clean_wav, job_id="demo-001", use_llm="auto", model="medasr",
    ollama_model="llama3.2:3b", strict: bool = False, validate_terms="auto",
):
    stt = transcribe(clean_wav, job_id, model=model, strict=strict)
    norm = normalize_text(stt["text"])
    # refresh segment lang tags with demo LID
    for s in stt["segments"]:
        s["lang"] = detect_lang_tag(s["text"]) if len(s["text"]) < 200 else norm["lang_tag"]
    ent, llm_reason = None, ""
    if use_llm in (True, "auto"):
        try:
            from llm_extract import extract_llm_primary, merge_primary, ollama_available
            ok, why = ollama_available(ollama_model)
            if ok or use_llm is True:
                base = extract_entities(stt["text"], norm["normalized_en"], stt["segments"])
                llm = extract_llm_primary(stt["text"], norm["normalized_en"], stt["segments"], model=ollama_model)
                ent = merge_primary(base, llm, model=ollama_model)
            else:
                llm_reason = why
        except ValueError as e:
            ent, llm_reason = None, str(e)
    if ent is None:
        ent = extract_entities(stt["text"], norm["normalized_en"], stt["segments"])
        if use_llm is True:
            ent = ollama_tidy(ent, norm["normalized_en"], model=ollama_model)
        elif use_llm == "auto" and llm_reason:
            ent["llm_engine"] = f"regex-fallback ({llm_reason})"
    ent = _run_term_validation(ent, validate_terms, ollama_model)
    ent = _annotate_entity_spans(ent, norm["normalized_en"])
    transcript_json = {"job_id": job_id, "text": stt["text"], "language": norm["lang_tag"],
                       "segments": stt["segments"], "normalized_en": norm["normalized_en"],
                       "normalizations": norm["normalizations"], "stt_engine": stt["engine"],
                       "stt_provenance": stt["stt_provenance"]}
    entities_json = {"job_id": job_id, **ent}
    return transcript_json, entities_json

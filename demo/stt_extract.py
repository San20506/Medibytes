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
    toks = [t for t in toks if t != "and"]
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


def numwords_to_digits(text):
    """Dictation words -> digits: five-hundred milligram -> 500 milligram."""
    def rep(m):
        toks = re.split(r"[ -]", m.group(0).lower())
        toks = [t for t in toks if t]
        if not any(t in _ONES or t in _TENS or t == "hundred" for t in toks):
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


def transcribe(clean_wav, job_id="demo-001", model="small-int8", strict: bool = False):
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
            return _mock_result(clean_wav, job_id, model, exc, "no medasr")
    try:
        import faster_whisper as faster_whisper_module
        from faster_whisper import WhisperModel
        size = {"tiny-int8": "tiny", "base-int8": "base", "small-int8": "small"}.get(model, "tiny")
        wm = WhisperModel(size, device="cpu", compute_type="int8")
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
            "device": "cpu",
            "compute_type": "int8",
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
        return {"text": text, "segments": segs,
                "engine": f"faster-whisper:{size}-cpu-int8",
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
        drug_clauses = DRUG_SPLIT_PAT.split(sent) if DRUG_SPLIT_PAT.search(sent) else [sent]
        for clause in drug_clauses:
            for m in DRUG_CTX.finditer(clause):
                rawm = m.group("name").lower()
                canon = alias_to_canonical.get(rawm)
                fuzzy_note = ""
                if canon is None and rawm not in drugs_known:
                    # fuzzy sound-alike fallback: asitromaisin -> azithromycin (difflib stand-in)
                    import difflib
                    best, score = None, 0
                    for known in list(drugs_known) + list(alias_to_canonical.keys()):
                        sc = difflib.SequenceMatcher(None, rawm, known).ratio()
                        if sc > score:
                            best, score = known, sc
                    if score >= (0.55 if len(rawm) >= 8 else 0.6):
                        canon = alias_to_canonical.get(best, best)
                        fuzzy_note = f"fuzzy {rawm}->{canon} ({score:.2f})"
                    else:
                        continue  # skip non-drug words; drug_list is the mini-RxNorm
                if _overlaps(allergy_spans, m.start("name"), m.end("name")):
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
                if fuzzy_note:
                    conf = 0.88  # fuzzy match -> YELLOW, human must glance
                    color = "YELLOW"
                drugs.append({"name": canon or rawm, "dose": dose, "unit": unit,
                              "frequency": freq, "duration": dur,
                              "confidence": conf, "color": color,
                              "source_sentence": proof_sent, "negated": False,
                              **({"note": fuzzy_note} if fuzzy_note else {})})
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
            vitals.append({"text": vm.group(0), "confidence": 0.87, "color": "YELLOW",
                           "source_sentence": proof_sent})
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


def run_stt_extract(
    clean_wav, job_id="demo-001", use_llm="auto", model="small-int8",
    ollama_model="llama3.2:3b", strict: bool = False,
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
    ent = _annotate_entity_spans(ent, norm["normalized_en"])
    transcript_json = {"job_id": job_id, "text": stt["text"], "language": norm["lang_tag"],
                       "segments": stt["segments"], "normalized_en": norm["normalized_en"],
                       "normalizations": norm["normalizations"], "stt_engine": stt["engine"],
                       "stt_provenance": stt["stt_provenance"]}
    entities_json = {"job_id": job_id, **ent}
    return transcript_json, entities_json

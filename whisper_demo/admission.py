"""Admission form: fill a structured admission note from a transcript.

Shape follows the discharge note: an LLM proposes values, then a deterministic gate keeps only what
the transcript proves. Fields that were not dictated stay null; nothing is guessed.

DATA LEAVES THE MACHINE: the transcript is sent to the chat-completions server named by
WHISPER_DEMO_FORM_URL (default https://api.openai.com/v1) using OPENAI_API_KEY. `provenance.sent_to`
records the host on every result.

`FORM` is the single description of the form; the empty form, the prompt, the gate and
`admission_schema.json` all derive from it (a test keeps the schema file in sync).
"""
import json
import os
import re
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from . import drug_reference
from .extract import _load_drug_list, numwords_to_digits

DEFAULT_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4.1-mini"
CONFIDENCE_CAP = 0.88
ATTEMPTS = 3

# name -> {"leaf": True} | {"scalars": [...], "lists": {name: [item fields]}} | {"list": [item fields]}
FORM = {
    "patient": {"scalars": ["name", "age", "sex"]},
    "admission": {"scalars": ["reason_for_admission", "mode", "department_or_ward"]},
    "chief_complaints": {"list": ["complaint", "duration"]},
    "history_of_present_illness": {"list": ["text"]},
    "pertinent_negatives": {"list": ["symptom"]},
    "past_medical_history": {"list": ["condition", "duration_or_status"]},
    "past_surgical_history": {"list": ["procedure"]},
    "drug_history": {"list": ["drug", "dose", "unit", "frequency"]},
    "allergies": {"scalars": ["no_known_allergies"], "lists": {"items": ["allergen", "reaction"]}},
    "family_history": {"list": ["text"]},
    "personal_history": {"scalars": ["smoking", "alcohol", "diet", "occupation"]},
    "vitals": {"scalars": ["bp", "pulse", "temperature", "respiratory_rate", "spo2", "weight"]},
    "general_examination": {"leaf": True},
    "systemic_examination": {"scalars": ["cvs", "respiratory", "abdomen", "cns"]},
    "provisional_diagnosis": {"list": ["diagnosis"]},
    "investigations_advised": {"list": ["investigation"]},
    "admission_plan": {"scalars": ["diet", "follow_up"],
                       "lists": {"orders": ["drug", "dose", "unit", "frequency", "duration", "route"],
                                 "advice": ["text"]}},
}

# Values that are normalised rather than quoted ("gentleman" -> male, "twice daily" -> BD).
NORMALISED = {"sex", "mode", "no_known_allergies", "unit", "frequency", "route", "bp"}
# A negated sentence may not fill these; a negated complaint is a pertinent negative instead.
SYMPTOM_LISTS = {"chief_complaints", "history_of_present_illness"}
HISTORY_LISTS = {"past_medical_history", "past_surgical_history", "family_history",
                 "provisional_diagnosis", "drug_history"}
HISTORY_LISTS |= {"allergies.items"}
UNITS = {"mg", "mcg", "g", "ml", "u", "iu", "meq", "%"}
ORDER_CUES = re.compile(r"\b(start(ed)?|give|given|prescri\w+|advis\w+|treatment plan|plan includes?|initiate\w*|"
                        r"begin|commence\w*|administer\w*|add|to receive|will receive)\b", re.I)
HISTORY_CUES = re.compile(r"\b(on|taking|takes|taken|known case|k/c|since|currently|history of|regularly|"
                          r"already|ongoing)\b", re.I)
NO_ALLERGY = re.compile(r"\b(no known (drug |food )?allerg|nkda|no (drug |food )?allerg|not allergic|"
                        r"denies (any )?allerg)", re.I)
ALLERGY_EXCEPTION = re.compile(r"\b(except|other than|apart from|but)\b", re.I)
FREQUENCY_LIKE = re.compile(r"(daily|times|once|twice|thrice|\bbd\b|\btds\b|\bod\b|\bqid\b|hourly|per day|every)", re.I)
DURATION_LIKE = re.compile(r"\b\d+\s*(days?|weeks?|months?|years?)\b", re.I)
CUES = {  # normalised fields must be backed by explicit words in the cited sentence
    "sex": lambda s, v: bool(re.search(
        r"\b(female|woman|girl|lady|she|her|mrs|ms|mother)\b" if str(v).lower().startswith("f")
        else r"\b(male|man|boy|gentleman|he|his|him|mr|father)\b", s, re.I)),
    "mode": lambda s, v: bool(re.search(
        r"emergenc|casualty|\ber\b|accident|trauma" if str(v).lower().startswith("emerg")
        else r"elective|planned|scheduled|routine admission", s, re.I)),
    "no_known_allergies": lambda s, v: v is not True or bool(NO_ALLERGY.search(s)),
}
NEGATION_CUES = re.compile(r"\b(no|not|nil|never|denies|denied|deny|without|absent|negative for)\b", re.I)
CLAUSE_BREAK = re.compile(r"[.,;:]|\b(but|however|although|except)\b", re.I)


def _lists(spec):
    """[(list name or None for a top-level list, item fields)] for one FORM entry."""
    if "list" in spec:
        return [(None, spec["list"])]
    return list(spec.get("lists", {}).items())


def empty_form():
    form = {}
    for name, spec in FORM.items():
        if spec.get("leaf"):
            form[name] = None
        elif "list" in spec:
            form[name] = []
        else:
            form[name] = {f: None for f in spec.get("scalars", [])}
            for lname, _ in _lists(spec):
                form[name][lname] = []
    return form


def build_schema():
    """Draft-07 schema generated from FORM (admission_schema.json is this, written out)."""
    meta = {"source_sentence": {"type": "string"}, "confidence": {"type": "number"},
            "status": {"type": "string", "enum": ["verified", "needs_review"]}}
    leaf = {"type": ["object", "null"], "required": ["value", "source_sentence"],
            "additionalProperties": False,
            "properties": {"value": {"type": ["string", "number", "boolean"]}, **meta}}

    def item(fields):
        return {"type": "object", "required": ["source_sentence"], "additionalProperties": False,
                "properties": {**{f: {"type": ["string", "number", "null"]} for f in fields}, **meta}}

    props = {}
    for name, spec in FORM.items():
        if spec.get("leaf"):
            props[name] = leaf
        elif "list" in spec:
            props[name] = {"type": "array", "items": item(spec["list"])}
        else:
            inner = {f: leaf for f in spec.get("scalars", [])}
            inner.update({ln: {"type": "array", "items": item(fs)} for ln, fs in _lists(spec)})
            props[name] = {"type": "object", "additionalProperties": False, "properties": inner}
    return {"$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Admission note (heard-only, proof-grounded)", "type": "object",
            "required": ["form"], "additionalProperties": False,
            "properties": {"form": {"type": "object", "additionalProperties": False,
                                    "required": list(FORM), "properties": props},
                           "provenance": {"type": "object"}}}


def _skeleton():
    """The prompt's description of the form: FORM rendered as an example JSON shape."""
    def item(fields):
        body = ", ".join(f'"{f}": "..."' for f in fields)
        return "{" + body + ', "source_sentence": "exact sentence heard", "confidence": 0.9}'

    leaf = '{"value": "...", "source_sentence": "exact sentence heard", "confidence": 0.9}'
    out = {}
    for name, spec in FORM.items():
        if spec.get("leaf"):
            out[name] = leaf
        elif "list" in spec:
            out[name] = "[" + item(spec["list"]) + ", ...]"
        else:
            inner = {f: leaf for f in spec.get("scalars", [])}
            inner.update({ln: "[" + item(fs) + ", ...]" for ln, fs in _lists(spec)})
            out[name] = inner
    return json.dumps(out, indent=1).replace('\\"', '"').replace('"{', "{").replace('}"', "}") \
        .replace('"[', "[").replace(']"', "]")


# ---- model call ---------------------------------------------------------------------------

class FormLLMError(ValueError):
    def __init__(self, message, retryable=True):
        super().__init__(message)
        self.retryable = retryable


def _settings():
    url = os.environ.get("WHISPER_DEMO_FORM_URL", DEFAULT_URL).rstrip("/")
    return {"url": url, "model": os.environ.get("WHISPER_DEMO_FORM_MODEL", DEFAULT_MODEL),
            "key": os.environ.get("OPENAI_API_KEY") or os.environ.get("CHATGPT_API_KEY")}


def status():
    """What `/api/health` reports. Never includes the key."""
    s = _settings()
    host = urlparse(s["url"]).netloc
    return {"provider": "openai" if host == "api.openai.com" else host, "model": s["model"],
            "key_present": bool(s["key"])}


def _chat(prompt, *, model, url, key, timeout=90):
    """One chat-completions call: (reply text, usage dict)."""
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "response_format": {"type": "json_object"}, "temperature": 0,
                       "max_tokens": 4000}).encode("utf-8")
    req = urllib.request.Request(
        url + "/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            reply = json.loads(resp.read().decode("utf-8", errors="ignore"))
    except urllib.error.HTTPError as e:
        raise FormLLMError(f"form LLM HTTP {e.code}", retryable=e.code in (429, 500, 502, 503, 504)) from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise FormLLMError(f"form LLM unreachable ({type(e).__name__})") from None
    except json.JSONDecodeError:
        raise FormLLMError("form LLM returned non-JSON") from None
    try:
        return reply["choices"][0]["message"]["content"] or "", reply.get("usage") or {}
    except (KeyError, IndexError, TypeError):
        raise FormLLMError("form LLM reply had no content") from None


def _facts_lines(verified):
    lines = []
    for f in (verified or {}).get("facts", []):
        if f.get("status") == "rejected":
            continue
        if f["kind"] == "drug":
            dose = f"{f['dose']:g} {f.get('unit') or ''}".strip() if f.get("dose") not in (None, "") else "no dose"
            lines.append(f"- drug: {f['name']} {dose}")
        elif f["kind"] in ("vital", "symptom", "allergy"):
            lines.append(f"- {f['kind']}: {f.get('text', '')}")
    return "\n".join(lines) or "(none)"


def _prompt(transcript, normalized, verified):
    return (
        "You are a clinical scribe filling a hospital ADMISSION FORM from a dictated transcript.\n"
        "Return STRICT JSON only: one object with exactly the keys of the schema below.\n"
        "Schema (leaf = one scalar field, item = one list entry; use null for a leaf not dictated, [] for an empty list):\n"
        f"{_skeleton()}\n\n"
        "Rules (must follow):\n"
        "- Output ONLY what is spoken in the transcript. NEVER invent a name, age, value, drug, dose or diagnosis.\n"
        "- source_sentence MUST be copied EXACTLY, character for character, from the transcript.\n"
        "- A field that was not dictated is null (or an empty list). Do not write 'unknown' or 'not stated'.\n"
        "- Anything the patient DENIES or does not have (\"denies cough\", \"no fever\") goes ONLY in "
        "pertinent_negatives, never in chief_complaints or history_of_present_illness.\n"
        "- drug_history = medicines the patient ALREADY takes (\"on\", \"taking\", \"known case\", \"since\"). "
        "admission_plan.orders = medicines newly started or advised in this note (\"start\", \"give\", \"advise\").\n"
        "- allergies.no_known_allergies: true only if the speaker says there are no known allergies; "
        "otherwise null. List named allergens in allergies.items.\n"
        "- vitals as dictated (bp as \"150/90\"). sex as male/female/other. mode as emergency/elective only if said.\n"
        "- Drug names, doses and units must match the VERIFIED FACTS where they appear there.\n"
        "- history_of_present_illness: short statements copied from the speaker's own words.\n"
        "- Drug fields: dose is the NUMBER, unit is mg/mcg/g/ml/U, frequency is e.g. BD or twice daily, "
        "duration is e.g. 5 days (orders only).\n"
        "- Every leaf and item MUST be an object with value/source_sentence/confidence as in the schema; "
        "never a bare string.\n\n"
        f"VERIFIED FACTS (machine-extracted, may be incomplete):\n{_facts_lines(verified)}\n\n"
        f"Transcript:\n{transcript[:8000]}\n\n"
        f"Normalized:\n{normalized[:8000]}"
    )


# ---- gate ---------------------------------------------------------------------------------

def _is_empty(v):
    return v is None or (isinstance(v, str) and not v.strip())


def _scalar(v):
    return isinstance(v, (str, int, float, bool)) and not _is_empty(v)


def _conf(value, default=0.9):
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return default


def _normalise(raw):
    """Model reply -> a full form shape. A bare scalar or a missing source gets `source_sentence: None`,
    which the gate then tries to derive from the transcript or drops; nothing is discarded silently."""
    form = empty_form()
    raw = raw if isinstance(raw, dict) else {}
    raw = raw["form"] if isinstance(raw.get("form"), dict) else raw

    def leaf(x):
        if isinstance(x, dict) and _scalar(x.get("value")):
            src = x.get("source_sentence")
            return {"value": x["value"], "source_sentence": src if isinstance(src, str) and src.strip() else None,
                    "confidence": _conf(x.get("confidence"), 0.7 if not src else 0.9)}
        if _scalar(x):
            return {"value": x, "source_sentence": None, "confidence": 0.7}
        return None

    def items(xs, fields):
        out = []
        for x in xs if isinstance(xs, list) else []:
            if not isinstance(x, dict):
                continue
            row = {f: (x.get(f) if _scalar(x.get(f)) and not isinstance(x.get(f), bool) else None)
                   for f in fields}
            if all(v is None for v in row.values()):
                continue
            src = x.get("source_sentence")
            row.update(source_sentence=src if isinstance(src, str) and src.strip() else None,
                       confidence=_conf(x.get("confidence"), 0.7 if not src else 0.9))
            out.append(row)
        return out

    for name, spec in FORM.items():
        block = raw.get(name)
        if spec.get("leaf"):
            form[name] = leaf(block)
        elif "list" in spec:
            form[name] = items(block, spec["list"])
        else:
            block = block if isinstance(block, dict) else {}
            for f in spec.get("scalars", []):
                form[name][f] = leaf(block.get(f))
            for lname, fields in _lists(spec):
                form[name][lname] = items(block.get(lname), fields)
    return form


def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(text).strip()) if s.strip()]


def _stem(w):
    return w.lower().rstrip("s") if len(w) > 3 else w.lower()


def _negated(sentence, value):
    """Is `value` under a negation cue within its own clause of `sentence`?"""
    if NEGATION_CUES.search(str(value)):  # the value itself says it ("They deny any back pain")
        return True
    words = re.findall(r"[A-Za-z]{3,}", str(value))
    low = sentence.lower()
    positions = [low.find(w.lower()) for w in words if low.find(w.lower()) >= 0]
    if not positions:
        return bool(NEGATION_CUES.search(sentence))
    before = sentence[:min(positions)]
    breaks = list(CLAUSE_BREAK.finditer(before))
    clause = before[breaks[-1].end():] if breaks else before
    return bool(NEGATION_CUES.search(clause))


def _allergy_negated(sentence, allergen):
    """A negated allergen: "no known allergies" (unless an exception follows), or a cue in its clause.

    A comma does not end the scope here ("No known drug allergies, specifically to sulfa"), so the
    window runs to the last full stop / semicolon / "but", and "allergic" between cue and value
    re-opens the scope ("No fever, allergic to penicillin").
    """
    if NO_ALLERGY.search(sentence) and not ALLERGY_EXCEPTION.search(sentence):
        return True
    low = sentence.lower()
    pos = [low.find(w.lower()) for w in re.findall(r"[A-Za-z]{3,}", str(allergen)) if low.find(w.lower()) >= 0]
    if not pos:
        return False
    before = sentence[:min(pos)]
    cut = [m.end() for m in re.finditer(r"[.;:]|\b(but|however|although|except)\b", before, re.I)]
    window = before[cut[-1]:] if cut else before
    cue = NEGATION_CUES.search(window)
    return bool(cue) and not re.search(r"allerg", window[cue.end():], re.I)


def _clean_drug(row, fields):
    """Fix obvious misfiling of dose/unit/frequency/duration. Returns True if the row needs review."""
    review = False
    dose, parsed_unit = row.get("dose"), None
    if isinstance(dose, str):
        m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*([A-Za-z%]+)?\s*$", dose)
        if m:
            row["dose"] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
            parsed_unit = m.group(2)
            if parsed_unit and not row.get("unit"):
                row["unit"] = parsed_unit
    unit, freq = row.get("unit"), row.get("frequency")
    if unit is not None and str(unit).lower() not in UNITS:
        if FREQUENCY_LIKE.search(str(unit)) and freq is None:
            row["frequency"], row["unit"], freq = unit, None, unit
        elif FREQUENCY_LIKE.search(str(unit)) and DURATION_LIKE.search(str(freq or "")):
            row["frequency"], row["unit"] = unit, None
            if "duration" in fields and row.get("duration") is None:
                row["duration"] = freq
            freq = row["frequency"]
        else:
            review = True
    if row.get("unit") is None and parsed_unit:  # the unit the dose carried, once a misfiled one moved out
        row["unit"] = parsed_unit
    if freq is not None and DURATION_LIKE.search(str(freq)) and not FREQUENCY_LIKE.search(str(freq)):
        if "duration" in fields and row.get("duration") is None:
            row["duration"], row["frequency"] = freq, None
        else:
            review = True
    return review


class _Gate:
    def __init__(self, transcript, normalized, verified):
        self.sources = [transcript, normalized]
        self.full = " ".join([transcript, normalized, numwords_to_digits(transcript),
                              numwords_to_digits(normalized)])
        seen, self.sentences = set(), []
        for s in _sentences(transcript) + _sentences(normalized):
            if s not in seen:
                seen.add(s)
                self.sentences.append(s)
        self.facts = [f for f in (verified or {}).get("facts", []) if f.get("status") != "rejected"]
        self.drug_facts = {str(f.get("name", "")).lower(): f for f in self.facts if f.get("kind") == "drug"}
        self.known = set()
        for d in _load_drug_list():
            for a in [d["name"], *d.get("aliases", [])]:
                self.known.add(str(a).lower())
        self.dropped = 0
        self.drops = []
        self.reason = ""
        self.moved_to_negatives = 0
        self.misfiled = 0
        self.derived = 0

    def _drug_ok(self, name):
        words = re.findall(r"[A-Za-z][A-Za-z\-]+", str(name))
        return any(w.lower() in self.drug_facts or w.lower() in self.known or
                   drug_reference.lookup(w.lower(), block=drug_reference.STOPWORDS, brands=False)
                   for w in words)

    def _drug_heard(self, name, sentence):
        """A real drug is not enough: it must be in the cited sentence, or the extractor tied it there.

        The second case covers a repaired name (heard "pantaprazole", charted "pantoprazole").
        """
        low = sentence.lower()
        for w in re.findall(r"[A-Za-z][A-Za-z\-]+", str(name)):
            if w.lower() in low:
                return True
            fact = self.drug_facts.get(w.lower())
            if fact is not None and str(fact.get("source_sentence", "")).strip() == sentence:
                return True
        return False

    def containing(self, cited):
        """The full transcript sentence a cited quote sits in (negation and role live there)."""
        return next((s for s in self.sentences if cited in s), cited)

    def fail(self, reason):
        self.reason = reason
        return False, None

    def drop(self, key, value, reason=None):
        self.dropped += 1
        if len(self.drops) < 25:
            self.drops.append({"field": key, "value": str(value)[:60], "reason": reason or self.reason})

    def _derive(self, row, fields):
        """Source sentence for a value the model gave without one: the first transcript sentence that
        holds all its numbers and content words. None if no sentence does."""
        values = {f: row[f] for f in fields if row.get(f) is not None}
        if values.get("no_known_allergies") is True:
            return next((s for s in self.sentences if NO_ALLERGY.search(s)), None)
        digits = [d for v in values.values() for d in re.findall(r"\d+(?:\.\d+)?", str(v))]
        words = [_stem(w) for f, v in values.items() if f not in NORMALISED and not isinstance(v, (int, float))
                 for w in re.findall(r"[A-Za-z]{3,}", str(v))]
        if not digits and not words:
            return None
        for s in self.sentences:
            if all(d in numwords_to_digits(s) for d in digits) and all(w in s.lower() for w in words):
                return s
        return None

    def check(self, row, fields):
        """(ok, status) for one leaf-or-item row. May fill in `source_sentence` and lower `confidence`."""
        status = "verified"
        sent = (row.get("source_sentence") or "").strip()
        if not sent:
            sent = self._derive(row, fields)
            if not sent:
                return self.fail("no source sentence given and none found in the transcript")
            row["source_sentence"], row["confidence"] = sent, min(float(row.get("confidence", 0.7)), 0.7)
            self.derived += 1
            status = "needs_review"
        elif not any(sent in s for s in self.sources if s):
            return self.fail("source_sentence is not a verbatim quote of the transcript")
        elif len(sent.split()) < 4 and sent.rstrip(".!? ") not in {s.rstrip(".!? ") for s in self.sentences}:
            return self.fail("citation too short to be a sentence")  # "pain" is a substring, not a sentence
        for f, v in {f: row[f] for f in fields if row.get(f) is not None}.items():
            text = str(v)
            if any(n not in self.full for n in re.findall(r"\d+(?:\.\d+)?", text)):
                return self.fail(f"{f}: a number in the value was never spoken")
            if f in CUES and not CUES[f](self.containing(sent), v):
                return self.fail(f"{f}: no cue word in the cited sentence backs this value")
            if f == "drug":
                if not self._drug_ok(text):
                    return self.fail("drug is not a known drug")
                if not self._drug_heard(text, sent):
                    return self.fail("drug name is not in the cited sentence")
            elif f not in NORMALISED and not isinstance(v, (int, float)):
                content = [_stem(w) for w in re.findall(r"[A-Za-z]{3,}", text)]
                if content and not any(w in sent.lower() for w in content):
                    return self.fail(f"{f}: value words are not in the cited sentence")
        fact = None
        if "drug" in fields and row.get("drug") is not None:
            for w in re.findall(r"[A-Za-z][A-Za-z\-]+", str(row["drug"])):
                fact = self.drug_facts.get(w.lower()) or fact
        if fact is not None and row.get("dose") is not None and fact.get("dose") not in (None, ""):
            try:
                if float(row["dose"]) != float(fact["dose"]):
                    status = "needs_review"
            except (TypeError, ValueError):
                status = "needs_review"
        if fact is not None and fact.get("status") != "verified":
            status = "needs_review"
        return True, status


def gate(form, transcript, normalized, verified):
    """Keep only what the transcript proves. Returns (form, counters)."""
    g = _Gate(transcript, normalized, verified)
    out = empty_form()

    def keep(row, status):
        row = dict(row)
        row["confidence"] = min(float(row.get("confidence", 0.9)), CONFIDENCE_CAP)
        row["status"] = status
        return row

    def leaf(block, fname):
        if block is None:
            return None
        probe = {"source_sentence": block["source_sentence"], fname: block["value"],
                 "confidence": block["confidence"]}
        ok, status = g.check(probe, [fname])
        if not ok:
            g.drop(fname, block["value"])
            return None
        return keep({**block, "source_sentence": probe["source_sentence"],
                     "confidence": probe["confidence"]}, status)

    def items(rows, fields, key):
        kept = []
        for row in rows:
            row = dict(row)
            review = _clean_drug(row, fields) if "drug" in fields else False
            ok, status = g.check(row, fields)
            main = next((row[f] for f in fields if row.get(f) is not None), "")
            if not ok:
                g.drop(key, main)
                continue
            cited = row["source_sentence"]
            sent = g.containing(cited)  # polarity and role are read from the whole transcript sentence
            if key in SYMPTOM_LISTS and _negated(sent, main):
                out["pertinent_negatives"].append(keep(
                    {"symptom": main, "source_sentence": cited, "confidence": row["confidence"]}, status))
                g.moved_to_negatives += 1
                continue
            if key == "allergies.items" and _allergy_negated(sent, main):
                g.drop(key, main, "allergen is negated in its sentence")
                continue
            if key in HISTORY_LISTS and key != "allergies.items" and _negated(sent, main):
                g.drop(key, main, "negated: a denial is not a history")  # "denies diabetes"
                continue
            if key == "drug_history" and ORDER_CUES.search(sent) and not HISTORY_CUES.search(sent):
                g.misfiled += 1  # a drug just ordered is not one the patient already takes
                g.drop(key, main, "order sentence filed as drug history")
                continue
            if key == "orders" and HISTORY_CUES.search(sent) and not ORDER_CUES.search(sent):
                g.misfiled += 1
                g.drop(key, main, "history sentence filed as an order")
                continue
            kept.append(keep(row, "needs_review" if review else status))
        return kept

    for name, spec in FORM.items():
        block = form.get(name)
        if spec.get("leaf"):
            out[name] = leaf(block, name)
        elif "list" in spec:
            out[name] = out[name] + items(block, spec["list"], name)
        else:
            for f in spec.get("scalars", []):
                out[name][f] = leaf(block.get(f), f)
            for lname, fields in _lists(spec):
                out[name][lname] = items(block.get(lname, []), fields,
                                         f"{name}.{lname}" if name == "allergies" else lname)
    # a whole-sentence negation moved from the HPI duplicates the symptoms the model listed itself
    negs = out["pertinent_negatives"]
    out["pertinent_negatives"] = [n for n in negs if not (
        n["symptom"] == n["source_sentence"] and any(o is not n and o["source_sentence"] == n["source_sentence"]
                                                      and o["symptom"] != o["source_sentence"] for o in negs))]
    # a drug the extractor verified but the model left out is added, placed by the cues in its sentence
    have = {w.lower() for r in out["drug_history"] + out["admission_plan"]["orders"]
            for w in re.findall(r"[A-Za-z][A-Za-z\-]+", str(r.get("drug") or ""))}
    backfilled = 0
    for fact in g.facts:
        name, sent = str(fact.get("name", "")), str(fact.get("source_sentence", "")).strip()
        if fact.get("kind") != "drug" or fact.get("negated") or not name or name.lower() in have:
            continue
        if not any(sent in s for s in g.sources if s):
            continue
        order, hist = ORDER_CUES.search(g.containing(sent)), HISTORY_CUES.search(g.containing(sent))
        if bool(order) == bool(hist):
            continue  # no cue, or both: the role is ambiguous, so do not guess
        target, fields = ((out["admission_plan"]["orders"], dict(FORM["admission_plan"]["lists"])["orders"])
                          if order else (out["drug_history"], FORM["drug_history"]["list"]))
        row = {f: None for f in fields}
        dose = fact.get("dose")
        row.update(drug=name, unit=fact.get("unit") or None,
                   dose=(int(dose) if isinstance(dose, float) and dose.is_integer() else dose) or None,
                   source_sentence=sent, confidence=min(_conf(fact.get("confidence"), 0.8), CONFIDENCE_CAP),
                   status="needs_review")  # frequency and duration are not in a fact
        target.append(row)
        have.add(name.lower())
        backfilled += 1
    # an explicit "no known allergies" sets the field even if the model left it null
    if out["allergies"]["no_known_allergies"] is None:
        sent = next((s for s in g.sentences if NO_ALLERGY.search(s) and not ALLERGY_EXCEPTION.search(s)), None)
        if sent:
            out["allergies"]["no_known_allergies"] = {"value": True, "source_sentence": sent,
                                                      "confidence": 0.8, "status": "verified"}
    return out, {"dropped_unproven": g.dropped, "moved_to_pertinent_negatives": g.moved_to_negatives,
                 "misfiled_dropped": g.misfiled, "source_derived": g.derived,
                 "backfilled_drugs": backfilled, "drops": g.drops}


# ---- public ---------------------------------------------------------------------------------

def _schema():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "admission_schema.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def stamp_schema(result):
    """Attach provenance.schema = valid|INVALID by validating {form} against the shipped schema."""
    import jsonschema

    prov = dict(result.get("provenance", {}))
    try:
        jsonschema.validate({"form": result["form"]}, _schema())
        prov["schema"] = "valid"
    except jsonschema.ValidationError as e:
        prov["schema"] = "INVALID"
        prov["schema_error"] = e.message[:200]
    return {"form": result["form"], "provenance": prov}


def fill(transcript, verified, *, model=None):
    """The admission form for a transcript dict: {form, provenance}. Never raises on model trouble.

    No key, an HTTP error, or unusable JSON after ATTEMPTS tries gives an all-null form labelled
    `form-skipped (reason)`: there is no fallback that writes fields from anywhere else.
    """
    s = _settings()
    model = model or s["model"]
    text, norm = str(transcript.get("text", "") or ""), str(transcript.get("normalized_en", "") or "")
    prov = {"model": model, "sent_to": urlparse(s["url"]).netloc, "attempts": 0}

    def skipped(reason):
        prov["llm_engine"] = f"form-skipped ({reason})"
        return stamp_schema({"form": empty_form(), "provenance": prov})

    if not s["key"]:
        return skipped("no OPENAI_API_KEY set")
    started = time.perf_counter()
    last = "no attempt"
    for attempt in range(1, ATTEMPTS + 1):
        prov["attempts"] = attempt
        try:
            reply, usage = _chat(_prompt(text, norm, verified), model=model, url=s["url"], key=s["key"])
        except FormLLMError as e:
            last = str(e)
            if not e.retryable:
                return skipped(last)
            continue
        match = re.search(r"\{.*\}", reply, re.S)
        try:
            raw = json.loads(match.group(0)) if match else None
        except json.JSONDecodeError:
            raw = None
        if not isinstance(raw, dict):
            last = "form LLM JSON unusable"
            continue
        form, counters = gate(_normalise(raw), text, norm, verified)
        prov.update(counters, llm_engine=f"form:{model}", usage=usage,
                    seconds=round(time.perf_counter() - started, 2))
        return stamp_schema({"form": form, "provenance": prov})
    return skipped(last)

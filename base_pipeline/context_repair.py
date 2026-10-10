"""Stage 4d - context-based repair of drug names the string matcher gave up on.

The regex extractor repairs a misheard drug name only when it scores
`DOSELESS_FUZZY_FLOOR` (0.84) against the national list. That bar is right for a
row that is created from spelling alone, and it leaves two gaps:

  * a mishearing between about 0.70 and 0.84 never becomes a row at all
    (`paracetamine` -> paracetamol scores 0.783, tied with the chemotherapy
    drug altretamine), and
  * the matcher cannot use the rest of the sentence - "given for fever",
    "contains an antibiotic", "a zero-calorie sweetener" - which is exactly what
    separates a tie.

This stage reads the sentence. For each word that could be a misheard medicine it
shortlists substances by spelling at a lower floor, then asks a local LLM which
shortlisted substance - if any - the sentence is about. The model is advisory,
under the same rules as `term_validate`:

1. **Closed set.** It may only pick a name from the shortlist the matcher built.
   Anything else voids the verdict.
2. **Never above RED.** A repaired row says it was repaired, carries the heard
   word and the shortlist, and always needs a human.
3. **None is a normal answer.** Most sentences contain no misheard medicine, and a
   word that is not a medicine must come back as `none`.
4. **Fail quiet.** No server, bad JSON, a timeout: the entities are returned
   untouched with the reason recorded.

Every LLM call is timed so its cost is reported apart from the ASR it follows.
"""
from __future__ import annotations

import json
import os
import re
import time

from .llm_client import default_model

# Spelling floor for a candidate to be *offered*: the reference module's own
# `FUZZY_FLOOR`, the bar a dosed mention already clears. A row built from
# spelling alone still needs `DOSELESS_FUZZY_FLOOR` (0.84); between the two the
# sentence decides. (A first version used 0.70 and let a 3B model approve
# `certain` -> ceritinib and `boosting` -> bosutinib; see PIPELINE_EVAL_REPORT.)
CONTEXT_FLOOR = 0.78
CONTEXT_BAND = 0.10         # keep candidates within this of the best
MAX_CANDIDATES = 4
MIN_TOKEN_LEN = 6
MIN_CONFIDENCE = 0.85       # an LLM answer at or under this is a "none"
MAX_CALLS_PER_CLIP = 40     # a runaway guard, not a tuning knob

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")
# A sentence that is not about a medicine cannot contain a misheard one. Without
# this the stage would shortlist every long English word in a ward handover.
_CUE = re.compile(
    r"\b(?:take|takes|taking|taken|medicine|medication|medications|drug|drugs|tablet|tablets|"
    r"tab|capsule|capsules|syrup|injection|infusion|prescri\w+|contain\w*|regimen|dose|doses|"
    r"dosage|therapy|treatment|given|give|advised|started|receiving|received|used|component|"
    r"antibiotic|supplement|sweetener|ointment|solution|drops)\b", re.I)


def _prompt(token: str, sentence: str, candidates: list[dict]) -> str:
    listing = "\n".join(
        f"- {c['name']} (forms: {', '.join(c.get('forms') or []) or 'n/a'})" for c in candidates)
    return (
        "You are a clinical speech-recognition post-processor. A speech recogniser\n"
        "may have misheard ONE word in this sentence as a non-word. A spelling matcher\n"
        "shortlisted medicines whose names look like that word.\n\n"
        "Decide whether the word is a mishearing of exactly one shortlisted medicine,\n"
        "using what the SENTENCE says about it: what it treats, what it is, how it is\n"
        "given. Spelling similarity alone is not evidence.\n\n"
        "Return STRICT JSON only:\n"
        '{"decision": "candidate" | "none", "chosen": "<name copied from the shortlist, or empty>",\n'
        ' "confidence": 0.0, "reason": "<one short clause>"}\n\n'
        "Rules:\n"
        '- "chosen" must be copied exactly from the shortlist. Never name a medicine that is not listed.\n'
        '- Answer "none" if the word is an ordinary word, a product brand name, or if the\n'
        "  sentence does not clearly point to one shortlisted medicine. A wrong drug on a\n"
        '  chart injures a patient; "none" only costs a clinician a glance.\n'
        "- Two shortlisted medicines that both fit the sentence also means \"none\".\n\n"
        f"Sentence: {sentence[:500]}\n"
        f"Word: {token}\n"
        f"Shortlist:\n{listing}\n")


def _llm_runner(model, timeout):
    checked: list[bool] = []

    def run(prompt: str) -> dict:
        from .llm_client import complete_json, llm_available
        from .llm_extract import _extract_json_from_output
        if not checked:
            ok, reason = llm_available(model)
            if not ok:
                raise ValueError(reason)
            checked.append(True)
        raw = _extract_json_from_output(complete_json(prompt, model, timeout, max_tokens=200))
        if not raw:
            raise ValueError("no JSON in context-repair output")
        return json.loads(raw)

    return run


_ENGLISH: set[str] | None = None
_SUFFIXES = ("ing", "ed", "es", "s", "ly", "er", "ers", "ion", "ions")


def _english() -> set[str]:
    global _ENGLISH
    if _ENGLISH is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "english_words.txt")
        try:
            with open(path, encoding="utf-8") as f:
                _ENGLISH = {line.strip() for line in f if line.strip()}
        except OSError:
            _ENGLISH = set()
    return _ENGLISH


def is_ordinary_word(low: str) -> bool:
    """An everyday English word, or an inflection of one (`boosting` <- `boost`)."""
    words = _english()
    if low in words:
        return True
    for suffix in _SUFFIXES:
        if low.endswith(suffix) and len(low) - len(suffix) >= 4:
            stem = low[:-len(suffix)]
            if stem in words or stem + "e" in words:
                return True
    return False


def eligible_tokens(sentence: str, claimed: set[str], stopwords) -> list[str]:
    """Words worth shortlisting.

    Long, not a stopword, not an ordinary English word, not already an exact
    reference name, and not one half of a multi-word name (`fenofibric` in
    `fenofibric acid`, `etabonate` in `loteprednol etabonate`).
    """
    from . import drug_reference
    found = [(m.group(0), m.group(0).lower()) for m in _WORD.finditer(sentence)]
    in_name: set[int] = set()
    for i in range(len(found)):
        for n in (3, 2):
            if i + n <= len(found) and drug_reference.is_substance_name(
                    " ".join(low for _, low in found[i:i + n])):
                in_name.update(range(i, i + n))
    out, seen = [], set()
    for i, (w, low) in enumerate(found):
        if (i in in_name or len(low) < MIN_TOKEN_LEN or low in stopwords or low in claimed
                or low in seen or is_ordinary_word(low)
                or drug_reference.is_substance_name(low)
                or drug_reference.lookup(low, block=stopwords, brands=False)):
            continue
        seen.add(low)
        out.append(w)
    return out


def judge(token: str, sentence: str, candidates: list[dict], run) -> dict:
    """One verdict. Raises ValueError when the model could not be consulted."""
    closed = {c["name"].lower(): c["name"] for c in candidates}
    verdict = run(_prompt(token, sentence, candidates))
    if not isinstance(verdict, dict):
        raise ValueError("context-repair JSON is not an object")
    try:
        conf = min(max(float(verdict.get("confidence", 0.0)), 0.0), 1.0)
    except (TypeError, ValueError):
        conf = 0.0
    reason = str(verdict.get("reason", "") or "")[:160]
    chosen = str(verdict.get("chosen", "") or "").strip()
    if str(verdict.get("decision", "")).strip().lower() != "candidate" or not chosen:
        return {"status": "none", "confidence": conf, "reason": reason}
    if chosen.lower() not in closed:
        raise ValueError(f"model chose {chosen!r}, outside the shortlist")
    if conf < MIN_CONFIDENCE:
        return {"status": "none", "confidence": conf, "reason": reason or f"confidence {conf:.2f}"}
    return {"status": "repaired", "chosen": closed[chosen.lower()], "confidence": conf, "reason": reason}


def apply_context_repair(entities: dict, normalized_text: str, *, chart_name, stopwords,
                         model=None, timeout=30, runner=None, skip_sentence=None) -> dict:
    """Add RED rows for misheard medicines the LLM can place from context. In place.

    `skip_sentence` lets the caller veto a sentence outright. The extractor passes its
    lab-request test, because "requested cardiac troponin levels" is the one place a
    3B model reliably approves `troponin` -> tiopronin: the sentence names a
    test, not a medicine, and the doseless path already refuses it for that reason.
    """
    from . import drug_reference
    if not drug_reference.available():
        entities["context_repair"] = {"status": "skipped", "reason": "no drug reference"}
        return entities
    model = model or default_model()
    run = runner or _llm_runner(model, timeout)
    have = {str(d.get("name", "")).lower() for d in entities.get("drugs", []) if isinstance(d, dict)}
    claimed = {str(d.get("raw_token") or "").lower() for d in entities.get("drugs", [])
               if isinstance(d, dict)}
    log: dict = {"engine": "injected" if runner else f"llm:{model}", "considered": 0, "calls": 0,
                 "llm_seconds": 0.0, "repaired": [], "declined": [], "errors": []}
    for sentence in _SENT_SPLIT.split(normalized_text or ""):
        if not _CUE.search(sentence) or (skip_sentence and skip_sentence(sentence)):
            continue
        for token in eligible_tokens(sentence, claimed, stopwords):
            cands = drug_reference.candidates(token, floor=CONTEXT_FLOOR, band=CONTEXT_BAND,
                                              limit=MAX_CANDIDATES)
            if not cands:
                continue
            log["considered"] += 1
            shortlist = []
            for c in cands:
                shown = chart_name(c["name"])
                if shown.lower() not in {s["name"].lower() for s in shortlist}:
                    shortlist.append({**c, "name": shown})
            if log["calls"] >= MAX_CALLS_PER_CLIP:
                log["errors"].append("call cap reached")
                break
            t0 = time.perf_counter()
            try:
                verdict = judge(token, sentence, shortlist, run)
            except Exception as err:
                log["errors"].append(f"{token}: {type(err).__name__}: {err}"[:160])
                if "no LLM server" in str(err) or "not served" in str(err) or "pulled" in str(err):
                    entities["context_repair"] = {**log, "status": "skipped", "reason": str(err)[:160]}
                    return entities
                continue
            finally:
                log["calls"] += 1
                log["llm_seconds"] = round(log["llm_seconds"] + time.perf_counter() - t0, 3)
            top = next(c for c in shortlist if c["name"] == verdict.get("chosen")) \
                if verdict["status"] == "repaired" else shortlist[0]
            record = {"heard": token, "shortlist": [(c["name"], c["score"]) for c in shortlist],
                      "confidence": verdict["confidence"], "reason": verdict["reason"]}
            if verdict["status"] != "repaired":
                log["declined"].append(record)
                continue
            chosen = verdict["chosen"]
            if chosen.lower() in have:
                continue
            have.add(chosen.lower())
            entities.setdefault("drugs", []).append({
                "name": chosen, "dose": None, "unit": None, "frequency": "", "duration": "",
                "confidence": 0.65, "color": "RED", "negated": False,
                "source_sentence": sentence.strip(),
                "note": (f"context repair {token}->{chosen} (spelling {top['score']:.2f}, "
                         f"LLM {verdict['confidence']:.2f}: {verdict['reason']}) - physician must confirm"),
                "context_repair": {"heard": token, "model": model,
                                   "shortlist": record["shortlist"]},
            })
            log["repaired"].append({**record, "chosen": chosen})
    entities["context_repair"] = {**log, "status": "ran"}
    return entities

"""Stage 4c - deterministic verification of extracted facts (no LLM).

Sits between ``extract_entities`` and any LLM structuring step. Every check
here is a pure function of the transcript text plus the shipped datasets, so
its verdicts are reproducible and auditable:

* **numerals** - ``numwords_to_digits`` audit: each dose / vital number must
  occur as digits in the digitized transcript. A number the decoder never
  emitted (in words or digits) cannot be verified.
* **drug names** - exact mini-list alias, exact substance
  (``lookup(brands=False)``), or fuzzy top-candidate agreement. Brand-only
  names and names no candidate supports are rejected, never guessed.
* **doses/units** - UCUM units only; ``unit_fits`` (dimension) +
  ``dose_conflicts`` (20x magnitude) from the national pack as evidence.
* **vitals** - ``vital_span_plausible`` (VITALS_RANGES); violations keep the
  row but force review, mirroring the extractor.
* **proof** - every row's ``source_sentence`` must occur verbatim in the
  transcript (raw or normalized); unproven rows are capped, never promoted.
* **polarity** - ``negated`` flags are preserved as-is; verification never
  flips them.

Statuses: ``verified`` (all gates pass), ``needs_review`` (usable with human
check, confidence capped), ``rejected`` (must not reach any output). Nothing
here promotes confidence; caps only lower it, mirroring ``term_validate``.
"""

from __future__ import annotations

import re

UNITS_OK = {"mg", "mcg", "g", "ml", "U"}

STATUS_VERIFIED = "verified"
STATUS_NEEDS_REVIEW = "needs_review"
STATUS_REJECTED = "rejected"


def _haystacks(transcript):
    """(raw, normalized, digitized) transcript views for proof/numeral checks."""
    from .extract import numwords_to_digits

    raw = str(transcript.get("text", "") or "")
    normalized = str(transcript.get("normalized_en", "") or raw)
    return raw, normalized, numwords_to_digits(raw), numwords_to_digits(normalized)


def _proof_ok(source_sentence, raw, normalized):
    """Is this proof sentence a verbatim substring of the transcript?"""
    sent = str(source_sentence or "").strip()
    if not sent:
        return False
    return sent in raw or sent in normalized


def _number_present(haystack, value):
    """Do these digits occur in this text (integer-aware, no 98-in-98.6)?"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    if number.is_integer():
        return re.search(rf"(?<!\d){int(number)}(?!\d)(?!\.\d)", haystack) is not None
    whole, _, frac = f"{number:.10g}".partition(".")
    return any(
        re.search(p, haystack, re.I)
        for p in (
            rf"(?<!\d){re.escape(whole)}\.{re.escape(frac)}(?!\d)",
            rf"(?<!\d){re.escape(whole)}\s*point\s*{re.escape(frac)}(?!\d)",
        )
    )


_BRAND_TOKEN = re.compile(r"[a-z][a-z0-9'-]*|\d+(?:\.\d+)?")
_NUMERAL = re.compile(r"\d+(?:\.\d+)?")
BRAND_MAX_WORDS = 3  # mirrors extract.MAX_NAME_WORDS
# A number followed by one of these is a count, a duration or a vital, not a
# strength: `site 4 times`, `came 3 days ago`, `ten 98 percent`.
_NOT_STRENGTH = frozenset("""
time times day days daily week weeks month months year years hour hours hourly
minute minutes min mins second seconds am pm o'clock percent % degree degrees
mmhg bpm beats breaths per by over x
""".split())


def brand_mentions(text):
    """Dosed brand mentions in `text`: ``[{brand, number, substances}]``.

    Brands are evidence only - they never create a row (``lookup`` docstring).
    And a bare exact brand hit is not evidence either: across this project's
    54 real transcripts 16 of 17 distinct hits were English (`monitor`, `came`,
    `ten`, `okay`, `site`). What a dictated brand has that a collision lacks is
    a strength straight after it - `Dolo 650`, `Pan 40` - so a mention counts
    only when the next token is a number inside the brand's listed mass range
    (bare number read as mg, the way strengths are spoken) and not followed by
    a count/duration/vital word (`site 4 times`). On the corpus this leaves
    zero hits; `Monitor 2` fails the range.
    """
    from . import drug_reference
    from .extract import numwords_to_digits

    words = _BRAND_TOKEN.findall(numwords_to_digits(str(text or "")).lower())
    out, seen = [], set()
    for n in range(BRAND_MAX_WORDS, 0, -1):
        for i in range(len(words) - n):
            gram, nxt = words[i:i + n], words[i + n]
            if not _NUMERAL.fullmatch(nxt) or any(_NUMERAL.fullmatch(w) for w in gram):
                continue
            if i + n + 1 < len(words) and words[i + n + 1] in _NOT_STRENGTH:
                continue
            name = " ".join(gram)
            rec = drug_reference.lookup(name, block=drug_reference.STOPWORDS, brands=True)
            if not rec or rec.get("kind") != "brand":
                continue
            lo_hi = rec.get("mass_mg")
            if not lo_hi or not (lo_hi[0] <= float(nxt) <= lo_hi[1]):
                continue
            if (i, nxt) in seen:  # longest n-gram at this position wins
                continue
            seen.add((i, nxt))
            subs = sorted({drug_reference.inn_name(s) for s in rec.get("substances") or []})
            out.append({"brand": name, "number": nxt, "substances": subs})
    return out


def _brand_evidence(name, dose, source_sentence):
    """(check, note) for a drug row from dosed brands in its own sentence.

    ``brand:corroborated`` - a brand of this substance was dictated with it.
    ``brand:conflict``     - this row's own dose is the strength of a brand of
                             a different substance (`Dolo 650` charted as
                             amoxicillin 650 mg). A different brand with a
                             different number is just another drug in a list,
                             and goes to ``brand_mentions`` instead.
    Neither changes confidence; conflict only lowers status (caller).
    """
    from . import drug_reference

    label = str(name or "").strip().lower()
    if not label:
        return None, ""
    own = {label, str(drug_reference.inn_name(label)).lower()}
    hits = brand_mentions(source_sentence)
    if not hits:
        return None, ""
    match = [h for h in hits if own & set(h["substances"])]
    if match:
        h = match[0]
        return "brand:corroborated", f"{h['brand']} {h['number']} ({', '.join(h['substances'])}) in sentence"
    try:
        own_dose = float(dose)
    except (TypeError, ValueError):
        return None, ""
    clash = [h for h in hits if float(h["number"]) == own_dose]
    if not clash:
        return None, ""
    h = clash[0]
    return "brand:conflict", (f"dose {h['number']} belongs to {h['brand']} "
                              f"({', '.join(h['substances'])}), not {label}")


def verify_numerals(transcript):
    """Audit word->digit conversion: which numbers survive as digits?

    Returns ``{digits: [...], Thus the extractor's dose/vital regexes (which
    run on the digitized copy) can only see numbers listed here.
    """
    _, _, digitized_raw, digitized_norm = _haystacks(transcript)
    found = sorted({m.group(0) for m in re.finditer(r"\d+(?:\.\d+)?", digitized_raw + " " + digitized_norm)},
                   key=lambda s: (float(s), s))
    return {"digits": found, "count": len(found)}


def _drug_name_status(name, raw_token, candidates, matcher_pick):
    """(status, note) for one drug name against mini-list + reference."""
    from . import drug_reference
    from .extract import _load_drug_list

    label = str(name or "").strip().lower()
    if not label:
        return STATUS_REJECTED, "empty drug name"
    known = {d["name"].lower() for d in _load_drug_list()}
    aliases = {}
    for d in _load_drug_list():
        for a in [d["name"]] + d.get("aliases", []):
            aliases[str(a).lower()] = str(d["name"]).lower()
    if label in known or label in aliases:
        return STATUS_VERIFIED, "mini-list exact (name or observed alias)"
    rec = drug_reference.lookup(label, block=drug_reference.STOPWORDS, brands=False)
    if rec:
        return STATUS_VERIFIED, f"reference substance: {rec.get('name')}"
    # Brand-only hit means the token is a trade name, not a chartable row.
    brand = drug_reference.lookup(label, block=frozenset(), brands=True)
    if brand and brand.get("kind") == "brand":
        subs = ", ".join((brand.get("substances") or [])[:3])
        return STATUS_REJECTED, f"brand-only name (of {subs}) - never creates a row"
    cands = list(candidates or [])
    if cands:
        top = str((cands[0] or {}).get("name", "")).lower()
        pick = str(matcher_pick or top).lower()
        if top == label or pick == label:
            return STATUS_VERIFIED, "fuzzy top-candidate agreement"
        return STATUS_NEEDS_REVIEW, f"name disagrees with top candidate {top}"
    ref_cands = drug_reference.candidates(label)
    if ref_cands and str(ref_cands[0].get("name", "")).lower() == label:
        return STATUS_VERIFIED, "reference fuzzy top-candidate agreement"
    if ref_cands:
        return STATUS_NEEDS_REVIEW, f"unsupported name; closest is {ref_cands[0].get('name')}"
    return STATUS_REJECTED, "name matches no mini entry and no reference substance"


def _drug_dose_status(name, dose, unit):
    """(status, note, evidence) for one dose+unit against the pack."""
    from . import drug_reference

    if dose is None or dose == "":
        return STATUS_NEEDS_REVIEW, "no dose dictated", ""
    if unit not in UNITS_OK:
        return STATUS_NEEDS_REVIEW, f"non-UCUM unit {unit!r}", ""
    rec = drug_reference.lookup(str(name or "").lower(), block=drug_reference.STOPWORDS, brands=False)
    if not rec:
        cands = drug_reference.candidates(str(name or "").lower())
        rec = cands[0] if cands else None
    if not rec:
        return STATUS_NEEDS_REVIEW, "no pack record; magnitude unchecked", ""
    fits = drug_reference.unit_fits(rec, unit)
    if fits is False:
        return STATUS_REJECTED, f"unit {unit} is a different dimension for {name}", drug_reference.evidence(rec, dose, unit)
    if drug_reference.dose_conflicts(rec, dose, unit):
        return STATUS_REJECTED, f"dose {dose} {unit} far outside listed range", drug_reference.evidence(rec, dose, unit)
    return STATUS_VERIFIED, "dose/unit plausible", drug_reference.evidence(rec, dose, unit)


def verify_drug(row, raw, normalized, digitized):
    """Verify one drug row. Never promotes; caps on failure."""
    checks, notes = [], []
    name = row.get("name", "")
    status_name, note_name = _drug_name_status(
        name, row.get("raw_token"), row.get("candidates"), row.get("matcher_pick"))
    checks.append(f"name:{status_name}")
    notes.append(note_name)
    status = status_name
    dose, unit = row.get("dose"), row.get("unit")
    status_dose, note_dose, ev = _drug_dose_status(name, dose, unit)
    checks.append(f"dose:{status_dose}")
    notes.append(note_dose + (f" [{ev}]" if ev else ""))
    if status_dose == STATUS_REJECTED or status == STATUS_REJECTED:
        status = STATUS_REJECTED
    elif status_dose == STATUS_NEEDS_REVIEW or status == STATUS_NEEDS_REVIEW:
        status = STATUS_NEEDS_REVIEW
    if dose is not None and dose != "":
        if _number_present(digitized, dose):
            checks.append("numeral:verified")
        else:
            checks.append("numeral:needs_review")
            notes.append(f"dose {dose} not found as digits in transcript")
            status = STATUS_NEEDS_REVIEW if status == STATUS_VERIFIED else status
    if not _proof_ok(row.get("source_sentence"), raw, normalized):
        checks.append("proof:needs_review")
        notes.append("source_sentence not verbatim in transcript")
        status = STATUS_NEEDS_REVIEW if status == STATUS_VERIFIED else status
    else:
        checks.append("proof:verified")
    brand_check, brand_note = _brand_evidence(name, dose, row.get("source_sentence"))
    if brand_check:
        checks.append(brand_check)
        notes.append(brand_note)
        if brand_check == "brand:conflict" and status == STATUS_VERIFIED:
            status = STATUS_NEEDS_REVIEW
    if row.get("negated"):
        checks.append("polarity:negated-preserved")
    conf = row.get("confidence")
    cap = None
    if status == STATUS_REJECTED:
        cap = 0.70
    elif status == STATUS_NEEDS_REVIEW:
        cap = 0.88 if str(row.get("color", "")).upper() != "RED" else 0.70
    if isinstance(conf, (int, float)) and cap is not None and conf > cap:
        conf = cap
    return {"kind": "drug", "name": name, "dose": dose, "unit": unit,
            "negated": bool(row.get("negated", False)), "status": status,
            "checks": checks, "notes": "; ".join(n for n in notes if n),
            "confidence": conf, "source_sentence": row.get("source_sentence", "")}


def verify_vital(row, raw, normalized):
    """Verify one vital span: digits present + physiologically plausible."""
    from .extract import vital_span_plausible

    checks, notes = [], []
    span = str(row.get("text", "") or "")
    ok, note = vital_span_plausible(span)
    status = STATUS_VERIFIED if ok else STATUS_NEEDS_REVIEW
    checks.append(f"plausibility:{'verified' if ok else 'needs_review'}")
    if note:
        notes.append(note)
    nums = re.findall(r"\d+(?:\.\d+)?", span)
    if not nums:
        status = STATUS_REJECTED
        checks.append("numeral:rejected")
        notes.append("no digits in vital span")
    if not _proof_ok(row.get("source_sentence", span), raw, normalized):
        checks.append("proof:needs_review")
        notes.append("vital proof not verbatim in transcript")
        status = STATUS_NEEDS_REVIEW if status == STATUS_VERIFIED else status
    else:
        checks.append("proof:verified")
    conf = row.get("confidence")
    if status == STATUS_NEEDS_REVIEW and isinstance(conf, (int, float)) and conf > 0.70:
        conf = 0.70
    return {"kind": "vital", "text": span, "status": status,
            "checks": checks, "notes": "; ".join(notes),
            "confidence": conf, "source_sentence": row.get("source_sentence", "")}


def verify_allergy(row, raw, normalized):
    """Verify one allergy row: proof present, polarity preserved as-is."""
    checks = ["polarity:preserved"]
    status = STATUS_VERIFIED
    if not _proof_ok(row.get("source_sentence", row.get("text", "")), raw, normalized):
        checks.append("proof:needs_review")
        status = STATUS_NEEDS_REVIEW
        notes = "allergy proof not verbatim in transcript"
    else:
        checks.append("proof:verified")
        notes = "DENIED (negated)" if row.get("negated") else "ASSERTED"
    return {"kind": "allergy", "text": row.get("text", ""), "negated": bool(row.get("negated", False)),
            "status": status, "checks": checks, "notes": notes,
            "confidence": row.get("confidence"), "source_sentence": row.get("source_sentence", "")}


def verify_simple(kind, row, raw, normalized):
    """Proof-only check for symptoms / diagnosis / followup rows."""
    if _proof_ok(row.get("source_sentence", row.get("text", "")), raw, normalized):
        return {"kind": kind, "text": str(row.get("text", "")), "status": STATUS_VERIFIED,
                "checks": ["proof:verified"], "notes": "",
                "confidence": row.get("confidence"), "source_sentence": row.get("source_sentence", "")}
    return {"kind": kind, "text": str(row.get("text", "")), "status": STATUS_NEEDS_REVIEW,
            "checks": ["proof:needs_review"], "notes": "proof not verbatim in transcript",
            "confidence": row.get("confidence"), "source_sentence": row.get("source_sentence", "")}


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_PROOF_STOP = frozenset("the a an and or of for to in on at is was has have with his her he she".split())


def _sentences(text):
    return [x.strip() for x in _SENTENCE_END.split(str(text or "")) if x.strip()]


def recover_proof(text, raw, normalized):
    """The one transcript sentence that states `text`, or "" if not exactly one.

    The LLM-primary path emits diagnosis / follow-up / symptom rows with no
    proof sentence, and an unproven row can never reach the note - so a
    correctly heard `community-acquired pneumonia` was lost. The sentence is
    looked up, never written: it must contain the row's text (or every one of
    its content words) and be the ONLY such sentence in that transcript view.
    Two candidates is ambiguity, and ambiguity stays unproven.
    """
    want = str(text or "").strip().lower()
    if not want:
        return ""
    words = [w for w in re.findall(r"[a-z0-9][a-z0-9'-]*", want) if w not in _PROOF_STOP]
    for view in (raw, normalized):
        sents = _sentences(view)
        hits = [x for x in sents if want in x.lower()]
        if not hits and words:
            hits = [x for x in sents
                    if all(re.search(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])", x.lower())
                           for w in words)]
        unique = list(dict.fromkeys(hits))
        if len(unique) == 1:
            return unique[0]
        if len(unique) > 1:
            return ""
    return ""


def _with_proof(row, raw, normalized):
    """(row, recovered): fill an EMPTY source_sentence by lookup, else as-is."""
    if str(row.get("source_sentence", "") or "").strip():
        return row, False
    text = row.get("text") or row.get("raw_token") or row.get("name") or ""
    sent = recover_proof(text, raw, normalized)
    if not sent:
        return row, False
    return dict(row, source_sentence=sent), True


def verify(transcript, entities):
    """Verify every extracted fact. Returns ``{facts, summary}``.

    ``facts`` is one entry per row (see the per-kind verifiers); ``summary``
    counts them by ``{kind}_{status}`` plus ``numerals`` from the audit.
    Rejected drug/vital rows must not reach chart or LLM output - callers
    filter on ``status != "rejected"`` and send the rest to human review.
    """
    raw, normalized, digitized_raw, digitized_norm = _haystacks(transcript)
    digitized = digitized_raw + " " + digitized_norm
    facts = []

    def run(fn, row, *args):
        row, recovered = _with_proof(row, raw, normalized)
        fact = fn(row, *args)
        if recovered:
            fact["checks"] = list(fact.get("checks", [])) + ["proof:recovered"]
        facts.append(fact)

    for d in entities.get("drugs", []) or []:
        if isinstance(d, dict):
            run(verify_drug, d, raw, normalized, digitized)
    for v in entities.get("vitals", []) or []:
        if isinstance(v, dict):
            run(verify_vital, v, raw, normalized)
    for a in entities.get("allergies", []) or []:
        if isinstance(a, dict):
            run(verify_allergy, a, raw, normalized)
    for s in entities.get("symptoms", []) or []:
        if isinstance(s, dict):
            run(lambda r, *x: verify_simple("symptom", r, *x), s, raw, normalized)
    for key in ("diagnosis", "followup"):
        row = entities.get(key) or {}
        if isinstance(row, dict) and row.get("text"):
            run(lambda r, *x, k=key: verify_simple(k, r, *x), row, raw, normalized)
    numerals = verify_numerals(transcript)
    summary = {"numerals": numerals, "total": len(facts)}
    for f in facts:
        key = f"{f['kind']}_{f['status']}"
        summary[key] = summary.get(key, 0) + 1
    # Dosed brands no drug row accounts for: shown to the reviewer, never
    # charted and never handed to the LLM as a fact.
    from .drug_reference import inn_name

    charted = set()
    for f in facts:
        if f["kind"] == "drug":
            name = str(f.get("name", "")).lower()
            charted |= {name, str(inn_name(name)).lower()}
    unrowed = [m for m in brand_mentions(normalized or raw)
               if not charted & set(m["substances"])]
    if unrowed:
        summary["brand_mentions"] = len(unrowed)
    return {"facts": facts, "summary": summary, "brand_mentions": unrowed}

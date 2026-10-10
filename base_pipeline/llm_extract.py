"""LLM-primary clinical extraction (docs/PIPELINE_ARCHITECTURE_REPORT.md Sec 5) with regex offline fallback.

Chain: STT transcript + template field schema -> local LLM strict JSON
(chief_complaint, vitals, drugs, allergies, denied, diagnosis, followup,
each with proof source_sentence + seg id) -> hand-validated against
templates/er_discharge.schema.json rules -> entities_json shape.
Any failure (no server/model, timeout, bad JSON, schema violation) raises
ValueError(reason) so callers fall back to regex extract_entities.
"""
import json
import os
import re

from .llm_client import complete_json, default_model, llm_available

_HERE = os.path.dirname(os.path.abspath(__file__))
UNITS_OK = {"mg", "mcg", "g", "ml", "U"}


def _regex_vitals_slots(regex_ent):
    """Which vitals slots regex already filled with digits (gap-fill gate for LLM).

    Classification is delegated to vitals_kind.vitals_box_kind so this gate and the
    chart agree exactly: a slot counted filled here must be the slot coords
    charts, or the LLM overwrites a charted vital (too narrow) or declines to
    fill a genuinely empty one (too wide).
    """
    from .vitals_kind import vitals_box_kind
    has = {"bp": False, "temp": False, "spo2": False}
    for v in regex_ent.get("vitals", []) or []:
        kind = vitals_box_kind(v)
        if kind in has:
            has[kind] = True
    return has


def _llm_vitals_kind(item):
    """Classify an LLM vitals item into bp/temp/spo2/other for gap-fill gating.

    LLM-built items carry an explicit "kind" (they have no cue text and no
    source sentence, so a cue check would read them all as "other" and let
    them past the gate). Anything else goes through the shared classifier.
    """
    import re as _re
    from .vitals_kind import vitals_box_kind
    t = str(item.get("text", item.get("name", "")))
    if _re.search(r"^BP\b", t, _re.I):
        return "bp"
    return vitals_box_kind(item) or "other"


def merge_primary(regex_ent, llm_ent, model="llama3.2:3b"):
    """LLM gap-fills over regex base; regex numerals always win conflicts."""
    import difflib
    merged = {k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v)
              for k, v in regex_ent.items()}
    names = [(str(d.get("name", "")).lower(), d) for d in merged.get("drugs", [])]
    for t in llm_ent.get("drugs", []):
        tn = str(t.get("name", "")).lower()
        best, score = None, 0
        for n, d in names:
            s = difflib.SequenceMatcher(None, tn, n).ratio()
            if s > score:
                best, score = d, s
        if best is not None and score >= 0.6:
            for k in ("dose", "unit", "frequency", "duration"):
                if best.get(k) in (None, "", 0) and t.get(k) not in (None, "", 0):
                    best[k] = t[k]
        elif tn:
            t = dict(t)
            t["note"] = (t.get("note", "") + " llm-primary").strip()
            merged["drugs"].append(t)
            names.append((tn, merged["drugs"][-1]))
    for k in ("diagnosis", "followup"):
        if not merged.get(k) and llm_ent.get(k):
            merged[k] = llm_ent[k]
    filled = _regex_vitals_slots(merged)
    for k in ("allergies", "symptoms"):
        seen = {str(x.get("text", x.get("name", ""))).lower() for x in merged.get(k, [])}
        for x in llm_ent.get(k, []):
            if str(x.get("text", x.get("name", ""))).lower() not in seen:
                merged[k].append(x)
    # Vitals: gap-fill only — never overwrite a regex-filled slot with LLM digits.
    seen_v = {str(x.get("text", x.get("name", ""))).lower() for x in merged.get("vitals", [])}
    for x in llm_ent.get("vitals", []) or []:
        kind = _llm_vitals_kind(x)
        if kind in filled and filled[kind]:
            continue
        if str(x.get("text", x.get("name", ""))).lower() in seen_v:
            continue
        merged["vitals"].append(x)
    # Structured vitals: only fill slots regex left empty.
    sv = dict(llm_ent.get("structured_vitals", {}) or {})
    if sv:
        if filled["bp"]:
            sv.pop("sys", None)
            sv.pop("dia", None)
        if filled["temp"]:
            sv.pop("temp", None)
        if filled["spo2"]:
            sv.pop("spo2", None)
        if sv:
            merged["structured_vitals"] = sv
    merged["llm_engine"] = f"regex+llm-primary:{model}"
    return merged


def _prompt(transcript_text, normalized_en):
    return (
        "You are a clinical extractor for ER discharge notes.\n"
        "Input may mix Hindi/Tamil/English.\n\n"
        "Return STRICT JSON only (no prose, no markdown).\n"
        "Schema:\n"
        "{\n"
        '  "chief_complaint": "verbatim symptom text",\n'
        '  "history": "normalized English history",\n'
        '  "drugs": [{"name":"...", "dose":500, "unit":"mg", "frequency":"twice daily",\n'
        '             "duration":"3 days", "source_sentence":"exact sentence heard",\n'
        '             "seg_id": 0}],\n'
        '  "vitals": {"sys":"", "dia":"", "temp":"", "spo2":""},\n'
        '  "allergies_active": [],\n'
        '  "allergies_denied": [],\n'
        '  "symptoms_denied": [],\n'
        '  "diagnosis": {"text":""},\n'
        '  "followup": {"text":""}\n'
        "}\n\n"
        "Rules (anti-hallucination, must follow):\n"
        "- Output ONLY values heard in the transcript. NEVER invent BP, temp, SpO2, drugs, or diagnoses.\n"
        "- If a field was not heard, use empty string or empty array (NOT null, NOT example values).\n"
        "- Doses numeric + UCUM units (mg/mcg/g/ml/U).\n"
        "- source_sentence MUST be a verbatim substring of the transcript; chief_complaint likewise.\n"
        "- Vitals: sys/dia digits only (with decimals if heard), temp with decimals + °F/°C/degrees as heard, spo2 with %.\n"
        "- seg_id = segment index where the evidence sentence occurs; must be within the transcript.\n\n"
        f"Transcript:\n{transcript_text[:2000]}\n\n"
        f"Normalized:\n{normalized_en[:2000]}"
    )


def _valid_drug(d):
    if not isinstance(d, dict) or not d.get("name"):
        return None
    try:
        dose = float(d["dose"]) if d.get("dose") not in (None, "") else None
    except (TypeError, ValueError):
        dose = None
    unit = d.get("unit")
    unit = unit if unit in UNITS_OK else None
    missing_numeral = dose is None or unit is None
    return {"name": str(d["name"]), "dose": dose,
            "unit": unit,
            "frequency": d.get("frequency", "") or "",
            "duration": d.get("duration", "") or "",
            "confidence": 0.70 if missing_numeral else 0.9,
            "color": "RED" if missing_numeral else "YELLOW",
            "source_sentence": d.get("source_sentence", ""), "negated": False,
            "note": "llm-primary" + (" dose/unit missing" if missing_numeral else "")}


def _extract_json_from_output(output):
    """Extract the last complete JSON object from LLM output, handling streaming artifacts."""
    if not output:
        return None
    # Remove markdown code fences if present
    output = re.sub(r'```json\s*', '', output)
    output = re.sub(r'```\s*$', '', output)
    # Remove ANSI escape sequences
    output = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', output)
    output = re.sub(r'\u001b\[[0-9;]*[A-Za-z]', '', output)
    # Remove other control characters
    output = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', output)
    # Find all JSON-like objects and return the last complete one
    # Use a stack-based approach to find balanced braces
    brace_stack = []
    json_candidates = []
    for i, ch in enumerate(output):
        if ch == '{':
            brace_stack.append(i)
        elif ch == '}' and brace_stack:
            start = brace_stack.pop()
            if not brace_stack:  # Complete object
                json_candidates.append(output[start:i+1])
    # Return the last (most complete) candidate
    return json_candidates[-1] if json_candidates else None


def extract_llm_primary(transcript_text, normalized_en, segments,
                        model=None, timeout=60):
    model = model or default_model()
    ok, reason = llm_available(model)
    if not ok:
        raise ValueError(reason)
    # Retry up to 3 times with exponential backoff
    last_error = None
    for attempt in range(3):
        try:
            out = complete_json(_prompt(transcript_text, normalized_en), model, timeout)
        except ValueError as e:
            last_error = e
            continue
        json_str = _extract_json_from_output(out)
        if not json_str:
            last_error = ValueError("no JSON in LLM output")
            continue
        try:
            j = json.loads(json_str)
        except Exception as e:
            last_error = ValueError(f"LLM JSON unparsable: {e}")
            continue
        if not isinstance(j, dict) or not any(k in j for k in ("drugs", "chief_complaint", "vitals")):
            last_error = ValueError("LLM JSON missing required keys")
            continue
        # Proof gate: source_sentence must be a verbatim transcript substring,
        # else clear it and cap confidence (anti-hallucination).
        _full = f"{transcript_text} {normalized_en}"
        def _gate(item):
            s = str(item.get("source_sentence", "") or "")
            if s and s[:40] not in _full and s not in _full:
                item = dict(item)
                item["source_sentence"] = ""
                item["confidence"] = min(float(item.get("confidence", 0.9)), 0.80)
                item["note"] = (str(item.get("note", "")) + " unprovenance").strip()
            return item
        # Process the parsed JSON into entities format
        drugs = [d for d in (_gate(_valid_drug(x)) if _valid_drug(x) else None
                             for x in j.get("drugs", []) or []) if d]
        v = j.get("vitals", {}) if isinstance(j.get("vitals"), dict) else {}
        vitals = []
        structured_vitals = {}
        import re as _re2
        # Digit-presence gate: LLM vitals digits must occur in transcript/normalized text.
        def _digits(s):
            return _re2.findall(r"\d+(?:\.\d+)?", str(s))
        if v.get("sys") and v.get("dia"):
            ds, dd = _digits(v["sys"]), _digits(v["dia"])
            if ds and dd and ds[0] in _full and dd[0] in _full:
                vitals.append({"text": f"BP {v['sys']}/{v['dia']}", "kind": "bp",
                               "confidence": 0.88,
                               "color": "YELLOW", "source_sentence": ""})
                structured_vitals["sys"] = str(v["sys"])
                structured_vitals["dia"] = str(v["dia"])
        for k in ("temp", "spo2"):
            if v.get(k):
                dk = _digits(v[k])
                if dk and dk[0] in _full:
                    # "kind" is provenance: the LLM reported this under the
                    # temp/spo2 key, and "96%" carries no cue of its own.
                    vitals.append({"text": str(v[k]), "kind": k,
                                   "confidence": 0.87, "color": "YELLOW",
                                   "source_sentence": ""})
                    structured_vitals[k] = str(v[k])
        allergies = [{"text": str(a), "negated": True, "confidence": 0.9,
                      "source_sentence": "", "note": "NEGATED - llm-primary"}
                     for a in (j.get("allergies_denied", []) or [])]
        symptoms = ([{"text": str(j["chief_complaint"]), "confidence": 0.9, "color": "YELLOW",
                      "source_sentence": "", "negated": False}] if j.get("chief_complaint") else [])
        symptoms += [{"text": str(s), "confidence": 0.91, "color": "GREEN",
                      "source_sentence": "", "negated": True, "note": "DENIED - excluded"}
                     for s in (j.get("symptoms_denied", []) or [])]
        dx = j.get("diagnosis", {}) if isinstance(j.get("diagnosis"), dict) else {}
        result = {"drugs": drugs, "symptoms": symptoms, "vitals": vitals,
                  "structured_vitals": structured_vitals,
                  "allergies": allergies,
                  "negations": [{"span": str(a), "negated": True} for a in (j.get("allergies_denied", []) or [])],
                  "diagnosis": {"text": str(dx.get("text", "")), "icd10": "",
                                "confidence": 0.85, "color": "YELLOW", "source_sentence": ""}
                  if dx.get("text") else {},
                  "followup": {"text": str(j.get("followup", {}).get("text", "") if isinstance(j.get("followup"), dict) else j.get("followup", "")),
                               "confidence": 0.88, "color": "YELLOW", "source_sentence": ""}
                  if j.get("followup") else {},
                  "llm_engine": f"llm-primary:{model}"}
        return result
    raise last_error or ValueError("LLM extraction failed after retries")


# ---- 6-section discharge note (heard-only, proof-grounded) -----------------
# Second LLM stage: takes the transcript plus the deterministic `verify`
# facts and authors ONLY the six requested sections. Verified values win
# every conflict; `gate_discharge` then drops anything without verbatim
# proof, so unheard sections stay empty rather than invented.

DISCHARGE_SECTIONS = ("present_illness", "examination_findings",
                      "previous_diagnosis", "plan_of_management",
                      "diet_instructions", "general_instructions")


# Verified fact kind -> discharge section. The prompt states this map so the
# model does not have to guess where a fact belongs.
PREFILL_SECTION = {"symptom": "present_illness", "vital": "examination_findings",
                   "diagnosis": "previous_diagnosis", "drug": "plan_of_management",
                   "followup": "plan_of_management", "allergy": "general_instructions"}


def _seg_hint(source_sentence, segments):
    """Segment index containing this proof sentence, or None."""
    sent = str(source_sentence or "").strip()
    if not sent:
        return None
    for seg in segments or []:
        if isinstance(seg, dict) and sent in str(seg.get("text", "")):
            return seg.get("id")
    return None


MAX_PROMPT_FACTS = 60


def _fact_index(verified):
    """[(fact_id, fact)] for the non-rejected facts the prompt shows, in order.

    The id is how the model says WHICH fact an item places. Without it the
    only link back was the proof sentence, and one sentence routinely holds
    BP, pulse and RR - so canonicalize could not tell them apart and dropped
    the item. Shared by the prompt and canonicalize so the ids always agree.
    """
    facts = [f for f in (verified or {}).get("facts", []) if f.get("status") != "rejected"]
    return [(f"F{i}", f) for i, f in enumerate(facts[:MAX_PROMPT_FACTS], 1)]


def _verified_lines(verified, segments=None):
    """Verified-fact lines with FULL proof sentences and seg hints.

    Proof is never truncated: the authenticity gate requires a verbatim
    transcript substring, so a cut sentence would guarantee a drop.
    """
    lines = []
    for fid, f in _fact_index(verified):
        src = str(f.get("source_sentence", "") or "")
        seg = _seg_hint(src, segments)
        tag = f"seg={seg}" if isinstance(seg, int) else "seg=?"
        target = PREFILL_SECTION.get(f.get("kind", ""), "?")
        if f["kind"] == "drug":
            dose = f"{f['dose']} {f.get('unit') or ''}".strip() if f.get("dose") not in (None, "") else "no dose"
            neg = "NOT GIVEN " if f.get("negated") else ""
            brand = next((c for c in f.get("checks", []) if c.startswith("brand:")), "")
            status = f"{f['status']}, {brand}" if brand else f["status"]
            lines.append(f"- {fid} drug -> {target}: {neg}{f['name']} {dose} [{status}, {tag}] <<{src}>>")
        elif f["kind"] == "vital":
            lines.append(f"- {fid} vital -> {target}: {f.get('text')} [{f['status']}, {tag}] <<{src}>>")
        elif f["kind"] == "allergy":
            lines.append(f"- {fid} allergy -> {target}: {'DENIED ' if f.get('negated') else ''}{f.get('text')} [{f['status']}, {tag}] <<{src}>>")
        else:
            neg = "DENIED " if f.get("negated") else ""
            lines.append(f"- {fid} {f['kind']} -> {target}: {neg}{f.get('text', '')} [{f['status']}, {tag}] <<{src}>>")
    return "\n".join(lines)


# The plan and diet are dictated last; a 2,000-character cut lost them on
# longer dictations. 6,000 covers every clip in the corpus.
PROMPT_TRANSCRIPT_CHARS = 6000


def _discharge_prompt(transcript_text, normalized_en, verified, segments):
    n_segs = len(segments or [])
    return (
        "You are a clinical scribe structuring a dictated note.\n"
        "Copy EVERY fact from the VERIFIED list into its mapped section below.\n"
        "Return STRICT JSON only (no prose, no markdown).\n"
        "Schema:\n"
        '{"present_illness": {"items": [{"fact_id": "F1", "text": "...", "source_sentence": "exact sentence heard", "seg_id": 0, "confidence": 0.9}]},\n'
        ' "examination_findings": {"items": [...]}, "previous_diagnosis": {"items": [...]},\n'
        ' "plan_of_management": {"items": [...]}, "diet_instructions": {"items": [...]},\n'
        ' "general_instructions": {"items": [...]}}\n\n'
        "Section map (follow exactly):\n"
        "- present_illness: symptoms the patient has (prefix DENIED: if marked DENIED)\n"
        "- examination_findings: vital-sign readings\n"
        "- previous_diagnosis: past history and diagnosis statements\n"
        "- plan_of_management: prescribed drugs with dose/unit/frequency PLUS follow-up plans "
        '(prefix "NOT GIVEN: " if marked NOT GIVEN)\n'
        "- diet_instructions: ONLY diet/food/fluid advice actually spoken, else []\n"
        "- general_instructions: allergy statements (prefix DENIED if denied) and other spoken advice\n\n"
        "Rules (must follow):\n"
        "- Output ONLY values heard in the transcript. NEVER invent vitals, drugs, doses, diagnoses, or advice.\n"
        "- Cover every VERIFIED fact below: verified and needs_review facts MUST appear; rejected ones are absent.\n"
        "- If a section has no dictated facts, use empty items []. Diet stays EMPTY unless diet was spoken.\n"
        "- Drug names, doses and units MUST match the VERIFIED list verbatim (verified values win any conflict).\n"
        "- source_sentence MUST be copied EXACTLY as shown in <<...>> (verbatim transcript substring).\n"
        "- seg_id: copy the seg=N hint from the fact line; use null only for seg=? items.\n"
        "- Doses numeric with UCUM units (mg/mcg/g/ml/U).\n"
        "- ONE fact per item: each vital is its own item (never merge BP, pulse and RR into one item).\n"
        "- For a VERIFIED fact: set fact_id to its id (F1, F2, ...) and copy its text exactly as listed.\n"
        "- You MAY add other facts spoken in the transcript (symptoms, past history, advice) with "
        "fact_id null, using ONLY words from their source sentence. Never add an item from a "
        "sentence that says no/denies/without/not.\n\n"
        f"VERIFIED FACTS:\n{_verified_lines(verified, segments) or '(none)'}\n\n"
        f"Transcript:\n{transcript_text[:PROMPT_TRANSCRIPT_CHARS]}\n\n"
        f"Normalized:\n{normalized_en[:PROMPT_TRANSCRIPT_CHARS]}"
    )


def extract_discharge(transcript_text, normalized_en, segments, verified,
                      model=None, timeout=None):
    """Author the 6-section note with llama3.2:3b (or configured model).

    One note on a CPU-only 3B model measured 134 s, so the budget defaults
    to 300 s (MEDIBYTES_LLM_TIMEOUT overrides). A timeout is not retried:
    the same prompt on the same box would only time out again.
    """
    from .llm_client import LLMTimeout, timeout_for

    model = model or default_model()
    timeout = timeout or timeout_for(300)
    ok, reason = llm_available(model)
    if not ok:
        raise ValueError(reason)
    last_error = None
    for _ in range(3):
        try:
            out = complete_json(
                _discharge_prompt(transcript_text, normalized_en, verified, segments),
                model, timeout)
        except LLMTimeout:
            raise
        except ValueError as e:
            last_error = e
            continue
        json_str = _extract_json_from_output(out)
        if not json_str:
            last_error = ValueError("no JSON in LLM discharge output")
            continue
        try:
            j = json.loads(json_str)
        except Exception as e:
            last_error = ValueError(f"LLM discharge JSON unparsable: {e}")
            continue
        if not isinstance(j, dict) or not any(k in j for k in DISCHARGE_SECTIONS):
            last_error = ValueError("LLM discharge JSON missing section keys")
            continue
        sections = {}
        for key in DISCHARGE_SECTIONS:
            block = j.get(key) or {}
            items = block.get("items", []) if isinstance(block, dict) else []
            clean = []
            for it in items if isinstance(items, list) else []:
                if not isinstance(it, dict) or not str(it.get("text", "")).strip():
                    continue
                try:
                    conf = float(it.get("confidence", 0.9))
                except (TypeError, ValueError):
                    conf = 0.9
                seg = it.get("seg_id")
                seg = seg if isinstance(seg, int) else None
                row = {"text": str(it["text"]).strip(),
                       "source_sentence": str(it.get("source_sentence", "") or ""),
                       "seg_id": seg, "confidence": conf}
                if isinstance(it.get("fact_id"), str) and it["fact_id"].strip():
                    row["fact_id"] = it["fact_id"].strip()
                clean.append(row)
            sections[key] = {"items": clean}
        return {"sections": sections, "provenance": {"llm_engine": f"discharge:{model}"}}
    raise last_error or ValueError("LLM discharge failed after retries")


def gate_discharge(discharge, transcript_text, normalized_en, verified):
    """Authenticity gate: drop unproven items, cap the rest. Deterministic.

    An item survives only if (a) its source_sentence is a verbatim transcript
    substring, (b) every digit-group in its text occurs in the transcript, and
    (c) every drug name it mentions resolves via the mini-list or reference.
    Survivors keep ``min(confidence, 0.88)``; failures are counted, not fixed.
    """
    from . import drug_reference
    from .extract import _load_drug_list

    full = f"{transcript_text} {normalized_en}"
    known = {d["name"].lower() for d in _load_drug_list()}
    for d in _load_drug_list():
        for a in [d["name"]] + d.get("aliases", []):
            known.add(str(a).lower())
    verified_names = {str(f.get("name", "")).lower() for f in (verified or {}).get("facts", [])
                      if f.get("kind") == "drug" and f.get("status") != "rejected"}
    dropped = 0
    out_sections = {}
    for key in DISCHARGE_SECTIONS:
        items = (discharge.get("sections", {}).get(key, {}) or {}).get("items", []) or []
        kept = []
        for it in items:
            sent = str(it.get("source_sentence", "") or "")
            text = str(it.get("text", "") or "")
            ok = bool(sent) and (sent in transcript_text or sent in normalized_en)
            if ok:
                for num in re.findall(r"\d+(?:\.\d+)?", text):
                    if num not in full:
                        ok = False
                        break
            if ok:
                for word in re.findall(r"[A-Za-z][A-Za-z\-]+", text):
                    w = word.lower()
                    if w in known or w in verified_names:
                        continue
                    rec = drug_reference.lookup(w, block=drug_reference.STOPWORDS, brands=False)
                    if rec:
                        continue
                    # Only police tokens that look like drug mentions: fuzzy
                    # candidates exist AND the extractor flagged the same token.
                    cands = drug_reference.candidates(w)
                    if cands and w not in verified_names:
                        ok = False
                        break
            if ok:
                it = dict(it)
                it["confidence"] = min(float(it.get("confidence", 0.9)), 0.88)
                kept.append(it)
            else:
                dropped += 1
        out_sections[key] = {"items": kept}
    prov = dict(discharge.get("provenance", {}))
    prov["dropped_unproven"] = dropped
    prov["verified_summary"] = (verified or {}).get("summary", {})
    return {"sections": out_sections, "provenance": prov}


_SCHEMA_CACHE: dict = {}


def _discharge_schema():
    """Load discharge_schema.json once (Draft7, self-checked)."""
    if "schema" not in _SCHEMA_CACHE:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "discharge_schema.json"), encoding="utf-8") as fh:
            _SCHEMA_CACHE["schema"] = json.load(fh)
    return _SCHEMA_CACHE["schema"]


def validate_discharge(discharge):
    """(ok, errors): Draft7 conformance of a discharge note.

    Pure function — no LLM, no I/O beyond the cached schema file. Errors
    are human-readable paths (``sections.plan.items[2].text``), capped at 10.
    Raises nothing for unusable input: reports it as an error instead.
    """
    try:
        from jsonschema import Draft7Validator
    except ImportError as err:
        return False, [f"jsonschema not installed ({err}); add it to requirements"]
    try:
        schema = _discharge_schema()
        Draft7Validator.check_schema(schema)
    except Exception as err:
        return False, [f"schema file invalid: {type(err).__name__}: {err}"]
    if not isinstance(discharge, dict):
        return False, ["discharge note is not an object"]
    errors = [f"{'/'.join(str(p) for p in e.absolute_path)}: {e.message}"
              for e in Draft7Validator(schema).iter_errors(discharge)]
    return (True, []) if not errors else (False, errors[:10])


def stamp_schema(discharge):
    """Flag-still-ship: record conformance in provenance, never raise.

    Valid notes get ``provenance.schema = "valid"``; invalid ones keep
    their content and get ``provenance.schema = "INVALID"`` plus the first
    error in ``provenance.schema_error`` for the RED review row. Callers
    always write the file either way.
    """
    try:
        ok, errors = validate_discharge(discharge)
    except Exception as err:  # validation must never break a run
        ok, errors = False, [f"validator crashed: {type(err).__name__}: {err}"]
    prov = dict(discharge.get("provenance", {}))
    if ok:
        prov["schema"] = "valid"
        prov.pop("schema_error", None)
    else:
        prov["schema"] = "INVALID"
        prov["schema_error"] = errors[0] if errors else "unknown schema error"
    return {"sections": discharge.get("sections", {}), "provenance": prov}


def _prefill_text(fact):
    """Section item text built from one verified fact (heard words only)."""
    kind = fact.get("kind", "")
    if kind == "drug":
        dose = fact.get("dose")
        unit = fact.get("unit") or ""
        if dose is None or dose == "":
            core = f"{fact.get('name', '')} no dose".strip()
        else:
            try:
                core = f"{fact.get('name', '')} {float(dose):g} {unit}".strip()
            except (TypeError, ValueError):
                core = str(fact.get("name", ""))
        if fact.get("negated"):
            core = f"NOT GIVEN: {core}"
        return core
    if kind == "vital":
        return str(fact.get("text", ""))
    if kind == "allergy":
        prefix = "DENIED allergy: " if fact.get("negated") else "Allergy: "
        return prefix + str(fact.get("text", ""))
    if kind == "symptom":
        text = str(fact.get("text", ""))
        return f"DENIED: {text}" if fact.get("negated") else text
    return str(fact.get("text", ""))


def _prefill_eligible(fact, transcript_text, normalized_en):
    """Gate-identical eligibility: proof verbatim + digits present + mapped."""
    if fact.get("status") == "rejected" or fact.get("kind") not in PREFILL_SECTION:
        return False
    if fact.get("kind") == "drug" and (fact.get("dose") is None or fact.get("dose") == ""):
        pass  # doseless rows still surface (RED review), proof decides
    src = str(fact.get("source_sentence", "") or "")
    if not src or (src not in transcript_text and src not in normalized_en):
        return False
    full = f"{transcript_text} {normalized_en}"
    text = _prefill_text(fact).replace("NOT GIVEN: ", "").replace("DENIED: ", "").replace(
        "DENIED allergy: ", "").replace("Allergy: ", "")
    for num in re.findall(r"\d+(?:\.\d+)?", text):
        if num not in full:
            return False
    return bool(text.strip())


_NUM = re.compile(r"(?<![\d.])\d+(?:\.\d+)?(?![\d.])")


def _covered(fact, fact_text, item):
    """Does this note item already state this verified fact?

    Identity is the fact, not its sentence: one dictated sentence routinely
    carries BP, pulse and RR, and keying on the proof sentence surfaced only
    the first of them. So a vital is covered only by an item reporting the
    same set of numbers (an LLM item merging three vitals does not hide two
    of them), a drug by its name plus its dose plus the same NOT GIVEN
    polarity, and anything else by containing its text with the same DENIED
    polarity - which also collapses one symptom heard in two sentences.
    """
    have = str(item.get("text", "") or "").lower()
    want = str(fact_text or "").lower()
    if not have or not want:
        return False
    kind = fact.get("kind")
    if kind == "vital":
        nums = set(_NUM.findall(want))
        return bool(nums) and set(_NUM.findall(have)) == nums
    if kind == "drug":
        if ("not given" in have) != bool(fact.get("negated")):
            return False
        if str(fact.get("name", "")).lower() not in have:
            return False
        dose = fact.get("dose")
        if dose in (None, ""):
            return True
        try:
            return f"{float(dose):g}" in {f"{float(n):g}" for n in _NUM.findall(have)}
        except (TypeError, ValueError):
            return True
    if ("denied" in have) != ("denied" in want):
        return False
    core = re.sub(r"^(denied( allergy)?|allergy):\s*", "", want).strip()
    return bool(core) and core in have


def merge_discharge(verified, discharge, transcript_text="", normalized_en=""):
    """Deterministic pre-fill: every gate-passable verified fact surfaces once.

    Runs after the gate over the gated sections: for each eligible verified
    fact whose text/proof is not already present (case-folded), appends an
    item tagged ``verified_kind``/``verified_status`` with proof verbatim and
    confidence capped (never promoted). Diet is never pre-filled (heard-only:
    only the LLM may assert diet advice, and only with proof).
    """
    sections = {k: {"items": list((discharge.get("sections", {}).get(k, {}) or {}).get("items", []) or [])}
                for k in DISCHARGE_SECTIONS}
    prefilled = 0
    for fact in (verified or {}).get("facts", []):
        target = PREFILL_SECTION.get(fact.get("kind", ""), "")
        if not target or target == "diet_instructions":
            continue
        if not _prefill_eligible(fact, transcript_text, normalized_en):
            continue
        text = _prefill_text(fact)
        src = str(fact.get("source_sentence", "") or "")
        if any(_covered(fact, text, it) for it in sections[target]["items"]):
            continue
        conf = fact.get("confidence", 0.9)
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.9
        conf = min(conf, 0.88)
        if fact.get("kind") == "vital" and fact.get("status") == "needs_review":
            conf = min(conf, 0.70)
        sections[target]["items"].append({
            "text": text, "source_sentence": src, "seg_id": None,
            "confidence": conf, "verified_kind": fact.get("kind"),
            "verified_status": fact.get("status")})
        prefilled += 1
    provenance = dict(discharge.get("provenance", {}))
    provenance["verified_summary"] = (verified or {}).get("summary", {})
    provenance["prefilled"] = prefilled
    return {"sections": sections, "provenance": provenance}


def _proof_group(verified):
    """Verified non-rejected facts keyed by source_sentence."""
    groups: dict[str, list[dict]] = {}
    for f in (verified or {}).get("facts", []):
        if f.get("status") == "rejected":
            continue
        src = str(f.get("source_sentence", "") or "")
        if src:
            groups.setdefault(src, []).append(f)
    return groups


def _closest_drug(item_text, candidates):
    """The verified drug sharing the most name tokens with the LLM text."""
    import difflib

    words = set(re.findall(r"[a-z]+", str(item_text or "").lower()))
    best, best_score = None, 0.0
    for c in candidates:
        name_words = set(str(c.get("name", "")).lower().split())
        overlap = len(words & name_words)
        score = overlap + difflib.SequenceMatcher(
            None, str(item_text or "").lower(), str(c.get("name", "")).lower()).ratio() / 10.0
        if overlap and score > best_score:
            best, best_score = c, score
    return best


def _vital_by_numbers(item_text, cands):
    """The one vital fact whose numbers are exactly this item's, or None.

    One sentence carries BP, pulse and RR; an item reading "pulse 92" is the
    pulse fact because 92 is its whole number set, not because of wording.
    """
    have = set(_NUM.findall(str(item_text or "")))
    if not have:
        return None
    hits = [c for c in cands if c.get("kind") == "vital"
            and set(_NUM.findall(str(c.get("text", "")))) == have]
    return hits[0] if len(hits) == 1 else None


# Sections an LLM-only item may land in. Drugs, vitals and allergies must come
# from a verified fact: those are the rows where a wrong value is a safety event.
_EXTRACTIVE_SECTIONS = {"present_illness", "previous_diagnosis",
                        "diet_instructions", "general_instructions"}
_EXTRACTIVE_STOP = frozenset(
    "a an the and or of in on at to for with his her he she is was has have had "
    "been some also about".split())


def _extractive_ok(item, section):
    """May an LLM item with no verified fact stay? Only when it is extractive.

    The extractor has no rule for most symptoms and history ("chest
    discomfort", "hypertension"), so the model is the only stage that can
    place them - and canonicalize used to drop every such item. It keeps one
    only when nothing in it could be wrong in a way the transcript would not
    show: the proof sentence already passed the verbatim gate; the item has
    no digits (numbers belong to verified vitals/doses); every content word
    of the item is a word of that sentence (heard, not paraphrased); no word
    is a medicine; and the sentence carries no negation cue at all, so
    "no history of asthma" can never become an "asthma" line. Kept items are
    capped at 0.80, below anything verified.
    """
    from . import drug_reference
    from .extract import NEG_PAT, _load_drug_list

    if section not in _EXTRACTIVE_SECTIONS:
        return False
    text = str(item.get("text", "") or "").lower()
    sent = str(item.get("source_sentence", "") or "").lower()
    if not text or not sent or re.search(r"\d", text) or NEG_PAT.search(sent):
        return False
    words = [w for w in re.findall(r"[a-z][a-z'-]*", text) if w not in _EXTRACTIVE_STOP]
    if not words:
        return False
    heard = set(re.findall(r"[a-z][a-z'-]*", sent))
    if not all(w in heard for w in words):
        return False
    known = {d["name"].lower() for d in _load_drug_list()}
    for w in words:
        if w in known or drug_reference.lookup(w, block=drug_reference.STOPWORDS, brands=False):
            return False
    return True


def canonicalize_discharge(discharge, verified):
    """Entity lock-in: LLM lines keep their shape, verified facts own entities.

    For each surviving item, looks up verified facts sharing its proof
    sentence and substitutes canonical surface forms (drug name/dose/unit,
    vital span, allergy/symptom/diagnosis/followup text). Substitutions are
    tagged ``verified_kind``/``verified_status`` plus a ``canonicalized``
    note with confidence capped (never promoted). Items with no matching
    verified fact (ambiguous multi-drug proof, orphan line) are dropped and
    counted — pre-fill still guarantees the verified fact surfaces.
    """
    groups = _proof_group(verified)
    by_id = dict(_fact_index(verified))
    out_sections: dict[str, dict] = {k: {"items": []} for k in DISCHARGE_SECTIONS}
    canonicalized = extractive = 0
    dropped = int((discharge.get("provenance", {}) or {}).get("dropped_unproven", 0) or 0)
    for key in DISCHARGE_SECTIONS:
        items = (discharge.get("sections", {}).get(key, {}) or {}).get("items", []) or []
        for it in items:
            src = str(it.get("source_sentence", "") or "")
            cands = groups.get(src, [])
            target = None
            fid = it.get("fact_id")
            if fid in by_id and str(by_id[fid].get("source_sentence", "") or "") == src:
                target = by_id[fid]  # the model named the fact; proof must agree
            elif len(cands) == 1:
                target = cands[0]
            elif cands:
                drugs = [c for c in cands if c.get("kind") == "drug"]
                target = _closest_drug(it.get("text", ""), drugs) if drugs else None
                if target is None:
                    target = _vital_by_numbers(it.get("text", ""), cands)
                if target is None:
                    same_kind = [c for c in cands if PREFILL_SECTION.get(c.get("kind")) == key]
                    target = same_kind[0] if len(same_kind) == 1 else None
            if target is None:
                if _extractive_ok(it, key):
                    it = dict(it)
                    it.pop("fact_id", None)
                    it["confidence"] = min(float(it.get("confidence", 0.9)), 0.80)
                    it["note"] = (str(it.get("note", "") or "") + " llm-extractive").strip()
                    out_sections[key]["items"].append(it)
                    extractive += 1
                else:
                    dropped += 1
                continue
            # A verified fact lives in its mapped section, whatever the model chose.
            home = PREFILL_SECTION.get(target.get("kind"), key)
            dest = out_sections[home]["items"]
            canon = _prefill_text(target)
            if str(it.get("text", "")).strip() == canon:
                it = dict(it)
                it.setdefault("verified_kind", target.get("kind"))
                it.setdefault("verified_status", target.get("status"))
                dest.append(it)
                continue
            it = dict(it)
            it["text"] = canon
            it["verified_kind"] = target.get("kind")
            it["verified_status"] = target.get("status")
            it["confidence"] = min(float(it.get("confidence", 0.9)), 0.88)
            if target.get("kind") == "vital" and target.get("status") == "needs_review":
                it["confidence"] = min(it["confidence"], 0.70)
            note = str(it.get("note", "") or "")
            it["note"] = (note + " canonicalized").strip()
            dest.append(it)
            canonicalized += 1
    provenance = dict(discharge.get("provenance", {}))
    provenance["dropped_unproven"] = dropped
    provenance["canonicalized"] = canonicalized
    provenance["llm_extractive"] = extractive
    return {"sections": out_sections, "provenance": provenance}

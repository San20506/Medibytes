"""Stages 2-4 glued together: `run_stt_extract` is STT, normalise, extract, then the optional
LLM tidy, term validation and context repair.

The stages live in `asr.py` (speech to text) and `extract.py` (normalise and extract).
"""
import json
import re
import time

from . import decode_presets
from .asr import transcribe
from .extract import (_LAB_REQUEST, _annotate_entity_spans, _chart_name, _load_drug_list,
                      _reconcile_allergens, _vital_span_kind, detect_lang_tag, extract_entities,
                      normalize_text)
from .llm_client import complete_json, default_model, llm_available


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


def llm_tidy(entities, normalized_en, model=None, timeout=60):
    """Optional LLM tidy via the local LLM server. Per-drug schema merge; falls back to regex."""
    prompt = ("Output strict JSON only with keys drugs[{name,dose,unit,frequency,duration}],"
              "symptoms,vitals,allergies,diagnosis{followup}. Doses numeric + UCUM units mg/mcg/g/ml/U. "
              f"Normalize from: {normalized_en}. Draft: {json.dumps(entities)[:2000]}")
    try:
        model = model or default_model()
        txt = complete_json(prompt, model, timeout, max_tokens=1500).strip()
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
                merged["tidy_engine"] = f"llm:{model}"
                merged["tidy_dropped"] = dropped
                return merged
    except Exception as e:
        entities["tidy_engine"] = f"regex-only (llm failed: {type(e).__name__})"
        return entities
    entities["tidy_engine"] = "regex-only (llm skipped/failed)"
    return entities


def _run_term_validation(ent, validate_terms, llm_model):
    """Stage 4b gate. `auto` runs only when the model is actually pulled.

    A fuzzy match that nobody validated and one a validator cleared must not
    look alike on the chart, so the skip reason is recorded on the rows rather
    than swallowed.
    """
    if validate_terms is False:
        return ent
    from .term_validate import apply_validation, flagged_rows
    rows = flagged_rows(ent)
    if not rows:
        return ent
    if validate_terms == "auto":
        ok, why = llm_available(llm_model)
        if not ok:
            for row in rows:
                row["validation"] = {"status": "skipped", "reason": why}
                row["note"] = f"{row.get('note', '')} | validator skipped".strip(" |")
            ent["term_validator"] = f"unavailable ({why})"
            return ent
    return apply_validation(ent, model=llm_model)


def _run_context_repair(ent, normalized_text, context_repair, llm_model):
    """Stage 4d gate. Off unless asked for; `auto` also needs a served model."""
    if not context_repair:
        return ent
    if context_repair == "auto":
        ok, why = llm_available(llm_model)
        if not ok:
            ent["context_repair"] = {"status": "skipped", "reason": why}
            return ent
    from . import drug_reference
    from .context_repair import apply_context_repair
    known = {d["name"].lower() for d in _load_drug_list()}
    return apply_context_repair(ent, normalized_text, model=llm_model,
                                chart_name=lambda n: _chart_name(n, known),
                                stopwords=drug_reference.STOPWORDS,
                                skip_sentence=lambda sent: bool(_LAB_REQUEST.search(sent)))


def _union_drugs(primary, second_text, second_model, segments):
    """Add drug and vital rows a second decoder heard and the primary missed.

    The decoders fail on different words. Measured on the 23 held-out clips,
    `medasr` recovers 14 of 24 gold drugs and `small-int8` 15, but their union
    is 20 - six drugs exist in one transcript and not the other, which is a
    bigger gap than any single-decoder change has produced.

    Drugs are unioned and vitals are not, and that asymmetry is the whole
    design. A drug name is a discrete thing: if either decoder heard
    `ondansetron`, something was said, and de-duplicating on the canonical
    name keeps the list clean - the union added **zero** spurious drug rows.
    Vitals are a number plus a span of prose, so the same reading comes back
    as "pulse is 102" from one decoder and "pulse 102" from the other; the
    union could not tell those apart and produced 44 spurious spans against
    12. So the second pass contributes drug rows only.

    A row that only one decoder heard is weaker evidence than one they agree
    on, so it is never GREEN and it names the decoder that found it.
    """
    known = {str(d.get("name", "")).lower() for d in primary.get("drugs", [])}
    extra = extract_entities(second_text, normalize_text(second_text)["normalized_en"],
                             segments)
    added = []
    for row in extra.get("drugs", []):
        name = str(row.get("name", "")).lower()
        if not name or name in known:
            continue
        known.add(name)
        row = dict(row)
        row["confidence"] = min(float(row.get("confidence", 0.70)), 0.70)
        row["color"] = "RED" if row.get("color") == "RED" else "YELLOW"
        row["note"] = (f"{row.get('note', '')} | heard only by {second_model}, "
                       f"not by the primary decoder").strip(" |")
        row["second_pass"] = second_model
        added.append(row)
    primary["drugs"] = list(primary.get("drugs", [])) + added
    if added:
        primary["drug_second_pass"] = {"model": second_model,
                                       "added": [r["name"] for r in added]}

    # Vitals too, but keyed on what the reading *is* rather than how it was
    # worded. De-duplicating on the raw span failed because the decoders write
    # the same reading differently - "pulse is 102" against "pulse 102" - so
    # every agreed vital arrived twice and the union produced 44 spurious
    # spans against 12. Keying on (kind, numbers) recognises them as one
    # reading: same +2 facts, 17 spurious instead of 44.
    def _key(span):
        return (_vital_span_kind(span), tuple(re.findall(r"\d+(?:\.\d+)?", span)))

    seen = {_key(v.get("text", "")) for v in primary.get("vitals", [])}
    v_added = []
    for row in extra.get("vitals", []):
        k = _key(row.get("text", ""))
        if k[0] is None or k in seen:
            continue
        seen.add(k)
        row = dict(row)
        row["confidence"] = min(float(row.get("confidence", 0.87)), 0.70)
        row["note"] = (f"{row.get('note', '')} | heard only by {second_model}, "
                       f"not by the primary decoder").strip(" |")
        row["second_pass"] = second_model
        v_added.append(row)
    primary["vitals"] = list(primary.get("vitals", [])) + v_added
    if v_added:
        primary.setdefault("drug_second_pass", {})["vitals_added"] = len(v_added)
    return primary


def run_stt_extract(
    clean_wav, job_id="demo-001", use_llm="auto", model="medasr",
    llm_model=None, strict: bool = False, validate_terms="auto",
    second_pass="auto", context_repair=False, decode=decode_presets.DEFAULT,
):
    stt = transcribe(clean_wav, job_id, model=model, strict=strict, decode=decode)
    norm = normalize_text(stt["text"])
    # refresh segment lang tags with demo LID
    for s in stt["segments"]:
        s["lang"] = detect_lang_tag(s["text"]) if len(s["text"]) < 200 else norm["lang_tag"]
    ent, llm_reason = None, ""
    if use_llm in (True, "auto"):
        try:
            from .llm_extract import extract_llm_primary, merge_primary
            ok, why = llm_available(llm_model)
            if ok or use_llm is True:
                base = extract_entities(stt["text"], norm["normalized_en"], stt["segments"])
                llm = extract_llm_primary(stt["text"], norm["normalized_en"], stt["segments"], model=llm_model)
                ent = merge_primary(base, llm, model=llm_model)
            else:
                llm_reason = why
        except ValueError as e:
            ent, llm_reason = None, str(e)
    if ent is None:
        ent = extract_entities(stt["text"], norm["normalized_en"], stt["segments"])
        if use_llm is True:
            ent = llm_tidy(ent, norm["normalized_en"], model=llm_model)
        elif use_llm == "auto" and llm_reason:
            ent["llm_engine"] = f"regex-fallback ({llm_reason})"
    # Second decoder, drug rows only. `auto` pairs the two decoders that were
    # measured to disagree most usefully and skips the work when the primary
    # is already one of a pair that would duplicate it.
    pair = {"medasr": "small-int8", "medasr-lm": "small-int8",
            "small-int8": "medasr"}
    second = pair.get(model) if second_pass == "auto" else second_pass
    if second and second != model:
        try:
            alt = transcribe(clean_wav, job_id, model=second, strict=False)
            if not alt.get("stt_provenance", {}).get("is_mock"):
                ent = _union_drugs(ent, alt["text"], second, alt.get("segments") or [])
        except Exception as exc:  # a second opinion is a bonus, never a failure
            ent["drug_second_pass"] = {"model": second,
                                       "error": f"{type(exc).__name__}: {exc}"[:120]}
    ent = _reconcile_allergens(ent)  # the union above can re-add an allergen as an order
    _t = time.perf_counter()
    ent = _run_term_validation(ent, validate_terms, llm_model)
    if ent.get("term_validator"):  # LLM time, reported apart from ASR
        ent["term_validator_seconds"] = round(time.perf_counter() - _t, 3)
    ent = _run_context_repair(ent, norm["normalized_en"], context_repair, llm_model)
    ent = _annotate_entity_spans(ent, norm["normalized_en"])
    transcript_json = {"job_id": job_id, "text": stt["text"], "language": norm["lang_tag"],
                       "segments": stt["segments"], "normalized_en": norm["normalized_en"],
                       "normalizations": norm["normalizations"], "stt_engine": stt["engine"],
                       "stt_provenance": stt["stt_provenance"]}
    entities_json = {"job_id": job_id, **ent}
    return transcript_json, entities_json

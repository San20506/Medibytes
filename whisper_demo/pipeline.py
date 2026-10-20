# Forked from base_pipeline/stt_extract.py and run.run_clip at c68d7a9; not kept in sync.
"""Whisper large-v3 -> normalise -> extract -> optional LLM steps -> verify -> discharge note.

The same stages as the shipped chain with the denoiser, MedASR and the second decoder pass
removed. `run` writes transcript.json, entities.json, verify.json and discharge.json.

Differences from the shipped chain: the unused `llm_tidy` path is dropped, so `use_llm` means
LLM-primary extraction (regex fallback, labelled) and the discharge note, nothing else.
"""
import json
import time
from pathlib import Path

from . import admission, audio, stt
from .extract import (_LAB_REQUEST, _annotate_entity_spans, _chart_name, _load_drug_list,
                      _reconcile_allergens, detect_lang_tag, extract_entities, normalize_text)
from .llm_client import default_model, llm_available

SECTIONS = ("present_illness", "examination_findings", "previous_diagnosis",
            "plan_of_management", "diet_instructions", "general_instructions")


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


def _empty_discharge(reason):
    return {"sections": {k: {"items": []} for k in SECTIONS},
            "provenance": {"llm_engine": reason}}


def _discharge(transcript, entities, use_llm, llm_model):
    """Stage 5: the six-section note. LLM, heard-only and gated; empty and labelled without one."""
    if not use_llm:
        return _empty_discharge("discharge-skipped (use_llm is off)")
    try:
        from .llm_extract import (canonicalize_discharge, extract_discharge, gate_discharge,
                                  merge_discharge)

        text, norm_en = transcript.get("text", ""), transcript.get("normalized_en", "")
        raw = extract_discharge(text, norm_en, transcript.get("segments", []),
                                entities["verify"], model=llm_model)
        gated = gate_discharge(raw, text, norm_en, entities["verify"])
        locked = canonicalize_discharge(gated, entities["verify"])
        return merge_discharge(entities["verify"], locked, text, norm_en)
    except Exception as err:
        return _empty_discharge(f"discharge-skipped ({type(err).__name__}: {err})")


def extract_from_transcript(stt_result, *, job_id, use_llm=False, validate_terms="auto",
                            context_repair=False, llm_model=None):
    """Stages 3-4 on an STT result: (transcript_json, entities_json)."""
    norm = normalize_text(stt_result["text"])
    for s in stt_result["segments"]:  # segment language tags from the demo LID, as in the shipped chain
        s["lang"] = detect_lang_tag(s["text"]) if len(s["text"]) < 200 else norm["lang_tag"]
    ent, llm_reason = None, ""
    if use_llm:
        try:
            from .llm_extract import extract_llm_primary, merge_primary

            ok, why = llm_available(llm_model)
            if ok:
                base = extract_entities(stt_result["text"], norm["normalized_en"], stt_result["segments"])
                llm = extract_llm_primary(stt_result["text"], norm["normalized_en"],
                                          stt_result["segments"], model=llm_model)
                ent = merge_primary(base, llm, model=llm_model)
            else:
                llm_reason = why
        except ValueError as e:
            ent, llm_reason = None, str(e)
    if ent is None:
        ent = extract_entities(stt_result["text"], norm["normalized_en"], stt_result["segments"])
        if use_llm:
            ent["llm_engine"] = f"regex-fallback ({llm_reason or 'llm extraction failed'})"
    ent = _reconcile_allergens(ent)
    t = time.perf_counter()
    ent = _run_term_validation(ent, validate_terms, llm_model)
    if ent.get("term_validator"):
        ent["term_validator_seconds"] = round(time.perf_counter() - t, 3)
    ent = _run_context_repair(ent, norm["normalized_en"], context_repair, llm_model)
    ent = _annotate_entity_spans(ent, norm["normalized_en"])
    transcript = {"job_id": job_id, "text": stt_result["text"], "language": norm["lang_tag"],
                  "segments": stt_result["segments"], "normalized_en": norm["normalized_en"],
                  "normalizations": norm["normalizations"], "stt_engine": stt_result["engine"],
                  "stt_provenance": stt_result["stt_provenance"]}
    return transcript, {"job_id": job_id, **ent}


TEMPLATES = ("discharge", "admission")


def run(audio_path, out_dir, *, job_id=None, language=None, use_llm=False,
        validate_terms="auto", context_repair=False, llm_model=None, template="discharge"):
    """One clip through the whole chain. Raises on a decode or STT failure; never mocks.

    `template` picks the structured output: `discharge` (six-section note, local LLM) or
    `admission` (admission form, sent to the form LLM; see admission.py).
    """
    from .verify import verify

    if template not in TEMPLATES:
        raise ValueError(f"unknown template {template!r}; valid: {', '.join(TEMPLATES)}")

    audio_path, out_dir = Path(audio_path), Path(out_dir)
    llm_model = llm_model or default_model()  # so provenance names the model, not None
    out_dir.mkdir(parents=True, exist_ok=True)
    job_id = job_id or audio_path.stem
    seconds = {}
    t = time.perf_counter()
    samples, meta = audio.load_audio(audio_path)
    seconds["load_audio"] = round(time.perf_counter() - t, 2)
    t = time.perf_counter()
    stt_result = stt.transcribe(samples, language=language, job_id=job_id)
    seconds["stt"] = round(time.perf_counter() - t, 2)
    t = time.perf_counter()
    transcript, entities = extract_from_transcript(
        stt_result, job_id=job_id, use_llm=use_llm, validate_terms=validate_terms,
        context_repair=context_repair, llm_model=llm_model)
    seconds["extract"] = round(time.perf_counter() - t, 2)
    try:
        entities["verify"] = verify(transcript, entities)
    except Exception as err:  # verification must never break a run
        entities["verify"] = {"facts": [], "summary": {"error": f"{type(err).__name__}: {err}"}}
    t = time.perf_counter()
    form = None
    if template == "admission":
        form = admission.fill(transcript, entities["verify"])
    else:
        entities["discharge"] = _discharge(transcript, entities, use_llm, llm_model)
        try:
            from .llm_extract import stamp_schema

            entities["discharge"] = stamp_schema(entities["discharge"])
        except Exception:
            pass
    seconds["llm_note" if template == "discharge" else "form"] = round(time.perf_counter() - t, 2)
    result = {"template": template, "audio": meta, "transcript": transcript, "entities": entities,
              "verify": entities["verify"], "discharge": entities.get("discharge"), "form": form,
              "seconds": seconds}
    outputs = [("transcript.json", transcript), ("entities.json", entities),
               ("verify.json", entities["verify"]), ("audio_meta.json", meta)]
    outputs.append(("admission.json", form) if form else ("discharge.json", entities["discharge"]))
    for name, obj in outputs:
        (out_dir / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return result

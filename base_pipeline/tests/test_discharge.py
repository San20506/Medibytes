"""6-section discharge: prompt map + deterministic pre-fill + gate interplay.

No LLM server needed: prompt/merge/gate are pure functions of transcript +
verified facts. The live-Ollama shape is covered by the manual clip run.
"""
import copy
import json as _json
import os
import sys

import pytest
from jsonschema import Draft7Validator, ValidationError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from base_pipeline.llm_extract import (
    DISCHARGE_SECTIONS,
    _discharge_prompt,
    gate_discharge,
    merge_discharge,
    stamp_schema,
    validate_discharge,
)

TEXT = "Patient has fever. Pulse 92. Give paracetamol 500 mg twice daily. No penicillin allergy."
NORM = TEXT


def _verified():
    return {"facts": [
        {"kind": "symptom", "text": "fever", "status": "verified", "confidence": 0.9,
         "source_sentence": "Patient has fever."},
        {"kind": "vital", "text": "pulse 92", "status": "verified", "confidence": 0.87,
         "source_sentence": "Pulse 92."},
        {"kind": "drug", "name": "paracetamol", "dose": 500.0, "unit": "mg",
         "negated": False, "status": "verified", "confidence": 0.96,
         "source_sentence": "Give paracetamol 500 mg twice daily."},
        {"kind": "allergy", "text": "penicillin", "negated": True, "status": "verified",
         "confidence": 0.82, "source_sentence": "No penicillin allergy."},
    ], "summary": {}}


def _empty():
    return {"sections": {k: {"items": []} for k in DISCHARGE_SECTIONS}, "provenance": {}}


def test_prompt_states_section_map_and_full_proof():
    prompt = _discharge_prompt(TEXT, NORM, _verified(), [])
    for section in ("present_illness", "examination_findings", "plan_of_management"):
        assert section in prompt
    assert "Patient has fever." in prompt  # full proof, never truncated
    assert "seg=" in prompt


def test_prefill_covers_gated_empty_sections_with_tags():
    gated = gate_discharge(_empty(), TEXT, NORM, _verified())
    merged = merge_discharge(_verified(), gated, TEXT, NORM)
    plan = merged["sections"]["plan_of_management"]["items"]
    assert any(i["text"] == "paracetamol 500 mg" and i["verified_kind"] == "drug" for i in plan), plan
    exam = merged["sections"]["examination_findings"]["items"]
    assert any(i["text"] == "pulse 92" for i in exam), exam
    assert merged["provenance"]["prefilled"] >= 3
    assert all(i["confidence"] <= 0.88 for sec in merged["sections"].values() for i in sec["items"])


def test_prefill_never_adds_rejected_or_diet():
    verified = {"facts": [
        {"kind": "drug", "name": "Space", "dose": None, "unit": None, "negated": False,
         "status": "rejected", "confidence": 0.7, "source_sentence": "space today"},
    ], "summary": {}}
    merged = merge_discharge(verified, gate_discharge(_empty(), "space today", "space today", verified),
                             "space today", "space today")
    assert all(sec["items"] == [] for sec in merged["sections"].values())
    assert merged["sections"]["diet_instructions"]["items"] == []


def test_golden_tagged_discharge_validates():
    from base_pipeline.llm_extract import _discharge_schema

    schema = _discharge_schema()
    Draft7Validator.check_schema(schema)
    verified = _verified()
    gated = gate_discharge(_empty(), TEXT, NORM, verified)
    merged = merge_discharge(verified, gated, TEXT, NORM)
    stamped = stamp_schema(merged)
    ok, errors = validate_discharge(stamped)
    assert ok, errors
    assert stamped["provenance"]["schema"] == "valid"
    assert os.path.isfile(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "discharge_schema.json"))


def test_schema_rejects_missing_sentence_extra_section_bad_seg():
    from base_pipeline.llm_extract import _discharge_schema

    schema = _discharge_schema()
    verified = _verified()
    gated = gate_discharge(_empty(), TEXT, NORM, verified)
    good = stamp_schema(merge_discharge(verified, gated, TEXT, NORM))

    bad_keys = copy.deepcopy(good)
    del bad_keys["sections"]["diet_instructions"]
    with pytest.raises(ValidationError):
        Draft7Validator(schema).validate(bad_keys)

    bad_item = copy.deepcopy(good)
    bad_item["sections"]["plan_of_management"]["items"].append({"text": "x"})
    with pytest.raises(ValidationError):
        Draft7Validator(schema).validate(bad_item)

    bad_seg = copy.deepcopy(good)
    bad_seg["sections"]["present_illness"]["items"].append(
        {"text": "fever", "source_sentence": TEXT, "seg_id": "zero"})
    with pytest.raises(ValidationError):
        Draft7Validator(schema).validate(bad_seg)

    bad_tag = copy.deepcopy(good)
    bad_tag["sections"]["present_illness"]["items"].append(
        {"text": "fever", "source_sentence": TEXT, "verified_status": "maybe"})
    with pytest.raises(ValidationError):
        Draft7Validator(schema).validate(bad_tag)


def test_stamp_flags_invalid_but_ships():
    broken = {"sections": {"present_illness": {"items": []}}, "provenance": {}}
    stamped = stamp_schema(broken)
    assert stamped["provenance"]["schema"] == "INVALID"
    assert "present_illness" in stamped["sections"]  # content preserved
    ok, errors = validate_discharge(stamped)
    assert not ok and errors


def test_review_rows_flags_invalid_schema():
    from base_pipeline.run import review_rows

    ents = {"drugs": [], "vitals": [], "allergies": [],
            "discharge": {"sections": {}, "provenance": {
                "schema": "INVALID", "schema_error": "sections: missing" }}}
    rows = list(review_rows(ents))
    assert ("discharge", "schema INVALID", 0.70, "RED", "sections: missing") in rows


def test_prefill_dedups_llm_items():
    llm = _empty()
    llm["sections"]["plan_of_management"]["items"] = [
        {"text": "paracetamol 500 mg", "source_sentence": "Give paracetamol 500 mg twice daily.",
         "seg_id": 0, "confidence": 0.88}]
    merged = merge_discharge(_verified(), gate_discharge(llm, TEXT, NORM, _verified()), TEXT, NORM)
    plan = merged["sections"]["plan_of_management"]["items"]
    assert len(plan) == 1, plan


# ---- dedup is by fact identity, not by shared proof sentence ----

VITALS_SENT = "His blood pressure is 130/85, pulse is around 92 beats per minute, and respiratory rate is 22."
RX_SENT = "He has been advised paracetamol 650 mg and the doctor also considered amoxicillin."


def _empty_note():
    return {"sections": {k: {"items": []} for k in DISCHARGE_SECTIONS}, "provenance": {}}


def _vfact(text):
    return {"kind": "vital", "text": text, "status": "verified", "confidence": 0.87,
            "source_sentence": VITALS_SENT}


def _items(merged, section):
    return [it["text"] for it in merged["sections"][section]["items"]]


def test_three_vitals_in_one_sentence_all_reach_the_note():
    verified = {"facts": [_vfact("blood pressure is 130/85"), _vfact("pulse is around 92"),
                          _vfact("respiratory rate is 22")]}
    merged = merge_discharge(verified, _empty_note(), VITALS_SENT, VITALS_SENT)
    assert _items(merged, "examination_findings") == [
        "blood pressure is 130/85", "pulse is around 92", "respiratory rate is 22"]


def test_an_llm_item_merging_vitals_does_not_hide_the_others():
    note = _empty_note()
    note["sections"]["examination_findings"]["items"].append(
        {"text": "BP 130/85, pulse 92", "source_sentence": VITALS_SENT})
    verified = {"facts": [_vfact("blood pressure is 130/85"), _vfact("pulse is around 92")]}
    texts = _items(merge_discharge(verified, note, VITALS_SENT, VITALS_SENT), "examination_findings")
    assert "blood pressure is 130/85" in texts and "pulse is around 92" in texts


def test_two_drugs_in_one_sentence_both_reach_the_plan():
    verified = {"facts": [
        {"kind": "drug", "name": "paracetamol", "dose": 650, "unit": "mg", "status": "verified",
         "confidence": 0.96, "source_sentence": RX_SENT},
        {"kind": "drug", "name": "amoxicillin", "dose": None, "unit": None,
         "status": "needs_review", "confidence": 0.7, "source_sentence": RX_SENT}]}
    texts = _items(merge_discharge(verified, _empty_note(), RX_SENT, RX_SENT), "plan_of_management")
    assert texts == ["paracetamol 650 mg", "amoxicillin no dose"]


def test_one_symptom_heard_in_two_sentences_surfaces_once():
    text = "He has fever and cough. For the fever, give paracetamol 650 mg."
    verified = {"facts": [
        {"kind": "symptom", "text": "fever", "status": "verified", "source_sentence": "He has fever and cough."},
        {"kind": "symptom", "text": "fever", "status": "verified",
         "source_sentence": "For the fever, give paracetamol 650 mg."}]}
    assert _items(merge_discharge(verified, _empty_note(), text, text), "present_illness") == ["fever"]


def test_an_llm_item_already_stating_the_drug_suppresses_the_prefill():
    note = _empty_note()
    note["sections"]["plan_of_management"]["items"].append(
        {"text": "Paracetamol 650 mg twice daily", "source_sentence": RX_SENT})
    verified = {"facts": [{"kind": "drug", "name": "paracetamol", "dose": 650, "unit": "mg",
                           "status": "verified", "confidence": 0.96, "source_sentence": RX_SENT}]}
    assert len(_items(merge_discharge(verified, note, RX_SENT, RX_SENT), "plan_of_management")) == 1


def test_a_not_given_item_does_not_cover_an_active_order():
    note = _empty_note()
    note["sections"]["plan_of_management"]["items"].append(
        {"text": "NOT GIVEN: paracetamol 650 mg", "source_sentence": RX_SENT})
    verified = {"facts": [{"kind": "drug", "name": "paracetamol", "dose": 650, "unit": "mg",
                           "status": "verified", "confidence": 0.96, "source_sentence": RX_SENT}]}
    assert len(_items(merge_discharge(verified, note, RX_SENT, RX_SENT), "plan_of_management")) == 2


# ---- fact ids: the model names the fact; canonicalize trusts it only with the right proof ----

from base_pipeline.llm_extract import _fact_index, canonicalize_discharge


def _three_vitals():
    return {"facts": [_vfact("blood pressure is 130/85"), _vfact("pulse is around 92"),
                      _vfact("respiratory rate is 22")]}


def _llm_note(*items, section="examination_findings"):
    note = _empty_note()
    note["sections"][section]["items"] = [dict(i) for i in items]
    return note


def test_prompt_numbers_every_shown_fact_and_skips_rejected():
    verified = _three_vitals()
    verified["facts"].insert(0, {"kind": "drug", "name": "space", "status": "rejected",
                                 "source_sentence": "x"})
    assert [fid for fid, _ in _fact_index(verified)] == ["F1", "F2", "F3"]
    prompt = _discharge_prompt(VITALS_SENT, VITALS_SENT, verified, [])
    assert "- F2 vital -> examination_findings: pulse is around 92" in prompt
    assert "space" not in prompt


def test_fact_id_resolves_one_of_three_vitals_in_a_shared_sentence():
    note = _llm_note({"fact_id": "F2", "text": "Pulse 92 bpm", "source_sentence": VITALS_SENT})
    out = canonicalize_discharge(note, _three_vitals())
    assert _items(out, "examination_findings") == ["pulse is around 92"]


def test_without_a_fact_id_the_numbers_pick_the_vital():
    note = _llm_note({"text": "RR 22", "source_sentence": VITALS_SENT})
    out = canonicalize_discharge(note, _three_vitals())
    assert _items(out, "examination_findings") == ["respiratory rate is 22"]


def test_a_fact_id_with_the_wrong_proof_sentence_is_not_trusted():
    note = _llm_note({"fact_id": "F2", "text": "pulse 92", "source_sentence": "Some other sentence."})
    out = canonicalize_discharge(note, _three_vitals())
    assert _items(out, "examination_findings") == []
    assert out["provenance"]["dropped_unproven"] == 1


def test_a_verified_fact_placed_in_the_wrong_section_moves_home():
    note = _llm_note({"fact_id": "F1", "text": "BP 130/85", "source_sentence": VITALS_SENT},
                     section="general_instructions")
    out = canonicalize_discharge(note, _three_vitals())
    assert _items(out, "general_instructions") == []
    assert _items(out, "examination_findings") == ["blood pressure is 130/85"]


# ---- LLM-only items survive only when purely extractive ----

COMPLAINT = "The cough is productive and he is also complaining about some chest discomfort and shortness of breath."
HISTORY = "He already has a history of hypertension and type two diabetes mellitus."
DENIAL = "There is no history of asthma or chest discomfort."


def _canon_one(text, sentence, section):
    note = _llm_note({"text": text, "source_sentence": sentence}, section=section)
    out = canonicalize_discharge(note, {"facts": []})
    return _items(out, section), out


def test_heard_symptom_with_verbatim_proof_is_kept_and_capped():
    items, out = _canon_one("chest discomfort", COMPLAINT, "present_illness")
    assert items == ["chest discomfort"]
    kept = out["sections"]["present_illness"]["items"][0]
    assert kept["confidence"] <= 0.80 and "llm-extractive" in kept["note"]
    assert out["provenance"]["llm_extractive"] == 1


def test_heard_history_is_kept_in_previous_diagnosis():
    assert _canon_one("hypertension", HISTORY, "previous_diagnosis")[0] == ["hypertension"]


def test_negated_sentence_never_yields_an_extractive_item():
    assert _canon_one("asthma", DENIAL, "previous_diagnosis")[0] == []
    assert _canon_one("chest discomfort", DENIAL, "present_illness")[0] == []


def test_paraphrase_is_not_extractive():
    # "fatigue" was never said; "tired" was.
    assert _canon_one("fatigue", "He says he has been feeling pretty tired.", "present_illness")[0] == []


def test_numbers_and_drugs_need_a_verified_fact():
    assert _canon_one("temperature 101", "His temperature is 101.", "present_illness")[0] == []
    assert _canon_one("paracetamol", "He takes paracetamol for pain.", "general_instructions")[0] == []


def test_extractive_items_never_enter_safety_sections():
    for sec in ("examination_findings", "plan_of_management"):
        assert _canon_one("chest discomfort", COMPLAINT, sec)[0] == []

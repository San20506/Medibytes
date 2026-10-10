"""Entity lock-in: LLM lines keep shape, verified facts own entity values.

Each case is one leak path from the gate audit: the gate alone would keep
the divergent line, the canonicalizer must repair or drop it. No LLM needed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from base_pipeline.llm_extract import DISCHARGE_SECTIONS, canonicalize_discharge

SENT_DRUG = "Give paracetamol 500 mg twice daily."
SENT_VITAL = "Pulse 92."
SENT_SYM = "Patient has fever."


def _verified():
    return {"facts": [
        {"kind": "drug", "name": "paracetamol", "dose": 500.0, "unit": "mg",
         "negated": False, "status": "verified", "confidence": 0.96,
         "source_sentence": SENT_DRUG},
        {"kind": "vital", "text": "pulse 92", "status": "verified",
         "confidence": 0.87, "source_sentence": SENT_VITAL},
        {"kind": "symptom", "text": "fever", "status": "verified",
         "confidence": 0.9, "source_sentence": SENT_SYM},
    ], "summary": {}}


def _note(sections, **kw):
    base = {k: {"items": []} for k in DISCHARGE_SECTIONS}
    base.update(sections)
    prov = {"llm_engine": "discharge:test", "dropped_unproven": 0}
    prov.update(kw)
    return {"sections": base, "provenance": prov}


def _item(text, sent):
    return {"text": text, "source_sentence": sent, "seg_id": 0, "confidence": 0.9}


def test_legal_drug_swap_substituted():
    d = _note({"plan_of_management": {"items": [_item("amoxicillin 500 mg", SENT_DRUG)]}})
    out = canonicalize_discharge(d, _verified())
    plan = out["sections"]["plan_of_management"]["items"]
    assert len(plan) == 1 and plan[0]["text"] == "paracetamol 500 mg", plan
    assert plan[0]["verified_kind"] == "drug" and "canonicalized" in plan[0].get("note", "")
    assert out["provenance"]["canonicalized"] == 1


def test_invented_drug_without_candidates_dropped():
    d = _note({"plan_of_management": {"items": [_item("blorptinib 10 mg", SENT_DRUG)]}})
    out = canonicalize_discharge(d, _verified())
    plan = out["sections"]["plan_of_management"]["items"]
    # closest verified drug by proof shares the sentence: substituted, never invented
    assert all("blorptinib" not in i["text"] for i in plan), plan


def test_dose_mutation_inside_substring_substituted():
    d = _note({"plan_of_management": {"items": [_item("paracetamol 50 mg", SENT_DRUG)]}})
    out = canonicalize_discharge(d, _verified())
    assert out["sections"]["plan_of_management"]["items"][0]["text"] == "paracetamol 500 mg"


def test_unit_flip_substituted():
    d = _note({"plan_of_management": {"items": [_item("paracetamol 500 g", SENT_DRUG)]}})
    out = canonicalize_discharge(d, _verified())
    assert out["sections"]["plan_of_management"]["items"][0]["text"] == "paracetamol 500 mg"


def test_vital_paraphrase_and_implausible_value_substituted():
    d = _note({"examination_findings": {"items": [
        _item("pulse 29", SENT_VITAL), _item("heart beating at ninety-two", SENT_VITAL)]}})
    out = canonicalize_discharge(d, _verified())
    exam = out["sections"]["examination_findings"]["items"]
    assert [i["text"] for i in exam] == ["pulse 92", "pulse 92"], exam


def test_symptom_swap_with_reused_proof_substituted():
    d = _note({"present_illness": {"items": [_item("cough", SENT_SYM)]}})
    out = canonicalize_discharge(d, _verified())
    assert out["sections"]["present_illness"]["items"][0]["text"] == "fever"


def test_orphan_line_with_a_number_is_dropped():
    d = _note({"present_illness": {"items": [_item("rash for 3 days", "Patient has rash for 3 days.")]}})
    out = canonicalize_discharge(d, _verified())
    assert out["sections"]["present_illness"]["items"] == []
    assert out["provenance"]["dropped_unproven"] == 1


def test_heard_orphan_line_is_kept_capped_and_tagged():
    # Signed off 2026-10-08: extractive LLM-only items survive (see _extractive_ok).
    d = _note({"present_illness": {"items": [_item("rash", "Patient has rash.")]}})
    out = canonicalize_discharge(d, _verified())
    [kept] = out["sections"]["present_illness"]["items"]
    assert kept["text"] == "rash" and kept["confidence"] <= 0.80
    assert "llm-extractive" in kept["note"]


def test_equal_items_pass_untouched_with_tags():
    d = _note({"plan_of_management": {"items": [_item("paracetamol 500 mg", SENT_DRUG)]}})
    out = canonicalize_discharge(d, _verified())
    plan = out["sections"]["plan_of_management"]["items"]
    assert len(plan) == 1 and "canonicalized" not in plan[0].get("note", "")
    assert plan[0]["verified_kind"] == "drug"
    assert out["provenance"]["canonicalized"] == 0

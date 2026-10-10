"""Deterministic verification gates: numerals, names, doses, vitals, proof.

Every case here must hold without any LLM server running: `verify` is a
pure function of transcript text plus the shipped datasets.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from base_pipeline import verify as V
from base_pipeline.stt_extract import extract_entities, normalize_text


def _t(text):
    norm = normalize_text(text)["normalized_en"]
    return {"text": text, "normalized_en": norm}


def test_numeral_audit_finds_word_converted_doses():
    text = "give paracetamol six hundred and fifty milligrams twice daily"
    audit = V.verify_numerals(_t(text))
    assert "650" in audit["digits"]


def test_exact_dosed_drug_verifies():
    text = "give paracetamol 500 mg twice daily for 3 days"
    t = _t(text)
    ents = extract_entities(text, t["normalized_en"], [])
    out = V.verify(t, ents)
    para = [f for f in out["facts"] if f["kind"] == "drug" and f["name"] == "paracetamol"]
    assert para and para[0]["status"] == "verified", para


def test_brand_only_name_rejects():
    t = _t("patient feels space today")
    row = {"name": "Space", "dose": None, "unit": None, "confidence": 0.88,
           "color": "YELLOW", "source_sentence": "patient feels space today"}
    fact = V.verify_drug(row, t["text"], t["normalized_en"], t["text"])
    assert fact["status"] == "rejected", fact


def test_unsupported_name_rejects_or_reviews():
    t = _t("give blorptinib 10 mg daily")
    row = {"name": "blorptinib", "dose": 10.0, "unit": "mg", "confidence": 0.88,
           "color": "YELLOW", "source_sentence": "give blorptinib 10 mg daily"}
    fact = V.verify_drug(row, t["text"], t["normalized_en"], t["text"])
    assert fact["status"] in ("rejected", "needs_review"), fact


def test_implausible_dose_rejects_with_evidence():
    t = _t("give levothyroxine 25 mg daily")
    row = {"name": "levothyroxine", "dose": 25.0, "unit": "mg", "confidence": 0.96,
           "color": "GREEN", "source_sentence": "give levothyroxine 25 mg daily"}
    fact = V.verify_drug(row, t["text"], t["normalized_en"], t["text"])
    assert fact["status"] == "rejected", fact
    assert fact["confidence"] <= 0.70


def test_missing_proof_caps_but_keeps_name():
    t = _t("give paracetamol 500 mg daily")
    row = {"name": "paracetamol", "dose": 500.0, "unit": "mg", "confidence": 0.96,
           "color": "GREEN", "source_sentence": "a sentence never dictated"}
    fact = V.verify_drug(row, t["text"], t["normalized_en"], t["text"])
    assert fact["status"] == "needs_review"
    assert fact["confidence"] <= 0.88


def test_implausible_vital_needs_review():
    t = _t("pulse 940 today")
    row = {"text": "pulse 940", "confidence": 0.87, "color": "YELLOW",
           "source_sentence": "pulse 940 today"}
    fact = V.verify_vital(row, t["text"], t["normalized_en"])
    assert fact["status"] == "needs_review", fact
    assert fact["confidence"] <= 0.70


def test_allergy_polarity_preserved():
    t = _t("no penicillin allergy")
    row = {"text": "penicillin", "negated": True, "confidence": 0.82,
           "source_sentence": "no penicillin allergy"}
    fact = V.verify_allergy(row, t["text"], t["normalized_en"])
    assert fact["negated"] is True and fact["status"] in ("verified", "needs_review")


def test_verify_never_promotes():
    t = _t("give azithromycin 500 mg daily")
    row = {"name": "azithromycin", "dose": 500.0, "unit": "mg", "confidence": 0.70,
           "color": "RED", "source_sentence": "give azithromycin 500 mg daily"}
    fact = V.verify_drug(row, t["text"], t["normalized_en"], t["text"])
    assert fact["confidence"] <= 0.70


def test_summary_counts():
    text = "give paracetamol 500 mg daily, pulse 92"
    t = _t(text)
    ents = extract_entities(text, t["normalized_en"], [])
    out = V.verify(t, ents)
    assert out["summary"]["total"] == len(out["facts"]) > 0


def test_gate_discharge_keeps_proven_drops_invented():
    from base_pipeline.llm_extract import gate_discharge

    text = "give paracetamol 500 mg twice daily for fever"
    verified = {"facts": [{"kind": "drug", "name": "paracetamol", "status": "verified"}],
                "summary": {}}
    discharge = {"sections": {
        "plan_of_management": {"items": [
            {"text": "paracetamol 500 mg twice daily", "source_sentence": text,
             "seg_id": 0, "confidence": 0.9},
            {"text": "amoxicillin 250 mg", "source_sentence": "a sentence never said",
             "seg_id": 0, "confidence": 0.9}]},
        "present_illness": {"items": []}, "examination_findings": {"items": []},
        "previous_diagnosis": {"items": []}, "diet_instructions": {"items": []},
        "general_instructions": {"items": []}}, "provenance": {}}
    gated = gate_discharge(discharge, text, text, verified)
    kept = gated["sections"]["plan_of_management"]["items"]
    assert len(kept) == 1 and kept[0]["confidence"] <= 0.88
    assert gated["provenance"]["dropped_unproven"] == 1


# ---- brands: evidence only, never a row, never a promotion ----

def _drug(name, dose, sentence, conf=0.96, color="GREEN"):
    return {"name": name, "dose": dose, "unit": "mg", "confidence": conf,
            "color": color, "source_sentence": sentence}


def test_dosed_brand_of_same_substance_corroborates_without_promoting():
    s = "give paracetamol 650 mg that is dolo 650 twice daily"
    t = _t(s)
    fact = V.verify_drug(_drug("paracetamol", 650, s, conf=0.88, color="YELLOW"),
                         t["text"], t["normalized_en"], t["text"])
    assert "brand:corroborated" in fact["checks"], fact
    assert "dolo" in fact["notes"]
    assert fact["confidence"] == 0.88  # evidence, not promotion


def test_brand_strength_charted_as_another_drug_goes_to_review():
    s = "give amoxicillin dolo 650 twice daily"
    t = _t(s)
    fact = V.verify_drug(_drug("amoxicillin", 650, s), t["text"], t["normalized_en"], t["text"])
    assert "brand:conflict" in fact["checks"], fact
    assert fact["status"] != "verified"
    assert fact["confidence"] <= 0.88


def test_another_drugs_brand_in_a_list_is_not_a_conflict():
    s = "give azithromycin 500 mg and pan 40 before breakfast"
    t = _t(s)
    fact = V.verify_drug(_drug("azithromycin", 500, s), t["text"], t["normalized_en"], t["text"])
    assert not any(c.startswith("brand:") for c in fact["checks"]), fact


def test_english_word_brands_are_not_mentions():
    # `monitor`, `came`, `site` are registered brands; with no fitting strength
    # straight after them they are just words.
    s = "we will monitor 2 hourly, he came 3 days ago, check the site 4 times"
    assert V.brand_mentions(s) == []


def test_unrowed_dosed_brand_is_listed_for_review_but_never_becomes_a_fact():
    text = "start dolo 650 twice daily"
    t = _t(text)
    out = V.verify(t, {"drugs": [], "vitals": [], "allergies": []})
    assert out["facts"] == []
    assert [m["brand"] for m in out["brand_mentions"]] == ["dolo"]
    assert "paracetamol" in out["brand_mentions"][0]["substances"]


def test_brand_covered_by_a_row_is_not_listed_again():
    text = "give paracetamol 650 mg that is dolo 650"
    t = _t(text)
    ents = {"drugs": [_drug("paracetamol", 650, text)], "vitals": [], "allergies": []}
    assert V.verify(t, ents)["brand_mentions"] == []


# ---- proof recovery: looked up, never written; ambiguity stays unproven ----

DICTATION = ("He has been having fever and cough for about three days now. "
             "I am thinking this possibly could be community-acquired pneumonia. "
             "For the fever, he has been advised paracetamol 650 mg.")


def test_diagnosis_without_proof_recovers_its_one_sentence_and_verifies():
    t = _t(DICTATION)
    out = V.verify(t, {"drugs": [], "vitals": [], "allergies": [],
                       "diagnosis": {"text": "community-acquired pneumonia", "source_sentence": ""}})
    dx = [f for f in out["facts"] if f["kind"] == "diagnosis"][0]
    assert dx["status"] == "verified"
    assert dx["source_sentence"] == "I am thinking this possibly could be community-acquired pneumonia."
    assert "proof:recovered" in dx["checks"]


def test_text_in_two_sentences_is_ambiguous_and_stays_unproven():
    assert V.recover_proof("fever", DICTATION, DICTATION) == ""
    t = _t(DICTATION)
    out = V.verify(t, {"symptoms": [{"text": "fever", "source_sentence": ""}]})
    assert out["facts"][0]["status"] == "needs_review"


def test_text_never_said_is_not_given_a_proof():
    assert V.recover_proof("pulmonary embolism", DICTATION, DICTATION) == ""


def test_content_words_match_the_digitized_view():
    # LLM row says "3 days"; raw says "three days", normalized says "3 days".
    norm = DICTATION.replace("three", "3")
    sent = V.recover_proof("fever and cough for about 3 days now", DICTATION, norm)
    assert sent == "He has been having fever and cough for about 3 days now."


def test_an_existing_proof_is_never_replaced():
    row = {"text": "community-acquired pneumonia", "source_sentence": "made up sentence"}
    t = _t(DICTATION)
    out = V.verify(t, {"diagnosis": row})
    assert out["facts"][0]["source_sentence"] == "made up sentence"
    assert out["facts"][0]["status"] == "needs_review"

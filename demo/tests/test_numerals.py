"""Golden numeral tests: medication + vitals extraction (regex path, no Ollama).

Run:  python -m pytest demo/tests/test_numerals.py -q
Covers: decimals, BP 'of'/long-form, BPA/AC phonetic (narrow), multi-drug
split, freq vocab (twice a day / every 8 hours), generic allergy candidate.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEMO = os.path.dirname(_HERE)
sys.path.insert(0, _DEMO)

from stt_extract import extract_entities, normalize_text
from coords import resolve_slots


def _run(text):
    norm = normalize_text(text)
    segs = [{"id": 0, "text": text, "start": 0.0, "end": 5.0,
             "lang": "mix", "confidence": 0.9, "words": []}]
    ent = extract_entities(text, norm["normalized_en"], segs)
    tj = {"job_id": "t", "text": text, "language": "mix",
          "segments": segs, "normalized_en": norm["normalized_en"]}
    ent = {"job_id": "t", **ent}
    slots = {s["key"]: s for s in resolve_slots(ent, tj, "er_discharge")["slots"]}
    return ent, slots


def test_demo001_temp_only_no_invented_bp():
    ent, slots = _run(
        "Patient ko fever hai, bukhar 101 degree, give paracetamol 500 mg BID, do time, 3 din tak. No allergy.")
    assert slots["vitals_temp"]["text"] == "101 degree"
    assert slots["vitals_bp_sys"]["text"].startswith("NIL")
    assert slots["vitals_bp_dia"]["text"].startswith("NIL")
    assert ent["drugs"][0]["name"] == "paracetamol"
    assert ent["drugs"][0]["dose"] == 500.0


def test_demo002_bp_split():
    ent, slots = _run(
        "Khansi hai, cough for 5 days, BP 130 by 80, give azithromycin 500 mg once daily 3 days.")
    assert slots["vitals_bp_sys"]["text"] == "130"
    assert slots["vitals_bp_dia"]["text"] == "80"
    assert ent["drugs"][0]["frequency"] == "once daily"
    assert ent["drugs"][0]["duration"] == "3 days"


def test_luvvoice_bpa_ac_phonetic_narrow():
    ent, slots = _run(
        "Khansi hai, cough for 5 days, BPA 130 by AC, give asitromaisin 500 mg once daily 3 days.")
    assert slots["vitals_bp_sys"]["text"] == "130"
    assert slots["vitals_bp_dia"]["text"] == "80"
    assert ent["drugs"][0]["name"] == "azithromycin"


def test_live_two_drugs_freq_and_vitals():
    ent, slots = _run(
        "Looking at their vitals, their blood pressure is sitting at 122 of 78. "
        "Their temp is currently 100.4 degrees Fahrenheit, and their oxygen saturation is "
        "looking good at 99% on room air. We are starting them on amoxicillin, 875 mg "
        "twice a day for seven days, along with ibuprofen, for 100 mg every eight hours as needed.")
    names = [d["name"] for d in ent["drugs"]]
    assert "amoxicillin" in names
    assert "ibuprofen" in names
    amox = next(d for d in ent["drugs"] if d["name"] == "amoxicillin")
    ibu = next(d for d in ent["drugs"] if d["name"] == "ibuprofen")
    assert amox["dose"] == 875.0
    assert amox["frequency"] == "twice a day"
    assert amox["duration"] == "7 days"
    assert ibu["dose"] == 100.0
    assert "8 hours" in ibu["frequency"] or "eight hours" in ibu["frequency"]
    assert slots["vitals_bp_sys"]["text"] == "122"
    assert slots["vitals_bp_dia"]["text"] == "78"
    assert "100.4" in slots["vitals_temp"]["text"]
    assert "99" in slots["vitals_spo2"]["text"]


def test_live_allergy_candidate():
    """A misheard allergy is canonicalised, the way a misheard drug always was.

    This used to assert the raw spelling "moxasillin". An allergy table that
    says "moxasillin" does not protect anyone from amoxicillin, so the candidate
    is now resolved through the same alias map the drug rows use.
    """
    ent, _ = _run("They don't have any known allergy to a moxasillin.")
    assert any(a["text"] == "amoxicillin" and a["negated"] for a in ent["allergies"])


# --- SpO2 cue scoping -------------------------------------------------------
# A percentage is only a saturation when a saturation cue is actually present.
# The cue regex was once widened to treat the bare word "percent" as a cue, to
# rescue spans like "95 percent" whose cue sits in the sentence rather than the
# span. That rescued the real reading and also charted every other percentage
# as an SpO2 - including 50, a peri-arrest value that passes the plausibility
# range, so nothing downstream caught it.

def test_percent_without_saturation_cue_is_not_spo2():
    """"100 percent compliance" is not a saturation; the box stays NIL/RED."""
    _, slots = _run("Give 500 mg paracetamol, 100 percent compliance expected.")
    assert slots["vitals_spo2"]["text"] == "NIL"
    assert slots["vitals_spo2"]["color"] == "RED"


def test_percent_of_cases_is_not_spo2():
    """SpO2 50% is peri-arrest and 50 is inside the plausible range, so a
    mischarted "50 percent of cases" would reach the chart unchallenged."""
    _, slots = _run("Discharge in 50 percent of cases.")
    assert slots["vitals_spo2"]["text"] == "NIL"
    assert slots["vitals_spo2"]["color"] == "RED"


def test_saturation_cue_in_span_charts_spo2():
    _, slots = _run("Oxygen saturation is 98 percent on room air.")
    assert "98" in slots["vitals_spo2"]["text"]
    assert slots["vitals_spo2"]["color"] == "YELLOW"


def test_saturation_cue_in_sentence_charts_spo2():
    """MB_MED_008: the vitals pattern captures a bare "95 percent"; the cue
    ("her saturation is maintaining") is in the sentence, not the span."""
    _, slots = _run("She's receiving oxygen through a face mask at six litres "
                    "per minute, and her saturation is maintaining around "
                    "ninety-five percent.")
    assert "95" in slots["vitals_spo2"]["text"]


def test_saturation_cue_in_other_clause_does_not_leak():
    """A cue elsewhere in the sentence must not promote an unrelated percentage."""
    _, slots = _run("Her saturation was checked on admission; "
                    "discharge follows in 50 percent of cases.")
    assert "50" not in slots["vitals_spo2"]["text"]


def test_regex_vitals_gate_sees_sentence_level_cue():
    """merge_primary must not let the LLM overwrite a sentence-cued SpO2."""
    from llm_extract import merge_primary
    regex_ent = {"drugs": [], "symptoms": [], "allergies": [], "vitals": [
        {"text": "95 percent",
         "source_sentence": "her saturation is maintaining around ninety-five percent."}]}
    merged = merge_primary(regex_ent, {"structured_vitals": {"spo2": "88%"},
                                       "vitals": [{"text": "88%", "kind": "spo2"}]})
    assert "spo2" not in merged.get("structured_vitals", {})
    assert all(v.get("text") != "88%" for v in merged["vitals"])


def test_llm_fills_spo2_when_regex_percent_was_not_a_saturation():
    """The gate must not be so narrow that a genuinely empty slot stays empty."""
    from llm_extract import merge_primary
    regex_ent = {"drugs": [], "symptoms": [], "allergies": [], "vitals": [
        {"text": "100 percent",
         "source_sentence": "Give 500 mg paracetamol, 100 percent compliance expected."}]}
    merged = merge_primary(regex_ent, {"structured_vitals": {"spo2": "97%"},
                                       "vitals": [{"text": "97%", "kind": "spo2"}]})
    assert merged["structured_vitals"]["spo2"] == "97%"


def test_decimal_saturation_keeps_its_sentence_cue():
    """Live STT writes digits; "95.5" must not be split into two clauses."""
    _, slots = _run("Her saturation is 95.5 percent.")
    assert "95.5" in slots["vitals_spo2"]["text"]

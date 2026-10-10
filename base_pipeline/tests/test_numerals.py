"""Golden numeral tests: medication + vitals extraction (regex path, no LLM).

Run:  python -m pytest base_pipeline/tests/test_numerals.py -q
Covers: decimals, BP 'of'/long-form, multi-drug split, freq vocab (twice a day /
every 8 hours), misheard drug and allergy names, and which vitals box a span fills
(`vitals_kind.vitals_box_kind`, the classifier the chart and the LLM gap-fill share).
"""
from base_pipeline.llm_extract import merge_primary
from base_pipeline.extract import extract_entities, normalize_text
from base_pipeline.vitals_kind import vitals_box_kind


def _run(text):
    norm = normalize_text(text)
    segs = [{"id": 0, "text": text, "start": 0.0, "end": 5.0,
             "lang": "mix", "confidence": 0.9, "words": []}]
    return extract_entities(text, norm["normalized_en"], segs)


def _boxes(ent):
    """{box kind: [span text, ...]} for every vitals row that fills a box."""
    boxes = {}
    for v in ent["vitals"]:
        kind = vitals_box_kind(v)
        if kind:
            boxes.setdefault(kind, []).append(v["text"].strip())
    return boxes


def test_demo001_temp_only_no_invented_bp():
    ent = _run("Patient ko fever hai, bukhar 101 degree, give paracetamol 500 mg BID, "
               "do time, 3 din tak. No allergy.")
    assert _boxes(ent) == {"temp": ["101 degree"]}
    assert ent["drugs"][0]["name"] == "paracetamol"
    assert ent["drugs"][0]["dose"] == 500.0


def test_demo002_bp_row_and_drug_details():
    ent = _run("Khansi hai, cough for 5 days, BP 130 by 80, give azithromycin 500 mg once daily 3 days.")
    assert _boxes(ent) == {"bp": ["BP 130 by 80"]}
    assert ent["drugs"][0]["frequency"] == "once daily"
    assert ent["drugs"][0]["duration"] == "3 days"


def test_misheard_drug_name_is_canonicalised_beside_a_garbled_bp():
    ent = _run("Khansi hai, cough for 5 days, BPA 130 by AC, give asitromaisin 500 mg once daily 3 days.")
    assert ent["drugs"][0]["name"] == "azithromycin"


def test_live_two_drugs_freq_and_vitals():
    ent = _run(
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
    boxes = _boxes(ent)
    assert any("122" in t and "78" in t for t in boxes["bp"])
    assert any("100.4" in t for t in boxes["temp"])
    assert any("99" in t for t in boxes["spo2"])


def test_live_allergy_candidate():
    """A misheard allergy is canonicalised, the way a misheard drug always was.

    This used to assert the raw spelling "moxasillin". An allergy table that
    says "moxasillin" does not protect anyone from amoxicillin, so the candidate
    is now resolved through the same alias map the drug rows use.
    """
    ent = _run("They don't have any known allergy to a moxasillin.")
    assert any(a["text"] == "amoxicillin" and a["negated"] for a in ent["allergies"])


# --- SpO2 cue scoping -------------------------------------------------------
# A percentage is only a saturation when a saturation cue is actually present.
# The cue regex was once widened to treat the bare word "percent" as a cue, to
# rescue spans like "95 percent" whose cue sits in the sentence rather than the
# span. That rescued the real reading and also charted every other percentage
# as an SpO2 - including 50, a peri-arrest value that passes the plausibility
# range, so nothing downstream caught it.

def test_percent_without_saturation_cue_is_not_spo2():
    """"100 percent compliance" is not a saturation."""
    ent = _run("Give 500 mg paracetamol, 100 percent compliance expected.")
    assert ent["vitals"], "the extractor still emits the span; the classifier must refuse it"
    assert "spo2" not in _boxes(ent)


def test_percent_of_cases_is_not_spo2():
    """SpO2 50% is peri-arrest and 50 is inside the plausible range, so a
    mischarted "50 percent of cases" would reach the chart unchallenged."""
    ent = _run("Discharge in 50 percent of cases.")
    assert ent["vitals"]
    assert "spo2" not in _boxes(ent)


def test_saturation_cue_in_span_charts_spo2():
    ent = _run("Oxygen saturation is 98 percent on room air.")
    assert any("98" in t for t in _boxes(ent)["spo2"])


def test_saturation_cue_in_sentence_charts_spo2():
    """MB_MED_008: the vitals pattern captures a bare "95 percent"; the cue
    ("her saturation is maintaining") is in the sentence, not the span."""
    ent = _run("She's receiving oxygen through a face mask at six litres "
               "per minute, and her saturation is maintaining around "
               "ninety-five percent.")
    assert any("95" in t for t in _boxes(ent)["spo2"])


def test_saturation_cue_in_other_clause_does_not_leak():
    """A cue elsewhere in the sentence must not promote an unrelated percentage."""
    ent = _run("Her saturation was checked on admission; "
               "discharge follows in 50 percent of cases.")
    assert not any("50" in t for t in _boxes(ent).get("spo2", []))


def test_decimal_saturation_keeps_its_sentence_cue():
    """Live STT writes digits; "95.5" must not be split into two clauses."""
    ent = _run("Her saturation is 95.5 percent.")
    assert any("95.5" in t for t in _boxes(ent)["spo2"])


def test_regex_vitals_gate_sees_sentence_level_cue():
    """merge_primary must not let the LLM overwrite a sentence-cued SpO2."""
    regex_ent = {"drugs": [], "symptoms": [], "allergies": [], "vitals": [
        {"text": "95 percent",
         "source_sentence": "her saturation is maintaining around ninety-five percent."}]}
    merged = merge_primary(regex_ent, {"structured_vitals": {"spo2": "88%"},
                                       "vitals": [{"text": "88%", "kind": "spo2"}]})
    assert "spo2" not in merged.get("structured_vitals", {})
    assert all(v.get("text") != "88%" for v in merged["vitals"])


def test_llm_fills_spo2_when_regex_percent_was_not_a_saturation():
    """The gate must not be so narrow that a genuinely empty slot stays empty."""
    regex_ent = {"drugs": [], "symptoms": [], "allergies": [], "vitals": [
        {"text": "100 percent",
         "source_sentence": "Give 500 mg paracetamol, 100 percent compliance expected."}]}
    merged = merge_primary(regex_ent, {"structured_vitals": {"spo2": "97%"},
                                       "vitals": [{"text": "97%", "kind": "spo2"}]})
    assert merged["structured_vitals"]["spo2"] == "97%"

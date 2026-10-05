"""Rules added for ward-dictation grammar: vitals cues, doseless drugs, allergy
polarity, spoken decimals.

Every sentence here is written for the test, not copied from the evaluation
corpus. The 7-clip clinical dataset is both the bug report and the scoreboard
for these rules, so a rule that only works on its exact strings would score as a
gain and generalise to nothing; these cases are the check against that.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from stt_extract import extract_entities, normalize_text, numwords_to_digits


def _ents(text):
    return extract_entities(text, normalize_text(text)["normalized_en"], [])


def _vitals(text):
    return [v["text"] for v in _ents(text)["vitals"]]


@pytest.mark.parametrize("sentence, value", [
    ("Her pulse rate is 76 beats per minute.", "76"),
    ("His pulse was around 112.", "112"),
    ("Heart rate was 64 at rest.", "64"),
    ("The pulse 88 was regular.", "88"),
    ("Respiratory rate is 16 breaths per minute.", "16"),
    ("Her respiratory rate was around 28.", "28"),
])
def test_pulse_and_respiratory_rate_are_captured(sentence, value):
    assert any(value in span for span in _vitals(sentence)), _vitals(sentence)


@pytest.mark.parametrize("sentence", [
    "Her blood pressure is 118/76.",
    "Her blood pressure was 118/76.",
    "Her blood pressure was around 118/76.",
    "BP 118 over 76 on arrival.",
    "Blood pressure were recorded as 118/76.",
])
def test_blood_pressure_survives_any_copula(sentence):
    spans = _vitals(sentence)
    assert any("118" in s and "76" in s for s in spans), spans


def test_blood_pressure_tolerates_a_number_run_into_the_copula():
    """Decoders run a stray number into the preceding word; the pair still reads."""
    spans = _vitals("His blood pressure is07 142/88 today.")
    assert any("142" in s and "88" in s for s in spans), spans


@pytest.mark.parametrize("sentence, value", [
    ("His blood glucose was 63 mg/dL.", "63"),
    ("We checked her blood sugar and it had fallen to 48.", "48"),
    ("The reading came back at 11.2 mmol/L.", "11.2"),
])
def test_glucose_is_captured(sentence, value):
    assert any(value in span for span in _vitals(sentence)), _vitals(sentence)


def test_a_bare_number_is_not_a_vital():
    """The cue carries the rule; a loose number must not become a reading."""
    assert _vitals("He has been waiting 45 minutes in the corridor.") == []


def test_doseless_drug_is_recorded_red():
    drugs = _ents("He is continuing ceftriaxone as per the postoperative order.")["drugs"]
    assert [(d["name"], d["dose"], d["color"]) for d in drugs] == [
        ("ceftriaxone", None, "RED")
    ]


def test_dosed_row_wins_when_a_drug_is_named_twice():
    drugs = _ents("She takes metformin regularly. She takes metformin 500 mg daily.")["drugs"]
    assert len(drugs) == 1
    assert (drugs[0]["name"], drugs[0]["dose"], drugs[0]["unit"]) == ("metformin", 500.0, "mg")


def test_a_drug_named_only_as_an_allergy_is_not_prescribed():
    ents = _ents("She has a known allergy to warfarin.")
    assert ents["drugs"] == []


def test_allergy_asserted_when_the_drug_precedes_the_word():
    ents = _ents("Please record ibuprofen as a suspected drug allergy.")
    assert [(a["text"], a["negated"]) for a in ents["allergies"]] == [("ibuprofen", False)]


def test_assertion_survives_a_negation_later_in_the_sentence():
    """"... so that it is not given again" must not flip the polarity."""
    ents = _ents(
        "Record cetirizine as a suspected drug allergy so that it is not given again."
    )
    assert [(a["text"], a["negated"]) for a in ents["allergies"]] == [("cetirizine", False)]


def test_denial_before_the_drug_still_reads_as_denied():
    ents = _ents("She has no known ibuprofen allergy.")
    assert all(a["negated"] for a in ents["allergies"]), ents["allergies"]


@pytest.mark.parametrize("spoken, digits", [
    ("ninety-eight point six degrees Fahrenheit", "98.6"),
    ("one hundred point two degrees Fahrenheit", "100.2"),
    ("thirty-seven point five degrees celsius", "37.5"),
])
def test_spoken_decimals_become_numbers(spoken, digits):
    assert digits in numwords_to_digits(spoken)


def test_spoken_integers_are_unaffected():
    assert "130" in numwords_to_digits("one hundred and thirty over eighty-five")


@pytest.mark.parametrize("spoken, digits", [
    ("one thirty over eighty-five", "130"),
    ("one ten over seventy", "110"),
    ("two fifteen over ninety", "215"),
    ("one nineteen", "119"),
])
def test_hundreds_spoken_without_the_word_hundred(spoken, digits):
    assert digits in numwords_to_digits(spoken)


def test_small_numbers_are_still_small():
    assert numwords_to_digits("ten days") == "10 days"
    assert numwords_to_digits("nineteen") == "19"


@pytest.mark.parametrize("sentence, expected", [
    ("She has a known allergy to warfarin.", [("warfarin", False)]),
    ("She is allergic to sulfa.", [("sulfa", False)]),
    ("She reports an allergy to aspirin.", [("aspirin", False)]),
    ("She has no known allergy to warfarin.", [("warfarin", True)]),
    ("Record ibuprofen as a suspected drug allergy.", [("ibuprofen", False)]),
])
def test_allergy_is_read_in_either_word_order(sentence, expected):
    ents = _ents(sentence)
    assert [(a["text"], a["negated"]) for a in ents["allergies"]] == expected


def test_a_denial_of_all_allergies_names_no_substance():
    """"patient denies drug allergies" must not record an allergy to "denies"."""
    assert _ents("Patient denies drug allergies.")["allergies"] == []


@pytest.mark.parametrize("sentence", [
    "He has had heart failure for 10 years.",
    "Advised a low sugar diet, review after 15 days.",
    "He was diagnosed with hypertension in 2019.",
    "Pulse oximetry was unavailable for 30 minutes.",
])
def test_cue_words_do_not_invent_readings(sentence):
    assert _vitals(sentence) == [], _vitals(sentence)


@pytest.mark.parametrize("sentence, value", [
    ("The reading was 54 mg/dL.", "54"),
    ("The reading was fifty-four milligrams per decilitre.", "54"),
    ("The reading was 7 millimoles per litre.", "7"),
])
def test_a_spoken_unit_is_its_own_cue(sentence, value):
    """Dictation says "milligrams per decilitre" as often as it says mg/dL."""
    assert any(value in span for span in _vitals(sentence)), _vitals(sentence)


@pytest.mark.parametrize("sentence, expected", [
    # An allergy asserted in a sentence that also contains a negation elsewhere.
    ("Known penicillin allergy, no other drug allergies.", [("penicillin", False)]),
    ("She has a penicillin allergy and cannot take amoxicillin.", [("penicillin", False)]),
    # A contrast marker ends the negation's reach.
    ("She does not have asthma but has a penicillin allergy.", [("penicillin", False)]),
    # A comma does not - every item in the list is denied.
    ("No history of asthma, diabetes, or penicillin allergy.", [("penicillin", True)]),
    ("No penicillin allergy.", [("penicillin", True)]),
])
def test_allergy_polarity_is_scoped_not_sentence_wide(sentence, expected):
    assert [(a["text"], a["negated"]) for a in _ents(sentence)["allergies"]] == expected


@pytest.mark.parametrize("sentence", [
    "She is allergic to a lot of things.",
    "She is allergic to nothing.",
    "Patient denies drug allergies.",
])
def test_an_unnamed_allergy_records_no_substance(sentence):
    assert _ents(sentence)["allergies"] == []


def test_a_medication_beside_an_allergy_denial_is_still_a_medication():
    """A proximity window used to drop this insulin; only captured spans do now."""
    ents = _ents("He takes insulin and has no known drug allergies.")
    assert [d["name"] for d in ents["drugs"]] == ["insulin"]
    assert ents["allergies"] == []


def test_a_prohibited_drug_is_not_an_order():
    assert _ents("He has not been given ceftriaxone yet.")["drugs"] == []

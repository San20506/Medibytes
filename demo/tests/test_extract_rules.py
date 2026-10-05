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


# ---- a prohibition is not a prescription ----
#
# `neg` was computed per sentence and used only for symptoms, so the dosed-drug
# branch hardcoded `negated: False`. The doseless branch had guarded this all
# along, which is what makes it an inconsistency rather than a design choice.
# The chart renders its active medication list as `[d for d in drugs if not
# d["negated"]]`, so the colour and the flag both matter: the flag removes the
# row from the list, and RED stops a mis-scoped negation deleting a real order
# in silence.

@pytest.mark.parametrize("sentence, drug", [
    ("Do not give ibuprofen 400 mg.", "ibuprofen"),
    ("Please do not give amoxicillin 500 mg.", "amoxicillin"),
    ("Stop metformin 500 mg.", "metformin"),
    ("Avoid ibuprofen 400 mg.", "ibuprofen"),
    ("Withhold metformin 500 mg today.", "metformin"),
    ("Discontinue metformin 500 mg.", "metformin"),
    ("Patient refused paracetamol 500 mg.", "paracetamol"),
    ("She cannot take ibuprofen 400 mg.", "ibuprofen"),
])
def test_a_prohibited_or_stopped_drug_is_not_an_active_order(sentence, drug):
    rows = _ents(sentence)["drugs"]
    assert [r["name"] for r in rows] == [drug], rows
    assert rows[0]["negated"] is True
    assert rows[0]["color"] == "RED"
    assert rows[0]["confidence"] <= 0.70


@pytest.mark.parametrize("sentence", [
    "Give paracetamol 500 mg BID.",
    # A denial of something *else* must not cancel the order that follows it.
    # Commas are deliberately transparent to negation scope so that "no history
    # of asthma, diabetes, or penicillin allergy" denies all three - which is
    # exactly what makes these sentences the hard case.
    "She has no fever, give paracetamol 500 mg.",
    "No penicillin allergy, give amoxicillin 500 mg.",
    "No allergies, start azithromycin 500 mg OD.",
    "Afebrile, continue metformin 500 mg BD.",
    "Patient denies chest pain. Give paracetamol 650 mg.",
])
def test_a_denial_of_something_else_still_prescribes(sentence):
    rows = _ents(sentence)["drugs"]
    assert rows, sentence
    assert rows[0]["negated"] is False
    assert rows[0]["color"] == "GREEN"


def test_negation_keeps_its_reach_across_a_multi_drug_split():
    # The clause splitter used to hand each clause to the matcher on its own,
    # so the second drug lost the "not" that governed the whole sentence.
    rows = _ents("Do not give amoxicillin 500 mg along with ibuprofen 400 mg.")["drugs"]
    assert {r["name"] for r in rows} == {"amoxicillin", "ibuprofen"}
    assert all(r["negated"] is True and r["color"] == "RED" for r in rows), rows


# ---- spoken digit-by-digit numbers ----

@pytest.mark.parametrize("spoken, digits", [
    ("six two five", "625"),
    ("five zero zero", "500"),
    ("one two zero", "120"),
    ("eight zero", "80"),
])
def test_a_run_of_single_digits_is_concatenated_not_summed(spoken, digits):
    # `_parse_numwords` summed the run, so "six two five" was 13. Dictating a
    # strength digit by digit is ordinary, and 13 mg of amoxicillin reached the
    # chart at GREEN - a 48x underdose at the pipeline's highest confidence.
    assert numwords_to_digits(spoken) == digits


@pytest.mark.parametrize("spoken, digits", [
    ("five hundred", "500"),
    ("one thirty", "130"),      # spoken hundreds with the "hundred" dropped
    ("one ten", "110"),         # ...and the same with a teen remainder
    ("two fifteen", "215"),
    ("ninety eight", "98"),
    ("ninety-eight point six", "98.6"),
    ("three", "3"),
])
def test_the_other_spoken_number_forms_are_unchanged(spoken, digits):
    assert numwords_to_digits(spoken) == digits


def test_a_conjunction_between_digits_is_left_alone():
    # "and" is in the number vocabulary so that "a hundred and one" works, and
    # it used to swallow conjunctions between unrelated numbers. Neither 3 nor
    # 12 is a defensible reading of "one and two", so the words stay as they
    # are and the dose simply fails to parse.
    assert numwords_to_digits("one and two") == "one and two"
    assert numwords_to_digits("a hundred and one") == "a 101"


def test_a_digit_by_digit_dose_reaches_the_chart_correctly():
    rows = _ents("Give amoxicillin six two five milligram BD.")["drugs"]
    assert (rows[0]["name"], rows[0]["dose"], rows[0]["unit"]) == ("amoxicillin", 625.0, "mg")


def test_a_digit_by_digit_blood_pressure_is_recovered():
    # Previously summed to "3 by 8", which matched no BP pattern at all, so the
    # reading was lost rather than merely wrong.
    assert any("120" in s and "80" in s for s in _vitals("BP one two zero by eight zero."))


def test_a_spoken_range_is_not_read_as_a_two_digit_number():
    # "for two three days" is two to three days, which Indian-English
    # dictation says constantly. Summing gave 5 and concatenating gives 23;
    # both are wrong and 23 is the more dangerous, so the words are left alone
    # and the duration simply does not parse.
    assert numwords_to_digits("for two three days") == "for two three days"
    assert numwords_to_digits("for three four weeks") == "for three four weeks"
    assert numwords_to_digits("two three times a day") == "two three times a day"
    # A dose is as ambiguous, and is left alone for the same reason.
    assert numwords_to_digits("two three milligram") == "two three milligram"


def test_the_range_guard_does_not_block_a_dictated_dose_or_pressure():
    assert numwords_to_digits("six two five milligram") == "625 milligram"
    assert numwords_to_digits("five zero zero milligram") == "500 milligram"
    # A diastolic "eight zero" is not two ascending digits, so it still converts.
    assert numwords_to_digits("one two zero by eight zero") == "120 by 80"


def test_an_ambiguous_two_digit_dose_goes_red_rather_than_guessing():
    rows = _ents("Give amoxicillin two three milligram BD.")["drugs"]
    assert rows[0]["dose"] is None
    assert rows[0]["color"] == "RED"


def test_a_stopped_drug_is_rendered_rather_than_silently_removed():
    # The active medication table skips negated rows, so without a NOT GIVEN
    # line a "stop metformin" would leave no trace on the document at all -
    # trading a wrong order for a missing one.
    from coords import resolve_slots
    text = "Stop metformin 500 mg, start insulin 10 units."
    ents = _ents(text)
    ents["job_id"] = "t"
    out = resolve_slots(ents, {"job_id": "t", "text": text, "segments": [],
                               "normalized_en": text})
    assert any("metformin" in d["text"] for d in out["denied"]), out["denied"]
    active = [s["text"] for s in out["slots"]
              if s["key"].startswith("drug_") and s["key"].endswith("_name")]
    assert "insulin" in active and "metformin" not in active


@pytest.mark.parametrize("sentence, drug", [
    # "Stop taking X" is the commonest way a stop is dictated, and it was the
    # last hole: the scope reset at `taking` because only `NEG_PAT` counted as
    # governing a verb, and "stop" is not a negation particle.
    ("Stop taking metformin 500 mg.", "metformin"),
    ("Patient refused to take paracetamol 500 mg.", "paracetamol"),
    ("Avoid giving ibuprofen 400 mg.", "ibuprofen"),
    ("Patient has not been given paracetamol 500 mg.", "paracetamol"),
    ("Do not continue metformin 500 mg.", "metformin"),
])
def test_a_stop_word_governs_the_verb_that_follows_it(sentence, drug):
    rows = _ents(sentence)["drugs"]
    assert [r["name"] for r in rows] == [drug], rows
    assert rows[0]["negated"] is True and rows[0]["color"] == "RED"


@pytest.mark.parametrize("sentence, stopped, active", [
    ("Stop amoxicillin 500 mg and start azithromycin 500 mg OD.",
     "amoxicillin", "azithromycin"),
    ("Stop metformin 500 mg, start insulin 10 units.", "metformin", "insulin"),
])
def test_a_drug_switch_stops_only_the_drug_being_switched_from(sentence, stopped, active):
    # The dangerous direction: marking the new drug stopped removes the
    # medication the patient is actually on.
    rows = {r["name"]: r for r in _ents(sentence)["drugs"]}
    assert rows[stopped]["negated"] is True
    assert rows[active]["negated"] is False


def test_a_stopped_drug_outranks_a_denied_symptom_for_the_last_slot():
    # The premium renderer caps the DENIED floats at three. A stopped
    # medication must not be the one that falls off the end.
    from coords import resolve_slots
    text = ("No penicillin allergy, no chest pain, no breathlessness. "
            "Stop metformin 500 mg.")
    ents = _ents(text)
    ents["job_id"] = "t"
    out = resolve_slots(ents, {"job_id": "t", "text": text, "segments": [],
                               "normalized_en": text})
    assert any("metformin" in d["text"] for d in out["denied"]), out["denied"]

"""What the clinical scoreboard is allowed to call a hit.

The harness had no tests, and three defects in it all pointed the same way:
they made the pipeline look better than it was.  Recall was the only thing
measured, so emitting more could never cost anything; a vital span matched on
its number alone, so one span could satisfy several unrelated gold facts; and
an integer matched inside a decimal, so a temperature of 98.6 satisfied a gold
SpO2 of 98.

These tests exist because a scoreboard that cannot go down is not a
scoreboard.  Every case below fails against the previous implementation.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import pytest

from clinical_eval import gold
from clinical_eval.score import (_number_present, _score_entities, _vital_hit,
                                 span_kind)

CLIP = gold.BY_ID["MB_MED_001_clean"]


def _kind(kind):
    return next(v for v in CLIP.vitals if v.kind == kind)


# ---- precision: emitting more must be able to cost something ----

def test_spurious_drug_rows_are_counted():
    one = {"vitals": [], "allergies": [],
           "drugs": [{"name": "paracetamol", "dose": 650.0, "unit": "mg"}]}
    assert _score_entities(CLIP, one)["drugs_spurious"] == 0
    noisy = {**one, "drugs": one["drugs"] + [{"name": f"bogus{i}"} for i in range(50)]}
    scored = _score_entities(CLIP, noisy)
    assert scored["drugs_spurious"] == 50
    # ...and recall is unmoved, which is precisely why precision had to exist:
    # the old harness reported only this number.
    assert scored["drugs_hit"] == 1


def test_dumping_the_whole_formulary_no_longer_looks_like_a_perfect_score():
    # Clip 001's three gold drugs are all in the mini formulary, so emitting
    # every name in it scored 3/3 with nothing counting against it.
    names = ["paracetamol", "amoxicillin", "clavulanic acid",
             "azithromycin", "ibuprofen", "metformin", "atorvastatin"]
    scored = _score_entities(CLIP, {"vitals": [], "allergies": [],
                                    "drugs": [{"name": n} for n in names]})
    assert scored["drugs_hit"] == 3            # recall still perfect...
    assert scored["drugs_spurious"] == 4       # ...and the padding is visible


def test_a_spurious_allergy_is_counted_even_when_gold_has_none():
    assert not CLIP.allergies
    scored = _score_entities(CLIP, {"vitals": [], "drugs": [],
                                    "allergies": [{"text": "penicillin"}]})
    assert scored["allergies_spurious"] == 1


# ---- a vital span must report the right kind of thing ----

def test_a_span_of_the_wrong_vital_kind_is_not_a_hit():
    assert _vital_hit({"vitals": [{"text": "respiratory rate 92"}]}, _kind("pulse")) is False
    assert _vital_hit({"vitals": [{"text": "pulse 92"}]}, _kind("pulse")) is True


def test_one_span_cannot_satisfy_every_gold_vital():
    # A span containing all the numbers - the degenerate case of matching on
    # value alone - scored four of this clip's five vitals.
    blob = " ".join(str(v.value if v.value is not None else v.systolic)
                    for v in CLIP.vitals)
    scored = _score_entities(CLIP, {"vitals": [{"text": blob}], "drugs": [],
                                    "allergies": []})
    assert scored["vitals_hit"] == 0
    assert scored["vitals_spurious"] == 1


def test_each_gold_vital_consumes_its_own_span():
    # Two genuine readings, two gold facts, no double-counting either way.
    ents = {"vitals": [{"text": "pulse 92"}, {"text": "respiratory rate 22"}],
            "drugs": [], "allergies": []}
    scored = _score_entities(CLIP, ents)
    assert scored["vitals_hit"] == 2
    assert scored["vitals_spurious"] == 0


@pytest.mark.parametrize("span, kind", [
    ("BP 130/80", "bp"),
    ("blood pressure 130 by 80", "bp"),
    ("SpO2 94", "spo2"),
    ("94 percent", "spo2"),
    ("temp 101 degrees", "temp_f"),
    ("respiratory rate 22", "rr"),
    ("pulse 92", "pulse"),
    ("heart rate 92", "pulse"),
    ("54 milligrams per decilitre", "glucose"),
    ("glucose 180 mg/dL", "glucose"),
    ("nothing numeric here", None),
])
def test_span_kind_classification(span, kind):
    assert span_kind(span) == kind


def test_rate_is_not_enough_to_make_a_span_a_pulse():
    # "rate" appears in both cues, so ordering is load-bearing.
    assert span_kind("respiratory rate 22") == "rr"


# ---- an integer is not a prefix of a decimal ----

def test_a_decimal_does_not_satisfy_an_integer():
    assert _number_present("98.6 degrees Fahrenheit", 98) is False
    assert _number_present("SpO2 98 percent", 98) is True


def test_the_decimal_itself_still_matches():
    assert _number_present("98.6 degrees Fahrenheit", 98.6) is True
    assert _number_present("ninety-eight point six degrees", 98.6) is True


def test_the_real_collision_this_prevents():
    # Clip 002 carries a temperature of 98.6 and an SpO2 of 98. The
    # temperature span alone used to score the SpO2 fact.
    clip2 = gold.BY_ID["MB_MED_002_clean"]
    spo2 = next(v for v in clip2.vitals if v.kind == "spo2")
    assert spo2.value == 98.0
    assert _vital_hit({"vitals": [{"text": "98.6 degrees Fahrenheit"}]}, spo2) is False


# ---- a drug the extractor says was NOT given is not an active order ----

def test_a_negated_row_does_not_satisfy_an_active_gold_drug():
    # `_extracted_drug` matches on name alone, so a row flagged "not given"
    # counted as finding the drug. That left the scoreboard blind to the one
    # failure the negation fix can introduce: wrongly negating a real order.
    active = {"vitals": [], "allergies": [],
              "drugs": [{"name": "paracetamol", "dose": 650.0, "unit": "mg",
                         "negated": False}]}
    scored = _score_entities(CLIP, active)
    assert scored["drugs_hit"] == 1 and scored["drugs_wrong_polarity"] == 0

    negated = {**active, "drugs": [{**active["drugs"][0], "negated": True}]}
    scored = _score_entities(CLIP, negated)
    assert scored["drugs_hit"] == 0
    assert scored["drugs_wrong_polarity"] == 1


def test_a_row_with_no_negated_key_is_treated_as_active():
    # Gold drugs on this clip are all active, and most emitters omit the key.
    scored = _score_entities(CLIP, {"vitals": [], "allergies": [],
                                    "drugs": [{"name": "paracetamol"}]})
    assert scored["drugs_hit"] == 1


def test_a_wrongly_negated_drug_scores_no_dose_either():
    # A dose is only correct for an order that exists. Scoring it while the
    # drug itself counts as missed would let half the fact survive a polarity
    # error.
    scored = _score_entities(CLIP, {"vitals": [], "allergies": [],
                                    "drugs": [{"name": "paracetamol", "dose": 650.0,
                                               "unit": "mg", "negated": True}]})
    assert scored["drugs_hit"] == 0
    assert scored["doses_hit"] == 0
    assert scored["drugs_wrong_polarity"] == 1

"""Stage 4b: what the term validator is allowed to do to a chart.

Every test here drives the validator with an injected runner, so the model's
answer is the test input rather than something that has to be sampled. The
point of the stage is not that an LLM is consulted - it is that no answer an
LLM can give is permitted to put an unreviewed drug name on a discharge note,
including an answer that is confident, well-formed and wrong.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from stt_extract import extract_entities, normalize_text
from term_validate import (MIN_CONFIDENCE, apply_validation, build_request,
                           flagged_rows, judge)


def _ents(text):
    return extract_entities(text, normalize_text(text)["normalized_en"], [])


FUZZY = "Give asithromysin 500 mg once daily for 3 days."


def _row(text=FUZZY):
    rows = flagged_rows(_ents(text))
    assert rows, f"no fuzzy row produced for {text!r}"
    return rows[0]


def _runner(**verdict):
    def run(_prompt):
        return verdict
    return run


# ---- the matcher now hands the validator something to judge ----

def test_fuzzy_match_carries_the_heard_token_and_its_candidates():
    row = _row()
    assert row["raw_token"] == "asithromysin"
    assert "azithromycin" in [c["name"] for c in row["candidates"]]
    assert row["candidates"] == sorted(row["candidates"],
                                       key=lambda c: -c["score"])


def test_exact_and_alias_hits_are_not_flagged():
    # An exact name and a listed alias never reach the validator: there is no
    # fuzzy decision to review, and sending them would invite a model to
    # second-guess a certain match.
    assert flagged_rows(_ents("Give paracetamol 500 mg twice daily.")) == []
    assert flagged_rows(_ents("Give asitromaisin 500 mg once daily.")) == []


def test_request_exposes_only_the_closed_candidate_set():
    request = build_request(_row())
    assert request["flagged_token"] == "asithromysin"
    assert request["target_sentence"] == FUZZY
    assert all(isinstance(s, str) for s in request["suggestions"])


# ---- the three rules that make the model advisory ----

def test_a_name_outside_the_candidate_set_is_refused():
    # The dangerous hallucination: a plausible drug nobody proposed.
    with pytest.raises(ValueError, match="outside the candidate set"):
        judge(_row(), _runner(status="corrected", chosen_token="erythromycin",
                              semantic_confidence_score=0.99))


def test_an_edit_beyond_the_flagged_token_is_refused():
    row = _row()
    with pytest.raises(ValueError, match="more than the flagged token"):
        judge(row, _runner(status="corrected", chosen_token="azithromycin",
                           semantic_confidence_score=0.95,
                           final_output_sentence="Give azithromycin 1000 mg once daily for 3 days."))


def test_a_verdict_whose_sentence_disagrees_with_the_swap_is_refused():
    # The sentence is rebuilt here from the chosen token; the model's own string
    # is only ever a cross-check, so disagreement kills the verdict.
    with pytest.raises(ValueError, match="more than the flagged token"):
        judge(_row(), _runner(status="corrected", chosen_token="azithromycin",
                              semantic_confidence_score=0.95,
                              final_output_sentence="totally different text"))


def test_a_matcher_tie_is_never_broken_by_the_model():
    row = _row()
    top = row["candidates"][0]["score"]
    row["candidates"] = [row["candidates"][0],
                         {"name": "amoxicillin", "score": top - 0.01, "matched_alias": "amox"}]

    def never_called(_prompt):  # pragma: no cover
        raise AssertionError("the model was asked to break a matcher tie")

    record = judge(row, never_called)
    assert record["status"] == "ambiguous"
    assert "matcher tie" in record["reason"]


def test_low_confidence_becomes_ambiguous_not_a_correction():
    record = judge(_row(), _runner(status="corrected", chosen_token="azithromycin",
                                   semantic_confidence_score=MIN_CONFIDENCE - 0.01))
    assert record["status"] == "ambiguous"
    assert record["applied_corrections"] == []
    assert record["final_output_sentence"] == record["original_sentence"]


def test_a_confident_correction_is_applied_within_the_closed_set():
    record = judge(_row(), _runner(status="corrected", chosen_token="azithromycin",
                                   semantic_confidence_score=0.95, reason="macrolide dose fits"))
    assert record["status"] == "corrected"
    assert record["applied_corrections"] == [
        {"replaced_token": "asithromysin", "substituted_with": "azithromycin"}]
    assert record["final_output_sentence"] == "Give azithromycin 500 mg once daily for 3 days."
    assert record["original_sentence"] == FUZZY


# ---- what lands on the chart ----

def test_validation_never_promotes_a_row_to_green():
    ents = _ents(FUZZY)
    before = ents["drugs"][0]["confidence"]
    apply_validation(ents, runner=_runner(status="corrected", chosen_token="azithromycin",
                                          semantic_confidence_score=1.0))
    row = ents["drugs"][0]
    assert row["color"] == "YELLOW"
    assert row["confidence"] <= before

def test_ambiguity_sends_the_row_to_a_human():
    ents = _ents(FUZZY)
    apply_validation(ents, runner=_runner(status="ambiguous",
                                          semantic_confidence_score=0.4,
                                          chosen_token="asithromysin",
                                          reason="two macrolides fit"))
    row = ents["drugs"][0]
    assert row["color"] == "RED"
    assert row["confidence"] <= 0.70
    assert "ambiguous" in row["note"]
    # The matcher's name is kept so the proof chain still points somewhere.
    assert row["name"] == "azithromycin"


def test_an_unreachable_validator_leaves_the_matcher_untouched_and_says_so():
    ents = _ents(FUZZY)
    before = dict(ents["drugs"][0])

    def boom(_prompt):
        raise ValueError("no ollama binary")

    apply_validation(ents, runner=boom)
    row = ents["drugs"][0]
    assert (row["name"], row["color"], row["confidence"]) == (
        before["name"], before["color"], before["confidence"])
    assert row["validation"]["status"] == "skipped"
    assert "validator skipped" in row["note"]


def test_rows_with_no_fuzzy_match_are_left_alone():
    ents = _ents("Give paracetamol 500 mg twice daily for 3 days.")
    before = dict(ents["drugs"][0])

    def never_called(_prompt):  # pragma: no cover
        raise AssertionError("validator ran on a certain match")

    apply_validation(ents, runner=never_called)
    assert ents["drugs"][0] == before


# ---- the model disagreeing with the matcher ----

HAZARD = "Start hydroxazine 25 mg twice daily."


def test_rejecting_every_candidate_pulls_the_wrong_drug_off_the_chart():
    # A model rejecting every candidate must not leave the row looking like an
    # ordinary fuzzy match. The candidate set is injected rather than matched,
    # so this tests the rejection mechanism and not the lexicon's contents -
    # which is the point: the mechanism has to hold for whatever the matcher
    # proposes, including drugs no reference list happens to contain.
    ents = _ents(HAZARD)
    row = ents["drugs"][0]
    row["name"] = "levothyroxine"
    row["candidates"] = [{"name": "levothyroxine", "score": 0.80,
                          "matched_alias": "levothyroxine"}]
    apply_validation(ents, runner=_runner(status="verified",
                                          chosen_token="hydroxazine",
                                          semantic_confidence_score=0.97,
                                          reason="an antihistamine, not a thyroid hormone"))
    row = ents["drugs"][0]
    assert row["validation"]["status"] == "rejected"
    assert row["color"] == "RED"
    assert row["confidence"] <= 0.70
    assert row["name"] == "hydroxazine"          # what was said, not what was guessed
    assert "levothyroxine" in row["note"]        # the rejected pick stays readable


def test_overruling_the_matchers_top_pick_goes_to_a_human():
    row_ents = _ents(FUZZY)
    row = row_ents["drugs"][0]
    top = row["candidates"][0]["score"]
    row["candidates"] = [row["candidates"][0],
                         {"name": "amoxicillin", "score": top - 0.5, "matched_alias": "amox"}]
    apply_validation(row_ents, runner=_runner(status="corrected",
                                              chosen_token="amoxicillin",
                                              semantic_confidence_score=0.99))
    # Two signals disagree about which drug was said; neither one wins alone.
    assert row["name"] == "amoxicillin"
    assert row["color"] == "RED"
    assert row["confidence"] <= 0.70
    assert "overrules matcher" in row["note"]


def test_agreeing_with_the_matcher_is_the_only_path_that_stays_yellow():
    ents = _ents(FUZZY)
    apply_validation(ents, runner=_runner(status="corrected",
                                          chosen_token="azithromycin",
                                          semantic_confidence_score=0.99))
    row = ents["drugs"][0]
    assert (row["name"], row["color"]) == ("azithromycin", "YELLOW")
    assert row["confidence"] == 0.88

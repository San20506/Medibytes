"""Context repair: closed set, never above RED, `none` is a normal answer."""
from base_pipeline import drug_reference
from base_pipeline.context_repair import apply_context_repair, judge

SENT = "Paracetamine has been given for fever as prescribed."


def _run(entities, text, verdicts):
    calls = []

    def runner(prompt):
        calls.append(prompt)
        return verdicts(prompt) if callable(verdicts) else verdicts

    out = apply_context_repair(entities, text, chart_name=lambda n: "paracetamol" if n == "acetaminophen" else n,
                               stopwords=drug_reference.STOPWORDS, runner=runner)
    return out, calls


def test_repairs_from_context_and_stays_red():
    ent, calls = _run({"drugs": []}, SENT, {"decision": "candidate", "chosen": "paracetamol",
                                            "confidence": 0.93, "reason": "antipyretic for fever"})
    assert [d["name"] for d in ent["drugs"]] == ["paracetamol"]
    row = ent["drugs"][0]
    assert row["color"] == "RED" and row["confidence"] <= 0.7
    assert row["context_repair"]["heard"] == "Paracetamine"
    assert "paracetamol" in calls[0] and ent["context_repair"]["calls"] == 1


def test_none_adds_nothing():
    ent, _ = _run({"drugs": []}, SENT, {"decision": "none", "chosen": "", "confidence": 0.9, "reason": "x"})
    assert ent["drugs"] == [] and ent["context_repair"]["declined"]


def test_low_confidence_is_none():
    ent, _ = _run({"drugs": []}, SENT, {"decision": "candidate", "chosen": "paracetamol",
                                        "confidence": 0.5, "reason": "guess"})
    assert ent["drugs"] == []


def test_name_outside_shortlist_is_refused():
    ent, _ = _run({"drugs": []}, SENT, {"decision": "candidate", "chosen": "morphine",
                                        "confidence": 0.99, "reason": "invented"})
    assert ent["drugs"] == [] and ent["context_repair"]["errors"]


def test_sentence_without_a_medicine_cue_is_never_sent():
    ent, calls = _run({"drugs": []}, "Paracetamine looks nice today.", {"decision": "none"})
    assert calls == [] and ent["drugs"] == []


def test_existing_row_is_not_duplicated():
    ent, _ = _run({"drugs": [{"name": "paracetamol", "color": "RED"}]}, SENT,
                  {"decision": "candidate", "chosen": "paracetamol", "confidence": 0.95, "reason": "x"})
    assert len(ent["drugs"]) == 1


def test_exactly_spelled_names_are_not_candidates():
    ent, calls = _run({"drugs": []}, "Take dapagliflozin as prescribed by your doctor.", {"decision": "none"})
    assert calls == []  # `dapagliflozin` is a listed name, so it was not misheard


def test_judge_raises_on_non_object():
    import pytest
    with pytest.raises(ValueError):
        judge("x", "s", [{"name": "a"}], lambda p: ["no"])


def test_ordinary_english_and_inflections_are_not_candidates():
    from base_pipeline.context_repair import eligible_tokens, is_ordinary_word
    assert is_ordinary_word("certain") and is_ordinary_word("boosting") and is_ordinary_word("treating")
    assert not is_ordinary_word("paracetamine")
    got = eligible_tokens("A certain medication for boosting energy and treating conditions.",
                          set(), drug_reference.STOPWORDS)
    assert got == []


def test_half_of_a_multiword_name_is_not_a_candidate():
    from base_pipeline.context_repair import eligible_tokens
    got = eligible_tokens("Start fenofibric acid for heart health.", set(), drug_reference.STOPWORDS)
    assert "fenofibric" not in got


def test_candidates_below_the_floor_are_not_offered():
    from base_pipeline.context_repair import CONTEXT_FLOOR
    assert CONTEXT_FLOOR == drug_reference.FUZZY_FLOOR


def test_vetoed_sentence_is_never_sent():
    calls = []
    out = apply_context_repair({"drugs": []}, "The doctor has requested cardiac troponin levels as prescribed.",
                               chart_name=lambda n: n, stopwords=drug_reference.STOPWORDS,
                               runner=lambda p: calls.append(p) or {"decision": "none"},
                               skip_sentence=lambda s: "troponin" in s)
    assert calls == [] and out["drugs"] == []

"""Release gate: an allergen is never an active order (allergy-as-order == 0).

The defect was real: a clip that said "she was started on amoxicillin" and, later,
"document amoxicillin as a suspected drug allergy" charted amoxicillin as an
order to give AND as an allergy. Each case below is one way the two statements
can meet, plus the controls that must NOT be blocked.
"""
from base_pipeline import extract as ex


def _ents(text):
    return ex.extract_entities(text, ex.normalize_text(text)["normalized_en"], [])


def _drug(ents, name):
    rows = [d for d in ents["drugs"] if d["name"] == name]
    assert rows, (name, ents["drugs"])
    return rows[0]


def test_the_clip_005_pattern_blocks_the_order():
    ents = _ents("She was started on amoxicillin this morning for a throat infection. "
                 "I have stopped her the medication for now. "
                 "We will document amoxicillin as a suspected drug allergy in her record.")
    row = _drug(ents, "amoxicillin")
    assert row["negated"] is True and row["allergen"] is True and row["color"] == "RED"
    assert "ALLERGEN" in row["note"]
    assert any(a["text"] == "amoxicillin" and not a["negated"] for a in ents["allergies"])


def test_a_dosed_order_followed_by_its_allergy_is_blocked_too():
    ents = _ents("Give amoxicillin 500 mg twice daily. She is allergic to amoxicillin.")
    row = _drug(ents, "amoxicillin")
    assert row["negated"] is True and row["allergen"] is True
    assert row["dose"] == 500.0                      # the dictated dose is kept, only the polarity changes


def test_the_allergy_may_come_first():
    ents = _ents("She has a known allergy to azithromycin. She is currently taking azithromycin.")
    assert _drug(ents, "azithromycin")["negated"] is True


def test_an_alias_spelling_of_the_allergen_still_matches():
    ents = _ents("Give amoxicillin 250 mg. She has an allergy to amoxycillin.")
    assert _drug(ents, "amoxicillin")["negated"] is True


def test_confidence_is_capped_not_promoted():
    ents = _ents("Give amoxicillin 500 mg twice daily. She is allergic to amoxicillin.")
    assert _drug(ents, "amoxicillin")["confidence"] <= 0.7


# ---- controls: these must stay active orders -------------------------------

def test_a_denied_allergy_blocks_nothing():
    ents = _ents("No penicillin allergy. Give penicillin 500 mg four times daily.")
    row = _drug(ents, "penicillin")
    assert not row["negated"] and not row.get("allergen")


def test_an_allergy_to_a_different_drug_blocks_nothing():
    ents = _ents("She is allergic to penicillin. Give azithromycin 500 mg once daily.")
    row = _drug(ents, "azithromycin")
    assert not row["negated"] and not row.get("allergen")


def test_the_pass_is_idempotent_and_catches_a_row_added_later():
    # the second-decoder union runs after extraction and can re-add the allergen
    ents = _ents("She is allergic to amoxicillin.")
    ents["drugs"].append({"name": "amoxicillin", "dose": None, "negated": False,
                          "color": "YELLOW", "confidence": 0.9})
    once = ex._reconcile_allergens(ents)
    snapshot = [dict(d) for d in once["drugs"]]
    assert [d for d in once["drugs"] if d["name"] == "amoxicillin"][0]["negated"] is True
    assert [dict(d) for d in ex._reconcile_allergens(once)["drugs"]] == snapshot


def test_a_dictation_with_no_allergy_is_untouched():
    ents = _ents("Give paracetamol 500 mg twice daily.")
    assert not _drug(ents, "paracetamol")["negated"]
    assert not any(d.get("allergen") for d in ents["drugs"])

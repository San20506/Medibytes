import json
from pathlib import Path

import jsonschema
import pytest

from whisper_demo import admission, extract, verify

TEXT = ("Patient is a 45 year old male. Complaining of chest pain for two days and fever since yesterday. "
        "Denies cough. Known case of diabetes on metformin 500 mg twice daily. No known drug allergies. "
        "BP 150 by 90, pulse 102. Impression acute coronary syndrome. Start aspirin 325 mg stat and admit to ICU.")


def S(sentence):
    assert sentence in TEXT, sentence
    return sentence


@pytest.fixture(scope="module")
def ctx():
    norm = extract.normalize_text(TEXT)["normalized_en"]
    transcript = {"text": TEXT, "normalized_en": norm, "segments": []}
    entities = extract.extract_entities(TEXT, norm, [])
    return transcript, verify.verify(transcript, {"job_id": "x", **entities})


def L(value, sent, conf=0.95):
    return {"value": value, "source_sentence": S(sent), "confidence": conf}


def good_reply():
    return {
        "patient": {"age": L("45 years", "Patient is a 45 year old male."),
                    "sex": L("male", "Patient is a 45 year old male.")},
        "chief_complaints": [
            {"complaint": "chest pain", "duration": "2 days",
             "source_sentence": S("Complaining of chest pain for two days and fever since yesterday."), "confidence": 0.9},
            # the model wrongly files a denial as a complaint
            {"complaint": "cough", "source_sentence": S("Denies cough."), "confidence": 0.9}],
        "past_medical_history": [{"condition": "diabetes",
                                  "source_sentence": S("Known case of diabetes on metformin 500 mg twice daily."),
                                  "confidence": 0.9}],
        "drug_history": [{"drug": "metformin", "dose": 500, "unit": "mg", "frequency": "BD",
                          "source_sentence": S("Known case of diabetes on metformin 500 mg twice daily."),
                          "confidence": 0.9}],
        "allergies": {"no_known_allergies": {"value": True, "source_sentence": S("No known drug allergies."),
                                              "confidence": 0.9}},
        "vitals": {"bp": L("150/90", "BP 150 by 90, pulse 102."), "pulse": L("102", "BP 150 by 90, pulse 102.")},
        "provisional_diagnosis": [{"diagnosis": "acute coronary syndrome",
                                   "source_sentence": S("Impression acute coronary syndrome."), "confidence": 0.9}],
        "admission_plan": {"orders": [{"drug": "aspirin", "dose": 325, "unit": "mg",
                                       "source_sentence": S("Start aspirin 325 mg stat and admit to ICU."),
                                       "confidence": 0.9}]},
    }


def run_gate(ctx, raw):
    transcript, verified = ctx
    return admission.gate(admission._normalise(raw), transcript["text"], transcript["normalized_en"], verified)


def test_schema_file_matches_the_form_spec():
    on_disk = json.loads((Path(admission.__file__).parent / "admission_schema.json").read_text())
    assert on_disk == admission.build_schema()


def test_empty_form_is_valid_and_all_null():
    form = admission.empty_form()
    jsonschema.validate({"form": form}, admission.build_schema())
    assert form["patient"] == {"name": None, "age": None, "sex": None}
    assert form["chief_complaints"] == [] and form["general_examination"] is None


def test_grounded_values_survive_and_the_form_is_schema_valid(ctx):
    form, counts = run_gate(ctx, good_reply())
    assert form["patient"]["age"]["status"] == "verified"
    assert form["patient"]["name"] is None            # not dictated
    assert form["vitals"]["bp"]["value"] == "150/90"
    assert form["drug_history"][0]["drug"] == "metformin"
    assert form["admission_plan"]["orders"][0]["drug"] == "aspirin"
    assert form["allergies"]["no_known_allergies"]["value"] is True
    assert form["provisional_diagnosis"][0]["diagnosis"] == "acute coronary syndrome"
    assert counts["dropped_unproven"] == 0
    jsonschema.validate({"form": form}, admission.build_schema())


def test_denied_symptom_filed_as_a_complaint_moves_to_pertinent_negatives(ctx):
    form, counts = run_gate(ctx, good_reply())
    assert [c["complaint"] for c in form["chief_complaints"]] == ["chest pain"]
    assert [n["symptom"] for n in form["pertinent_negatives"]] == ["cough"]
    assert counts["moved_to_pertinent_negatives"] == 1


def test_a_negation_in_another_clause_does_not_flip_the_complaint(ctx):
    transcript, verified = ctx
    text = "Fever for 3 days but denies cough."
    raw = {"chief_complaints": [{"complaint": "Fever", "duration": "3 days", "source_sentence": text},
                                {"complaint": "cough", "source_sentence": text}]}
    form, _ = admission.gate(admission._normalise(raw), text, text, {"facts": []})
    assert [c["complaint"] for c in form["chief_complaints"]] == ["Fever"]
    assert [n["symptom"] for n in form["pertinent_negatives"]] == ["cough"]


@pytest.mark.parametrize("mutate,field", [
    (lambda r: r["vitals"]["pulse"].update(source_sentence="Pulse was one hundred and two."), "vitals.pulse"),
    (lambda r: r["vitals"]["pulse"].update(value="120"), "vitals.pulse"),                 # digits not spoken
    (lambda r: r["drug_history"][0].update(drug="warfarin"), "drug_history"),             # not heard
    (lambda r: r["provisional_diagnosis"][0].update(diagnosis="pneumonia"), "provisional_diagnosis"),
])
def test_unproven_values_are_dropped_and_counted(ctx, mutate, field):
    raw = good_reply()
    mutate(raw)
    form, counts = run_gate(ctx, raw)
    assert counts["dropped_unproven"] >= 1
    if field == "vitals.pulse":
        assert form["vitals"]["pulse"] is None
    elif field == "drug_history":
        # the unheard drug is gone; the verified metformin is backfilled in its place
        assert [d["drug"] for d in form["drug_history"]] == ["metformin"]
    else:
        assert form[field] == []


def test_a_denied_history_is_not_a_history(ctx):
    text = "Denies diabetes."
    raw = {"past_medical_history": [{"condition": "diabetes", "source_sentence": text}]}
    form, counts = admission.gate(admission._normalise(raw), text, text, {"facts": []})
    assert form["past_medical_history"] == [] and counts["dropped_unproven"] == 1


def test_dose_that_disagrees_with_the_verified_fact_needs_review(ctx):
    raw = good_reply()
    raw["drug_history"][0]["dose"] = 850
    raw["drug_history"][0]["source_sentence"] = S("Known case of diabetes on metformin 500 mg twice daily.")
    form, _ = run_gate(ctx, raw)
    # 850 is not in the transcript, so the model's row is dropped; the verified 500 mg takes its place
    assert [(d["drug"], d["dose"], d["status"]) for d in form["drug_history"]] == [("metformin", 500, "needs_review")]


def test_fill_end_to_end_with_a_fake_model(ctx, monkeypatch):
    transcript, verified = ctx
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(admission, "_chat", lambda prompt, **kw: (json.dumps(good_reply()), {"total_tokens": 7}))
    out = admission.fill(transcript, verified)
    assert out["provenance"]["llm_engine"].startswith("form:")
    assert out["provenance"]["sent_to"] == "api.openai.com"
    assert out["provenance"]["schema"] == "valid"
    assert out["provenance"]["usage"] == {"total_tokens": 7}
    assert out["form"]["vitals"]["pulse"]["value"] == "102"


def test_prompt_carries_the_rules_and_the_transcript(ctx):
    transcript, verified = ctx
    prompt = admission._prompt(transcript["text"], transcript["normalized_en"], verified)
    assert "pertinent_negatives" in prompt and "NEVER invent" in prompt
    assert "metformin" in prompt and "Denies cough." in prompt


@pytest.mark.parametrize("setup,reason", [
    ("nokey", "no OPENAI_API_KEY"),
    ("auth", "HTTP 401"),
    ("garbage", "JSON unusable"),
])
def test_model_trouble_gives_an_all_null_labelled_form(ctx, monkeypatch, setup, reason):
    transcript, verified = ctx
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CHATGPT_API_KEY", raising=False)
    calls = []
    if setup != "nokey":
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    if setup == "auth":
        def chat(prompt, **kw):
            calls.append(1)
            raise admission.FormLLMError("form LLM HTTP 401", retryable=False)
    else:
        def chat(prompt, **kw):
            calls.append(1)
            return "not json at all", {}
    monkeypatch.setattr(admission, "_chat", chat)
    out = admission.fill(transcript, verified)
    assert out["provenance"]["llm_engine"].startswith("form-skipped")
    assert reason in out["provenance"]["llm_engine"]
    assert out["form"] == admission.empty_form()
    assert out["provenance"]["schema"] == "valid"
    assert len(calls) == {"nokey": 0, "auth": 1, "garbage": admission.ATTEMPTS}[setup]


def test_health_status_never_contains_the_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    s = admission.status()
    assert s["key_present"] is True and "sk-secret-value" not in json.dumps(s)


# ---- regressions found by running the real local model on a UTI dictation -----------------------

UTI = ("Patient presents today reporting a 3-day history of dysuria, accompanied by increased urinary urgency "
       "that began 48 hours ago. They deny any lower back pain or hematuria. No known drug allergies, "
       "specifically to sulfa drugs. On examination, vitals are stable with a blood pressure of 118 over 74, "
       "temperature at 99.2 Fahrenheit, and heart rate of 82. The primary clinical diagnosis is an "
       "uncomplicated UTI. The treatment plan includes nitrofurantoin 100 mg twice daily for 5 days, and "
       "finazopyridine 200 mg 3 times a day for 2 days for symptom relief. The patient is instructed to "
       "return to the clinic in 5 days if symptoms fail to resolve.")


@pytest.fixture(scope="module")
def uti():
    norm = extract.normalize_text(UTI)["normalized_en"]
    tr = {"text": UTI, "normalized_en": norm, "segments": []}
    return tr, verify.verify(tr, {"job_id": "x", **extract.extract_entities(UTI, norm, [])})


def gate_uti(uti, raw):
    tr, ver = uti
    return admission.gate(admission._normalise(raw), tr["text"], tr["normalized_en"], ver)


def test_bare_values_get_a_source_from_the_transcript_not_silently_dropped(uti):
    form, counts = gate_uti(uti, {"vitals": {"bp": "118/74", "pulse": "82", "temperature": "99.2"}})
    assert form["vitals"]["bp"]["value"] == "118/74"
    assert "118 over 74" in form["vitals"]["bp"]["source_sentence"]
    assert form["vitals"]["bp"]["status"] == "needs_review" and form["vitals"]["bp"]["confidence"] <= 0.7
    assert counts["source_derived"] == 3 and counts["dropped_unproven"] == 0


def test_a_bare_value_the_transcript_never_says_is_dropped_and_counted(uti):
    form, counts = gate_uti(uti, {"patient": {"sex": "female", "age": "62"}})
    assert form["patient"] == {"name": None, "age": None, "sex": None}
    assert counts["dropped_unproven"] == 2


def test_an_invented_sex_cited_to_a_real_sentence_is_dropped(uti):
    raw = {"patient": {"sex": {"value": "female", "source_sentence": "They deny any lower back pain or hematuria.",
                                 "confidence": 0.9}}}
    form, _ = gate_uti(uti, raw)
    assert form["patient"]["sex"] is None


def test_a_one_word_citation_is_not_a_sentence(uti):
    raw = {"history_of_present_illness": [{"text": "pain", "source_sentence": "pain"}]}
    form, counts = gate_uti(uti, raw)
    assert form["history_of_present_illness"] == [] and counts["dropped_unproven"] == 1


def test_a_whole_sentence_denial_in_the_hpi_becomes_one_negative_not_a_duplicate(uti):
    denial = "They deny any lower back pain or hematuria."
    raw = {"history_of_present_illness": [{"text": denial, "source_sentence": denial}],
           "pertinent_negatives": [{"symptom": "lower back pain", "source_sentence": denial},
                                   {"symptom": "hematuria", "source_sentence": denial}]}
    form, _ = gate_uti(uti, raw)
    assert form["history_of_present_illness"] == []
    assert [n["symptom"] for n in form["pertinent_negatives"]] == ["lower back pain", "hematuria"]


def test_a_whole_sentence_denial_is_kept_as_a_negative_when_the_model_listed_none(uti):
    denial = "They deny any lower back pain or hematuria."
    form, _ = gate_uti(uti, {"history_of_present_illness": [{"text": denial, "source_sentence": denial}]})
    assert [n["symptom"] for n in form["pertinent_negatives"]] == [denial]


def test_no_known_allergies_is_not_an_allergen_and_sets_the_flag_itself(uti):
    sent = "No known drug allergies, specifically to sulfa drugs."
    form, counts = gate_uti(uti, {"allergies": {"items": [{"allergen": "sulfa drugs", "source_sentence": sent}]}})
    assert form["allergies"]["items"] == [] and counts["dropped_unproven"] == 1
    assert form["allergies"]["no_known_allergies"]["value"] is True        # prefilled from the sentence
    assert form["allergies"]["no_known_allergies"]["source_sentence"] == sent


@pytest.mark.parametrize("text,allergen,kept", [
    ("No known allergies except penicillin.", "penicillin", True),
    ("No fever, allergic to penicillin.", "penicillin", True),
    ("Not allergic to penicillin.", "penicillin", False),
    ("Allergic to penicillin.", "penicillin", True),
])
def test_allergy_polarity(text, allergen, kept):
    raw = {"allergies": {"items": [{"allergen": allergen, "source_sentence": text}]}}
    form, _ = admission.gate(admission._normalise(raw), text, text, {"facts": []})
    assert bool(form["allergies"]["items"]) is kept
    if "except" in text:
        assert form["allergies"]["no_known_allergies"] is None


def test_the_treatment_plan_is_not_drug_history_and_an_old_medicine_is_not_an_order(uti, ctx):
    plan = "The treatment plan includes nitrofurantoin 100 mg twice daily for 5 days, and finazopyridine 200 mg 3 times a day for 2 days for symptom relief."
    form, counts = gate_uti(uti, {"drug_history": [{"drug": "nitrofurantoin", "dose": 100, "unit": "mg",
                                                    "source_sentence": plan}]})
    assert form["drug_history"] == [] and counts["misfiled_dropped"] == 1
    hist = "Known case of diabetes on metformin 500 mg twice daily."
    form, counts = run_gate(ctx, {"admission_plan": {"orders": [{"drug": "metformin", "dose": 500, "unit": "mg",
                                                                 "source_sentence": hist}]}})
    assert [o["drug"] for o in form["admission_plan"]["orders"]] == ["aspirin"]   # metformin is history, not an order
    assert counts["misfiled_dropped"] == 1


def test_misfiled_dose_unit_frequency_and_duration_are_put_back(uti):
    plan = "The treatment plan includes nitrofurantoin 100 mg twice daily for 5 days, and finazopyridine 200 mg 3 times a day for 2 days for symptom relief."
    raw = {"admission_plan": {"orders": [{"drug": "nitrofurantoin", "dose": "100 mg", "unit": "twice daily",
                                          "frequency": "5 days", "source_sentence": plan}]}}
    form, _ = gate_uti(uti, raw)
    o = form["admission_plan"]["orders"][0]
    assert (o["dose"], o["unit"], o["frequency"], o["duration"]) == (100, "mg", "twice daily", "5 days")
    assert o["status"] == "verified"


def test_an_unrecognised_unit_is_flagged_for_review(uti):
    plan = "The treatment plan includes nitrofurantoin 100 mg twice daily for 5 days, and finazopyridine 200 mg 3 times a day for 2 days for symptom relief."
    raw = {"admission_plan": {"orders": [{"drug": "nitrofurantoin", "dose": 100, "unit": "tablets",
                                          "source_sentence": plan}]}}
    form, _ = gate_uti(uti, raw)
    assert form["admission_plan"]["orders"][0]["status"] == "needs_review"


def test_polarity_and_role_are_read_from_the_whole_sentence_not_the_quoted_clause(uti):
    raw = {"allergies": {"items": [{"allergen": "sulfa drugs", "source_sentence": "specifically to sulfa drugs"}]},
           "drug_history": [{"drug": "nitrofurantoin", "dose": 100, "unit": "mg",
                             "source_sentence": "nitrofurantoin 100 mg twice daily for 5 days"}]}
    form, counts = gate_uti(uti, raw)
    assert form["allergies"]["items"] == [] and form["drug_history"] == []
    assert counts["misfiled_dropped"] == 1 and counts["dropped_unproven"] == 2


def test_every_drop_is_listed_with_its_reason(uti):
    raw = {"patient": {"sex": {"value": "female", "source_sentence": "They deny any lower back pain or hematuria.",
                               "confidence": 0.9}},
           "history_of_present_illness": [{"text": "pain", "source_sentence": "pain"}],
           "vitals": {"pulse": {"value": "120", "source_sentence": "heart rate of 82", "confidence": 0.9}}}
    _, counts = gate_uti(uti, raw)
    reasons = {(d["field"], d["reason"]) for d in counts["drops"]}
    assert ("sex", "sex: no cue word in the cited sentence backs this value") in reasons
    assert ("history_of_present_illness", "citation too short to be a sentence") in reasons
    assert any(f == "pulse" and "never spoken" in r for f, r in reasons)
    assert len(counts["drops"]) == counts["dropped_unproven"] == 3


PLAN = ("The treatment plan includes nitrofurantoin 100 mg twice daily for 5 days, and finazopyridine 200 mg "
        "3 times a day for 2 days for symptom relief.")


def test_a_verified_drug_the_model_left_out_is_backfilled_as_needs_review(uti):
    raw = {"admission_plan": {"orders": [{"drug": "nitrofurantoin", "dose": 100, "unit": "mg",
                                          "frequency": "twice daily", "duration": "5 days",
                                          "source_sentence": PLAN}]}}
    form, counts = gate_uti(uti, raw)
    orders = form["admission_plan"]["orders"]
    assert [o["drug"] for o in orders] == ["nitrofurantoin", "phenazopyridine"]
    added = orders[1]
    assert (added["dose"], added["unit"], added["status"]) == (200, "mg", "needs_review")
    assert added["frequency"] is None and added["source_sentence"] == PLAN
    assert counts["backfilled_drugs"] == 1


def test_backfill_places_a_history_drug_by_its_cue_and_never_duplicates(ctx):
    form, counts = run_gate(ctx, {})        # model returned nothing
    assert [d["drug"] for d in form["drug_history"]] == ["metformin"]
    assert [o["drug"] for o in form["admission_plan"]["orders"]] == ["aspirin"]
    assert counts["backfilled_drugs"] == 2
    form, counts = run_gate(ctx, good_reply())  # model already has both
    assert counts["backfilled_drugs"] == 0 and len(form["drug_history"]) == 1


def test_backfill_skips_a_negated_or_role_ambiguous_drug():
    text = "Aspirin was not given. Penicillin is on the shelf and start nothing."
    ver = {"facts": [{"kind": "drug", "name": "aspirin", "dose": None, "negated": True, "status": "verified",
                      "source_sentence": "Aspirin was not given.", "confidence": 0.9}]}
    form, counts = admission.gate(admission._normalise({}), text, text, ver)
    assert form["admission_plan"]["orders"] == [] and counts["backfilled_drugs"] == 0

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
    # 850 is not in the transcript at all, so it is dropped rather than flagged
    assert form["drug_history"] == []


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

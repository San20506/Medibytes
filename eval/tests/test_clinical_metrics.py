from __future__ import annotations

from eval.metrics import clinical_metrics, strict_empty_gold_entities


def test_nonclinical_corpus_never_encodes_clinical_metrics_as_zero() -> None:
    result = clinical_metrics(
        {
            "drugs": [],
            "symptoms": [],
            "vitals": [],
            "allergies": [],
            "negations": [],
            "diagnosis": [],
            "followup": [],
        }
    )
    for name in (
        "medical_wer",
        "medication_dose_accuracy",
        "clinical_ner_f1",
        "field_accuracy",
        "critical_error_rate",
    ):
        assert result[name] == {
            "status": "not_applicable",
            "reason": "non_clinical_source_reference_corpus",
        }
        assert not isinstance(result[name], (int, float))


def test_empty_gold_scoring_retains_regex_false_positive_diagnostics() -> None:
    entities = {
        "drugs": [{"name": "paracetamol"}],
        "symptoms": [{"name": "fever"}],
        "vitals": [{"name": "bp"}],
        "allergies": [],
        "negations": [{"text": "denies cough"}],
        "diagnosis": [],
        "followup": [],
    }
    result = strict_empty_gold_entities(entities)
    assert result["gold_status"] == "empty_nonclinical"
    assert result["false_positive_fields"] == 3
    assert result["entity_counts"]["negations"] == 1


def test_empty_gold_scoring_accepts_empty_scalar_diagnosis_and_followup() -> None:
    result = strict_empty_gold_entities(
        {
            "drugs": [],
            "symptoms": [],
            "vitals": [],
            "allergies": [],
            "negations": [],
            "diagnosis": {},
            "followup": {},
        }
    )
    assert result["entity_counts"]["diagnosis"] == 0
    assert result["entity_counts"]["followup"] == 0
    assert result["false_positive_fields"] == 0

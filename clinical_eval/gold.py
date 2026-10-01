"""Hand-authored clinical gold facts for the MediBytes 7-clip medical dataset.

**Authored by an agent from the dataset's own reference transcripts, not by a
clinician.** Every value below is traceable to a sentence in
`transcripts/MB_MED_00N_clean.txt`; nothing is inferred from audio or from model
output, and the file was written before any decoder was run on this corpus. A
clinician should review it before any number derived from it is quoted outside
this repo.

Why facts and not text: the shipped reference transcripts are **paraphrases of
the audio, not verbatim transcriptions** (see `clinical_eval/README.md`). The
clinical content matches; the wording does not. So WER is not computable here,
while the facts a scribe product must capture are, and those are what this file
pins down.

Matching rules, fixed here so a later result cannot redefine them:
  * a vital matches on **numeric equality** of its value(s); blood pressure is a
    (systolic, diastolic) pair and both must match.
  * a drug matches on **canonical name**; `aliases` lists the spellings a decoder
    may legitimately produce (including the ones the dataset's own speakers use).
  * a dose matches on numeric value **and** unit.
  * `negated=True` means the record asserts the finding is absent. 005 is the
    case that matters: amoxicillin is a *positive* suspected allergy while
    penicillin is a *negated* history, in adjacent sentences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Vital:
    kind: str                      # bp | pulse | temp_f | spo2 | rr | glucose
    value: float | None = None
    systolic: float | None = None
    diastolic: float | None = None
    unit: str = ""
    note: str = ""


@dataclass(frozen=True)
class Drug:
    name: str                      # canonical, lowercase
    dose: float | None = None
    unit: str | None = None
    frequency: str = ""
    aliases: tuple[str, ...] = ()
    negated: bool = False


@dataclass(frozen=True)
class Allergy:
    substance: str
    negated: bool                  # True = record asserts NO such allergy
    note: str = ""


@dataclass(frozen=True)
class Clip:
    clip_id: str
    speaker_role: str              # the voice the dictation is in
    vitals: tuple[Vital, ...] = ()
    drugs: tuple[Drug, ...] = ()
    allergies: tuple[Allergy, ...] = ()
    symptoms: tuple[str, ...] = ()
    negated_findings: tuple[str, ...] = ()
    diagnosis: str = ""
    history: tuple[str, ...] = ()


CLIPS: tuple[Clip, ...] = (
    Clip(
        clip_id="MB_MED_001_clean",
        speaker_role="clinician dictation",
        vitals=(
            Vital("temp_f", value=101.0, unit="F"),
            Vital("bp", systolic=130.0, diastolic=85.0, unit="mmHg"),
            Vital("pulse", value=92.0, unit="bpm"),
            Vital("rr", value=22.0, unit="breaths/min"),
            Vital("spo2", value=94.0, unit="%"),
        ),
        drugs=(
            Drug("paracetamol", dose=650.0, unit="mg", aliases=("paracetamol",)),
            Drug("amoxicillin", aliases=("amoxicillin", "amoxycillin", "co-amoxiclav"),
                 frequency="", dose=None, unit=None),
            Drug("clavulanic acid", aliases=("clavulanic acid", "clavulanate")),
        ),
        symptoms=("fever", "productive cough", "chest discomfort",
                  "shortness of breath", "fatigue", "reduced appetite"),
        diagnosis="community-acquired pneumonia",
        history=("hypertension", "type 2 diabetes mellitus"),
    ),
    Clip(
        clip_id="MB_MED_002_clean",
        speaker_role="clinician handover",
        vitals=(
            Vital("bp", systolic=150.0, diastolic=95.0, unit="mmHg"),
            Vital("pulse", value=86.0, unit="bpm"),
            Vital("temp_f", value=98.6, unit="F"),
            Vital("spo2", value=98.0, unit="%"),
        ),
        drugs=(
            Drug("amlodipine", dose=5.0, unit="mg", frequency="once daily",
                 aliases=("amlodipine",)),
        ),
        symptoms=("headache", "dizziness", "blurred vision", "tiredness"),
        diagnosis="hypertension",
        history=("hypertension", "high cholesterol"),
    ),
    Clip(
        clip_id="MB_MED_003_clean",
        speaker_role="clinician handover",
        vitals=(
            Vital("bp", systolic=160.0, diastolic=90.0, unit="mmHg"),
            Vital("pulse", value=88.0, unit="bpm"),
            Vital("spo2", value=97.0, unit="%"),
        ),
        drugs=(),
        symptoms=("right-sided weakness", "slurred speech", "facial droop"),
        diagnosis="acute ischemic stroke",
        history=("hypertension", "type 2 diabetes mellitus"),
    ),
    Clip(
        clip_id="MB_MED_004_clean",
        speaker_role="clinician handover",
        vitals=(
            Vital("bp", systolic=145.0, diastolic=90.0, unit="mmHg"),
            Vital("pulse", value=96.0, unit="bpm"),
            Vital("rr", value=20.0, unit="breaths/min"),
            Vital("spo2", value=95.0, unit="%"),
        ),
        drugs=(Drug("aspirin", aliases=("aspirin",)),),
        symptoms=("chest pain", "radiating left arm pain", "breathlessness", "sweating"),
        diagnosis="acute coronary syndrome",
        history=("hypertension", "high cholesterol", "smoker"),
    ),
    Clip(
        clip_id="MB_MED_005_clean",
        speaker_role="nurse escalation",
        vitals=(
            Vital("spo2", value=96.0, unit="%"),
            Vital("pulse", value=108.0, unit="bpm"),
            Vital("bp", systolic=110.0, diastolic=70.0, unit="mmHg"),
        ),
        drugs=(
            Drug("amoxicillin", aliases=("amoxicillin", "amoxycillin")),
            Drug("adrenaline", aliases=("adrenaline", "epinephrine")),
        ),
        # The safety-critical pair: a POSITIVE suspected allergy and a NEGATED
        # history, two sentences apart, in the same clip.
        allergies=(
            Allergy("amoxicillin", negated=False,
                    note="suspected drug allergy, must be documented"),
            Allergy("penicillin", negated=True,
                    note="patient does not recall a previous penicillin allergy"),
        ),
        symptoms=("itching", "rash", "lip swelling", "throat tightness"),
        diagnosis="acute allergic reaction",
    ),
    Clip(
        clip_id="MB_MED_006_clean",
        speaker_role="family caller",
        vitals=(
            Vital("glucose", value=54.0, unit="mg/dL", note="initial"),
            Vital("glucose", value=72.0, unit="mg/dL", note="after sugar"),
        ),
        drugs=(
            Drug("insulin", aliases=("insulin",)),
            Drug("metformin", aliases=("metformin",)),
        ),
        symptoms=("sweating", "shakiness", "dizziness", "confusion"),
        negated_findings=("vomiting", "fever"),
        diagnosis="hypoglycaemia",
        history=("type 2 diabetes mellitus",),
    ),
    Clip(
        clip_id="MB_MED_007_clean",
        speaker_role="nurse escalation",
        vitals=(
            Vital("temp_f", value=100.2, unit="F"),
            Vital("bp", systolic=124.0, diastolic=78.0, unit="mmHg"),
            Vital("pulse", value=98.0, unit="bpm"),
            Vital("rr", value=18.0, unit="breaths/min"),
            Vital("spo2", value=97.0, unit="%"),
        ),
        drugs=(
            Drug("paracetamol", aliases=("paracetamol",)),
            Drug("ceftriaxone", aliases=("ceftriaxone",)),
        ),
        symptoms=("surgical site pain", "nausea", "wound redness", "low-grade fever"),
        negated_findings=("active bleeding", "vomiting"),
        diagnosis="post-operative surgical site infection",
        history=("laparoscopic appendectomy", "acute appendicitis"),
    ),
)

BY_ID: Mapping[str, Clip] = {clip.clip_id: clip for clip in CLIPS}


def numeric_targets(clip: Clip) -> list[tuple[str, float]]:
    """Every number a transcript must preserve, as (label, value) pairs.

    This is the ASR-fidelity surface: if the decoder loses one of these, the
    downstream record is wrong regardless of how good the word error rate looks.
    """

    targets: list[tuple[str, float]] = []
    for vital in clip.vitals:
        if vital.kind == "bp":
            targets.append((f"bp_systolic{'_'+vital.note if vital.note else ''}", vital.systolic))
            targets.append((f"bp_diastolic{'_'+vital.note if vital.note else ''}", vital.diastolic))
        else:
            label = vital.kind + (f"_{vital.note}" if vital.note else "")
            targets.append((label, vital.value))
    for drug in clip.drugs:
        if drug.dose is not None:
            targets.append((f"dose_{drug.name}", drug.dose))
    return targets


def drug_names(clip: Clip) -> list[str]:
    return [drug.name for drug in clip.drugs]


__all__ = ["Vital", "Drug", "Allergy", "Clip", "CLIPS", "BY_ID",
           "numeric_targets", "drug_names"]

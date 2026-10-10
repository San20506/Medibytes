"""Ground truth v2 for the 30-clip MediBytes dataset, loaded from `gold_v2.json`.

How the data was made (details in the JSON's `meta`): three independent annotators
per clip, each given only the reference transcript and a written guideline (no
pipeline output, no earlier gold), every fact backed by a verbatim quote; facts
without unanimous agreement went to an adjudicator, then a critic and three
skeptics per critic claim. **Agent-authored, not clinician-reviewed.**

The judgment calls live in the data as flags, not in code:
  * `borderline`  - an item the annotators themselves doubted counts (oxygen, a
                    named supplement, a conditional drug, a class-level allergy denial)
  * `historic`    - a reading the text presents as superseded ("was 54 ... now 72")
  * `components`  - a combination product, scored as its component drugs
`load()` picks which of those count; the default scores every explicit reading and
every unambiguous drug.
"""
from __future__ import annotations

import json
from pathlib import Path

from clinical_eval.gold import Allergy, Clip, Drug, Vital

_PATH = Path(__file__).with_name("gold_v2.json")

# Legitimate alternative spellings of the same substance (INN/USAN/British), so a
# correct-but-differently-named row is not scored as a miss.
_ALIASES = {
    "paracetamol": ("acetaminophen",), "salbutamol": ("albuterol",),
    "adrenaline": ("epinephrine",), "levothyroxine": ("thyroxine",),
    "sodium valproate": ("valproate",), "ipratropium bromide": ("ipratropium",),
    "normal saline": ("saline",), "amoxicillin": ("amoxycillin", "co-amoxiclav"),
    "clavulanic acid": ("clavulanate", "co-amoxiclav"),
}


def _drug(rec: dict, name: str) -> Drug:
    return Drug(name, dose=rec.get("dose"), unit=rec.get("unit"),
                aliases=(name, *_ALIASES.get(name, ())), negated=bool(rec.get("negated")))


def load(*, include_borderline: bool = False, current_only: bool = False) -> dict[str, Clip]:
    data = json.loads(_PATH.read_text(encoding="utf-8"))["clips"]
    clips: dict[str, Clip] = {}
    for cid, c in data.items():
        vitals = tuple(
            Vital(v["kind"], value=v.get("value"), systolic=v.get("systolic"),
                  diastolic=v.get("diastolic"), unit=v["unit"],
                  note="historic" if v.get("historic") else "")
            for v in c["vitals"] if not (current_only and v.get("historic")))
        drugs = tuple(
            _drug(d, n) for d in c["drugs"]
            if include_borderline or not d.get("borderline")
            for n in d.get("components") or [d["name"]])
        allergies = tuple(
            Allergy(a["substance"], bool(a.get("negated")))
            for a in c["allergies"] if include_borderline or not a.get("borderline"))
        clips[cid] = Clip(clip_id=cid, speaker_role="dictation",
                          vitals=vitals, drugs=drugs, allergies=allergies)
    return clips

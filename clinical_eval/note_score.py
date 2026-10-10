"""C6 - note fidelity: does each gold fact reach the RIGHT section of the note?

`score.py` stops at extraction (C5). The product's output is the 6-section
discharge note, so this scores `discharge.json` from `base_pipeline.run`
output folders (`<runs>/<clip_id>/<denoiser>/`), against the same gold.

Per clip:

  hit         gold fact present in its expected section
  misplaced   present, but only in another section
  missing     nowhere in the note
  flip        drug present but NOT GIVEN disagrees with gold
  invented    an item carrying a number absent from the transcript (should
              be 0: the authenticity gate exists to guarantee it)
  duplicates  the same item text twice in one section

Separately, `verified_to_note` - of the facts `verify.json` did not reject,
how many reached their mapped section. That isolates the note stage from
ASR and extraction: a fact the pipeline verified but then lost is a note bug.

Matching reuses `score._assign_vitals` (one item claims at most one gold
vital - invariant #9) and `_number_present`. Symptoms and history match on
their head word (`productive cough` -> `cough`) because the note phrases
them in heard words; gold is agent-written paraphrase, so an exact phrase
match would score the gold's wording rather than the pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

from clinical_eval import gold
from clinical_eval.score import _assign_vitals, _number_present

SECTIONS = ("present_illness", "examination_findings", "previous_diagnosis",
            "plan_of_management", "diet_instructions", "general_instructions")
# Where verify.json fact kinds belong (mirrors llm_extract.PREFILL_SECTION;
# copied, not imported, so the scorer does not borrow the graded code's map).
VERIFY_SECTION = {"vital": "examination_findings", "drug": "plan_of_management",
                  "symptom": "present_illness", "diagnosis": "previous_diagnosis",
                  "followup": "plan_of_management", "allergy": "general_instructions"}
_STOP = {"of", "the", "a", "an", "and", "with", "type", "acute", "mild", "severe"}


def _items(note: Mapping[str, Any], section: str) -> list[str]:
    block = (note.get("sections") or {}).get(section) or {}
    return [str(it.get("text", "")) for it in block.get("items", []) or []]


def _head(phrase: str) -> str:
    words = [w for w in re.findall(r"[a-z0-9]+", phrase.lower()) if w not in _STOP]
    return words[-1] if words else phrase.lower()


def _contains(texts: list[str], phrase: str) -> bool:
    head = _head(phrase)
    return any(re.search(rf"\b{re.escape(head)}", t.lower()) for t in texts)


def _drug_in(texts: list[str], drug: gold.Drug) -> str | None:
    """The first item naming this drug (with its dose, when gold has one)."""
    names = [a.lower() for a in tuple(drug.aliases) + (drug.name,)]
    for t in texts:
        low = t.lower()
        if not any(n in low for n in names):
            continue
        if drug.dose is not None and not _number_present(t, drug.dose):
            continue
        return t
    return None


def _place(note: Mapping[str, Any], expected: str, found) -> str:
    """`hit` / `misplaced` / `missing` for a per-section predicate."""
    if found(_items(note, expected)):
        return "hit"
    if any(found(_items(note, s)) for s in SECTIONS if s != expected):
        return "misplaced"
    return "missing"


def _vital_status(clip: gold.Clip, note: Mapping[str, Any]) -> list[str]:
    exam = {"vitals": [{"text": t} for t in _items(note, "examination_findings")]}
    taken, _ = _assign_vitals(clip, exam)
    out = []
    for vital, idx in zip(clip.vitals, taken):
        if idx is not None:
            out.append("hit")
            continue
        one = gold.Clip(clip.clip_id, "", vitals=(vital,))
        elsewhere = any(
            _assign_vitals(one, {"vitals": [{"text": t} for t in _items(note, s)]})[0][0] is not None
            for s in SECTIONS if s != "examination_findings")
        out.append("misplaced" if elsewhere else "missing")
    return out


def score_note(clip: gold.Clip, note: Mapping[str, Any], transcript: str,
               verified: Mapping[str, Any] | None = None) -> dict[str, Any]:
    facts: list[dict[str, Any]] = []
    for vital, status in zip(clip.vitals, _vital_status(clip, note)):
        facts.append({"kind": "vital", "fact": vital.kind, "status": status})

    for drug in clip.drugs:
        status = _place(note, "plan_of_management", lambda ts, d=drug: _drug_in(ts, d))
        if status == "hit":
            item = _drug_in(_items(note, "plan_of_management"), drug) or ""
            if ("not given" in item.lower()) != drug.negated:
                status = "flip"
        facts.append({"kind": "drug", "fact": drug.name, "status": status})

    for sym in clip.symptoms:
        facts.append({"kind": "symptom", "fact": sym, "status": _place(
            note, "present_illness", lambda ts, p=sym: _contains(ts, p))})
    if clip.diagnosis:
        facts.append({"kind": "diagnosis", "fact": clip.diagnosis, "status": _place(
            note, "previous_diagnosis", lambda ts, p=clip.diagnosis: _contains(ts, p))})
    for hist in clip.history:
        facts.append({"kind": "history", "fact": hist, "status": _place(
            note, "previous_diagnosis", lambda ts, p=hist: _contains(ts, p))})

    from base_pipeline.stt_extract import numwords_to_digits

    heard = numwords_to_digits(transcript)
    invented, duplicates = [], 0
    for s in SECTIONS:
        seen = set()
        for t in _items(note, s):
            key = re.sub(r"\s+", " ", t.strip().lower())
            if key in seen:
                duplicates += 1
            seen.add(key)
            for num in re.findall(r"\d+(?:\.\d+)?", t):
                if not _number_present(heard, float(num)):
                    invented.append(f"{s}: {t}")
                    break

    out = {"clip_id": clip.clip_id, "facts": facts, "invented": invented,
           "duplicates": duplicates, "total": len(facts)}
    for k in ("hit", "misplaced", "missing", "flip"):
        out[k] = sum(1 for f in facts if f["status"] == k)
    if verified is not None:
        out["verified_to_note"] = verified_to_note(note, verified)
    return out


def verified_to_note(note: Mapping[str, Any], verified: Mapping[str, Any]) -> dict[str, Any]:
    """Of the non-rejected verified facts, how many reached their section."""
    reached, lost = 0, []
    for f in (verified or {}).get("facts", []) or []:
        if f.get("status") == "rejected":
            continue
        section = VERIFY_SECTION.get(f.get("kind", ""))
        if not section:
            continue
        texts = _items(note, section)
        if f["kind"] == "vital":
            nums = re.findall(r"\d+(?:\.\d+)?", str(f.get("text", "")))
            ok = bool(nums) and any(all(n in t for n in nums) for t in texts)
        elif f["kind"] == "drug":
            ok = any(str(f.get("name", "")).lower() in t.lower() for t in texts)
        else:
            ok = _contains(texts, str(f.get("text", "")))
        if ok:
            reached += 1
        else:
            lost.append(f"{f['kind']}: {f.get('name') or f.get('text')}")
    return {"reached": reached, "total": reached + len(lost), "lost": lost}


def score_runs(runs_dir: Path, denoiser: str = "none") -> dict[str, Any]:
    rows = []
    for clip in gold.CLIPS:
        arm = runs_dir / clip.clip_id / denoiser
        if not (arm / "discharge.json").exists():
            continue
        note = json.loads((arm / "discharge.json").read_text(encoding="utf-8"))
        tr = json.loads((arm / "transcript.json").read_text(encoding="utf-8"))
        ver_path = arm / "verify.json"
        ver = json.loads(ver_path.read_text(encoding="utf-8")) if ver_path.exists() else None
        text = f"{tr.get('text', '')} {tr.get('normalized_en', '')}"
        rows.append(score_note(clip, note, text, ver))
    totals = {k: sum(r[k] for r in rows)
              for k in ("hit", "misplaced", "missing", "flip", "total", "duplicates")}
    totals["invented"] = sum(len(r["invented"]) for r in rows)
    v2n = [r["verified_to_note"] for r in rows if r.get("verified_to_note")]
    totals["verified_reached"] = sum(v["reached"] for v in v2n)
    totals["verified_total"] = sum(v["total"] for v in v2n)
    by_kind: dict[str, dict[str, int]] = {}
    for r in rows:
        for f in r["facts"]:
            k = by_kind.setdefault(f["kind"], {"hit": 0, "total": 0})
            k["total"] += 1
            k["hit"] += f["status"] == "hit"
    return {"clips": rows, "totals": totals, "by_kind": by_kind}


def _pct(a: int, b: int) -> str:
    return f"{a}/{b} ({100.0 * a / b:.0f}%)" if b else "n/a"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="C6 note fidelity over base_pipeline.run output")
    ap.add_argument("runs_dir", help="the --out folder of a base_pipeline.run")
    ap.add_argument("--denoiser", default="none")
    ap.add_argument("--json", help="also write the full report here")
    a = ap.parse_args(argv)
    rep = score_runs(Path(a.runs_dir), a.denoiser)
    t = rep["totals"]
    print(f"C6 note fidelity, {len(rep['clips'])} clips")
    print(f"  gold facts in right section : {_pct(t['hit'], t['total'])}")
    for k, v in sorted(rep["by_kind"].items()):
        print(f"    {k:<10}: {_pct(v['hit'], v['total'])}")
    print(f"  misplaced {t['misplaced']}, missing {t['missing']}, flips {t['flip']}")
    print(f"  invented {t['invented']}, duplicates {t['duplicates']}")
    print(f"  verified -> note            : {_pct(t['verified_reached'], t['verified_total'])}")
    for r in rep["clips"]:
        miss = [f"{f['kind']}:{f['fact']}({f['status']})" for f in r["facts"] if f["status"] != "hit"]
        print(f"  {r['clip_id']}: {_pct(r['hit'], r['total'])}  not-hit: {', '.join(miss) or '-'}")
        if (r.get("verified_to_note") or {}).get("lost"):
            print(f"      verified but lost: {'; '.join(r['verified_to_note']['lost'])}")
        for inv in r["invented"]:
            print(f"      INVENTED {inv}")
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

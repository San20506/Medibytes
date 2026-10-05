"""Score each pipeline component against the clinical gold facts.

Word error rate is deliberately **not** the headline here, and on this corpus it
is not even computable: the shipped reference transcripts are paraphrases of the
audio, not verbatim transcriptions, so a word-level comparison measures the
rewrite rather than the decoder. What is computable, and what a scribe product
actually has to get right, is whether each clinical fact survives the pipeline.

Three metrics, each attributable to one stage:

  ASR fidelity (C2)   does the decoder's raw text contain the gold number / drug
                      name at all? A number the decoder never produced cannot be
                      extracted by any downstream stage.
  Extraction ceiling  does `extract_entities` recover the fact from the PERFECT
  (C4)                reference transcript? This is the extractor's own limit,
                      with ASR taken out of the picture.
  End to end (C5)     does the fact survive audio -> clean -> STT -> normalize ->
                      extract? C4 minus C5 is the ASR-induced loss.

Clinical-critical items (dose values, vital values, allergy polarity) are counted
separately from everything else, because they are the ones where an error is a
safety event rather than a cosmetic one. With n=7 the output is per-clip counts;
no bootstrap, no p-values.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from clinical_eval import gold
from clinical_eval.prepare import DATA_ROOT_DEFAULT, corpus_root


def _number_present(text: str, value: float) -> bool:
    """Is this clinical number recoverable from the text, in any spoken form?

    A decoder may write 98.6 as "98.6" or as "98 point 6"; 150/95 may appear as
    "150 over 95", "150/95" or "one fifty over ninety five". The pipeline's own
    `numwords_to_digits` converts spelled-out integers but not "point", so the
    decimal forms are matched explicitly rather than counted as misses.
    """

    from demo.stt_extract import numwords_to_digits

    haystack = numwords_to_digits(text)
    if float(value).is_integer():
        integral = str(int(value))
        # `(?!\d)` alone lets a decimal satisfy an integer: a temperature span
        # reading "98.6 degrees Fahrenheit" scored a gold SpO2 of 98, and clip
        # 002 carries both facts, so the vitals total was inflated by a
        # collision between two unrelated readings. A digit after the point is
        # a different number, not this one.
        return re.search(rf"(?<!\d){re.escape(integral)}(?!\d)(?!\.\d)",
                         haystack) is not None
    whole, frac = f"{value:.10g}".split(".")
    patterns = (
        rf"(?<!\d){re.escape(whole)}\.{re.escape(frac)}(?!\d)",
        rf"(?<!\d){re.escape(whole)}\s*point\s*{re.escape(frac)}(?!\d)",
    )
    return any(re.search(p, haystack, re.I) for p in patterns)


def _name_present(text: str, drug: gold.Drug) -> bool:
    lowered = text.lower()
    return any(alias.lower() in lowered for alias in (drug.aliases or (drug.name,)))


def _extracted_drug(entities: Mapping[str, Any], drug: gold.Drug) -> dict[str, Any] | None:
    for item in entities.get("drugs", []) or []:
        name = str(item.get("name", "")).lower()
        if name == drug.name.lower() or any(a.lower() == name for a in drug.aliases):
            return dict(item)
    return None


# Which gold kind an emitted vital span is talking about. The extractor emits
# untyped spans, so without this the scorer compares numbers alone - and a span
# reading "respiratory rate 92" scored a gold *pulse* of 92 as a hit. Worse, a
# single span containing every number in the transcript scored four of clip
# 001's five vitals. Deliberately written here rather than imported from
# `demo/`: the scorer should not borrow the judgement of the code it grades.
_KIND_CUES: tuple[tuple[str, str], ...] = (
    ("bp", r"\bBP\b|blood\s*pressure|\d+\s*(?:by|/|over)\s*\d+"),
    ("spo2", r"spo2|sp02|o2\s*sat|oxygen\s*sat|saturation|%|percent"),
    ("temp_f", r"temp|fahrenheit|celsius|degree|°"),
    ("rr", r"respiratory\s*rate|resp\b|respiration|breathing\s*rate"),
    ("pulse", r"pulse|heart\s*rate|\bHR\b"),
    # The concentration units are a cue in their own right: the decoder writes
    # "54 milligrams per decilitre" with no word "glucose" anywhere in the span.
    ("glucose", r"glucose|sugar|BGL|RBS|CBG|mg\s*/\s*d[lL]|mmol"
                r"|milligrams?\s+per\s+decilit(?:re|er)"
                r"|millimoles?\s+per\s+lit(?:re|er)"),
)


def span_kind(text: str) -> str | None:
    """The gold vital kind this span reports, or None if it names none.

    Order matters: `respiratory rate` is tested before `pulse` because "rate"
    appears in both cues, and a BP pair is tested first because "130/80"
    carries no word cue at all.
    """
    for kind, pattern in _KIND_CUES:
        if re.search(pattern, text, re.I):
            return kind
    return None


def _vital_candidates(entities: Mapping[str, Any], vital: gold.Vital) -> list[int]:
    """Indices of emitted spans that report this gold vital, value and kind."""
    out = []
    for i, item in enumerate(entities.get("vitals", []) or []):
        span = str(item.get("text", ""))
        if span_kind(span) != vital.kind:
            continue
        if vital.kind == "bp":
            ok = (_number_present(span, vital.systolic)
                  and _number_present(span, vital.diastolic))
        else:
            ok = _number_present(span, vital.value)
        if ok:
            out.append(i)
    return out


def _vital_hit(entities: Mapping[str, Any], vital: gold.Vital) -> bool:
    """Does any extracted vital span report this gold fact? (type and value)"""
    return bool(_vital_candidates(entities, vital))


def _allergy_polarity(entities: Mapping[str, Any], allergy: gold.Allergy) -> str:
    """`correct` / `wrong_polarity` / `missing` for one gold allergy."""

    for item in entities.get("allergies", []) or []:
        if str(item.get("text", "")).lower() != allergy.substance.lower():
            continue
        return "correct" if bool(item.get("negated")) == allergy.negated else "wrong_polarity"
    return "missing"


def _assign_vitals(clip: gold.Clip, entities: Mapping[str, Any]) -> tuple[list[int | None], set[int]]:
    """One emitted span per gold vital, and the span indices nothing claimed.

    Without consumption the scorer asks `any()` per gold fact, so one span can
    satisfy several gold vitals at once and a transcript-sized blob scores
    them all. Assignment is greedy over the scarcest gold fact first, which is
    exact whenever candidate sets nest and good enough for 5 facts a clip.
    """
    cands = {i: _vital_candidates(entities, v) for i, v in enumerate(clip.vitals)}
    taken: dict[int, int] = {}
    for gi in sorted(cands, key=lambda i: len(cands[i])):
        for si in cands[gi]:
            if si not in taken.values():
                taken[gi] = si
                break
    n = len(entities.get("vitals", []) or [])
    return ([taken.get(i) for i in range(len(clip.vitals))],
            {i for i in range(n) if i not in taken.values()})


def _spurious_drugs(clip: gold.Clip, entities: Mapping[str, Any]) -> list[str]:
    """Emitted drug rows that correspond to no gold drug.

    This is the half of the picture the harness never had: every total it
    reports is `hit / gold_total`, so emitting the entire 33-name formulary
    scored full drug recall and nothing counted against it.
    """
    known = {d.name.lower() for d in clip.drugs}
    for d in clip.drugs:
        known.update(a.lower() for a in (d.aliases or ()))
    return [str(item.get("name", "")) for item in entities.get("drugs", []) or []
            if str(item.get("name", "")).lower() not in known]


def _spurious_allergies(clip: gold.Clip, entities: Mapping[str, Any]) -> list[str]:
    known = {a.substance.lower() for a in clip.allergies}
    return [str(item.get("text", "")) for item in entities.get("allergies", []) or []
            if str(item.get("text", "")).lower() not in known]


def _score_entities(clip: gold.Clip, entities: Mapping[str, Any]) -> dict[str, Any]:
    assigned, unclaimed = _assign_vitals(clip, entities)
    vitals = [(f"{v.kind}{'_'+v.note if v.note else ''}", assigned[i] is not None)
              for i, v in enumerate(clip.vitals)]
    extra_drugs = _spurious_drugs(clip, entities)
    extra_allergies = _spurious_allergies(clip, entities)
    drugs = []
    for drug in clip.drugs:
        found = _extracted_drug(entities, drug)
        # Polarity is part of identity for a prescription. `_extracted_drug`
        # matches on name alone, so a row the extractor marked "not given"
        # still counted as finding an active gold drug - which left the
        # scoreboard blind to the one failure the negation fix can introduce.
        polarity_ok = bool(found) and bool(found.get("negated", False)) == drug.negated
        dose_ok = None
        if drug.dose is not None:
            dose_ok = bool(
                found
                and polarity_ok
                and found.get("dose") is not None
                and float(found["dose"]) == drug.dose
                and str(found.get("unit") or "").lower() == (drug.unit or "").lower()
            )
        drugs.append({"name": drug.name, "found": found is not None,
                      "polarity_ok": polarity_ok if found else None,
                      "dose_expected": drug.dose, "dose_ok": dose_ok})
    allergies = [{"substance": a.substance, "gold_negated": a.negated,
                  "outcome": _allergy_polarity(entities, a)} for a in clip.allergies]
    return {
        "vitals": vitals,
        "vitals_hit": sum(1 for _, ok in vitals if ok),
        "vitals_total": len(vitals),
        "drugs": drugs,
        "drugs_hit": sum(1 for d in drugs if d["found"] and d["polarity_ok"]),
        "drugs_wrong_polarity": sum(1 for d in drugs
                                    if d["found"] and not d["polarity_ok"]),
        "drugs_total": len(drugs),
        "doses_hit": sum(1 for d in drugs if d["dose_ok"]),
        "doses_total": sum(1 for d in drugs if d["dose_expected"] is not None),
        "allergies": allergies,
        "allergies_correct": sum(1 for a in allergies if a["outcome"] == "correct"),
        "allergies_total": len(allergies),
        # Precision. Recall alone is gameable: before these counters existed,
        # dumping every formulary name as a drug row scored 3/3 on clip 001.
        "vitals_spurious": len(unclaimed),
        "vitals_emitted": len(entities.get("vitals", []) or []),
        "drugs_spurious": len(extra_drugs),
        "drugs_spurious_names": extra_drugs,
        "drugs_emitted": len(entities.get("drugs", []) or []),
        "allergies_spurious": len(extra_allergies),
        "allergies_emitted": len(entities.get("allergies", []) or []),
    }


def _score_asr(clip: gold.Clip, text: str) -> dict[str, Any]:
    numbers = [(label, _number_present(text, value))
               for label, value in gold.numeric_targets(clip)]
    names = [(d.name, _name_present(text, d)) for d in clip.drugs]
    return {
        "numbers": numbers,
        "numbers_hit": sum(1 for _, ok in numbers if ok),
        "numbers_total": len(numbers),
        "drug_names": names,
        "drug_names_hit": sum(1 for _, ok in names if ok),
        "drug_names_total": len(names),
    }


def _totals(rows: Iterable[Mapping[str, Any]], *keys: str) -> dict[str, int]:
    out = {k: 0 for k in keys}
    for row in rows:
        for k in keys:
            out[k] += int(row.get(k, 0) or 0)
    return out


def score(data_root: Path, *, channel: str) -> dict[str, Any]:
    run_path = corpus_root(data_root) / f"run-{channel}.json"
    if not run_path.is_file():
        raise RuntimeError(f"no run to score: {run_path}")
    run = json.loads(run_path.read_text(encoding="utf-8"))

    report: dict[str, Any] = {
        "channel": channel,
        "clips": len(run["clips"]),
        "extraction_mode": "regex-only (use_llm=False; no Ollama model pulled)",
        "wer": "not computable - reference transcripts are paraphrases, not verbatim",
        "per_clip": {},
        "components": {},
    }

    ceiling_rows: list[dict[str, Any]] = []
    arm_rows: dict[str, list[dict[str, Any]]] = {}
    asr_rows: dict[str, list[dict[str, Any]]] = {}
    clean_rows: dict[str, list[dict[str, Any]]] = {}

    for clip_id, entry in sorted(run["clips"].items()):
        clip = gold.BY_ID[clip_id]
        ceiling = _score_entities(clip, entry["reference_pass"]["entities"])
        ceiling_rows.append(ceiling)
        per_clip: dict[str, Any] = {"extraction_ceiling": ceiling, "arms": {}}
        for key, arm in sorted(entry["arms"].items()):
            clean_rows.setdefault(key, []).append(arm["clean"])
            if not arm.get("pipeline"):
                per_clip["arms"][key] = {"failed": arm["stt"].get("error") or "clean failed"}
                continue
            asr = _score_asr(clip, arm["stt"]["text"])
            end = _score_entities(clip, arm["pipeline"]["entities"])
            asr_rows.setdefault(key, []).append(asr)
            arm_rows.setdefault(key, []).append(end)
            per_clip["arms"][key] = {"asr_fidelity": asr, "end_to_end": end,
                                     "stt_seconds": arm["stt"]["seconds"]}
        report["per_clip"][clip_id] = per_clip

    # Spurious counts ride alongside the hit counts so every arm reports
    # precision as well as recall.
    keys_e = ("drugs_wrong_polarity",
              "vitals_spurious", "vitals_emitted", "drugs_spurious",
              "drugs_emitted", "allergies_spurious", "allergies_emitted",
              "vitals_hit", "vitals_total", "drugs_hit", "drugs_total",
              "doses_hit", "doses_total", "allergies_correct", "allergies_total")
    report["components"]["C4_extraction_ceiling"] = _totals(ceiling_rows, *keys_e)
    report["components"]["C2_asr_fidelity"] = {
        key: _totals(rows, "numbers_hit", "numbers_total",
                     "drug_names_hit", "drug_names_total")
        for key, rows in sorted(asr_rows.items())
    }
    report["components"]["C5_end_to_end"] = {
        key: _totals(rows, *keys_e) for key, rows in sorted(arm_rows.items())
    }
    report["components"]["C1_ingest"] = {
        key: {
            "passed": sum(1 for r in rows if r["ok"]),
            "failed": sum(1 for r in rows if not r["ok"]),
            "errors": sorted({r["error"] for r in rows if not r["ok"]}),
            "mean_vad_ratio": (
                round(sum(r.get("vad_ratio") or 0 for r in rows if r["ok"])
                      / max(1, sum(1 for r in rows if r["ok"])), 4)
            ),
        }
        for key, rows in sorted(clean_rows.items())
    }

    out = corpus_root(data_root) / f"score-{channel}.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["output"] = str(out)
    return report


def _spurious_line(row: Mapping[str, Any]) -> str:
    """Rows emitted that matched no gold fact - the precision half.

    Printed on every arm rather than only when non-zero, so a regression that
    starts emitting junk is visible as a number changing rather than as a line
    appearing.
    """
    return ("spurious: vitals {vitals_spurious}/{vitals_emitted} "
            "drugs {drugs_spurious}/{drugs_emitted} "
            "allergies {allergies_spurious}/{allergies_emitted} emitted"
            ).format(**{k: row.get(k, 0) for k in (
                "vitals_spurious", "vitals_emitted", "drugs_spurious",
                "drugs_emitted", "allergies_spurious", "allergies_emitted")})


def _pct(hit: int, total: int) -> str:
    return f"{hit}/{total}" + (f" ({100*hit/total:.0f}%)" if total else "")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clinical_eval.score")
    parser.add_argument("--data-root", default=str(DATA_ROOT_DEFAULT))
    parser.add_argument("--channel", default="mix", choices=("mix", "left", "right"))
    arguments = parser.parse_args(argv)
    report = score(Path(arguments.data_root), channel=arguments.channel)

    print(f"\nclinical eval - {report['clips']} clips, channel={report['channel']}")
    print(f"WER: {report['wer']}")
    print(f"extraction: {report['extraction_mode']}\n")

    print("C1 ingest/clean")
    for key, row in report["components"]["C1_ingest"].items():
        print(f"  {key:<34} passed {row['passed']}  failed {row['failed']}  "
              f"mean VAD {row['mean_vad_ratio']}")
        for error in row["errors"]:
            print(f"      ! {error[:100]}")

    print("\nC2 ASR fidelity (is the gold value present in the raw transcript?)")
    for key, row in report["components"]["C2_asr_fidelity"].items():
        print(f"  {key:<34} numbers {_pct(row['numbers_hit'], row['numbers_total']):<14}"
              f" drug names {_pct(row['drug_names_hit'], row['drug_names_total'])}")

    c4 = report["components"]["C4_extraction_ceiling"]
    print("\nC4 extraction ceiling (perfect transcript in)")
    print(f"  vitals {_pct(c4['vitals_hit'], c4['vitals_total']):<14}"
          f" drugs {_pct(c4['drugs_hit'], c4['drugs_total']):<14}"
          f" doses {_pct(c4['doses_hit'], c4['doses_total']):<12}"
          f" allergy polarity {_pct(c4['allergies_correct'], c4['allergies_total'])}")
    print(f"  {_spurious_line(c4)}")

    print("\nC5 end to end (audio in, entities out)")
    for key, row in report["components"]["C5_end_to_end"].items():
        print(f"  {key:<34} vitals {_pct(row['vitals_hit'], row['vitals_total']):<14}"
              f" drugs {_pct(row['drugs_hit'], row['drugs_total']):<14}"
              f" doses {_pct(row['doses_hit'], row['doses_total']):<12}"
              f" allergy {_pct(row['allergies_correct'], row['allergies_total'])}")
        print(f"  {'':<34} {_spurious_line(row)}")
    print(f"\nwrote {report['output']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

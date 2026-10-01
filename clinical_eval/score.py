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
        return re.search(rf"(?<!\d){re.escape(integral)}(?!\d)", haystack) is not None
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


def _vital_hit(entities: Mapping[str, Any], vital: gold.Vital) -> bool:
    """Does any extracted vital string carry this gold value?

    The extractor emits vitals as raw spans, not typed fields, so the check is
    value-level: the span must contain the number (both numbers, for a BP pair).
    """

    spans = [str(item.get("text", "")) for item in entities.get("vitals", []) or []]
    if not spans:
        return False
    if vital.kind == "bp":
        return any(
            _number_present(s, vital.systolic) and _number_present(s, vital.diastolic)
            for s in spans
        )
    return any(_number_present(s, vital.value) for s in spans)


def _allergy_polarity(entities: Mapping[str, Any], allergy: gold.Allergy) -> str:
    """`correct` / `wrong_polarity` / `missing` for one gold allergy."""

    for item in entities.get("allergies", []) or []:
        if str(item.get("text", "")).lower() != allergy.substance.lower():
            continue
        return "correct" if bool(item.get("negated")) == allergy.negated else "wrong_polarity"
    return "missing"


def _score_entities(clip: gold.Clip, entities: Mapping[str, Any]) -> dict[str, Any]:
    vitals = [(f"{v.kind}{'_'+v.note if v.note else ''}", _vital_hit(entities, v))
              for v in clip.vitals]
    drugs = []
    for drug in clip.drugs:
        found = _extracted_drug(entities, drug)
        dose_ok = None
        if drug.dose is not None:
            dose_ok = bool(
                found
                and found.get("dose") is not None
                and float(found["dose"]) == drug.dose
                and str(found.get("unit") or "").lower() == (drug.unit or "").lower()
            )
        drugs.append({"name": drug.name, "found": found is not None,
                      "dose_expected": drug.dose, "dose_ok": dose_ok})
    allergies = [{"substance": a.substance, "gold_negated": a.negated,
                  "outcome": _allergy_polarity(entities, a)} for a in clip.allergies]
    return {
        "vitals": vitals,
        "vitals_hit": sum(1 for _, ok in vitals if ok),
        "vitals_total": len(vitals),
        "drugs": drugs,
        "drugs_hit": sum(1 for d in drugs if d["found"]),
        "drugs_total": len(drugs),
        "doses_hit": sum(1 for d in drugs if d["dose_ok"]),
        "doses_total": sum(1 for d in drugs if d["dose_expected"] is not None),
        "allergies": allergies,
        "allergies_correct": sum(1 for a in allergies if a["outcome"] == "correct"),
        "allergies_total": len(allergies),
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

    keys_e = ("vitals_hit", "vitals_total", "drugs_hit", "drugs_total",
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

    print("\nC5 end to end (audio in, entities out)")
    for key, row in report["components"]["C5_end_to_end"].items():
        print(f"  {key:<34} vitals {_pct(row['vitals_hit'], row['vitals_total']):<14}"
              f" drugs {_pct(row['drugs_hit'], row['drugs_total']):<14}"
              f" doses {_pct(row['doses_hit'], row['doses_total']):<12}"
              f" allergy {_pct(row['allergies_correct'], row['allergies_total'])}")
    print(f"\nwrote {report['output']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

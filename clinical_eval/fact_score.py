"""Fact-level scoring of `base_pipeline.run` output against the 30-clip ground truth.

    python -m clinical_eval.fact_score runs/v0
    python -m clinical_eval.fact_score runs/v1 --compare runs/v0      # paired, per fact

Reads `<runs>/<clip_id>/<denoiser>/entities.json` (the layout `base_pipeline.run`
writes) and scores it with `score._score_entities` against `gold_v2`.

Why not just a percentage: the 23 held-out clips hold ~123 facts, so one fact is
0.8 points and an interval of about +-5. Facts within a clip share a speaker,
a recording and an ASR run, so they are not independent: the headline interval
is a **clip-level cluster bootstrap**, with the Wilson interval beside it for
comparison. A change between two runs is judged **per fact, paired**
(exact McNemar on the facts one run got and the other did not), never by
comparing two percentages.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from clinical_eval import gold_v2
from clinical_eval.score import _score_entities

HELD_OUT_FROM = 8          # MB_MED_001..007 are the in-sample clips, 008+ held out
BOOTSTRAP_ROUNDS = 10_000
BOOTSTRAP_SEED = 1729


def clip_number(clip_id: str) -> int:
    return int(clip_id.split("_")[2])


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (centre - half), 100 * (centre + half))


def cluster_bootstrap(per_clip: Sequence[tuple[int, int]], rounds: int = BOOTSTRAP_ROUNDS,
                      seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """95% interval for sum(hit)/sum(total), resampling whole clips."""
    clips = [c for c in per_clip if c[1]]
    if not clips:
        return (0.0, 0.0)
    rng = random.Random(seed)
    stats = []
    for _ in range(rounds):
        draw = [clips[rng.randrange(len(clips))] for _ in clips]
        stats.append(100 * sum(h for h, _ in draw) / sum(t for _, t in draw))
    stats.sort()
    return (stats[int(0.025 * rounds)], stats[int(0.975 * rounds) - 1])


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact p for the discordant pairs (a-only vs b-only)."""
    n = only_a + only_b
    if not n:
        return 1.0
    k = min(only_a, only_b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def fact_hits(clip, score: Mapping[str, Any]) -> dict[tuple, bool]:
    """One boolean per gold fact, keyed so two runs can be paired."""
    hits: dict[tuple, bool] = {}
    for i, (_label, ok) in enumerate(score["vitals"]):
        hits[(clip.clip_id, "vital", i)] = bool(ok)
    for i, d in enumerate(score["drugs"]):
        hits[(clip.clip_id, "drug", i)] = bool(d["found"] and d["polarity_ok"])
        if d["dose_expected"] is not None:
            hits[(clip.clip_id, "dose", i)] = bool(d["dose_ok"])
    for i, a in enumerate(score["allergies"]):
        hits[(clip.clip_id, "allergy", i)] = a["outcome"] == "correct"
    return hits


def load_entities(runs_dir: Path, clip_id: str, denoiser: str) -> dict[str, Any] | None:
    path = runs_dir / clip_id / denoiser / "entities.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def score_runs(runs_dir: Path, *, denoiser: str = "none", include_borderline: bool = False,
               current_only: bool = False) -> dict[str, Any]:
    gold = gold_v2.load(include_borderline=include_borderline, current_only=current_only)
    clips: dict[str, Any] = {}
    hits: dict[tuple, bool] = {}
    missing = []
    for clip_id, clip in sorted(gold.items()):
        entities = load_entities(runs_dir, clip_id, denoiser)
        if entities is None:
            missing.append(clip_id)
            continue
        score = _score_entities(clip, entities)
        clips[clip_id] = score
        hits.update(fact_hits(clip, score))
    return {"gold": gold, "clips": clips, "hits": hits, "missing": missing}


def _slice(report: Mapping[str, Any], ids: Sequence[str]) -> dict[str, Any]:
    keys = ("vitals_hit", "vitals_total", "drugs_hit", "drugs_total", "doses_hit", "doses_total",
            "allergies_correct", "allergies_total", "vitals_spurious", "drugs_spurious",
            "drugs_wrong_polarity")
    tot = {k: 0 for k in keys}
    per_clip = []
    for cid in ids:
        sc = report["clips"].get(cid)
        if sc is None:
            continue
        for k in keys:
            tot[k] += sc[k]
        hit = sc["vitals_hit"] + sc["drugs_hit"] + sc["doses_hit"] + sc["allergies_correct"]
        n = sc["vitals_total"] + sc["drugs_total"] + sc["doses_total"] + sc["allergies_total"]
        per_clip.append((hit, n))
    k = sum(h for h, _ in per_clip)
    n = sum(t for _, t in per_clip)
    return {"hit": k, "total": n, "pct": round(100 * k / n, 1) if n else None,
            "wilson": tuple(round(x, 1) for x in wilson(k, n)),
            "cluster_bootstrap": tuple(round(x, 1) for x in cluster_bootstrap(per_clip)),
            "clips": len(per_clip), **tot}


def summarise(report: Mapping[str, Any]) -> dict[str, Any]:
    ids = sorted(report["clips"])
    return {"in_sample": _slice(report, [i for i in ids if clip_number(i) < HELD_OUT_FROM]),
            "held_out": _slice(report, [i for i in ids if clip_number(i) >= HELD_OUT_FROM]),
            "missing_clips": report["missing"]}


def compare(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    """a vs b on the facts both runs scored."""
    shared = a["hits"].keys() & b["hits"].keys()
    only_a = sorted(k for k in shared if a["hits"][k] and not b["hits"][k])
    only_b = sorted(k for k in shared if b["hits"][k] and not a["hits"][k])
    return {"facts_compared": len(shared), "a_only": only_a, "b_only": only_b,
            "net_a_minus_b": len(only_a) - len(only_b),
            "p_exact_mcnemar": round(mcnemar_exact(len(only_a), len(only_b)), 3)}


def misses(report: Mapping[str, Any]) -> list[str]:
    out = []
    for cid, sc in sorted(report["clips"].items()):
        bad = ([lbl for lbl, ok in sc["vitals"] if not ok]
               + [d["name"] + (" (wrong polarity)" if d["found"] and not d["polarity_ok"] else "")
                  for d in sc["drugs"] if not (d["found"] and d["polarity_ok"])]
               + [a["substance"] + ":" + a["outcome"] for a in sc["allergies"]
                  if a["outcome"] != "correct"])
        extra = []
        if sc["vitals_spurious"]:
            extra.append(f"spurious vitals {sc['vitals_spurious']}")
        if sc["drugs_spurious"]:
            extra.append("spurious drugs " + ",".join(sc["drugs_spurious_names"]))
        if bad or extra:
            out.append(f"{cid}: miss {bad}" + (f"  | {'; '.join(extra)}" if extra else ""))
    return out


def _line(name: str, s: Mapping[str, Any]) -> str:
    if not s["total"]:
        return f"  {name:10s} no clips scored"
    return (f"  {name:10s} {s['hit']}/{s['total']} = {s['pct']}%  "
            f"cluster-bootstrap {list(s['cluster_bootstrap'])}  wilson {list(s['wilson'])}  "
            f"[{s['clips']} clips]  vitals {s['vitals_hit']}/{s['vitals_total']} "
            f"drugs {s['drugs_hit']}/{s['drugs_total']}  spurious v={s['vitals_spurious']} "
            f"d={s['drugs_spurious']}  wrong-polarity={s['drugs_wrong_polarity']}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="clinical_eval.fact_score", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs_dir", help="the --out folder of a base_pipeline.run")
    ap.add_argument("--denoiser", default="none")
    ap.add_argument("--compare", help="a second runs folder to pair against (this one is 'a')")
    ap.add_argument("--include-borderline", action="store_true",
                    help="also score items the annotators flagged as doubtful (oxygen, iron, ...)")
    ap.add_argument("--current-only", action="store_true",
                    help="drop readings the transcript itself marks as superseded")
    ap.add_argument("--misses", action="store_true", help="list every miss and spurious span")
    ap.add_argument("--json", help="also write the summary here")
    a = ap.parse_args(argv)

    kw = dict(denoiser=a.denoiser, include_borderline=a.include_borderline,
              current_only=a.current_only)
    rep = score_runs(Path(a.runs_dir), **kw)
    summary = summarise(rep)
    print(f"{a.runs_dir}  (denoiser={a.denoiser}, borderline={'in' if a.include_borderline else 'out'}, "
          f"{'current-only' if a.current_only else 'all readings'})")
    print(_line("in-sample", summary["in_sample"]))
    print(_line("held-out", summary["held_out"]))
    if rep["missing"]:
        print(f"  NOT SCORED (no entities.json): {len(rep['missing'])} clips, e.g. {rep['missing'][:3]}")
    if a.misses:
        print("\n".join(["", *misses(rep)]))
    out: dict[str, Any] = {"summary": summary}
    if a.compare:
        other = score_runs(Path(a.compare), **kw)
        cmp_ = compare(rep, other)
        print(f"\npaired vs {a.compare}: {cmp_['facts_compared']} facts; this run only {len(cmp_['a_only'])}, "
              f"other only {len(cmp_['b_only'])}, net {cmp_['net_a_minus_b']:+d}, "
              f"exact McNemar p={cmp_['p_exact_mcnemar']}")
        out["compare"] = {**cmp_, "a_only": [list(k) for k in cmp_["a_only"]],
                          "b_only": [list(k) for k in cmp_["b_only"]]}
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2, default=list), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

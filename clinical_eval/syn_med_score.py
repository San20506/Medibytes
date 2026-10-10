"""Score `base_pipeline.run` output against the 40-clip United-Syn-Med sample.

    python -m clinical_eval.syn_med_score runs/syn_a_baseline
    python -m clinical_eval.syn_med_score runs/syn_c_context --compare runs/syn_a_baseline

Three things are measured per clip, because they fail independently:

  heard     did the ASR transcript contain the gold name (brand or substance)?
            fuzzy string match, >= HEARD_RATIO, so `Floxina OZ` still counts
  emitted   did the pipeline emit a drug row for each gold substance? (recall)
  spurious  drug rows that name nothing the clip says (precision)

Brands are scored at the `heard` level only: the pipeline never turns a brand
into a drug row, so there is no row to score. A *substance* row produced from a
brand is exactly the false positive this set exists to catch.

Facts are keyed so two runs can be paired (exact McNemar), as in `fact_score`.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path
from typing import Any, Sequence

from clinical_eval.fact_score import mcnemar_exact, wilson

GOLDS = {"dev": Path(__file__).with_name("syn_med_gold.json"),
         "heldout": Path(__file__).with_name("syn_med_gold_40b.json")}
HEARD_RATIO = 0.80
_WORD = re.compile(r"[a-z0-9]+")


def _norm(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def heard(name: str, transcript: str) -> float:
    """Best similarity of `name` to any same-length word window of the transcript."""
    target = _norm(name)
    words = _norm(transcript).split()
    k = len(target.split())
    best = 0.0
    for size in {max(1, k - 1), k, k + 1}:
        for i in range(len(words) - size + 1):
            window = " ".join(words[i:i + size])
            best = max(best, difflib.SequenceMatcher(None, target, window).ratio())
    return best


def row_matches(row_name: str, gold: str) -> bool:
    a, b = _norm(row_name), _norm(gold)
    return bool(a and b and (a == b or b in a.split() or a in b.split() or a.startswith(b)))


def score_run(runs_dir: Path, denoiser: str = "none", gold_set: str = "dev") -> dict[str, Any]:
    gold = json.loads(GOLDS[gold_set].read_text(encoding="utf-8"))["clips"]
    out: dict[str, Any] = {"clips": {}, "hits": {}}
    for clip_id, g in sorted(gold.items()):
        d = runs_dir / clip_id / denoiser
        if not (d / "entities.json").is_file():
            out["clips"][clip_id] = {"missing": True}
            continue
        ent = json.loads((d / "entities.json").read_text(encoding="utf-8"))
        text = json.loads((d / "transcript.json").read_text(encoding="utf-8"))["normalized_en"]
        rows = [r for r in ent.get("drugs", []) if isinstance(r, dict)]
        allowed = list(g["drugs"]) + list(g.get("borderline", []))
        rec: dict[str, Any] = {"transcript": text, "rows": [
            {"name": r["name"], "color": r.get("color"), "heard": r.get("raw_token"),
             "note": r.get("note", ""), "validation": (r.get("validation") or {}).get("status")}
            for r in rows]}
        if g["brand"]:
            ratio = heard(g["brand"], text)
            rec["brand_ratio"] = round(ratio, 3)
            out["hits"][(clip_id, "brand_heard", g["brand"])] = ratio >= HEARD_RATIO
        rec["substances"] = []
        for sub in g["drugs"]:
            ratio = heard(sub, text)
            emitted = any(row_matches(r["name"], sub) for r in rows)
            rec["substances"].append({"name": sub, "heard_ratio": round(ratio, 3), "emitted": emitted})
            out["hits"][(clip_id, "sub_heard", sub)] = ratio >= HEARD_RATIO
            out["hits"][(clip_id, "sub_emitted", sub)] = emitted
        rec["spurious"] = [r["name"] for r in rows
                           if not any(row_matches(r["name"], a) for a in allowed)]
        out["clips"][clip_id] = rec
    return out


def summarise(rep: dict[str, Any]) -> dict[str, Any]:
    def tally(kind: str) -> tuple[int, int]:
        ks = [k for k in rep["hits"] if k[1] == kind]
        return sum(rep["hits"][k] for k in ks), len(ks)
    spurious = [(c, n) for c, r in rep["clips"].items() for n in r.get("spurious", [])]
    s = {}
    for kind in ("brand_heard", "sub_heard", "sub_emitted"):
        k, n = tally(kind)
        s[kind] = {"hit": k, "total": n, "pct": round(100 * k / n, 1) if n else None,
                   "wilson": tuple(round(x, 1) for x in wilson(k, n))}
    s["spurious_rows"] = spurious
    s["rows_emitted"] = sum(len(r.get("rows", [])) for r in rep["clips"].values())
    return s


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    shared = a["hits"].keys() & b["hits"].keys()
    only_a = sorted(k for k in shared if a["hits"][k] and not b["hits"][k])
    only_b = sorted(k for k in shared if b["hits"][k] and not a["hits"][k])
    return {"a_only": only_a, "b_only": only_b, "p": round(mcnemar_exact(len(only_a), len(only_b)), 3)}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="clinical_eval.syn_med_score", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs_dir")
    ap.add_argument("--compare")
    ap.add_argument("--gold", choices=sorted(GOLDS), default="dev",
                    help="dev = the 40 clips the stage was built on; heldout = a disjoint 40")
    ap.add_argument("--detail", action="store_true", help="per-clip rows, misses and spurious rows")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    rep = score_run(Path(a.runs_dir), gold_set=a.gold)
    s = summarise(rep)
    print(a.runs_dir)
    for kind, label in (("brand_heard", "brand heard by ASR"), ("sub_heard", "substance heard by ASR"),
                        ("sub_emitted", "substance emitted as row")):
        v = s[kind]
        print(f"  {label:26s} {v['hit']}/{v['total']} = {v['pct']}%  wilson {list(v['wilson'])}")
    print(f"  drug rows emitted {s['rows_emitted']}, spurious {len(s['spurious_rows'])}: "
          f"{', '.join(f'{c[-3:]}:{n}' for c, n in s['spurious_rows']) or '-'}")
    if a.detail:
        for cid, r in rep["clips"].items():
            if r.get("missing"):
                print(cid, "MISSING")
                continue
            flags = []
            if "brand_ratio" in r and r["brand_ratio"] < HEARD_RATIO:
                flags.append(f"brand-miss({r['brand_ratio']})")
            flags += [f"{x['name']}: heard {x['heard_ratio']} emitted {x['emitted']}"
                      for x in r["substances"] if not (x["emitted"] and x["heard_ratio"] >= HEARD_RATIO)]
            flags += [f"SPURIOUS {n}" for n in r["spurious"]]
            if flags:
                print(f"{cid}: {'; '.join(flags)}\n    heard: {r['transcript'][:160]}")
    out: dict[str, Any] = {"summary": s}
    if a.compare:
        c = compare(rep, score_run(Path(a.compare), gold_set=a.gold))
        print(f"\npaired vs {a.compare}: this only {len(c['a_only'])}, other only {len(c['b_only'])}, "
              f"exact McNemar p={c['p']}")
        out["compare"] = {"a_only": [list(k) for k in c["a_only"]], "b_only": [list(k) for k in c["b_only"]], "p": c["p"]}
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=list), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

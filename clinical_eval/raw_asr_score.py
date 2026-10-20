"""Raw-transcript scoring for decoding presets (openspec/changes/raw-asr-decoding-presets).

    python -m clinical_eval.raw_asr_score runs/decode/greedy --set test30
    python -m clinical_eval.raw_asr_score runs/decode/beam5 --set test30 --compare runs/decode/greedy

Reads `<run>/<clip_id>/<arm>/transcript.json` text only - never the extracted entities - so the
number is the decoder's, not the post-processing's. Word error rate is pooled over the set
(total edits / total reference words); because one runaway clip can dominate that, runaway
clips (more than RUNAWAY_FACTOR x the reference words) are listed and the WER without them is
reported beside it.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any

from clinical_eval.syn_med_score import HEARD_RATIO, heard
from base_pipeline.paths import data_root

CORPUS = data_root() / "corpus"
HERE = Path(__file__).parent
RUNAWAY_FACTOR = 3
BOOTSTRAP_ROUNDS = 10_000
BOOTSTRAP_SEED = 1729


def tokens(text: str) -> list[str]:
    """Words of `text`: letters, combining marks and digits in any script.

    Marks (Unicode M*) stay inside the word: Devanagari and Tamil vowel signs, the anusvara and the
    virama are marks, and `\\w` or an `[a-z]` class would cut a word at each one. NFC makes the
    precomposed and decomposed spellings of a letter one word; joiners (ZWJ/ZWNJ) are removed so
    they neither split a word nor make two spellings differ. ASCII input tokenises as it always did.
    """
    text = unicodedata.normalize("NFC", text).lower().replace("'", "").replace("-", " ")
    text = text.replace("‌", "").replace("‍", "")
    words, word = [], []
    for ch in text:
        if unicodedata.category(ch)[0] in "LMN":
            word.append(ch)
        elif word:
            words.append("".join(word))
            word = []
    if word:
        words.append("".join(word))
    return words


def edits(ref: list[str], hyp: list[str]) -> int:
    """Word-level Levenshtein distance."""
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1]


def _samples(corpus: str) -> dict[str, str]:
    data = json.loads((CORPUS / corpus / "sample.json").read_text(encoding="utf-8"))
    return {c["clip_id"]: c["transcription"] for c in data["clips"]}


def load_set(name: str) -> dict[str, dict[str, Any]]:
    """clip_id -> {truth, brand, drugs} for a named set."""
    if name == "test30":
        clips = json.loads((HERE / "syn_med_test30.json").read_text(encoding="utf-8"))["clips"]
        return {k: {"truth": v["transcription"], "brand": v.get("brand"), "drugs": v["drugs"]}
                for k, v in clips.items()}
    corpus, gold_file = {"dev40": ("united-syn-med-40", "syn_med_gold.json"),
                         "held40": ("united-syn-med-40b", "syn_med_gold_40b.json")}[name]
    truth = _samples(corpus)
    gold = json.loads((HERE / gold_file).read_text(encoding="utf-8"))["clips"]
    return {k: {"truth": truth[k], "brand": gold[k].get("brand"), "drugs": gold[k]["drugs"]}
            for k in gold}


SETS = ("dev40", "held40", "test30")


def score_run(run: Path, set_name: str, arm: str = "none") -> dict[str, Any]:
    """Per-clip and pooled raw-transcript scores for the clips of `set_name` present in `run`."""
    clips, missing = {}, []
    for cid, g in load_set(set_name).items():
        path = Path(run) / cid / arm / "transcript.json"
        if not path.is_file():
            missing.append(cid)
            continue
        text = json.loads(path.read_text(encoding="utf-8"))["text"]
        ref, hyp = tokens(g["truth"]), tokens(text)
        clips[cid] = {
            "errors": edits(ref, hyp), "ref_words": len(ref), "hyp_words": len(hyp),
            "runaway": len(hyp) > RUNAWAY_FACTOR * len(ref),
            "brand": (None if not g["brand"] else heard(g["brand"], text) >= HEARD_RATIO),
            "substances": [heard(d, text) >= HEARD_RATIO for d in g["drugs"]],
            "text": text.strip(),
        }
    return {"set": set_name, "clips": clips, "missing": missing, **_pool(clips)}


def _pool(clips: dict[str, dict[str, Any]]) -> dict[str, Any]:
    def wer(rows):
        n = sum(r["ref_words"] for r in rows)
        return round(100 * sum(r["errors"] for r in rows) / n, 2) if n else None

    rows = list(clips.values())
    kept = [r for r in rows if not r["runaway"]]
    brands = [r["brand"] for r in rows if r["brand"] is not None]
    subs = [h for r in rows for h in r["substances"]]
    return {
        "n_clips": len(rows), "wer": wer(rows), "wer_excl_runaway": wer(kept),
        "word_accuracy": None if wer(rows) is None else round(100 - wer(rows), 2),
        "errors": sum(r["errors"] for r in rows), "ref_words": sum(r["ref_words"] for r in rows),
        "exact_sentences": sum(r["errors"] == 0 for r in rows),
        "runaway_clips": sorted(c for c, r in clips.items() if r["runaway"]),
        "brand_heard": f"{sum(brands)}/{len(brands)}", "substance_heard": f"{sum(subs)}/{len(subs)}",
    }


def compare(a: dict[str, Any], b: dict[str, Any], rounds: int = BOOTSTRAP_ROUNDS,
            seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """WER(a) - WER(b) on the clips both runs scored: per-clip tally and a clip-level bootstrap."""
    shared = sorted(set(a["clips"]) & set(b["clips"]))
    pairs = [(a["clips"][c]["errors"], b["clips"][c]["errors"], a["clips"][c]["ref_words"]) for c in shared]
    total = sum(p[2] for p in pairs)
    rng, diffs = random.Random(seed), []
    for _ in range(rounds):
        draw = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        n = sum(p[2] for p in draw)
        diffs.append(100 * sum(p[0] - p[1] for p in draw) / n)
    diffs.sort()
    lo, hi = diffs[int(0.025 * rounds)], diffs[int(0.975 * rounds) - 1]
    return {"clips": len(shared),
            "wer_diff_a_minus_b": round(100 * sum(p[0] - p[1] for p in pairs) / total, 2),
            "ci95": (round(lo, 2), round(hi, 2)), "a_better_clips": sum(p[0] < p[1] for p in pairs),
            "a_worse_clips": sum(p[0] > p[1] for p in pairs), "equal_clips": sum(p[0] == p[1] for p in pairs),
            "significant": lo > 0 or hi < 0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="clinical_eval.raw_asr_score", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--set", choices=SETS, required=True)
    ap.add_argument("--arm", default="none")
    ap.add_argument("--compare", help="a second run to pair against (this one is 'a')")
    ap.add_argument("--json", help="write the pooled result (and comparison) here")
    a = ap.parse_args(argv)
    rep = score_run(Path(a.run), a.set, a.arm)
    out = {k: v for k, v in rep.items() if k != "clips"}
    print(f"{a.run} [{a.set}] {rep['n_clips']} clips: WER {rep['wer']}% (word accuracy {rep['word_accuracy']}%), "
          f"WER excl. runaway {rep['wer_excl_runaway']}%, exact {rep['exact_sentences']}/{rep['n_clips']}, "
          f"brand heard {rep['brand_heard']}, substance heard {rep['substance_heard']}, runaway {rep['runaway_clips']}")
    if rep["missing"]:
        print(f"  NOT SCORED (no transcript): {len(rep['missing'])} clips")
    if a.compare:
        cmp = compare(rep, score_run(Path(a.compare), a.set, a.arm))
        out["compare"] = {"against": a.compare, **cmp}
        print(f"  vs {a.compare}: WER diff {cmp['wer_diff_a_minus_b']:+} pts, 95% CI {cmp['ci95']}, "
              f"better/worse/equal clips {cmp['a_better_clips']}/{cmp['a_worse_clips']}/{cmp['equal_clips']}, "
              f"{'SIGNIFICANT' if cmp['significant'] else 'not significant'}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

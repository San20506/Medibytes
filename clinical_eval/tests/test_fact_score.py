"""fact_score: the statistics are right, and a hand-built run scores as hand-computed."""
import json

import pytest

from clinical_eval import fact_score as fs
from clinical_eval import gold_v2


def test_wilson_matches_known_values():
    lo, hi = fs.wilson(30, 100)
    assert (round(lo, 1), round(hi, 1)) == (21.9, 39.6)
    assert fs.wilson(0, 0) == (0.0, 0.0)


def test_mcnemar_exact_known_values():
    assert fs.mcnemar_exact(0, 0) == 1.0
    assert fs.mcnemar_exact(5, 0) == pytest.approx(0.0625)      # 2 * (1/32)
    assert fs.mcnemar_exact(3, 3) == 1.0
    assert fs.mcnemar_exact(0, 5) == fs.mcnemar_exact(5, 0)


def test_cluster_bootstrap_is_deterministic_and_wider_than_a_point():
    clips = [(4, 4), (3, 4), (4, 4), (2, 4), (4, 4), (4, 4)]
    a = fs.cluster_bootstrap(clips, rounds=2000)
    assert a == fs.cluster_bootstrap(clips, rounds=2000)
    assert a[0] < 100 * 21 / 24 < a[1] <= 100.0
    # clip-level resampling is wider than treating the 24 facts as independent
    assert (a[1] - a[0]) > (fs.wilson(21, 24)[1] - fs.wilson(21, 24)[0]) * 0.5


def _write(runs, clip_id, entities, denoiser="none"):
    d = runs / clip_id / denoiser
    d.mkdir(parents=True)
    (d / "entities.json").write_text(json.dumps(entities), encoding="utf-8")


PERFECT_002 = {  # MB_MED_002: BP 150/95, pulse 86, temp 98.6 F, SpO2 98, amlodipine 5 mg
    "vitals": [{"text": "blood pressure 150 over 95", "color": "GREEN"},
               {"text": "pulse rate 86 beats per minute", "color": "GREEN"},
               {"text": "temperature 98.6 degrees Fahrenheit", "color": "GREEN"},
               {"text": "oxygen saturation 98 percent", "color": "GREEN"}],
    "drugs": [{"name": "amlodipine", "dose": 5.0, "unit": "mg", "negated": False}],
    "allergies": [],
}


def test_a_perfect_clip_scores_every_fact(tmp_path):
    _write(tmp_path, "MB_MED_002_clean", PERFECT_002)
    rep = fs.score_runs(tmp_path)
    s = fs.summarise(rep)["in_sample"]
    assert (s["hit"], s["total"], s["clips"]) == (6, 6, 1)      # 4 vitals + drug + its dose
    assert rep["missing"] and "MB_MED_001_clean" in rep["missing"]


def test_dropping_a_vital_and_adding_one_is_a_miss_and_a_spurious(tmp_path):
    bad = json.loads(json.dumps(PERFECT_002))
    bad["vitals"] = bad["vitals"][1:] + [{"text": "respiratory rate 40", "color": "GREEN"}]
    _write(tmp_path, "MB_MED_002_clean", bad)
    s = fs.summarise(fs.score_runs(tmp_path))["in_sample"]
    assert (s["hit"], s["total"]) == (5, 6) and s["vitals_spurious"] == 1


def test_paired_compare_names_the_fact_that_changed(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _write(a, "MB_MED_002_clean", PERFECT_002)
    worse = json.loads(json.dumps(PERFECT_002))
    worse["drugs"] = []
    _write(b, "MB_MED_002_clean", worse)
    cmp_ = fs.compare(fs.score_runs(a), fs.score_runs(b))
    assert cmp_["net_a_minus_b"] == 2                             # drug and its dose
    assert {k[1] for k in cmp_["a_only"]} == {"drug", "dose"} and not cmp_["b_only"]


def test_borderline_flag_changes_the_denominator(tmp_path):
    _write(tmp_path, "MB_MED_008_clean", {"vitals": [], "drugs": [], "allergies": []})
    base = fs.summarise(fs.score_runs(tmp_path))["held_out"]["total"]
    more = fs.summarise(fs.score_runs(tmp_path, include_borderline=True))["held_out"]["total"]
    assert more > base                                            # oxygen in clip 008
    assert gold_v2.load()["MB_MED_008_clean"].drugs

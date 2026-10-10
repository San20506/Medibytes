"""Spec: openspec/changes/raw-asr-decoding-presets (raw transcript scoring)."""
import json

from clinical_eval import raw_asr_score as r


def test_tokenisation_lowercases_drops_apostrophes_and_splits_hyphens():
    assert r.tokens("Doctor's L-Cet M Tab, 500 mg.") == ["doctors", "l", "cet", "m", "tab", "500", "mg"]


def test_edit_distance_counts_substitution_insertion_deletion():
    assert r.edits(["a", "b", "c"], ["a", "b", "c"]) == 0
    assert r.edits(["a", "b", "c"], ["a", "x", "c"]) == 1
    assert r.edits(["a", "b", "c"], ["a", "c"]) == 1
    assert r.edits(["a", "c"], ["a", "b", "c"]) == 1
    assert r.edits([], ["a", "b"]) == 2


def _clip(errors, ref, hyp, brand=None, subs=()):
    return {"errors": errors, "ref_words": ref, "hyp_words": hyp, "runaway": hyp > 3 * ref,
            "brand": brand, "substances": list(subs), "text": ""}


def test_a_runaway_clip_is_listed_and_does_not_hide_in_the_pooled_wer():
    pooled = r._pool({"ok1": _clip(1, 10, 10), "ok2": _clip(1, 10, 11), "run": _clip(60, 10, 70)})
    assert pooled["runaway_clips"] == ["run"]
    assert pooled["wer"] == round(100 * 62 / 30, 2)            # dominated by the runaway
    assert pooled["wer_excl_runaway"] == 10.0                  # the other two clips
    assert pooled["exact_sentences"] == 0


def test_paired_comparison_counts_clips_and_only_calls_a_clear_gap_significant():
    a = {"clips": {f"c{i}": _clip(0, 10, 10) for i in range(20)}}
    b = {"clips": {f"c{i}": _clip(2, 10, 10) for i in range(20)}}
    cmp = r.compare(a, b, rounds=500)
    assert cmp["a_better_clips"] == 20 and cmp["wer_diff_a_minus_b"] == -20.0 and cmp["significant"]
    same = r.compare(a, a, rounds=500)
    assert same["equal_clips"] == 20 and same["wer_diff_a_minus_b"] == 0 and not same["significant"]


def test_scoring_reads_the_transcript_text_and_never_the_entities(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "load_set", lambda name: {"SYN_001": {"truth": "Take paracetamol daily.",
                                                                 "brand": None, "drugs": ["paracetamol"]}})
    arm = tmp_path / "SYN_001" / "none"
    arm.mkdir(parents=True)
    (arm / "transcript.json").write_text(json.dumps({"text": "take paracetamol daily"}), encoding="utf-8")
    (arm / "entities.json").write_text("not json - must not be read", encoding="utf-8")
    rep = r.score_run(tmp_path, "test30")
    assert rep["errors"] == 0 and rep["exact_sentences"] == 1 and rep["substance_heard"] == "1/1"


def test_the_shipped_test_set_has_gold_for_every_clip_and_no_overlap_with_tuning_sets():
    test = r.load_set("test30")
    assert len(test) == 30 and all("truth" in v for v in test.values())
    sources = {c["source_file"] for c in json.loads((r.HERE / "syn_med_test30.json").read_text(encoding="utf-8"))["clips"].values()}
    for corpus in ("united-syn-med-40", "united-syn-med-40b"):
        sample = r.CORPUS / corpus / "sample.json"
        if sample.is_file():  # the corpus is machine-local
            used = {c["source_file"] for c in json.loads(sample.read_text(encoding="utf-8"))["clips"]}
            assert not (sources & used), corpus

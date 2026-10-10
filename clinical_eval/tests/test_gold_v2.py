"""The ground-truth file is frozen: its content hash is recorded in its own meta."""
import hashlib
import json

from clinical_eval import gold_v2


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if k != "audio_confirmed"}
    return [_strip(x) for x in o] if isinstance(o, list) else o


def test_the_file_matches_its_recorded_hash():
    data = json.loads(gold_v2._PATH.read_text(encoding="utf-8"))
    digest = hashlib.sha256(json.dumps(_strip({"clips": data["clips"]}), sort_keys=True).encode()).hexdigest()
    assert digest == data["meta"]["sha256_of_clips"]


def test_default_load_covers_all_thirty_clips_and_every_fact_has_a_quote():
    clips = gold_v2.load()
    assert len(clips) == 30
    for c in json.loads(gold_v2._PATH.read_text(encoding="utf-8"))["clips"].values():
        assert all(f.get("quote") for kind in c.values() for f in kind)


def test_borderline_items_are_excluded_by_default_and_included_on_request():
    held = lambda cs: sum(len(c.drugs) for k, c in cs.items() if int(k[7:10]) >= 8)
    assert held(gold_v2.load(include_borderline=True)) > held(gold_v2.load())
    names = {d.name for c in gold_v2.load().values() for d in c.drugs}
    assert "oxygen" not in names and "oxygen" in {
        d.name for c in gold_v2.load(include_borderline=True).values() for d in c.drugs}


def test_a_combination_product_is_scored_as_its_components():
    names = [d.name for d in gold_v2.load()["MB_MED_001_clean"].drugs]
    assert "amoxicillin" in names and "clavulanic acid" in names
    assert "amoxicillin with clavulanic acid" not in names


def test_current_only_drops_superseded_readings():
    full, cur = gold_v2.load(), gold_v2.load(current_only=True)
    assert len(cur["MB_MED_006_clean"].vitals) < len(full["MB_MED_006_clean"].vitals)

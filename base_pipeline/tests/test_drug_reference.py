"""Stage 4a-bis: what the national drug reference is allowed to change.

The reference exists to fix one failure - the matcher returning the nearest of
33 strings when the drug that was actually said is not among them - and it
arrives carrying 72k brand names, which is its own hazard.  These tests pin
both halves: the fix has to work, and the 72k names must not start answering
questions nobody asked.

Most tests run against a fixture lexicon rather than the shipped asset, so
they state the rule instead of the current contents of a 5 MB file.  The few
that need the real asset skip when it is absent, because a checkout without
the built asset must still pass.
"""
import json
import os
import sys


import pytest

from base_pipeline import drug_reference
from base_pipeline.term_validate import filter_by_unit, unit_conflicts

FIXTURE = {
    "substances": {
        # The sound-alike pair the whole stage exists for: same string shape,
        # different order of magnitude in the unit.
        "hydroxyzine": {"units": ["mg"], "forms": ["tablet"], "aliases": [],
                        "strengths": [{"value": 25.0, "unit": "mg", "form": "tablet"}]},
        "levothyroxine": {"units": ["mcg"], "forms": ["tablet"], "aliases": [],
                          "strengths": [{"value": 50.0, "unit": "mcg", "form": "tablet"}]},
        "acetaminophen": {"units": ["mg"], "forms": ["tablet"], "aliases": [],
                          "strengths": [{"value": 500.0, "unit": "mg", "form": "tablet"}]},
        "azithromycin": {"units": ["mg"], "forms": ["tablet"], "aliases": [],
                         "strengths": [{"value": 500.0, "unit": "mg", "form": "tablet"}]},
        # No units listed: the pack is silent, which must not read as a veto.
        "mystery": {"units": [], "forms": [], "aliases": [], "strengths": []},
    },
    "brands": {
        "azee": {"substances": ["azithromycin"], "units": ["mg"]},
        "dolo": {"substances": ["acetaminophen"], "units": ["mg"]},
        "pan": {"substances": ["pantoprazole"], "units": ["mg"]},
    },
    # The pack is named in USAN; a clinician says the INN name.
    "aliases": {"paracetamol": "acetaminophen"},
}


@pytest.fixture
def ref(tmp_path, monkeypatch):
    p = tmp_path / "drug_reference.json"
    p.write_text(json.dumps(FIXTURE), encoding="utf-8")
    monkeypatch.setattr(drug_reference, "_CACHE", None)
    monkeypatch.setattr(drug_reference, "_ASSET", str(p))
    yield drug_reference
    monkeypatch.setattr(drug_reference, "_CACHE", None)


# ---- the reference answers, and refuses to answer ----

def test_an_exact_name_is_a_drug_even_when_the_mini_list_never_heard_of_it(ref):
    assert ref.lookup("hydroxyzine")["kind"] == "substance"
    assert ref.lookup("azee")["substances"] == ["azithromycin"]


def test_the_inn_name_a_clinician_actually_says_resolves_to_the_packs_usan_row(ref):
    # Nobody in an Indian ward dictates "acetaminophen". Without the bridge a
    # perfectly-said `paracetamol` misses the reference entirely.
    assert ref.lookup("paracetamol")["name"] == "acetaminophen"


def test_the_misheard_token_reaches_the_real_drug(ref):
    names = [c["name"] for c in ref.candidates("hydroxazine")]
    assert names[0] == "hydroxyzine"


def test_an_ordinary_english_word_before_a_dose_is_never_fuzzed(ref):
    # `DRUG_CTX`'s name group is `[A-Za-z]+`, so "was 200 mg" offers `was` as a
    # drug name. Against a real brand table that finds something.
    for word in ("was", "takes", "started", "daily", "total"):
        assert ref.candidates(word) == [], word


def test_short_tokens_are_not_fuzzed_but_are_still_matched_exactly(ref):
    # `Pan` is a real brand, so an exact hit stands; a 3-letter *guess* does not.
    assert ref.lookup("pan")["substances"] == ["pantoprazole"]
    assert ref.candidates("pan") == []


def test_brands_of_one_generic_collapse_to_a_single_candidate(ref):
    # Fifty brands of one drug are one answer. Left as fifty they would read as
    # a tie and send every row to a human.
    cands = ref.candidates("azithromycin")
    keys = [tuple(c["substances"]) for c in cands]
    assert len(keys) == len(set(keys))


# ---- units as evidence ----

HYDROXYZINE = {"name": "hydroxyzine", "score": 0.91,
               "reference": {"kind": "substance", "units": ["mg"],
                             "mass_mg": [2.0, 100.0]}}
LEVOTHYROXINE = {"name": "levothyroxine", "score": 0.88,
                 "reference": {"kind": "substance", "units": ["mcg"],
                               "mass_mg": [0.005, 0.3]}}


def test_the_pack_rules_out_the_sound_alike_on_dose_magnitude(ref):
    # 25 mg is an ordinary hydroxyzine tablet and 83x the top of the range the
    # pack lists for levothyroxine. That is a fact about medicine, which is
    # exactly what the string matcher could not supply.
    row = {"unit": "mg", "dose": 25.0, "candidates": [HYDROXYZINE, LEVOTHYROXINE]}
    kept, dropped, survived = filter_by_unit(row)
    assert dropped == ["levothyroxine"]
    assert survived and [c["name"] for c in kept] == ["hydroxyzine"]


def test_the_same_rule_cuts_the_other_way_in_micrograms(ref):
    row = {"unit": "mcg", "dose": 50.0, "candidates": [HYDROXYZINE, LEVOTHYROXINE]}
    kept, dropped, _ = filter_by_unit(row)
    assert dropped == ["hydroxyzine"]
    assert [c["name"] for c in kept] == ["levothyroxine"]


def test_silence_from_the_pack_is_not_a_veto(ref):
    # A substance the pack lists no strengths for must not be filtered out;
    # "unknown" and "contradicted" are different answers.
    cand = {"name": "mystery", "score": 0.9,
            "reference": {"kind": "substance", "units": [], "mass_mg": None}}
    assert unit_conflicts(cand, "mg", 500.0) is False


def test_a_candidate_without_a_reference_record_is_left_alone(ref):
    assert unit_conflicts({"name": "whatever", "score": 0.9}, "mg", 500.0) is False


def test_a_dose_in_a_bigger_unit_is_not_a_conflict(ref):
    # The rule this replaced compared unit strings, so `azithromycin 1 g` -
    # a standard single dose - came back as a rejection because the pack
    # lists the drug in mg. Grams and milligrams are the same dimension.
    azithro = {"name": "azithromycin", "score": 0.9,
               "reference": {"kind": "substance", "units": ["mg", "ml"],
                             "mass_mg": [10.0, 600.0]}}
    assert unit_conflicts(azithro, "g", 1.0) is False
    assert unit_conflicts(azithro, "mg", 500.0) is False


def test_filtering_never_empties_the_shortlist(ref):
    # If every candidate conflicts, the caller must still see them - it sends
    # the row to a human rather than choosing from an empty set.
    row = {"unit": "mg", "dose": 25.0, "candidates": [LEVOTHYROXINE]}
    kept, dropped, survived = filter_by_unit(row)
    assert survived is False
    assert dropped == ["levothyroxine"] and len(kept) == 1


# ---- failure modes ----

def test_a_missing_asset_disables_the_module_rather_than_raising(tmp_path, monkeypatch):
    monkeypatch.setattr(drug_reference, "_CACHE", None)
    monkeypatch.setattr(drug_reference, "_ASSET", str(tmp_path / "absent.json"))
    assert drug_reference.available() is False
    assert drug_reference.lookup("hydroxyzine") is None
    assert drug_reference.candidates("hydroxazine") == []
    monkeypatch.setattr(drug_reference, "_CACHE", None)


def test_a_corrupt_asset_disables_the_module_rather_than_raising(tmp_path, monkeypatch):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(drug_reference, "_CACHE", None)
    monkeypatch.setattr(drug_reference, "_ASSET", str(p))
    assert drug_reference.available() is False
    monkeypatch.setattr(drug_reference, "_CACHE", None)


# ---- the shipped asset, when it is present ----

def _shipped():
    return os.path.isfile(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "assets", "drug_reference.json"))


shipped_only = pytest.mark.skipif(not _shipped(), reason="asset not built")


@shipped_only
def test_the_shipped_asset_knows_the_pair_that_motivated_it():
    drug_reference._CACHE = None
    assert drug_reference.lookup("hydroxyzine")["units"] == ["mg", "ml"]
    assert drug_reference.lookup("levothyroxine")["units"] == ["mcg"]


@shipped_only
def test_a_real_drug_outside_the_mini_list_now_reaches_the_chart():
    from base_pipeline.stt_extract import extract_entities
    drug_reference._CACHE = None
    text = "Give telmisartan 40 mg once daily."
    rows = extract_entities(text, text, [])["drugs"]
    assert [r["name"] for r in rows] == ["telmisartan"]
    # YELLOW, not GREEN: the name reached the chart through a list this
    # pipeline has not curated, and nobody has reviewed the match.
    assert rows[0]["color"] == "YELLOW"
    assert "reference-only" in rows[0]["note"]


@shipped_only
def test_a_filler_word_before_a_dose_still_produces_no_drug():
    from base_pipeline.stt_extract import extract_entities
    drug_reference._CACHE = None
    text = "Patient was 200 mg of nothing and takes 500 mg of nothing."
    assert extract_entities(text, text, [])["drugs"] == []


# ---- guards found by stress-testing the real asset ----

@shipped_only
def test_lab_results_are_not_prescriptions():
    # `DRUG_CTX` matches `<word> <num> <unit>` and its `\b` after the unit sits
    # happily before `/dL`, so a line of bloodwork offers four analytes as drug
    # names. The mini list dropped them by not knowing the words; the national
    # reference knows all of them.
    from base_pipeline.stt_extract import extract_entities
    drug_reference._CACHE = None
    for text in ("Glucose 180 mg/dL, calcium 9 mg/dL, albumin 3.2 g/dL, urea 40 mg/dL.",
                 "Hemoglobin 11 g/dL, sodium 138 mmol/L."):
        assert extract_entities(text, text, [])["drugs"] == [], text


@shipped_only
def test_a_concentration_that_really_is_a_dose_still_counts():
    # `mg/dL` is a lab value; `mg/ml` is how a suspension is ordered. The guard
    # has to tell them apart, or it trades false drugs for missing ones.
    from base_pipeline.stt_extract import extract_entities
    drug_reference._CACHE = None
    text = "Give amoxicillin 250 mg/ml suspension."
    assert [d["name"] for d in extract_entities(text, text, [])["drugs"]] == ["amoxicillin"]


@shipped_only
def test_an_ordinary_word_that_is_also_a_brand_is_refused_an_exact_hit():
    # The pack registers `Level`, `Exam`, `Fine` and `Pause` as trade names,
    # and `din` - Hindi for "day", and in this project's own demo scripts - is
    # a brand of drotaverine. An exact hit is not a guess about spelling, but
    # it is still a guess about intent.
    from base_pipeline.stt_extract import extract_entities
    drug_reference._CACHE = None
    for text in ("Patient din 500 mg tak.", "Sugar level 180 mg today.",
                 "Exam 5 mg done."):
        assert extract_entities(text, text, [])["drugs"] == [], text


@shipped_only
def test_the_chart_keeps_the_inn_spelling_a_clinician_dictated():
    # The pack is USAN-named. A clinician who said `frusemide` must not read
    # `furosemide` back off the note, and `Crocin` charts as paracetamol.
    from base_pipeline.stt_extract import extract_entities
    drug_reference._CACHE = None
    for text, want in (("Give frusemide 40 mg IV.", "frusemide"),
                       ("Start rifampicin 450 mg OD.", "rifampicin"),
                       ("Give lignocaine 50 mg.", "lignocaine")):
        assert [d["name"] for d in extract_entities(text, text, [])["drugs"]] == [want]


@shipped_only
def test_the_mini_lists_own_spelling_wins_over_the_synonym_table():
    # `furosemide` is the mini list's canonical name and its gold data is
    # written in it, so the INN bridge must not rewrite it to `frusemide`.
    from base_pipeline.stt_extract import extract_entities
    drug_reference._CACHE = None
    text = "Give furosemide 40 mg IV."
    assert [d["name"] for d in extract_entities(text, text, [])["drugs"]] == ["furosemide"]


@shipped_only
def test_a_coined_brand_name_is_never_reached_by_fuzzing():
    # 72k coined words collide densely and meaninglessly: `zofran` scores 0.909
    # against `ofran`, an ofloxacin brand, and `thromice` scores 0.80 against
    # `throcet`, an ambroxol-cetirizine brand. Both would be the wrong drug.
    drug_reference._CACHE = None
    assert "ofran" not in [c["name"] for c in drug_reference.candidates("zofran")]
    assert "throcet" not in [c["name"] for c in drug_reference.candidates("thromice")]
    # ...while the substance names the stage exists to reach still resolve.
    assert drug_reference.candidates("hydroxazine")[0]["name"] == "hydroxyzine"
    assert drug_reference.candidates("rosuvastatn")[0]["name"] == "rosuvastatin"


@shipped_only
def test_a_curated_alias_is_not_outranked_by_a_merely_closer_string():
    # `moxacillin` scores 0.95 against `oxacillin` and 0.90 against the mini
    # list's `amoxicillin`. The higher number is the wrong antibiotic.
    from base_pipeline.stt_extract import extract_entities
    drug_reference._CACHE = None
    text = "Start moxacillin 875 mg BID."
    rows = extract_entities(text, text, [])["drugs"]
    assert rows[0]["name"] == "amoxicillin"
    # oxacillin stays on the shortlist: the validator may still weigh it.
    assert "oxacillin" in [c["name"] for c in rows[0]["candidates"]]


@shipped_only
def test_the_dose_check_runs_without_a_model():
    # `term_validate` needs Ollama. A dose that is impossible for the matched
    # drug must be caught whether or not a model is pulled.
    from base_pipeline.stt_extract import extract_entities
    drug_reference._CACHE = None
    text = "Start hydroxazine 25 mg twice daily."
    row = extract_entities(text, text, [])["drugs"][0]
    assert row["name"] == "hydroxyzine"
    assert "levothyroxine" in row["note"]


@shipped_only
def test_a_standard_dose_in_a_larger_unit_is_not_vetoed():
    # The regression this guards: comparing unit strings made `azithromycin
    # 1 g`, an ordinary single dose, look like a contradiction.
    from base_pipeline.stt_extract import extract_entities
    drug_reference._CACHE = None
    text = "Give azithromysin 1 g stat."
    row = extract_entities(text, text, [])["drugs"][0]
    assert row["name"] == "azithromycin"
    assert "implausible" not in row.get("note", "")


@shipped_only
def test_every_candidate_implausible_sends_the_row_red_without_a_model():
    # `term_validate` would reject this, but it needs Ollama. With no model
    # pulled the row used to ship YELLOW - a thyroid hormone at 100x its
    # maximum dose, looking like an ordinary fuzzy match.
    from base_pipeline.stt_extract import extract_entities
    drug_reference._CACHE = None
    text = "Start levothyroksin 25 mg daily."
    row = extract_entities(text, text, [])["drugs"][0]
    assert row["color"] == "RED"
    assert row["confidence"] <= 0.70
    assert "implausible for every candidate" in row["note"]


@shipped_only
def test_a_dose_driven_swap_goes_red_rather_than_quietly_changing_the_drug():
    # The pack overruling the string matcher is two signals disagreeing about
    # which drug was said, which is `term_validate`'s own rule for going RED.
    # It must not swap the drug and leave the row looking merely fuzzy.
    from base_pipeline.stt_extract import extract_entities
    drug_reference._CACHE = None
    text = "Start hydroxazine 25 mcg daily."
    row = extract_entities(text, text, [])["drugs"][0]
    assert row["color"] == "RED"
    assert "ruled out on dose" in row["note"]
    # The matcher's own answer is preserved, so the validator can still tell
    # that the drug on this row is not the one the matcher proposed.
    assert row["matcher_pick"] == "hydroxyzine"


@shipped_only
def test_an_english_word_that_is_a_registered_brand_never_becomes_a_medicine():
    # 161 of the 6,943 distinct English words in this project's real ASR
    # output are registered Indian trade names.
    from base_pipeline.stt_extract import extract_entities
    drug_reference._CACHE = None
    for word in ("space", "bus", "gear", "lot", "cope", "making", "site"):
        text = f"Give {word} 500 mg daily."
        assert extract_entities(text, text, [])["drugs"] == [], word


@shipped_only
def test_a_brand_still_resolves_a_name_matched_some_other_way():
    # Brands are refused the power to create a row, not removed. The mini
    # list's own brand aliases must keep working.
    from base_pipeline.stt_extract import extract_entities
    drug_reference._CACHE = None
    text = "Give dolo 650 mg TDS."
    assert [d["name"] for d in extract_entities(text, text, [])["drugs"]] == ["paracetamol"]


@shipped_only
def test_the_national_reference_resolves_the_sound_alike_the_mini_list_could_not():
    from base_pipeline.stt_extract import extract_entities, normalize_text
    drug_reference._CACHE = None

    def _ents(text):
        return extract_entities(text, normalize_text(text)["normalized_en"], [])

    HAZARD = "Start hydroxazine 25 mg twice daily."
    # This is the case the reference pack was added for. `hydroxyzine` is not
    # in the 33-entry mini list, so difflib used to return `levothyroxine` -
    # the nearest of 33 strings, and a thyroid hormone in place of an
    # antihistamine. With the national list behind the matcher the real drug is
    # reachable, so it wins on its own score rather than by elimination.
    row = _ents(HAZARD)["drugs"][0]
    assert row["name"] == "hydroxyzine"
    assert row["color"] == "YELLOW"      # still a fuzzy match; a human glances
    # The hazard is not merely out-ranked, it is ruled out: the national list
    # dispenses levothyroxine in micrograms only and the sentence says 25 mg.
    kept, dropped, survived = filter_by_unit(row)
    assert "levothyroxine" in dropped
    assert survived and "hydroxyzine" in [c["name"] for c in kept]


# --- builder: the pack can arrive as a zip, a folder, or loose masters ------

_PACK = {
    "SubstanceMaster.txt": "Identifier\tSubstance Name\n"
                           "1\tParacetamol\t\n2\tAzithromycin dihydrate\t\n",
    "GenericMaster.txt": "Identifier\tGeneric Name\tSubstance Identifier\n"
                         "10\tParacetamol 650 mg oral tablet\t1\t\n"
                         "11\tAzithromycin 500 mg oral tablet\t2\t\n",
    "ProductMaster.txt": "Identifier\tProduct Name\n100\tDolo\t\n101\tAzee\t\n",
    "BrandMaster.txt": "Identifier\tBrand Name\tProduct Identifier\tSupplier Identifier"
                       "\tGeneric Identifier\tlast_updated_on\n"
                       "1000\tDolo 650 mg oral tablet\t100\t\t10\t20240101\t\n"
                       "1001\tAzee 500 mg oral tablet\t101\t\t11\t20260814\t\n",
    "License.txt": "CC BY 4.0\n",
}


def _write(folder, names):
    os.makedirs(folder, exist_ok=True)
    for name in names:
        with open(os.path.join(folder, name), "w", encoding="utf-8") as fh:
            fh.write(_PACK[name])


def _tables(out_dir):
    with open(os.path.join(out_dir, "drug_reference.json"), encoding="utf-8") as fh:
        asset = json.load(fh)
    return {k: asset[k] for k in ("substances", "brands", "aliases", "inn")}, asset


def test_builder_reads_zip_folder_and_loose_masters_identically(tmp_path, capsys):
    import zipfile
    from base_pipeline.tools import build_drug_reference as bdr

    zpath = tmp_path / "pack.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for name, body in _PACK.items():
            zf.writestr(bdr._PKG + name, body)
    folder = tmp_path / "pack"
    _write(str(folder), ["SubstanceMaster.txt", "GenericMaster.txt", "License.txt"])
    loose = tmp_path / "loose"
    _write(str(loose), ["ProductMaster.txt", "BrandMaster.txt"])

    bdr.build(str(zpath), out_dir=str(tmp_path / "a"))
    bdr.build(str(folder), product_master=str(loose / "ProductMaster.txt"),
              brand_master=str(loose / "BrandMaster.txt"), out_dir=str(tmp_path / "b"))
    from_zip, _ = _tables(str(tmp_path / "a"))
    from_loose, asset = _tables(str(tmp_path / "b"))

    assert from_zip == from_loose
    assert from_loose["brands"]["dolo"]["substances"] == ["paracetamol"]
    assert from_loose["brands"]["azee"]["substances"] == ["azithromycin"]
    assert asset["built_from"]["BrandMaster.txt"] == "BrandMaster.txt"
    assert asset["brand_last_updated"] == "20260814"


def test_builder_refuses_a_missing_master_rather_than_emptying_a_join(tmp_path):
    from base_pipeline.tools import build_drug_reference as bdr

    folder = tmp_path / "pack"
    _write(str(folder), ["SubstanceMaster.txt", "GenericMaster.txt", "License.txt"])
    with pytest.raises(FileNotFoundError, match="ProductMaster.txt"):
        bdr.build(str(folder), out_dir=str(tmp_path / "out"))


# ---- multi-word fuzzing: same word count, same last word ----

needs_asset = pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(drug_reference.__file__),
                                    "assets", "drug_reference.json")),
    reason="built asset absent")


@needs_asset
def test_a_garbled_first_word_of_a_two_word_substance_resolves():
    drug_reference._CACHE = None
    names = [c["name"] for c in drug_reference.candidates("clavulic acid")]
    assert names[:1] == ["clavulanic acid"]


@needs_asset
def test_an_ordinary_two_word_phrase_finds_no_drug():
    drug_reference._CACHE = None
    for phrase in ("his blood", "blood glucose", "chest discomfort", "room air", "oxygen therapy"):
        assert drug_reference.candidates(phrase) == [], phrase


def test_shape_rule_pins_word_count_and_last_word_but_not_single_words():
    assert drug_reference._same_shape("clavulic acid", "clavulanic acid")
    assert not drug_reference._same_shape("clavulic acid", "clavulanic")
    assert not drug_reference._same_shape("folic base", "folic acid")
    assert drug_reference._same_shape("hydroxazine", "hydroxyzine")
    assert drug_reference._same_shape("valproate", "sodium valproate")  # unchanged behaviour


# ---- salt-first names: the substance's own word is a fuzz target ----

def _unit(n):
    return {"units": ["mg"], "forms": ["tablet"], "aliases": [],
            "strengths": [{"value": 10.0, "unit": "mg", "form": "tablet"}]}


COMPONENT_FIXTURE = {
    "substances": {
        # `disodium` is shared by two names, so it is a modifier; the word left
        # over is the substance. Same shape as the pack's `disodium pamidronate`.
        "disodium zorlexidate": _unit(0), "disodium kantrelmide": _unit(0),
        # `chloride` shared by two names; `vantrexil` is the substance.
        "vantrexil chloride": _unit(0), "calcium chloride": _unit(0),
        # `normal` is an everyday word, so it must not become a drug.
        "normal fortexin": _unit(0), "human fortexin": _unit(0),
        # a name whose words are all unique has no modifier to strip.
        "zymbarok plexo": _unit(0),
    },
    "brands": {}, "aliases": {},
}


@pytest.fixture
def comp(tmp_path, monkeypatch):
    p = tmp_path / "drug_reference.json"
    p.write_text(json.dumps(COMPONENT_FIXTURE), encoding="utf-8")
    monkeypatch.setattr(drug_reference, "_CACHE", None)
    monkeypatch.setattr(drug_reference, "_ASSET", str(p))
    yield drug_reference
    monkeypatch.setattr(drug_reference, "_CACHE", None)


def test_a_misheard_substance_word_reaches_its_salt_first_name(comp):
    top = comp.candidates("zorlexadate", floor=0.84)[0]
    assert top["name"] == "disodium zorlexidate" and top["score"] >= 0.84


def test_the_substance_word_spelled_exactly_is_not_a_candidate_or_a_lookup(comp):
    # Spelled right it is a chemistry word like any other, not evidence of an order.
    assert comp.candidates("zorlexidate") == []
    assert comp.lookup("zorlexidate") is None and not comp.is_substance_name("zorlexidate")


def test_a_modifier_shared_between_names_is_never_a_component(comp):
    comps = comp.load()["_components"]
    assert "zorlexidate" in comps and "vantrexil" in comps
    assert not {"disodium", "chloride", "calcium"} & set(comps)


def test_an_everyday_english_word_is_never_a_component(comp):
    assert "normal" not in comp.load()["_components"]
    assert comp.candidates("normally") == []


def test_a_name_with_no_shared_modifier_gets_no_component_key(comp):
    comps = comp.load()["_components"]
    assert "zymbarok" not in comps and "plexo" not in comps


def test_shipped_reference_components_are_clean():
    # Property check on the real asset: every component is a fuzz-only,
    # non-English, non-modifier word that belongs to exactly one multi-word name.
    from base_pipeline.context_repair import is_ordinary_word

    drug_reference._CACHE = None
    ref = drug_reference.load()
    if not ref:
        pytest.skip("drug reference asset absent")
    comps = ref["_components"]
    assert comps, "no component keys were derived from the pack"
    for word, (kind, base) in comps.items():
        # a brand may share the spelling (`fenofibric`); a substance may not
        assert ref["_index"].get(word, ("", ""))[0] != "substance", word
        assert not is_ordinary_word(word), word
        assert word in base.split() and " " in base, (word, base)
        owners = [n for n in ref["substances"] if word in n.split()]
        assert owners == [base], (word, owners)

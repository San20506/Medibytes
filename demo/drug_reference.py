"""Stage 4a-bis - the national drug list behind the fuzzy matcher.

`drug_list_mini.json` holds 33 medicines.  That is enough to demo with and far
too few to post-process against: when a misheard token's real referent is not
one of the 33, difflib still returns its nearest neighbour, so the matcher's
answer is "the closest of 33 strings" dressed up as a drug name.  The injury
is specific - `hydroxazine` lands on `levothyroxine` because nothing closer
exists in the list.

This module puts the Common Drug Codes for India package (NRCeS / C-DAC Pune,
CC BY 4.0) behind that decision: ~2.5k substances and ~72k Indian brand names,
with the strengths and units the pack lists for each.  Two things follow.

**A real name stops being a near-miss.**  A token that matches the reference
exactly is a medicine, full stop, even when the mini list has never heard of
it.  No fuzzing is attempted and no sound-alike is proposed.

**Units become evidence.**  The pack knows `hydroxyzine` is dispensed in
milligrams and `levothyroxine` in micrograms.  A sentence saying `25 mg` is
therefore positive evidence for one and against the other - a signal about
medicine rather than about spelling, which is exactly what the string matcher
cannot supply and what `term_validate` is asked to weigh.

Three deliberate limits, because 72k names is as much a hazard as a resource:

1. **Augment, never replace.**  The mini list stays canonical: it is the
   vocabulary this project's gold data is written in, and its aliases are
   mishearings someone observed rather than strings that merely score well.
   The reference only adds candidates the mini list could not supply and
   attaches evidence to them.
2. **Brand names cannot put a drug on a chart.**  Of the 6,943 distinct
   English words in this project's real ASR output, 161 are registered Indian
   trade names - `Space` is a paracetamol, `Bus` a pantoprazole, `Lot` a
   lisinopril - while exactly two collide with a marketed substance.  Brands
   are coined to be short and memorable, which is the same thing as colliding
   with English, so no stopword list can cover them.  Neither `lookup` (when
   the caller passes `brands=False`) nor `candidates` will return one, and
   `DRUG_CTX`'s name group is `[A-Za-z]+`, so "was 200 mg" would otherwise
   offer `was` as a drug name.  The brand table is kept because it is what
   maps a trade name to its generic, which is the join any future handling of
   doseless brand mentions needs; today nothing in the extractor reads it.
3. **Candidates are deduped by substance.**  Fifty brands of one generic are
   one answer, not a fifty-way tie; left alone they would trip
   `term_validate`'s tie rule and send every row to a human.

A missing asset disables the module and leaves the matcher exactly as it was,
in the same spirit as the pipeline's other optional stages.
"""
import difflib
import json
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_ASSET = os.path.join(_HERE, "assets", "drug_reference.json")

# Below this length a token is fuzzed against nothing: at 4 characters the
# brand table has a plausible neighbour for almost any string.
MIN_FUZZY_LEN = 6
# Fuzzy candidates must beat this to be offered at all.
FUZZY_FLOOR = 0.78
# ...and stay within this of the best, so a long tail of weak names is dropped.
FUZZY_BAND = 0.06
MAX_CANDIDATES = 5

# `DRUG_CTX`'s name group is `[A-Za-z]+`, so the word before any dose lands in
# the drug slot - "was 200 mg", "takes 500 mg", "started 5 mg".  These are the
# words that reach it, refused both the fuzzy path and an exact substance hit.
# They are a backstop rather than the main defence: the list is hand-written
# and English is not, which is why brand names are excluded wholesale instead
# of being enumerated here.
STOPWORDS = frozenset("""
was were is are be been being am has have had having will would shall should
can could may might must do does did done take takes taken taking give gives
given giving start starts started starting stop stops stopped stopping
continue continues continued add adds added increase increased decrease
decreased reduce reduced change changed switch switched prescribe prescribed
advise advised advice receive received receiving order ordered need needs
needed want wants get gets got put puts send sent keep kept make makes made
use uses used using show shows showed seen see saw find found feel feels felt
about after before during since until while when where which what that this
these those there their then than with without within from into onto over
under above below between among through during again also only just even
still more most less least much many few more another other others same
daily twice thrice once night morning evening afternoon today tomorrow
yesterday week weeks month months year years day days hour hours time times
dose doses dosage tablet tablets capsule capsules syrup drops injection
patient patients doctor nurse ward bed history complaint complaints
blood sugar pressure pulse temperature weight height level levels count
pain fever cough cold rash nausea vomiting headache
back bid cin demo din don exam fine flow level mic path pause quiet
air blood glucose protein oxygen water saline urine stool urea sodium
albumin creatinine bilirubin
total normal stable better worse good bad high low mild severe acute chronic
please thanks note noted record recorded chart charted plan planned
""".split())

_CACHE = None


def load(path=None):
    """The reference, parsed once and cached. `None` when the asset is absent."""
    global _CACHE
    if _CACHE is None:
        p = path or _ASSET
        if not os.path.isfile(p):
            _CACHE = False
        else:
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                data.setdefault("substances", {})
                data.setdefault("brands", {})
                data.setdefault("aliases", {})
                data.setdefault("inn", {})
                # One flat name->kind index so lookup is a single dict hit, and
                # a length-bucketed list so fuzzing never scans all 74k names.
                # Order is precedence. Substances first, then the curated
                # INN/BAN aliases, and only then the 72k coined brand names:
                # `paracetamol`, `lignocaine`, `amoxycillin` and
                # `phenobarbitone` are all registered trade names as well as
                # the INN spelling a clinician says, and indexing them as
                # brands hid them from the substance table - a misheard
                # `paracetmol` found nothing at all.
                index = {}
                for nm in data["substances"]:
                    index[nm] = ("substance", nm)
                for alias, base in data["aliases"].items():
                    index.setdefault(alias, ("substance", base))
                for nm, rec in data["brands"].items():
                    index.setdefault(nm, ("brand", nm))
                # Only substances with at least one marketed product may be
                # fuzzed. The pack is SNOMED, so it also carries drug classes
                # ("analgesic", "antiviral"), excipients and bare chemicals -
                # 287 rows with no strength anywhere. `hydrazine` is one, and
                # at 0.900 against `hydroxazine` it tied with hydroxyzine's
                # 0.909 and sent the stage's own headline case to a human.
                # Nothing India dispenses is what "no product" means, so it is
                # not a thing a clinician can have said.
                subs = data["substances"]
                buckets, sub_buckets = {}, {}
                for nm, (kind, base) in index.items():
                    buckets.setdefault(len(nm), []).append(nm)
                    rec = subs.get(base) or {}
                    if kind == "substance" and (rec.get("units") or rec.get("mass_mg")):
                        sub_buckets.setdefault(len(nm), []).append(nm)
                data["_index"] = index
                data["_buckets"] = buckets
                data["_sub_buckets"] = sub_buckets
                _CACHE = data
            except (OSError, ValueError):
                _CACHE = False
    return _CACHE or None


def inn_name(name):
    """The INN spelling for a pack name, or the name unchanged.

    The pack is USAN-named; Indian charts, and this project's gold data, are
    written in INN.  A clinician who dictated `frusemide` must not read
    `furosemide` back off the discharge note, and a `Crocin` tablet charts as
    paracetamol rather than acetaminophen.
    """
    ref = load()
    if not ref or not name:
        return name
    return ref["inn"].get(str(name).strip().lower(), name)


def available():
    return load() is not None


def _resolve(ref, kind, name):
    """A reference hit as a record: canonical substance(s), units, strengths."""
    if kind == "substance":
        rec = ref["substances"].get(name, {})
        return {"name": name, "kind": "substance", "substances": [name],
                "units": rec.get("units", []), "strengths": rec.get("strengths", []),
                "forms": rec.get("forms", []), "mass_mg": rec.get("mass_mg")}
    rec = ref["brands"].get(name, {})
    subs = rec.get("substances", [])
    strengths = []
    for s in subs:
        strengths.extend(ref["substances"].get(s, {}).get("strengths", []))
    return {"name": name, "kind": "brand", "substances": subs,
            "units": rec.get("units", []), "strengths": strengths[:12],
            "forms": sorted({s.get("form") for s in strengths if s.get("form")})[:6],
            "mass_mg": rec.get("mass_mg")}


def lookup(token, block=frozenset(), brands=True):
    """An exact reference hit for `token`, or None. Case-insensitive.

    `brands=False` restricts the answer to substances.  Callers that turn an
    exact hit into a row on a chart must pass it.  Of the 6,943 distinct
    English words in this project's real ASR output, 161 are registered Indian
    trade names - `Space` is a paracetamol, `Bus` a pantoprazole, `Gear` an
    omeprazole, `Lot` a lisinopril - while exactly two collide with a marketed
    substance.  Brand names are coined to be short and memorable, which is the
    same thing as colliding with English, so no stopword list can be made to
    cover them.  INNs are not coined that way, which is why the substance
    table can be trusted where the brand table cannot.  Brands stay useful for
    resolving a name that was matched some other way to its generic; what they
    may not do is bring a row into existence.

    `block` refuses names that are also ordinary words.  The pack registers
    `Back`, `Level`, `Exam`, `Fine`, `Pause` and `Quiet` as Indian trade
    names, and `din` - which is Hindi for "day" and appears in this project's
    own demo scripts - is a brand of drotaverine.  An exact hit is not a guess
    about spelling, but it is still a guess about intent, and these words
    reach the drug slot constantly.  Analyte names go in the same bucket:
    `glucose` and `urea` are real substances and never prescriptions here.
    """
    ref = load()
    if not ref or not token:
        return None
    t = str(token).strip().lower()
    if t in block:
        return None
    hit = ref["_index"].get(t)
    if not hit:
        return None
    kind, base = hit
    if kind == "brand" and not brands:
        return None
    if kind == "substance":
        rec = ref["substances"].get(base) or {}
        if not (rec.get("units") or rec.get("mass_mg")):
            return None  # a class or a chemical, not something dispensed
    return _resolve(ref, kind, base)


def _is_fuzzable(token, stopwords):
    t = str(token or "").strip().lower()
    return (len(t) >= MIN_FUZZY_LEN and t.isalpha()
            and t not in (stopwords or frozenset()))


def candidates(token, stopwords=STOPWORDS, floor=FUZZY_FLOOR,
               band=FUZZY_BAND, limit=MAX_CANDIDATES, substances_only=True):
    """Reference names `token` could be, best first, deduped by substance.

    Only substance names are fuzzed, which is the whole of the third limit in
    this module's docstring.  Brand names are coined words, 72k of them, and
    near-collisions between them are dense and meaningless: `zofran` scores
    0.909 against `ofran`, an ofloxacin brand, where the ondansetron it is
    actually a brand of scores 0.353; `thromice` scores 0.80 against
    `throcet`, an ambroxol-cetirizine brand, against 0.60 for azithromycin.
    A near-match to a coined word is not evidence.  A near-match to an INN is,
    and every sound-alike this stage was built to fix - `hydroxazine`,
    `rosuvastatn`, `pantaprazol`, `metphormin` - resolves to a substance.
    Brands are still matched exactly by `lookup`, which is where `Dolo`,
    `Azee` and `Pan` are caught, and an exact hit is not a guess.

    The scan is bucketed by length before scoring: an edit-distance ratio of
    `floor` is unreachable once two strings differ in length by more than
    `(1 - floor)` of their total, so those buckets cannot contain a hit and are
    skipped outright.  `real_quick_ratio` then screens what is left, and the
    full ratio runs only on survivors - without the ladder this is a
    seconds-per-token scan of 74k names.
    """
    ref = load()
    t = str(token or "").strip().lower()
    if not ref or not _is_fuzzable(t, stopwords):
        return []
    # |len(a) - len(b)| <= (1 - floor) * (len(a) + len(b)) is the necessary
    # condition for SequenceMatcher.ratio() >= floor.
    span = int((1 - floor) / (1 + floor) * 2 * len(t)) + 1
    sm = difflib.SequenceMatcher()
    sm.set_seq2(t)
    scored = []
    buckets = ref["_sub_buckets"] if substances_only else ref["_buckets"]
    for length in range(max(1, len(t) - span), len(t) + span + 1):
        for name in buckets.get(length, ()):
            sm.set_seq1(name)
            if sm.real_quick_ratio() < floor or sm.quick_ratio() < floor:
                continue
            r = sm.ratio()
            if r >= floor:
                scored.append((r, name))
    if not scored:
        return []
    scored.sort(key=lambda p: (-p[0], p[1]))
    cutoff = scored[0][0] - band
    out, seen = [], set()
    for score, name in scored:
        if score < cutoff:
            break
        rec = _resolve(ref, *ref["_index"][name])
        # One generic with many brands is one candidate. Keying on the
        # substance set collapses them; an unjoined brand keys on itself.
        key = tuple(rec["substances"]) or (name,)
        if key in seen:
            continue
        seen.add(key)
        rec["score"] = round(score, 3)
        rec["matched_name"] = name
        out.append(rec)
        if len(out) == limit:
            break
    return out


# Mass units, in milligrams. Everything else is a different dimension.
MASS_MG = {"mcg": 0.001, "mg": 1.0, "g": 1000.0, "gram": 1000.0,
           "milligram": 1.0, "microgram": 0.001}
# How far outside the pack's listed strength range a dose may fall before it
# counts as evidence against the drug. The pack lists levothyroxine between
# 0.005 mg and 0.3 mg, so a sentence saying 25 mg is 83x its maximum; it lists
# azithromycin up to 600 mg, so a standard 1 g single dose is 1.7x. The factor
# has to sit between those, and 20x leaves room for the strengths the pack
# simply does not carry while still catching an order-of-magnitude error.
# It is deliberately generous: a missed veto costs a glance at a YELLOW row,
# and a false veto marks a correct drug as rejected.
DOSE_TOLERANCE = 20.0


def unit_fits(rec, unit):
    """Is `unit` the same *dimension* as the ones the pack lists? Tri-state.

    Dimension, not spelling: `mg` and `g` are the same kind of quantity, so a
    drug the pack lists in milligrams is not contradicted by a dose in grams -
    that was the bug this replaced, and it rejected `azithromycin 1 g`, an
    ordinary single dose. Magnitude is `dose_conflicts`' job.

    `None` means the pack cannot say - it lists no units, or the sentence's
    unit is one the pack does not quantify (`ml`, `%`). Silence is not
    disagreement.
    """
    if not rec or not unit:
        return None
    units = [u.lower() for u in (rec.get("units") or [])]
    if not units:
        return None
    u = _canon_unit(unit)
    if u is None:
        return None
    listed_mass = {x for x in units if x in MASS_MG}
    if u in MASS_MG:
        # Mass against a pack that lists only non-mass units (e.g. `U`) is a
        # real dimension clash; against any listed mass unit it fits.
        if listed_mass:
            return True
        return False if units else None
    if u == "u":
        return "u" in units if units else None
    return None  # ml, %, anything else: no opinion


def dose_conflicts(rec, dose, unit, tolerance=DOSE_TOLERANCE):
    """Is this dose implausible for this drug by an order of magnitude?

    This is the signal that separates a sound-alike pair when spelling cannot:
    `hydroxazine 25 mg` fits hydroxyzine (2-100 mg) and is 83x the top of
    levothyroxine's range, so the pack rules the thyroid hormone out on
    medicine rather than on string distance.

    False whenever the pack has no range for the drug, or the unit is not a
    mass - the check must never fire on an absence of data.
    """
    if not rec or dose is None or not unit:
        return False
    rng = rec.get("mass_mg")
    if not rng:
        return False
    u = _canon_unit(unit)
    if u not in MASS_MG:
        return False
    try:
        mg = float(dose) * MASS_MG[u]
    except (TypeError, ValueError):
        return False
    if mg <= 0:
        return False
    lo, hi = float(rng[0]), float(rng[1])
    return mg > hi * tolerance or mg < lo / tolerance


def _canon_unit(unit):
    u = str(unit).strip().lower()
    return {"milligram": "mg", "milligrams": "mg", "microgram": "mcg",
            "micrograms": "mcg", "ug": "mcg", "\u00b5g": "mcg",
            "gram": "g", "grams": "g", "milliliter": "ml", "millilitre": "ml",
            "milliliters": "ml", "unit": "u", "units": "u"}.get(u, u)


def evidence(rec, dose=None, unit=None):
    """A one-line, promptable summary of what the pack knows about `rec`."""
    if not rec:
        return ""
    bits = []
    if rec["kind"] == "brand" and rec["substances"]:
        bits.append("brand of " + ", ".join(rec["substances"]))
    if rec.get("units"):
        bits.append("dispensed in " + "/".join(rec["units"]))
    if rec.get("strengths"):
        shown = ", ".join(f"{_num(s['value'])} {s['unit']}"
                          + (f" {s['form']}" if s.get("form") else "")
                          for s in rec["strengths"][:4])
        bits.append("listed strengths: " + shown)
    if unit_fits(rec, unit) is False:
        bits.append(f"not dispensed in {unit} at all - a different kind of quantity")
    elif dose_conflicts(rec, dose, unit):
        rng = rec.get("mass_mg")
        bits.append(f"a dose of {_num(dose)} {unit} is far outside the listed "
                    f"range ({_num(rng[0])}-{_num(rng[1])} mg) - strong evidence "
                    f"against this drug")
    elif dose is not None and rec.get("mass_mg"):
        bits.append(f"{_num(dose)} {unit} is plausible for this drug")
    return "; ".join(bits)


def _num(v):
    return str(int(v)) if float(v).is_integer() else str(v)

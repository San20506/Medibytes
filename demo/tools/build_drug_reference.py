"""Build the compact drug-reference asset from the C-DAC flat-file package.

Input is `CommonDrugCodesForIndia_FlatFilePackage.zip` - the Common Drug Codes
for India package produced by NRCeS at C-DAC Pune, CC BY 4.0.  It is 11 MB of
SNOMED CT national-extension rows, which is the wrong shape for a post-
processing hot path: the pipeline needs to ask "is this misheard token a real
Indian medicine name, and does the dose in the sentence fit it?" and nothing
else.  So the parsing happens once, here, and the pipeline ships the derived
answer.

What the asset carries, and why each piece earns its place:

* **substances** - the spoken base name, salt stripped (`metformin
  hydrochloride` -> `metformin`), because a clinician dictates the base.  The
  salt form is kept as an alias so the written form still matches.
* **strengths** - every value/unit/form the pack lists for that substance,
  parsed out of `GenericMaster`'s fully-specified names.  This is the evidence
  that separates a sound-alike pair: `hydroxyzine` is dispensed in milligrams
  and `levothyroxine` in micrograms, so the unit in the dictated sentence
  decides between them where string distance cannot.
* **brands** - the Indian trade names, which is most of what gets said out
  loud on a ward, mapped to the substances they contain.

The brand->substance join runs ProductMaster -> BrandMaster.`Product
Identifier` -> `Generic Identifier` -> GenericMaster -> `Substance Identifier`
-> SubstanceMaster.  National-extension identifiers do not all resolve, so the
coverage of every hop is measured and printed rather than assumed; a silent
50%-join would quietly halve the reference's value.

Run:  python3 tools/build_drug_reference.py [path-to-zip]
Out:  assets/drug_reference.json  (+ assets/drug_reference.LICENSE.txt)
"""
import csv
import io
import json
import os
import re
import sys
import zipfile
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ASSETS = os.path.join(_HERE, "..", "assets")
_PKG = "CommonDrugCodesForIndia_FlatFilePackage/"
DEFAULT_ZIP = os.path.expanduser(
    "~/Downloads/CommonDrugCodesForIndia_FlatFilePackage.zip")

# Salt and ester suffixes stripped to reach the base name a clinician says.
# Order matters: the longest form has to win, so `sodium phosphate` is tried
# before `sodium`.
SALTS = [
    "hydrochloride", "hydrobromide", "hydroiodide", "dihydrochloride",
    "sodium phosphate", "sodium succinate", "sodium sulfate", "sodium sulphate",
    "calcium phosphate", "potassium clavulanate",
    "mesylate", "mesilate", "besylate", "besilate", "tosylate", "maleate",
    "fumarate", "tartrate", "bitartrate", "citrate", "succinate", "acetate",
    "phosphate", "sulfate", "sulphate", "nitrate", "gluconate", "lactate",
    "stearate", "palmitate", "valerate", "propionate", "butyrate", "benzoate",
    "salicylate", "oxalate", "pamoate", "embonate", "carbonate", "bicarbonate",
    "trihydrate", "dihydrate", "monohydrate", "anhydrous", "hemihydrate",
    "sodium", "potassium", "calcium", "magnesium", "zinc", "aluminium",
    "disodium", "dipotassium", "hemifumarate", "xinafoate", "furoate",
    "dipropionate", "propionate", "axetil", "proxetil", "trometamol",
    "arginine", "lysine", "meglumine", "olamine", "diolamine",
]
_SALT_PAT = re.compile(
    r"\s+(?:" + "|".join(sorted(map(re.escape, SALTS), key=len, reverse=True)) + r")$",
    re.I)

# "Amoxicillin 500 mg oral capsule" -> the strength and the form.
_STRENGTH = re.compile(
    r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>microgram|micrograms|mcg|milligram|milligrams"
    r"|mg|gram|grams|g|millilitre|milliliter|ml|unit|units|iu|%)\b", re.I)
_UNIT_CANON = {"microgram": "mcg", "micrograms": "mcg", "mcg": "mcg",
               "milligram": "mg", "milligrams": "mg", "mg": "mg",
               "gram": "g", "grams": "g", "g": "g",
               "millilitre": "ml", "milliliter": "ml", "ml": "ml",
               "unit": "U", "units": "U", "iu": "U", "%": "%"}
_FORMS = ("tablet", "capsule", "injection", "solution", "suspension", "syrup",
          "cream", "ointment", "gel", "drops", "inhaler", "powder", "lotion",
          "patch", "suppository", "spray", "granules", "sachet", "infusion")

# A brand name that is also an ordinary English word cannot be fuzzed against;
# `Was`, `For` and friends really are registered Indian trade names.
STOP_BRANDS = {
    "a", "an", "the", "was", "were", "is", "are", "be", "been", "for", "and",
    "or", "but", "not", "no", "yes", "on", "in", "at", "to", "of", "by", "with",
    "from", "as", "it", "he", "she", "they", "we", "you", "i", "his", "her",
    "him", "them", "us", "me", "my", "our", "your", "their", "this", "that",
    "these", "those", "all", "any", "some", "one", "two", "three", "new", "old",
    "day", "days", "night", "time", "dose", "take", "takes", "taken", "give",
    "given", "start", "started", "stop", "stopped", "add", "added", "also",
    "then", "now", "after", "before", "next", "last", "first", "well", "good",
    "best", "more", "most", "less", "same", "other", "if", "so", "up", "down",
    "out", "off", "over", "under", "per", "has", "have", "had", "will", "can",
    "may", "must", "should", "would", "did", "does", "do", "get", "got", "put",
    "see", "seen", "use", "used", "need", "needs", "want", "like", "plus",
    "pain", "fever", "cough", "cold", "rash", "sugar", "blood", "heart",
    "today", "daily", "once", "twice", "mg", "ml", "unit", "units",
}


# The pack is named in USAN; Indian clinicians dictate the INN/BAN name.  Every
# pair below is the same molecule under two spellings, and without the bridge a
# perfectly-said `paracetamol` or `salbutamol` simply misses the reference.
# Each entry is {spoken_name: pack_name}; the spoken name becomes an alias of
# the pack's substance record.
INN_TO_USAN = {
    "paracetamol": "acetaminophen",
    "salbutamol": "albuterol",
    "adrenaline": "epinephrine",
    "noradrenaline": "norepinephrine",
    "rifampicin": "rifampin",
    "frusemide": "furosemide",
    "lignocaine": "lidocaine",
    "cefalexin": "cephalexin",
    "amoxycillin": "amoxicillin",
    "glibenclamide": "glyburide",
    "pethidine": "meperidine",
    "thyroxine": "levothyroxine",
    "dothiepin": "dosulepin",
    "beclomethasone": "beclometasone",
    "hydroxyprogesterone": "hydroxyprogesterone caproate",
    "oestradiol": "estradiol",
    "oestrogen": "estrogen",
    "ciclosporin": "cyclosporine",
    "colistimethate": "colistin",
    "indometacin": "indomethacin",
    "chlorpheniramine": "chlorphenamine",
    "phenobarbitone": "phenobarbital",
    "sodium valproate": "valproate",
    "trimethoprim sulfamethoxazole": "sulfamethoxazole",
    "nifedipine retard": "nifedipine",
    "methylergometrine": "methylergonovine",
    "ergometrine": "ergonovine",
    "oxytetracycline": "oxytetracycline",
    "cotrimoxazole": "sulfamethoxazole",
    "metamizole": "dipyrone",
    "dicyclomine": "dicycloverine",
    "hyoscine": "scopolamine",
    "pentazocine": "pentazocine",
    "povidone iodine": "povidone-iodine",
}


def _rows(zf, name):
    """Tab-delimited rows as dicts, tolerating the pack's trailing tab and BOM.

    Every data line in the pack ends with a tab, so `csv` yields one extra
    empty field per row; the header has no such tab.  Zipping against the
    header and ignoring the surplus is what keeps the columns aligned.
    """
    raw = zf.read(_PKG + name).decode("utf-8-sig", "ignore")
    reader = csv.reader(io.StringIO(raw), delimiter="\t", quoting=csv.QUOTE_NONE)
    header = [h.strip() for h in next(reader)]
    for row in reader:
        if not any(f.strip() for f in row):
            continue
        yield {h: (row[i].strip() if i < len(row) else "")
               for i, h in enumerate(header)}


def base_name(name):
    """The spoken base of a substance name: `Metformin hydrochloride` -> `metformin`.

    Stripping repeats because the pack stacks modifiers (`...sodium anhydrous`),
    and stops before emptying the string - `Sodium chloride` is itself a drug,
    not a salt suffix waiting to be removed.
    """
    out = re.sub(r"\s*\(.*?\)\s*", " ", name).strip().lower()
    for _ in range(3):
        stripped = _SALT_PAT.sub("", out).strip()
        if not stripped or stripped == out:
            break
        out = stripped
    return out


def parse_strengths(generic_name):
    """Every strength in a fully-specified name, plus the dose form.

    A combination product names each component's strength in order, so the list
    is returned whole; the caller pairs it against that product's substance
    list positionally.
    """
    form = next((f for f in _FORMS if f in generic_name.lower()), "")
    out = []
    for m in _STRENGTH.finditer(generic_name):
        unit = _UNIT_CANON.get(m.group("unit").lower())
        if unit:
            out.append({"value": float(m.group("val")), "unit": unit, "form": form})
    return out


def build(zip_path):
    zf = zipfile.ZipFile(zip_path)
    stats = defaultdict(int)

    # --- substances -------------------------------------------------------
    sub_by_id, substances = {}, {}
    for r in _rows(zf, "SubstanceMaster.txt"):
        sid, nm = r.get("Identifier", ""), r.get("Substance Name", "")
        if not sid or not nm:
            continue
        base = base_name(nm)
        if not base or len(base) < 3 or not re.match(r"^[a-z]", base):
            continue
        sub_by_id[sid] = base
        rec = substances.setdefault(base, {"name": base, "aliases": set(),
                                           "strengths": [], "forms": set()})
        if nm.lower() != base:
            rec["aliases"].add(nm.lower())
        stats["substance_rows"] += 1

    # --- generics: strengths per substance, and the generic->substance map --
    gen_subs, gen_strengths = {}, {}
    for r in _rows(zf, "GenericMaster.txt"):
        gid, nm = r.get("Identifier", ""), r.get("Generic Name", "")
        if not gid:
            continue
        stats["generic_rows"] += 1
        ids = [s for s in r.get("Substance Identifier", "").split("+") if s]
        names = [sub_by_id[s] for s in ids if s in sub_by_id]
        if ids:
            stats["generic_with_subid"] += 1
            if names:
                stats["generic_subid_resolved"] += 1
        gen_subs[gid] = names
        strengths = parse_strengths(nm)
        gen_strengths[gid] = strengths
        # Pair component to strength only when the counts line up; a mismatched
        # combination row would otherwise attach one drug's dose to another.
        if names and strengths and len(names) == len(strengths):
            for base, st in zip(names, strengths):
                substances[base]["strengths"].append(st)
                if st["form"]:
                    substances[base]["forms"].add(st["form"])
        elif len(names) == 1 and strengths:
            for st in strengths:
                substances[names[0]]["strengths"].append(st)
                if st["form"]:
                    substances[names[0]]["forms"].add(st["form"])

    # --- brands: ProductMaster name, joined out to substances --------------
    product_name = {r["Identifier"]: r.get("Product Name", "")
                    for r in _rows(zf, "ProductMaster.txt") if r.get("Identifier")}
    brands = {}
    for r in _rows(zf, "BrandMaster.txt"):
        stats["brand_rows"] += 1
        pid, gid = r.get("Product Identifier", ""), r.get("Generic Identifier", "")
        nm = product_name.get(pid, "")
        if nm:
            stats["brand_product_resolved"] += 1
        else:
            # No clean trade name: fall back to the leading words of the long
            # BrandMaster string, cut at the first strength ("Safedox 200 mg
            # film-coated oral tablet" -> "safedox").
            long = r.get("Brand Name", "")
            m = _STRENGTH.search(long)
            nm = (long[:m.start()] if m else long).strip(" -+,")
        nm = nm.strip().lower()
        if not nm or len(nm) < 3 or not re.match(r"^[a-z][a-z0-9 .'-]*$", nm):
            continue
        if nm in STOP_BRANDS:
            stats["brand_stopword_dropped"] += 1
            continue
        subs = gen_subs.get(gid) or []
        if gid in gen_subs:
            stats["brand_generic_resolved"] += 1
        if subs:
            stats["brand_substance_resolved"] += 1
        rec = brands.setdefault(nm, {"name": nm, "substances": set(), "strengths": []})
        rec["substances"].update(subs)
        for st in gen_strengths.get(gid) or []:
            if st not in rec["strengths"]:
                rec["strengths"].append(st)

    # --- INN/BAN bridge ---------------------------------------------------
    # Applied as aliases rather than new records: the pack's row stays the one
    # source of strengths, and the spoken spelling becomes another way in.
    for spoken, pack in INN_TO_USAN.items():
        rec = substances.get(pack)
        if rec is None:
            stats["inn_unmatched"] += 1
            continue
        rec["aliases"].add(spoken)
        stats["inn_bridged"] += 1

    # --- compact ----------------------------------------------------------
    def dedupe(strengths, cap=12):
        seen, out = set(), []
        for st in strengths:
            key = (st["value"], st["unit"], st["form"])
            if key in seen:
                continue
            seen.add(key)
            out.append(st)
            if len(out) == cap:
                break
        return out

    # The strength list is capped for size, but the plausible-dose check must
    # not be: a truncated maximum would veto a legitimate large dose.  So the
    # mass range is computed over every strength the pack lists, before the
    # cap, and shipped as two numbers.
    MASS_MG = {"mcg": 0.001, "mg": 1.0, "g": 1000.0}

    def mass_range(strengths):
        vals = [s["value"] * MASS_MG[s["unit"]] for s in strengths
                if s["unit"] in MASS_MG and s["value"] > 0]
        return [min(vals), max(vals)] if vals else None

    subs_out = {}
    for base, rec in substances.items():
        subs_out[base] = {
            "units": sorted({s["unit"] for s in rec["strengths"]}),
            "strengths": dedupe(sorted(rec["strengths"], key=lambda s: (s["unit"], s["value"]))),
            "forms": sorted(rec["forms"])[:6],
            "aliases": sorted(rec["aliases"])[:4],
        }
        mr = mass_range(rec["strengths"])
        if mr:
            subs_out[base]["mass_mg"] = [round(mr[0], 6), round(mr[1], 6)]
    brands_out = {}
    for nm, rec in brands.items():
        subs = sorted(s for s in rec["substances"] if s)
        out = {"substances": subs,
               "units": sorted({s["unit"] for s in rec["strengths"]})}
        # A brand's plausible range is the union of its components' ranges.
        ranges = [subs_out[x]["mass_mg"] for x in subs
                  if x in subs_out and "mass_mg" in subs_out[x]]
        if ranges:
            out["mass_mg"] = [min(r[0] for r in ranges), max(r[1] for r in ranges)]
        brands_out[nm] = out

    # The alias index is what the runtime looks a spoken word up in, so the
    # bridge must survive the alias cap above; it is written from the full set.
    alias_index = {}
    for base, rec in substances.items():
        for a in rec["aliases"]:
            alias_index.setdefault(a, base)
    for spoken, pack in INN_TO_USAN.items():
        if pack in substances:
            alias_index[spoken] = pack

    asset = {
        "source": "Common Drug Codes for India (Flat File Package), NRCeS / "
                  "C-DAC Pune, CC BY 4.0",
        "license": "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)",
        "built_from": os.path.basename(zip_path),
        "substances": subs_out,
        "aliases": alias_index,
        # The bridge read backwards: the pack's USAN name -> the INN spelling
        # an Indian chart is written in. Charts say `paracetamol`, not
        # `acetaminophen`, whatever the pack calls the row.
        "inn": {pack: spoken for spoken, pack in INN_TO_USAN.items()
                if pack in subs_out and pack != spoken},
        "brands": brands_out,
    }
    os.makedirs(_ASSETS, exist_ok=True)
    out_path = os.path.normpath(os.path.join(_ASSETS, "drug_reference.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asset, f, separators=(",", ":"), sort_keys=True)
    with open(os.path.join(_ASSETS, "drug_reference.LICENSE.txt"), "w",
              encoding="utf-8") as f:
        f.write(zf.read(_PKG + "License.txt").decode("utf-8", "ignore"))

    def pct(a, b):
        return f"{100.0 * a / b:.1f}%" if b else "n/a"
    print(f"substances      : {len(subs_out)} from {stats['substance_rows']} rows")
    print(f"  with units    : {sum(1 for v in subs_out.values() if v['units'])}")
    print(f"generic rows    : {stats['generic_rows']}")
    print(f"  substance id  : {stats['generic_with_subid']} present, "
          f"{stats['generic_subid_resolved']} resolved "
          f"({pct(stats['generic_subid_resolved'], stats['generic_with_subid'])})")
    print(f"INN bridge      : {stats['inn_bridged']} bridged, "
          f"{stats['inn_unmatched']} unmatched")
    print(f"aliases         : {len(alias_index)}")
    print(f"brand rows      : {stats['brand_rows']} -> {len(brands_out)} names")
    print(f"  product name  : {pct(stats['brand_product_resolved'], stats['brand_rows'])}")
    print(f"  generic join  : {pct(stats['brand_generic_resolved'], stats['brand_rows'])}")
    print(f"  substance join: {pct(stats['brand_substance_resolved'], stats['brand_rows'])}")
    print(f"  stopword drop : {stats['brand_stopword_dropped']}")
    print(f"written         : {out_path} "
          f"({os.path.getsize(out_path) / 1e6:.2f} MB)")
    return asset


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ZIP)

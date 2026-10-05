"""Stage 4b - semantic validation of fuzzy-matched drug terms (new.md Sec 7).

The regex extractor's sound-alike fallback picks difflib's top scorer and ships
it at YELLOW.  That is the right call for `asithromysin -> azithromycin` and the
wrong one whenever the token's real referent is missing from the list: difflib
scores string shape, not medicine, so it still lands on whatever is nearest.
`hydroxazine -> levothyroxine` was the worked example until the national drug
reference (`drug_reference.py`) put hydroxyzine within reach; the hazard it
illustrates is unchanged for any drug that reference does not carry either.

This stage puts a language model between the match and the chart.  It sees the
sentence that was dictated and every candidate the matcher considered, and it
answers which candidate - if any - the sentence actually supports.

The model is advisory, never authoritative.  Three rules make that structural:

1. **Closed set.**  The only tokens it may choose are the candidates the matcher
   produced plus the word that was heard.  The corrected sentence is rebuilt
   here from the chosen token; the model's own sentence string is compared
   against that reconstruction and a mismatch rejects the whole verdict.
2. **No promotion.**  A validated term keeps the matcher's confidence; it never
   rises.  Only a human clears a fuzzy match to GREEN.
3. **Disagreement is not a decision.**  Three answers send the row RED for
   human review instead of to a chart: the model is unsure, the model rejects
   every candidate (a near-identical string is not the same medicine), or the
   model overrules the matcher's own top pick.  The only
   answer that leaves a row at the matcher's YELLOW is the two signals
   independently agreeing.  Sound-alike drug pairs are the whole reason this
   stage exists, and guessing between them is the injury it prevents.

Every failure path - no binary, no model, timeout, unparsable JSON, a verdict
that breaks a rule above - leaves the row exactly as the matcher produced it and
records why, mirroring `llm_extract`'s regex fallback.
"""
import json
import os
import re
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))

# A verdict at or under this is not a verdict; the row goes to a human.
MIN_CONFIDENCE = 0.85
# Two candidates this close on the matcher's own score are a tie, whatever the
# model says about them.  Sound-alike pairs sit exactly here.
TIE_BAND = 0.03

STATUS_CORRECTED = "corrected"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_REJECTED = "rejected"
STATUS_SKIPPED = "skipped"


def flagged_rows(entities):
    """The drug rows carrying a fuzzy match, i.e. the ones with a token to judge.

    One row is one flagged token, so each validator call weighs a single
    substitution site - the model is never asked to juggle two at once.
    """
    return [d for d in entities.get("drugs", [])
            if isinstance(d, dict) and d.get("raw_token") and d.get("candidates")]


def _candidate_evidence(cand, unit, dose=None):
    """What the national drug pack knows about one candidate, as a clause.

    Empty when the candidate carries no reference record - the mini list's own
    entries mostly do not - which keeps the prompt honest about where the
    evidence came from instead of implying the pack vouched for everything.
    """
    ref = cand.get("reference")
    if not ref:
        return ""
    try:
        import drug_reference
    except ImportError:
        return ""
    return drug_reference.evidence({"kind": ref.get("kind", "substance"),
                                    "substances": ref.get("substances", []),
                                    "units": ref.get("units", []),
                                    "strengths": ref.get("strengths", []),
                                    "mass_mg": ref.get("mass_mg"),
                                    "forms": ref.get("forms", [])},
                                   dose=dose, unit=unit)


def unit_conflicts(cand, unit, dose=None):
    """Does the pack contradict this candidate for this dose and unit?

    Two ways it can.  The unit may be the wrong *kind* of quantity - a drug
    the pack dispenses only in `U` against a sentence in milligrams.  Or the
    magnitude may be impossible: the pack lists levothyroxine between 0.005
    and 0.3 mg, so `25 mg` is 83x its maximum and the sentence is not about
    levothyroxine however close the spelling is.

    What it deliberately does not do is compare unit strings.  `mg` and `g`
    are the same dimension, and treating them as a mismatch rejected
    `azithromycin 1 g`, an ordinary single dose.
    """
    ref = cand.get("reference")
    if not ref or not unit:
        return False
    try:
        import drug_reference
    except ImportError:
        return False
    rec = {"units": ref.get("units", []), "mass_mg": ref.get("mass_mg")}
    if drug_reference.unit_fits(rec, unit) is False:
        return True
    return drug_reference.dose_conflicts(rec, dose, unit)


def build_request(row):
    """The input data structure for one flagged token."""
    unit = row.get("unit")
    cands = [c for c in row["candidates"] if c.get("name")]
    return {
        "target_sentence": str(row.get("source_sentence", "") or ""),
        "flagged_token": str(row["raw_token"]),
        "suggestions": [c["name"] for c in cands],
        "matcher_scores": {c["name"]: c.get("score") for c in cands},
        "dose": row.get("dose"),
        "unit": unit,
        "reference_evidence": {c["name"]: ev for c in cands
                               if (ev := _candidate_evidence(c, unit, row.get("dose")))},
    }


def _substitute(sentence, token, replacement):
    """Rebuild the sentence with `token` swapped for `replacement`, or None.

    The swap is whole-word and case-insensitive, and it must hit exactly once:
    a token appearing twice is not a single substitution site, so there is no
    unambiguous corrected sentence to return.
    """
    if not sentence or not token:
        return None
    hits = list(re.finditer(rf"\b{re.escape(token)}\b", sentence, re.I))
    if len(hits) != 1:
        return None
    start, end = hits[0].span()
    return sentence[:start] + replacement + sentence[end:]


def _prompt(request):
    suggestions = json.dumps(request["suggestions"])
    return (
        "You are the validation engine of a clinical NLP pipeline. A speech\n"
        "recogniser misheard one word in a dictated sentence. A fuzzy string\n"
        "matcher proposed the candidate drug names below. Decide which one the\n"
        "sentence actually supports.\n\n"
        "Return STRICT JSON only (no prose, no markdown):\n"
        "{\n"
        '  "status": "corrected" | "ambiguous",\n'
        '  "chosen_token": "<one entry from suggestions, or the flagged token>",\n'
        '  "semantic_confidence_score": 0.0,\n'
        '  "reason": "<one short clause>"\n'
        "}\n\n"
        "Rules you must follow:\n"
        '- "chosen_token" MUST be copied exactly from suggestions, or be the\n'
        "  flagged token itself. Never invent a drug name that is not listed,\n"
        "  even if you believe the real word is something else.\n"
        '- If the sentence does not clearly single out one candidate - two could\n'
        "  fit the dose, route or indication, or none fits - answer\n"
        '  "ambiguous". A wrong drug on a discharge note injures a patient; an\n'
        "  ambiguous answer only costs a clinician a glance.\n"
        '- "corrected" means a candidate replaces the flagged token. If none of\n'
        "  the candidates is the drug this sentence is about, copy the flagged\n"
        "  token back as chosen_token - that rejects them all, which is a valid\n"
        "  and often correct answer when the real drug is simply not listed.\n"
        "- Judge on medical sense: does the candidate match the dose, unit,\n"
        "  frequency and indication in this sentence? Spelling similarity is\n"
        "  what the matcher already did; it is not evidence.\n"
        "- semantic_confidence_score is your own certainty, 0.0 to 1.0.\n\n"
        f"Sentence: {request['target_sentence'][:600]}\n"
        f"Flagged token: {request['flagged_token']}\n"
        f"Dose in the sentence: {_dose_str(request)}\n"
        f"Suggestions: {suggestions}\n"
        + _evidence_block(request)
    )


def _dose_str(request):
    dose, unit = request.get("dose"), request.get("unit")
    if dose is None and not unit:
        return "(none stated)"
    shown = "" if dose is None else (str(int(dose)) if float(dose).is_integer()
                                     else str(dose))
    return f"{shown} {unit or ''}".strip()


def _evidence_block(request):
    """The pack's facts about each candidate, appended to the prompt.

    This is the evidence the matcher could not supply: what the drug is a brand
    of, and the units and strengths India's national list actually dispenses it
    in.  A unit that contradicts the sentence is the strongest single signal
    available for a sound-alike pair, so it is stated plainly rather than left
    for the model to infer from a strength table.
    """
    ev = request.get("reference_evidence") or {}
    if not ev:
        return ""
    lines = "\n".join(f"- {name}: {text}" for name, text in sorted(ev.items()))
    return ("\nReference facts from the Common Drug Codes for India list "
            "(authoritative on units and strengths, not on this sentence):\n"
            + lines + "\n")


def _ollama_runner(model, timeout):
    def run(prompt):
        from llm_extract import _extract_json_from_output, ollama_available
        ok, reason = ollama_available(model)
        if not ok:
            raise ValueError(reason)
        out = subprocess.run(["ollama", "run", "--format", "json", model, prompt],
                             capture_output=True, encoding="utf-8",
                             errors="ignore", timeout=timeout)
        raw = _extract_json_from_output(out.stdout or "")
        if not raw:
            raise ValueError("no JSON in validator output")
        return json.loads(raw)
    return run


def _tie(cands):
    """Candidates the matcher could not separate, best first.

    Only peers of the leader count.  A name from the 72k-entry national
    reference is not a rival to a curated mini-list pick merely by landing
    within `TIE_BAND` of it: in a table that large some brand is always
    within a hundredth of anything, and `sitromycin` - a cefoperazone brand -
    scores 0.818 against `asithromysin` where `azithromycin` scores 0.833.
    Letting that count as a tie sent every fuzzy azithromycin to a human.

    A reference name that clears `REFERENCE_MARGIN` over the leader has
    already taken first place and is judged on its own among its peers, so
    the rule costs nothing where the reference is genuinely the better
    answer.  Ties within one source are untouched - two curated candidates
    the matcher cannot separate are still a tie, which is the case the rule
    was written for.
    """
    top_score = cands[0].get("score") or 0.0
    top_source = cands[0].get("source", "mini")
    return [c["name"] for c in cands
            if (c.get("score") or 0.0) >= top_score - TIE_BAND
            and c.get("source", "mini") == top_source]


def filter_by_unit(row):
    """Candidates minus the ones the pack's units rule out. Also returns the cut.

    Spelling distance cannot tell `hydroxazine` from `levothyroxine`; the fact
    that the national list dispenses levothyroxine only in micrograms, against
    a sentence that says `25 mg`, can.  Removing those candidates before the
    tie check is what stops a sound-alike pair reading as a tie and sending an
    otherwise-decidable row to a human.

    The cut is never allowed to empty the set: if every candidate conflicts,
    the matcher has no survivors to offer and the caller sends the row to a
    human rather than picking from names the pack has already contradicted.
    """
    cands = [c for c in row.get("candidates", []) if c.get("name")]
    unit = row.get("unit")
    dose = row.get("dose")
    kept = [c for c in cands if not unit_conflicts(c, unit, dose)]
    dropped = [c["name"] for c in cands if unit_conflicts(c, unit, dose)]
    return (kept or cands), dropped, bool(kept)


def judge(row, runner):
    """Validate one flagged row. Returns the spec's output record.

    Raises ValueError when the model could not be consulted, which the caller
    treats as "leave the matcher's answer alone".
    """
    kept, dropped, any_survived = filter_by_unit(row)
    unit = row.get("unit")
    if not any_survived:
        # Every candidate is dispensed in units this sentence contradicts, so
        # the matcher's whole shortlist is wrong about the medicine even where
        # it is close on spelling. There is nothing here to clear.
        return _record(STATUS_REJECTED, str(row.get("source_sentence", "") or ""),
                       str(row.get("source_sentence", "") or ""), [], 0.0,
                       f"no candidate is dispensed in {unit}: "
                       f"{', '.join(c['name'] for c in kept)}"[:160])

    scoped = {**row, "candidates": kept}
    request = build_request(scoped)
    original = request["target_sentence"]
    heard = request["flagged_token"]
    closed_set = {s.lower(): s for s in request["suggestions"]}
    closed_set.setdefault(heard.lower(), heard)

    tied = _tie(kept)
    if len(tied) > 1:
        # The matcher itself could not separate these. No model verdict is
        # allowed to break that tie; it goes to a human.
        return _record(STATUS_AMBIGUOUS, original, original, [], 0.0,
                       f"matcher tie: {', '.join(tied)}")

    verdict = runner(_prompt(request))
    if not isinstance(verdict, dict):
        raise ValueError("validator JSON is not an object")

    chosen = str(verdict.get("chosen_token", "") or "").strip()
    if chosen.lower() not in closed_set:
        raise ValueError(f"validator chose {chosen!r}, outside the candidate set")
    chosen = closed_set[chosen.lower()]

    try:
        score = float(verdict.get("semantic_confidence_score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    score = min(max(score, 0.0), 1.0)
    reason = str(verdict.get("reason", "") or "")[:160]
    declared = str(verdict.get("status", "") or "").strip().lower()

    if declared == STATUS_AMBIGUOUS or score < MIN_CONFIDENCE:
        why = reason or (f"validator confidence {score:.2f} < {MIN_CONFIDENCE}")
        return _record(STATUS_AMBIGUOUS, original, original, [], score, why)

    if chosen.lower() == heard.lower():
        # The heard word is not a drug name - that is why the matcher fuzzed it
        # in the first place - so "the flagged token was already right" can only
        # mean the model rejected every candidate. That is the opposite of a
        # clearance and must not read like one.
        return _record(STATUS_REJECTED, original, original, [], score,
                       reason or f"no candidate fits {heard!r}")

    rebuilt = _substitute(original, heard, chosen)
    if rebuilt is None:
        raise ValueError(f"{heard!r} is not a single substitution site")
    # The model's own sentence is checked against the reconstruction rather than
    # used. The prompt does not ask for `final_output_sentence`, so this rarely
    # fires against a real model; the closed-set check and the reconstruction
    # above are the load-bearing guards, and this is defence in depth for a
    # model that volunteers a rewritten sentence anyway.
    offered = str(verdict.get("final_output_sentence", "") or "").strip()
    if offered and offered != rebuilt.strip():
        raise ValueError("validator rewrote more than the flagged token")
    record = _record(STATUS_CORRECTED, original, rebuilt,
                     [{"replaced_token": heard, "substituted_with": chosen}],
                     score, reason)
    # Agreeing with the matcher is corroboration; overruling it is two signals
    # disagreeing about which drug was said, which is the sound-alike hazard
    # itself and goes to a human rather than to whichever signal spoke last.
    # Measured against the matcher's *original* top pick, not the filtered
    # shortlist's. If the unit filter removed what the matcher proposed, then
    # any answer is a swap away from what the chart would otherwise have
    # said - exactly the disagreement this rule sends to a human, and
    # comparing against `kept[0]` would have hidden it behind a YELLOW.
    # `matcher_pick` is the string matcher's own answer, recorded before the
    # dose check reordered the shortlist. Reading the list's current head
    # instead would hide a swap that already happened at extraction.
    original_top = row.get("matcher_pick") or next(
        (c["name"] for c in row.get("candidates", []) if c.get("name")), "")
    record["overrules_matcher"] = chosen.lower() != str(original_top).lower()
    if dropped:
        # The pack removed candidates before the model saw them. Say so: a
        # verdict reached over a narrowed shortlist is not the same evidence as
        # one reached over the matcher's full list.
        record["unit_filtered"] = dropped
    return record


def _record(status, original, final, corrections, score, reason):
    return {
        "status": status,
        "original_sentence": original,
        "final_output_sentence": final,
        "applied_corrections": corrections,
        "semantic_confidence_score": round(score, 3),
        "reason": reason,
    }


def apply_validation(entities, model="llama3.2:3b", timeout=45, runner=None):
    """Run the validator over every fuzzy-matched drug row, in place.

    `entities` is returned either way; rows the validator could not judge keep
    the matcher's name, colour and confidence untouched, with the reason noted.
    """
    rows = flagged_rows(entities)
    if not rows:
        return entities
    run = runner or _ollama_runner(model, timeout)
    engine = "injected" if runner else f"ollama:{model}"
    for row in rows:
        try:
            record = judge(row, run)
        except Exception as e:
            # Unreachable model, bad JSON, a verdict that broke a rule: the
            # matcher's YELLOW stands, and the chart says the gate did not run.
            row["validation"] = {"status": STATUS_SKIPPED,
                                 "reason": f"{type(e).__name__}: {e}"[:160]}
            row["note"] = f"{row.get('note', '')} | validator skipped".strip(" |")
            continue
        row["validation"] = {**record, "engine": engine}
        if record["status"] == STATUS_REJECTED:
            # The matcher's pick was the nearest string, not the right drug.
            # Keeping it on the row invites a skim-approve, so the row carries
            # the word that was actually said and the rejected pick moves to
            # the note where a human has to read it.
            row["note"] = (f"{row.get('note', '')} | validator rejected "
                           f"{row['name']} for {row['raw_token']!r}").strip(" |")
            row["name"] = row["raw_token"]
            row["color"] = "RED"
            row["confidence"] = min(float(row.get("confidence", 0.88)), 0.70)
        elif record["status"] == STATUS_AMBIGUOUS:
            # Two plausible drugs is the sound-alike hazard itself. Keep the
            # matcher's name so the proof chain still points somewhere, but
            # demand a human before it reaches a chart.
            row["color"] = "RED"
            row["confidence"] = min(float(row.get("confidence", 0.88)), 0.70)
            row["note"] = f"{row.get('note', '')} | ambiguous: {record['reason']}".strip(" |")
        elif record["status"] == STATUS_CORRECTED:
            swap = record["applied_corrections"][0]
            row["name"] = swap["substituted_with"]
            row["note"] = (f"{row.get('note', '')} | validator "
                           f"{swap['replaced_token']}->{swap['substituted_with']}"
                           ).strip(" |")
            if record.get("overrules_matcher"):
                row["color"] = "RED"
                row["confidence"] = min(float(row.get("confidence", 0.88)), 0.70)
                row["note"] = f"{row['note']} (overrules matcher)"
            else:
                row["confidence"] = min(float(row.get("confidence", 0.88)), 0.88)
        else:
            row["confidence"] = min(float(row.get("confidence", 0.88)), 0.88)
        if row["color"] != "RED":
            row["color"] = "YELLOW"
    entities["term_validator"] = engine
    return entities

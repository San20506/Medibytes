"""Manual-test runner: audio -> denoise -> ASR -> extraction with fuzzy drug repair.

    python -m base_pipeline.run clip.wav more_clips/ --out runs/try1
    python -m base_pipeline.run noisy.wav --denoiser none --denoiser dpdfnet2-onnx

Every clip x denoiser arm gets its own folder holding the cleaned audio, the
transcript, the entities and a `report.md` to read next to the recording.
`review.csv` has one row per extracted item with empty `correct` / `notes`
columns, so a pass over new data ends with a tally instead of an impression.

Defaults are chosen for repeatable manual testing: regex extraction (no LLM
needed to reproduce a run) and the fuzzy-match validator only when a local LLM
server is actually serving the model.  See llm_client.py for pointing it at
Ollama, llama.cpp or vLLM.
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

from . import decode_presets
from .audio_clean import BackendId, EnhancementConfig, clean_audio
from .stt_extract import run_stt_extract

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".mp4", ".flac", ".ogg", ".aac", ".webm"}
_BACKENDS = Path(__file__).parent / "denoise_backends"


def find_audio(paths):
    found = []
    for p in map(Path, paths):
        if p.is_dir():
            found += sorted(f for f in p.iterdir() if f.suffix.lower() in AUDIO_EXT)
        elif p.is_file():
            found.append(p)
        else:
            raise FileNotFoundError(p)
    return found


def denoiser_config(backend):
    """The backend's own shipped config (model path + hash) if it has one."""
    raw = {"variant_id": backend.value}
    shipped = _BACKENDS / f"{backend.value}.config.json"
    if shipped.is_file():
        raw = json.loads(shipped.read_text(encoding="utf-8"))
        raw.pop("backend", None)
        if "model_path" in raw:
            raw["model_path"] = Path(raw["model_path"])
    return EnhancementConfig(backend=backend, **raw)


def _slug(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "clip"


def run_clip(src, arm_dir, denoiser, *, model, use_llm, validate_terms, second_pass,
             llm_model, context_repair=False, decode=decode_presets.DEFAULT):
    arm_dir.mkdir(parents=True, exist_ok=True)
    backend = BackendId(denoiser)
    cleaned = arm_dir / "cleaned.wav"
    t0 = time.perf_counter()
    meta = clean_audio(src, cleaned, backend=backend, config=denoiser_config(backend))
    t_clean = time.perf_counter() - t0
    t0 = time.perf_counter()
    transcript, entities = run_stt_extract(
        str(cleaned), job_id=_slug(f"{src.stem}-{denoiser}"), use_llm=use_llm,
        model=model, validate_terms=validate_terms, second_pass=second_pass,
        llm_model=llm_model, context_repair=context_repair, decode=decode)
    t_asr = time.perf_counter() - t0
    # Stage 4c: deterministic verification (no LLM) attaches to entities.
    try:
        from .verify import verify as _verify

        entities["verify"] = _verify(transcript, entities)
    except Exception as err:  # verification must never break a run
        entities["verify"] = {"facts": [], "summary": {"error": f"{type(err).__name__}: {err}"}}
    # Stage 5: 6-section discharge note (LLM, heard-only + gated). Without
    # --llm the sections stay empty and say why, mirroring other LLM stages.
    if use_llm:
        try:
            from .llm_extract import (canonicalize_discharge as _canon,
                                      extract_discharge as _extract_discharge,
                                      gate_discharge as _gate, merge_discharge as _merge)

            _raw = _extract_discharge(transcript.get("text", ""), transcript.get("normalized_en", ""),
                                      transcript.get("segments", []), entities["verify"],
                                      model=llm_model)
            _gated = _gate(_raw, transcript.get("text", ""),
                           transcript.get("normalized_en", ""), entities["verify"])
            _locked = _canon(_gated, entities["verify"])
            from .llm_extract import stamp_schema as _stamp
            entities["discharge"] = _stamp(_merge(
                entities["verify"], _locked,
                transcript.get("text", ""),
                transcript.get("normalized_en", "")))
        except Exception as err:
            entities["discharge"] = {"sections": {k: {"items": []} for k in (
                "present_illness", "examination_findings", "previous_diagnosis",
                "plan_of_management", "diet_instructions", "general_instructions")},
                "provenance": {"llm_engine": f"discharge-skipped ({type(err).__name__}: {err})"}}
    else:
        entities["discharge"] = {"sections": {k: {"items": []} for k in (
            "present_illness", "examination_findings", "previous_diagnosis",
            "plan_of_management", "diet_instructions", "general_instructions")},
            "provenance": {"llm_engine": "discharge-skipped (regex-only run; pass --llm)"}}
    try:
        from .llm_extract import stamp_schema as _stamp_empty

        entities["discharge"] = _stamp_empty(entities["discharge"])
    except Exception:
        pass
    for name, obj in (("transcript.json", transcript), ("entities.json", entities),
                      ("clean_meta.json", meta),
                      ("verify.json", entities["verify"]),
                      ("discharge.json", entities["discharge"])):
        (arm_dir / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
    result = {"meta": meta, "transcript": transcript, "entities": entities,
              "seconds": {"clean": round(t_clean, 2), "asr_extract": round(t_asr, 2)}}
    (arm_dir / "report.md").write_text(render_report(src, denoiser, result),
                                       encoding="utf-8")
    return result


def _dose(d):
    if d.get("dose") in (None, ""):
        return "no dose"
    return f"{d['dose']:g} {d.get('unit') or ''}".strip()


def _drug_provenance(d):
    """What a reviewer needs to judge a repaired drug: heard -> chosen, and why."""
    bits = []
    if d.get("raw_token") and d["raw_token"].lower() != d["name"].lower():
        bits.append(f"heard `{d['raw_token']}`")
    ref = d.get("reference") or {}
    if ref:
        bits.append(f"reference {ref.get('kind')}: {ref.get('name')}")
    cands = d.get("candidates") or []
    if cands:
        bits.append("candidates " + ", ".join(
            f"{c['name']} {c['score']:.2f}" for c in cands[:4] if c.get("name")
            and isinstance(c.get("score"), (int, float))))
    val = d.get("validation")
    if val:
        bits.append(f"validator {val.get('status')}")
    if d.get("second_pass"):
        bits.append(f"heard only by {d['second_pass']}")
    return "; ".join(bits)


def review_rows(entities):
    """(kind, item, confidence, colour, detail) for everything worth checking."""
    for d in entities.get("drugs", []):
        item = f"{'NOT GIVEN: ' if d.get('negated') else ''}{d['name']} {_dose(d)}"
        yield ("drug", item, d.get("confidence"), d.get("color"),
               _drug_provenance(d) or d.get("note", ""))
    for v in entities.get("vitals", []):
        yield ("vital", v.get("text", ""), v.get("confidence"), v.get("color"),
               v.get("note", ""))
    for a in entities.get("allergies", []):
        yield ("allergy", f"{'DENIED ' if a.get('negated') else ''}{a.get('text', '')}",
               a.get("confidence"), a.get("color"), a.get("note", ""))
    for b in (entities.get("verify") or {}).get("brand_mentions", []) or []:
        yield ("brand", f"{b['brand']} {b['number']}", 0.70, "RED",
               f"brand mention - not charted; is {', '.join(b['substances'])}")
    _prov = (entities.get("discharge") or {}).get("provenance", {}) or {}
    if _prov.get("schema") == "INVALID":
        yield ("discharge", "schema INVALID", 0.70, "RED", _prov.get("schema_error", ""))


def render_report(src, denoiser, result):
    m, t, e, s = (result[k] for k in ("meta", "transcript", "entities", "seconds"))
    out = [f"# {src.name} - denoiser `{denoiser}`", "",
           "## Stage 1 - clean", "",
           f"- backend `{m.get('actual_backend')}`, variant `{m.get('variant_id')}`, "
           f"VAD ratio {m.get('vad_ratio')}, RMS {m.get('rms_dbfs_before')} -> "
           f"{m.get('rms_dbfs_after')} dBFS, {s['clean']} s", "",
           "## Stage 2 - ASR", "",
           f"- engine `{t.get('stt_engine')}`, language {t.get('language')}, "
           f"{s['asr_extract']} s (ASR + extraction)",
           "", "**Raw transcript**", "", f"> {t.get('text', '')}", "",
           "**Normalised**", "", f"> {t.get('normalized_en', '')}", "",
           "## Stage 3 - extraction and fuzzy drug repair", ""]
    rows = list(review_rows(e))
    if rows:
        out += ["| kind | item | colour | conf | detail |", "|---|---|---|---|---|"]
        out += [f"| {k} | {i} | {c or ''} | {conf if conf is not None else ''} | "
                f"{d.replace('|', '/')} |" for k, i, conf, c, d in rows]
    else:
        out.append("_nothing extracted_")
    for key in ("llm_engine", "term_validator", "term_validator_seconds", "tidy_engine", "drug_second_pass",
                "context_repair"):
        if e.get(key):
            out += ["", f"- {key}: `{e[key]}`"]
    v = e.get("verify") or {}
    if v.get("summary"):
        out += ["", "## Stage 4c - verification",
                f"- facts {v['summary'].get('total', 0)}: " + ", ".join(
                    f"{k} {n}" for k, n in sorted(v["summary"].items())
                    if k not in ("total", "numerals", "error")) or "-",
                f"- numerals heard as digits: {(v['summary'].get('numerals') or {}).get('count', 0)}"]
    d = e.get("discharge") or {}
    if d.get("sections"):
        out += ["", "## Stage 5 - discharge note (6 sections, heard-only)"]
        for sec, block in d["sections"].items():
            items = (block or {}).get("items", []) or []
            out.append(f"### {sec} ({len(items)})")
            for it in items:
                tag = f" [deterministic:{it['verified_kind']}]" if it.get("verified_kind") else ""
                out.append(f"- {it.get('text', '')}{tag} [proof: {str(it.get('source_sentence', ''))[:100]}]")
            if not items:
                out.append("_not dictated_")
        if d.get("provenance"):
            out += ["", f"- discharge: `{d['provenance'].get('llm_engine')}`, "
                       f"dropped_unproven: {d['provenance'].get('dropped_unproven', 0)}, "
                       f"canonicalized: {d['provenance'].get('canonicalized', 0)}, "
                       f"prefilled: {d['provenance'].get('prefilled', 0)}, "
                       f"schema: `{d['provenance'].get('schema', 'unchecked')}`"]
            if d["provenance"].get("schema") == "INVALID":
                out += [f"  ! schema_error: {d['provenance'].get('schema_error', '')[:200]}"]
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m base_pipeline.run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="audio files and/or folders of audio")
    ap.add_argument("--out", default=f"runs/{time.strftime('%Y%m%d-%H%M%S')}",
                    help="output folder (default runs/<timestamp>)")
    ap.add_argument("--denoiser", action="append", choices=[b.value for b in BackendId],
                    help="repeat to compare arms side by side (default: none)")
    ap.add_argument("--model", default="medasr",
                    help="medasr (default) | small-int8 | base-int8 | medasr-lm | mock")
    ap.add_argument("--no-second-pass", action="store_true",
                    help="skip the second decoder that recovers drug names the first missed")
    ap.add_argument("--decode", default=decode_presets.DEFAULT, choices=list(decode_presets.PRESETS),
                    help="faster-whisper decoding preset (see decode_presets.py); medasr takes none")
    ap.add_argument("--second-pass", metavar="MODEL",
                    help="decoder for the second pass (default: auto-paired); e.g. large-v3 for medasr")
    ap.add_argument("--llm", action="store_true",
                    help="LLM-primary extraction (needs a local LLM server)")
    ap.add_argument("--no-validate", action="store_true",
                    help="skip LLM validation of fuzzy drug matches")
    ap.add_argument("--context-repair", action="store_true",
                    help="LLM places misheard medicine names from sentence context (RED rows only)")
    ap.add_argument("--variant", choices=["whisper-medasr"],
                    help="SNR-gated denoise -> Whisper -> MedASR on the medical clips -> extract "
                         "-> verify; ignores --denoiser/--model/--llm and every post-processing step")
    ap.add_argument("--snr-threshold", type=float, default=15.0,
                    help="--variant: denoise only when estimated SNR (dB) is below this")
    ap.add_argument("--whisper-model", default="small-int8",
                    help="--variant: Whisper size used to find the medical stretches")
    ap.add_argument("--llm-url", help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8080/v1")
    ap.add_argument("--llm-model", help="model name on that server")
    ap.add_argument("--llm-env", metavar="FILE",
                    help="KEY=VALUE file to load into the environment (e.g. CHATGPT_API_KEY=... "
                         "for --llm-url https://api.openai.com/v1)")
    a = ap.parse_args(argv)

    if a.decode != decode_presets.DEFAULT and a.model in ("medasr", "medasr-lm"):
        ap.error(f"--decode {a.decode} applies to faster-whisper models only, not --model {a.model}")
    if a.llm_env:
        for line in Path(a.llm_env).read_text(encoding="utf-8").splitlines():
            k, sep, v = line.partition("=")
            if sep and k.strip() and not k.lstrip().startswith("#"):
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))

    if a.llm_url:
        os.environ["MEDIBYTES_LLM_URL"] = a.llm_url
    try:
        clips = find_audio(a.inputs)
    except FileNotFoundError as err:
        print(f"not found: {err}", file=sys.stderr)
        return 2
    if not clips:
        print("no audio files found", file=sys.stderr)
        return 2

    out = Path(a.out)
    arms = [a.variant] if a.variant else a.denoiser or ["none"]
    review = []
    summary = ["# Run summary", "", "| clip | denoiser | engine | drugs | vitals | allergies | s |",
               "|---|---|---|---|---|---|---|"]
    for src in clips:
        for denoiser in arms:
            label = f"{src.name} [{denoiser}]"
            arm_dir = out / _slug(src.stem) / denoiser
            print(f"-> {label}", flush=True)
            try:
                if a.variant:
                    from .whisper_medasr import run_variant

                    r = run_variant(src, arm_dir, snr_threshold_db=a.snr_threshold,
                                    whisper_model=a.whisper_model)
                    print(f"   snr {r['gate']['snr_db']} dB, denoised={r['gate']['denoised']}",
                          flush=True)
                else:
                    r = run_clip(src, arm_dir, denoiser, model=a.model,
                                 use_llm=True if a.llm else False,
                                 validate_terms=False if a.no_validate else "auto",
                                 second_pass=None if a.no_second_pass else (a.second_pass or "auto"),
                                 llm_model=a.llm_model, context_repair=a.context_repair,
                                 decode=a.decode)
            except Exception as err:  # one bad clip must not end the batch
                print(f"   FAILED: {type(err).__name__}: {err}", file=sys.stderr)
                summary.append(f"| {src.name} | {denoiser} | FAILED: {type(err).__name__} | | | | |")
                continue
            e = r["entities"]
            summary.append(
                f"| [{src.name}]({_slug(src.stem)}/{denoiser}/report.md) | {denoiser} | "
                f"{r['transcript'].get('stt_engine')} | {len(e.get('drugs', []))} | "
                f"{len(e.get('vitals', []))} | {len(e.get('allergies', []))} | "
                f"{sum(r['seconds'].values()):.1f} |")
            review += [(src.name, denoiser, *row) for row in review_rows(e)]

    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    with open(out / "review.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip", "denoiser", "kind", "item", "confidence", "colour", "detail",
                    "correct (y/n)", "notes"])
        w.writerows([*row, "", ""] for row in review)
    print(f"\nwrote {out}/summary.md and review.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

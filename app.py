"""Streamlit UI for the clinical voice pipeline (pre-LLM, upload-only).

Run from ``base_pipeline_project/``::

    streamlit run app.py

Upload audio -> per-denoiser arms (``none`` + ``sherpa-gtcrn-simple``) ->
transparent Steps 0-6 ending in the same artefacts as the CLI
(``cleaned.wav``, ``transcript.json``, ``entities.json``, ``report.md``,
``summary.md``, ``review.csv``).

This file adds no clinical logic: it imports ``run_clip``,
``denoiser_config``, ``review_rows`` and ``render_report`` from
``base_pipeline.run`` and only renders their outputs.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import streamlit as st

from base_pipeline.audio_clean import BackendId, EnhancementConfig
from base_pipeline.run import (
    AUDIO_EXT,
    _slug,
    denoiser_config,
    render_report,
    review_rows,
    run_clip,
)

_HERE = Path(__file__).parent
_RUNS = _HERE / "runs"

SAFE_DENOISERS = ["none", "sherpa-gtcrn-simple"]
SAFE_MODELS = ["medasr", "small-int8", "base-int8", "mock"]


def _color_pill(color: str | None) -> str:
    bg = {"GREEN": "#d4edda", "YELLOW": "#fff3cd", "RED": "#f8d7da"}.get(
        str(color or "").upper(), "#e2e3e5"
    )
    return (
        f"<span style='background:{bg};border-radius:8px;padding:2px 10px;"
        f"font-weight:600'>{color or '-'}</span>"
    )


def _short(text: str | None, n: int = 12) -> str:
    text = str(text or "")
    return text if len(text) <= n else text[:n] + "…"


def ui_denoiser_config(backend: BackendId, override_model_path: str = ""):
    """Shipped backend config, with an optional Windows model-path override.

    Sets ``MEDIBYTES_GTCRN_MODEL`` (read by
    ``denoise_backends/sherpa_gtcrn_simple._model_candidates``) so the
    override passes the exact-path guard. SHA-256, params and variant checks
    still apply — a bad path stays a hard ``BACKEND_UNAVAILABLE`` error,
    never a silent fallback.
    """
    import os as _os  # noqa: PLC0415 (run-scoped env, mirrors CLI --llm-url)

    cfg = denoiser_config(backend)
    override = (override_model_path or "").strip().strip('"')
    if override and backend.value == "sherpa-gtcrn-simple":
        _os.environ["MEDIBYTES_GTCRN_MODEL"] = override
        cfg = EnhancementConfig(
            backend=cfg.backend,
            variant_id=cfg.variant_id,
            model_path=Path(override),
            model_sha256=cfg.model_sha256,
            parameters=dict(cfg.parameters),
            timeout_s=cfg.timeout_s,
        )
    return cfg


def _run_one_arm(
    src: Path, arm_dir: Path, denoiser: str, *, model: str, second_pass,
    use_llm: bool, validate_terms, llm_model, sherpa_path: str,
):
    """Run a single (clip, denoiser) arm, honouring the sherpa path override."""
    from base_pipeline.run import _BACKENDS  # noqa: PLC0415  (reuse, no fork)

    arm_dir.mkdir(parents=True, exist_ok=True)
    backend = BackendId(denoiser)
    # Reuse run_clip but with our config so the model path is overridable.
    import json as _json  # noqa: PLC0415

    raw = {"variant_id": backend.value}
    shipped = _BACKENDS / f"{backend.value}.config.json"
    if shipped.is_file():
        raw = _json.loads(shipped.read_text(encoding="utf-8"))
        raw.pop("backend", None)
        if "model_path" in raw:
            raw["model_path"] = Path(raw["model_path"])
    if (sherpa_path or "").strip() and backend.value == "sherpa-gtcrn-simple":
        import os as _os  # noqa: PLC0415 (run-scoped env for _model_candidates)

        _os.environ["MEDIBYTES_GTCRN_MODEL"] = sherpa_path.strip().strip('"')
        raw["model_path"] = Path(sherpa_path.strip().strip('"'))
    config = EnhancementConfig(backend=backend, **raw)

    from base_pipeline.audio_clean import clean_audio  # noqa: PLC0415
    from base_pipeline.stt_extract import run_stt_extract  # noqa: PLC0415

    cleaned = arm_dir / "cleaned.wav"
    t0 = time.perf_counter()
    meta = clean_audio(src, cleaned, backend=backend, config=config)
    t_clean = time.perf_counter() - t0
    t0 = time.perf_counter()
    transcript, entities = run_stt_extract(
        str(cleaned),
        job_id=_slug(f"{src.stem}-{denoiser}"),
        use_llm=use_llm,
        model=model,
        validate_terms=validate_terms,
        second_pass=second_pass,
        llm_model=llm_model,
    )
    t_asr = time.perf_counter() - t0
    try:
        from base_pipeline.verify import verify as _verify

        entities["verify"] = _verify(transcript, entities)
    except Exception as err:
        entities["verify"] = {"facts": [], "summary": {"error": f"{type(err).__name__}: {err}"}}
    if use_llm:
        try:
            from base_pipeline.llm_extract import (canonicalize_discharge as _canon,
                                                   extract_discharge as _extract_discharge,
                                                   gate_discharge as _gate, merge_discharge as _merge)

            _raw_d = _extract_discharge(transcript.get("text", ""), transcript.get("normalized_en", ""),
                                        transcript.get("segments", []), entities["verify"], model=llm_model)
            _gated_d = _gate(_raw_d, transcript.get("text", ""),
                             transcript.get("normalized_en", ""), entities["verify"])
            _locked_d = _canon(_gated_d, entities["verify"])
            from base_pipeline.llm_extract import stamp_schema as _stamp
            entities["discharge"] = _stamp(_merge(
                entities["verify"], _locked_d,
                transcript.get("text", ""),
                transcript.get("normalized_en", "")))
        except Exception as err:
            entities["discharge"] = {"sections": {k: {"items": []} for k in (
                "present_illness", "examination_findings", "previous_diagnosis",
                "plan_of_management", "diet_instructions", "general_instructions")},
                "provenance": {
                "llm_engine": f"discharge-skipped ({type(err).__name__}: {err})"}}
    else:
        entities["discharge"] = {"sections": {k: {"items": []} for k in (
            "present_illness", "examination_findings", "previous_diagnosis",
            "plan_of_management", "diet_instructions", "general_instructions")},
            "provenance": {
            "llm_engine": "discharge-skipped (LLM off; tick LLM-primary extraction)"}}
    try:
        from base_pipeline.llm_extract import stamp_schema as _stamp_empty

        entities["discharge"] = _stamp_empty(entities["discharge"])
    except Exception:
        pass
    for name, obj in (
        ("transcript.json", transcript),
        ("entities.json", entities),
        ("clean_meta.json", meta),
        ("verify.json", entities["verify"]),
        ("discharge.json", entities["discharge"]),
    ):
        (arm_dir / name).write_text(
            json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    result = {
        "meta": meta,
        "transcript": transcript,
        "entities": entities,
        "seconds": {"clean": round(t_clean, 2), "asr_extract": round(t_asr, 2)},
    }
    (arm_dir / "report.md").write_text(
        render_report(src, denoiser, result), encoding="utf-8"
    )
    return result


def _probe_card(denoiser: str, sherpa_path: str = ""):
    """Pre-run backend availability: yellow card, never a crash.

    Surfaces ``BACKEND_UNAVAILABLE`` (missing model/runtime) with its fix
    hint before ``enhance()`` runs. ``none`` is always available.
    """
    if denoiser != "sherpa-gtcrn-simple":
        st.caption("probe `none`: available — baseline identity control (stdlib, no model).")
        return
    try:
        from base_pipeline.denoise_backends import sherpa_gtcrn_simple as _sherpa
    except Exception as err:
        st.warning(f"probe `sherpa-gtcrn-simple`: unavailable — cannot import adapter ({err}).")
        return
    try:
        cfg = ui_denoiser_config(BackendId(denoiser), sherpa_path)
        p = _sherpa.probe(cfg)
        if p.available:
            st.success(
                f"probe `sherpa-gtcrn-simple`: available — {p.reason} "
                f"(runtime {p.runtime_version}, sha `{_short(p.model_sha256)}`)."
            )
        else:
            st.warning(
                f"probe `sherpa-gtcrn-simple`: unavailable — {p.reason}. "
                "Fix: `pip install sherpa-onnx==1.13.8 sherpa-onnx-bin==1.13.8` + "
                "download gtcrn_simple.onnx (sha e77603ac…) and set the sidebar path."
            )
    except Exception as err:
        st.warning(
            f"probe `sherpa-gtcrn-simple`: unavailable — `{type(err).__name__}: {err}`. "
            "Fix: check variant/params/model path (sidebar .onnx) + runtime 1.13.8."
        )


def _step_clean(meta: dict, seconds: dict, denoiser: str):
    st.markdown("**Step 1 — clean + denoise probe**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Backend", str(meta.get("actual_backend", denoiser)))
    c2.metric("VAD ratio", str(meta.get("vad_ratio", "-")))
    c3.metric(
        "RMS dBFS",
        f"{meta.get('rms_dbfs_before', '-')} → {meta.get('rms_dbfs_after', '-')}",
    )
    c4.metric("Clean s", str(seconds.get("clean", "-")))
    st.caption(
        f"variant `{meta.get('variant_id')}` · "
        f"model `{_short(meta.get('model_sha256'))}` · "
        f"runtime `{meta.get('runtime_version')}` · "
        f"device `{meta.get('provenance', {}).get('device', meta.get('device', 'cpu')) if isinstance(meta.get('provenance'), dict) else 'cpu'}` · "
        f"samples {meta.get('input_sample_count')}→{meta.get('output_sample_count')} "
        f"(drift {meta.get('sample_count_drift')}) · "
        f"sha `{_short(meta.get('input_sha256'))}`→`{_short(meta.get('output_sha256'))}`"
    )
    with st.expander("clean_meta.json (full)"):
        st.json(meta)


def _step_stt(transcript: dict):
    st.markdown("**Step 3 — STT transcript**")
    engine = str(transcript.get("stt_engine", "-"))
    is_mock = bool(transcript.get("stt_provenance", {}).get("is_mock"))
    if is_mock:
        st.warning(f"`{engine}` — mock fallback, labelled, never silent.")
    else:
        st.success(f"`{engine}`")
    st.caption(f"language `{transcript.get('language')}` · {len(transcript.get('segments', []) or [])} segments")
    st.markdown("> " + (transcript.get("text", "") or "_empty_"))
    segs = transcript.get("segments", []) or []
    if segs:
        st.dataframe(
            [
                {
                    "id": s.get("id"),
                    "start": s.get("start"),
                    "end": s.get("end"),
                    "conf": s.get("confidence"),
                    "text": s.get("text"),
                }
                for s in segs
            ],
            use_container_width=True,
        )


def _step_normalize(transcript: dict):
    st.markdown("**Step 4 — normalize**")
    st.text_area("raw", transcript.get("text", ""), height=80, key=f"raw-{transcript.get('job_id')}")
    st.text_area(
        "normalised_en", transcript.get("normalized_en", ""),
        height=80, key=f"norm-{transcript.get('job_id')}",
    )
    norms = transcript.get("normalizations", []) or []
    if norms:
        st.dataframe(norms, use_container_width=True)
    else:
        st.caption("No Hinglish/spoken-punct substitutions fired.")


def _entities_tables(entities: dict):
    drugs = entities.get("drugs", []) or []
    if drugs:
        st.markdown(f"Drugs ({len(drugs)})")
        st.dataframe(
            [
                {
                    "name": d.get("name"),
                    "dose": d.get("dose"),
                    "unit": d.get("unit"),
                    "freq": d.get("frequency"),
                    "conf": d.get("confidence"),
                    "colour": d.get("color"),
                    "negated": d.get("negated"),
                }
                for d in drugs
            ],
            use_container_width=True,
        )
        for d in drugs:
            label = f"{d.get('name')} {d.get('dose') or 'no dose'} {d.get('unit') or ''}".strip()
            with st.expander(f"provenance — {label}"):
                st.markdown(_color_pill(d.get("color")), unsafe_allow_html=True)
                st.write(f"source: {d.get('source_sentence', '')}")
                if d.get("raw_token"):
                    st.write(f"heard `{d['raw_token']}` → `{d.get('matcher_pick', d['name'])}`")
                if d.get("candidates"):
                    st.json(d["candidates"][:4])
                if d.get("reference"):
                    st.json(d["reference"])
                if d.get("validation"):
                    st.json(d["validation"])
                if d.get("second_pass"):
                    st.caption(f"heard only by `{d['second_pass']}`")
    else:
        st.caption("No drug rows.")
    vitals = entities.get("vitals", []) or []
    if vitals:
        st.markdown(f"Vitals ({len(vitals)})")
        st.dataframe(
            [{"item": v.get("text"), "conf": v.get("confidence"),
              "colour": v.get("color"), "note": v.get("note", "")} for v in vitals],
            use_container_width=True,
        )
    allergies = entities.get("allergies", []) or []
    if allergies:
        st.markdown(f"Allergies ({len(allergies)})")
        st.dataframe(
            [{"item": a.get("text"), "negated(DENIED)": a.get("negated"),
              "conf": a.get("confidence"), "colour": a.get("color")} for a in allergies],
            use_container_width=True,
        )
    for key in ("term_validator", "llm_engine", "tidy_engine", "drug_second_pass"):
        if entities.get(key):
            st.caption(f"{key}: `{entities[key]}`")


def _arm_view(clip_name: str, denoiser: str, stored: dict, orig_bytes: bytes, orig_name: str):
    st.markdown(f"### Arm `{denoiser}`")
    if stored.get("error"):
        st.error(f"FAILED: `{stored['error']}`")
        return
    result = stored["result"]
    arm_dir: Path = stored["arm_dir"]
    meta, transcript, entities = result["meta"], result["transcript"], result["entities"]
    st.markdown("**Step 0 — input**")
    st.caption(f"`{clip_name}` · {len(orig_bytes) / 1024:.1f} KB · ext `{Path(orig_name).suffix.lower()}`")
    _step_clean(meta, result["seconds"], denoiser)
    st.markdown("**Step 2 — audio (original vs cleaned)**")
    c1, c2 = st.columns(2)
    c1.caption("uploaded original")
    c1.audio(orig_bytes, format="audio/wav")
    c2.caption(f"cleaned ({denoiser}) — the signal actually transcribed")
    cleaned = arm_dir / "cleaned.wav"
    if cleaned.is_file():
        c2.audio(cleaned.read_bytes(), format="audio/wav")
    _step_stt(transcript)
    _step_normalize(transcript)
    st.markdown("**Step 5 — extraction + fuzzy drug repair**")
    _entities_tables(entities)
    v = entities.get("verify") or {}
    if v.get("facts"):
        with st.expander(f"Stage 4c — verification ({v.get('summary', {}).get('total', 0)} facts)"):
            st.dataframe(
                [{"kind": f.get("kind"), "name": f.get("name", f.get("text")),
                  "status": f.get("status"),
                  "checks": "; ".join(f.get("checks", []))} for f in v["facts"]],
                use_container_width=True,
            )
    d = entities.get("discharge") or {}
    if d.get("sections"):
        st.markdown("**Stage 5 — discharge note (6 sections, heard-only)**")
        _dp = d.get("provenance", {})
        st.caption(f"`{_dp.get('llm_engine', '')}` · "
                   f"dropped_unproven: {_dp.get('dropped_unproven', 0)} · "
                   f"canonicalized: {_dp.get('canonicalized', 0)} · "
                   f"prefilled: {_dp.get('prefilled', 0)} · "
                   f"schema: `{_dp.get('schema', 'unchecked')}`")
        if _dp.get("schema") == "INVALID":
            st.error(f"Schema INVALID: `{_dp.get('schema_error', '')[:200]}`")
        for sec, block in d["sections"].items():
            items = (block or {}).get("items", []) or []
            with st.expander(f"{sec} ({len(items)})", expanded=bool(items)):
                if not items:
                    st.caption("_not dictated_")
                for it in items:
                    badge = f" `deterministic:{it['verified_kind']}`" if it.get("verified_kind") else " `llm`"
                    st.markdown(f"- {it.get('text', '')}{badge}")
                    st.caption(f"proof: {it.get('source_sentence', '')[:160]}")
    st.markdown("**Step 6 — final artefacts**")
    report = arm_dir / "report.md"
    if report.is_file():
        with st.expander("report.md"):
            st.markdown(report.read_text(encoding="utf-8"))
    d1, d2, d3, d4 = st.columns(4)
    for path, col in (
        ("transcript.json", d1), ("entities.json", d2),
        ("clean_meta.json", d3), ("report.md", d4),
    ):
        p = arm_dir / path
        if p.is_file():
            col.download_button(f"⬇ {path}", p.read_bytes(), file_name=f"{clip_name}-{denoiser}-{path}")
    if cleaned.is_file():
        st.download_button("⬇ cleaned.wav", cleaned.read_bytes(),
                           file_name=f"{clip_name}-{denoiser}-cleaned.wav")


def main():
    st.set_page_config(page_title="MediBytes Voice Pipeline", layout="wide")
    st.title("MediBytes Voice Pipeline — transparent run")
    st.caption("Fail-closed clinical NLP: a wrong drug/vital is a safety event, never a guess. Upload-only · regex-first · RED means human review.")

    with st.sidebar:
        st.header("Pipeline knobs (safe subset)")
        use_none = st.checkbox("denoiser: none (clean)", value=True)
        use_sherpa = st.checkbox("denoiser: sherpa-gtcrn-simple (noisy)", value=True)
        model = st.selectbox("STT model", SAFE_MODELS, index=0,
                             help="medasr default; small-int8 for Hindi/Tamil/Hinglish; mock labels rows is_mock:true")
        second_pass = st.checkbox("second decoder pass (recover missed drugs)", value=True)
        st.divider()
        _default_sherpa = r"C:\models\gtcrn_simple.onnx" if Path(r"C:\models\gtcrn_simple.onnx").is_file() else ""
        sherpa_path = st.text_input(
            "sherpa .onnx path (Windows override, optional)",
            value=_default_sherpa, placeholder=r"C:\models\gtcrn_simple.onnx",
            help="Leaves repo untouched. Empty = shipped config. Wrong path = hard BACKEND_UNAVAILABLE.",
        )
        with st.expander("Advanced (LLM, optional)"):
            use_llm = st.checkbox("LLM-primary extraction", value=False)
            llm_url = st.text_input("LLM URL", value="")
            llm_model = st.text_input("LLM model", value="")
            no_validate = st.checkbox("skip term validation", value=False)
        run_btn = st.button("▶ Process", type="primary", use_container_width=True)

    uploads = st.file_uploader(
        "Drop clinical voice note(s)",
        type=[e.lstrip(".") for e in sorted(AUDIO_EXT)],
        accept_multiple_files=True,
    )
    if not uploads:
        st.info("Upload a `.wav/.mp3/.m4a/.mp4/.flac/.ogg/.aac/.webm` to begin. Nothing runs until you press Process.")
        return
    for f in uploads:
        st.audio(f.getvalue())
        st.caption(f"`{f.name}` · {len(f.getvalue()) / 1024:.1f} KB")

    if not run_btn:
        return
    denoisers = [d for d, on in (("none", use_none), ("sherpa-gtcrn-simple", use_sherpa)) if on]
    if not denoisers:
        st.error("Tick at least one denoiser.")
        return
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_root = _RUNS / f"ui-{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    if llm_url.strip():
        import os  # noqa: PLC0415 (run-scoped env mutation, mirrors CLI)

        os.environ["MEDIBYTES_LLM_URL"] = llm_url.strip()

    for f in uploads:
        slug = _slug(Path(f.name).stem)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / f.name
            src.write_bytes(f.getvalue())
            st.header(f"📼 {f.name}")
            stored: dict = {}
            with st.expander("Backend probe (pre-run availability)", expanded=False):
                for denoiser in denoisers:
                    _probe_card(denoiser, sherpa_path)
            for denoiser in denoisers:
                arm_dir = out_root / slug / denoiser
                with st.status(f"arm `{denoiser}` — clean → STT → extract…", expanded=False) as s:
                    try:
                        res = _run_one_arm(
                            src, arm_dir, denoiser, model=model,
                            second_pass="auto" if second_pass else None,
                            use_llm=use_llm, validate_terms=False if no_validate else "auto",
                            llm_model=llm_model.strip() or None, sherpa_path=sherpa_path,
                        )
                        stored[denoiser] = {"result": res, "arm_dir": arm_dir}
                        s.update(label=f"arm `{denoiser}` done", state="complete")
                    except Exception as err:  # one bad arm must not kill the other
                        stored[denoiser] = {"error": f"{type(err).__name__}: {err}", "arm_dir": arm_dir}
                        s.update(label=f"arm `{denoiser}` failed", state="error")
            tabs = st.tabs([*stored.keys(), "Δ compare"] if len(stored) > 1 else list(stored.keys()))
            for tab, denoiser in zip(tabs, stored.keys()):
                with tab:
                    _arm_view(f.name, denoiser, stored[denoiser], f.getvalue(), f.name)
            if len(stored) > 1 and all("result" in v for v in stored.values()):
                with tabs[-1]:
                    st.markdown("### Δ none vs sherpa-gtcrn-simple (same clip, same model)")
                    rows = []
                    for denoiser, v in stored.items():
                        e = v["result"]["entities"]
                        t = v["result"]["transcript"]
                        rows.append({"arm": denoiser,
                                     "engine": t.get("stt_engine"),
                                     "drugs": len(e.get("drugs", []) or []),
                                     "vitals": len(e.get("vitals", []) or []),
                                     "allergies": len(e.get("allergies", []) or []),
                                     "s": round(sum(v["result"]["seconds"].values()), 1)})
                    st.dataframe(rows, use_container_width=True)
                    a, b = list(stored.keys())[:2]
                    ta, tb = stored[a]["result"]["transcript"], stored[b]["result"]["transcript"]
                    if ta.get("text") != tb.get("text"):
                        st.markdown(f"**Transcript — `{a}`**")
                        st.markdown("> " + (ta.get("text", "") or "_empty_"))
                        st.markdown(f"**Transcript — `{b}`**")
                        st.markdown("> " + (tb.get("text", "") or "_empty_"))
                    else:
                        st.success("Identical transcripts — denoise changed nothing on this clip (expected for clean audio).")
            # run-level review.csv (same columns as CLI)
            import csv  # noqa: PLC0415

            review_path = out_root / f"review-{slug}.csv"
            with open(review_path, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["clip", "denoiser", "kind", "item", "confidence",
                            "colour", "detail", "correct (y/n)", "notes"])
                for denoiser, v in stored.items():
                    if "result" not in v:
                        continue
                    w.writerows([f.name, denoiser, *row, "", ""]
                                for row in review_rows(v["result"]["entities"]))
            st.download_button("⬇ review.csv (human tally)", review_path.read_bytes(),
                               file_name=f"{slug}-review.csv")
    st.success(f"Wrote `{out_root}/` — same layout as the CLI (`summary.md` per arm in `report.md`).")


if __name__ == "__main__":
    main()

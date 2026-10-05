"""Demo orchestrator Stage 0->4. Sequential, file-based (no Redis/DB).

Usage:
  python pipeline.py --in audio_in/sample1.wav --key demo-001 [--denoiser none]
  python pipeline.py --gen-samples   # create 2 synthetic wavs for offline testing
"""
import argparse
import hashlib
import json
import os
import sys
import uuid
import wave
import contextlib

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from audio_clean import BackendId, EnhancementConfig, clean_audio
from stt_extract import run_stt_extract


def _idempotency_get_or_create(key, state_path):
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    state = {}
    if os.path.isfile(state_path):
        try:
            with open(state_path, encoding="utf-8") as f:
                state = json.load(f)
        except Exception:
            state = {}
    if key in state:
        return state[key], True
    job_id = key or f"demo-{uuid.uuid4().hex[:6]}"
    state[key or job_id] = job_id
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    return job_id, False


def gen_samples(out_dir):
    """Generate 2 synthetic 8s 16k mono wavs (tone + silence) so pipeline runs with no mic."""
    os.makedirs(out_dir, exist_ok=True)
    sr = 16000
    for name, freq in (("sample1_hinglish_fever.wav", 440), ("sample2_cough_allergy.wav", 520)):
        t = np.arange(sr * 8) / sr
        tone = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        tone[int(6 * sr):] = 0.0  # trailing silence shows VAD
        pcm = (np.clip(tone, -1, 1) * 32767).astype(np.int16)
        p = os.path.join(out_dir, name)
        with contextlib.closing(wave.open(p, "wb")) as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm.tobytes())
        print(f"wrote {p}")


def _load_denoiser_config(path, backend):
    """Load an optional JSON backend config without changing CLI identity."""
    if not path:
        return EnhancementConfig(backend=backend, variant_id=backend.value)
    with open(path, encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError("--denoiser-config must contain a JSON object")
    requested = raw.pop("backend", backend.value)
    if requested != backend.value:
        raise ValueError(
            f"denoiser config backend {requested!r} does not match --denoiser {backend.value!r}"
        )
    allowed = {"variant_id", "model_path", "model_sha256", "parameters", "timeout_s"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown denoiser config fields: {sorted(unknown)}")
    raw.setdefault("variant_id", backend.value)
    return EnhancementConfig(backend=backend, **raw)


def run_pipeline(
    inp,
    key=None,
    use_llm="auto",
    model="small-int8",
    template="er_discharge",
    denoiser=BackendId.NONE.value,
    denoiser_config=None,
    validate_terms="auto",
    root=None,
    log=print,
):
    """Run stages 0->5 once and return every stage's output.

    This is the single orchestration the CLI and the HTTP server both call, so
    a server run cannot drift from the shipped chain.  ``root`` relocates the
    output tree (``_state``/``cleaned``/``transcripts``/``entities``/``exports``)
    so a server can keep its jobs out of the committed demo data.
    """
    root = _HERE if root is None else str(root)
    state_path = os.path.join(root, "_state", "idempotency.json")
    raw_key = key or hashlib.sha1(os.path.basename(inp).encode()).hexdigest()[:8]
    job_id, dup = _idempotency_get_or_create(raw_key, state_path)

    cleaned = os.path.join(root, "cleaned", f"{job_id}.wav")
    t_path = os.path.join(root, "transcripts", f"{job_id}.json")
    e_path = os.path.join(root, "entities", f"{job_id}.entities.json")

    log(f"[0] job {job_id} duplicate={dup} <- {inp}")
    backend = BackendId(denoiser)
    config = _load_denoiser_config(denoiser_config, backend)
    os.makedirs(os.path.dirname(cleaned), exist_ok=True)
    meta = clean_audio(inp, cleaned, backend=backend, config=config)
    log(
        f"[1] cleaned -> {cleaned} vad={meta['vad_ratio']} "
        f"backend={meta['actual_backend']} variant={meta['variant_id']} "
        f"rms {meta['rms_dbfs_before']}->{meta['rms_dbfs_after']}dBFS"
    )

    transcript_json, entities_json = run_stt_extract(
        cleaned, job_id, use_llm=use_llm, model=model, validate_terms=validate_terms
    )
    log(f"[2] STT engine={transcript_json['stt_engine']} lang={transcript_json['language']}")
    log(f"[3] normalized: {transcript_json['normalized_en'][:100]}")
    log(f"[4] drugs={len(entities_json['drugs'])} symptoms={len(entities_json['symptoms'])} "
        f"vitals={len(entities_json['vitals'])} allergies={len(entities_json['allergies'])}")

    for p, obj in ((t_path, transcript_json), (e_path, entities_json)):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
    meta_path = os.path.join(root, "cleaned", f"{job_id}.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log(f"wrote {t_path}\nwrote {e_path}")

    result = {
        "job_id": job_id,
        "duplicate": dup,
        "source": inp,
        "requested": {
            "denoiser": backend.value,
            "model": model,
            "use_llm": use_llm,
            "template": template,
            "validate_terms": validate_terms,
        },
        "clean": meta,
        "transcript": transcript_json,
        "entities": entities_json,
        "paths": {
            "cleaned_wav": cleaned,
            "clean_meta": meta_path,
            "transcript": t_path,
            "entities": e_path,
        },
        "template_error": "",
    }

    if template != "none":
        try:
            from fill_template import fill_template
            out = fill_template(entities_json, transcript_json, template)
            h_path = os.path.join(root, "exports", f"{job_id}.html")
            os.makedirs(os.path.dirname(h_path), exist_ok=True)
            with open(h_path, "w", encoding="utf-8") as f:
                f.write(out["html"])
            log(f"[5] template {template} -> {h_path}")
            result["paths"]["html"] = h_path
            if out.get("docx_bytes"):
                d_path = os.path.join(root, "exports", f"{job_id}.docx")
                with open(d_path, "wb") as f:
                    f.write(out["docx_bytes"])
                log(f"[5] docx -> {d_path}")
                result["paths"]["docx"] = d_path
        except Exception as e:
            result["template_error"] = f"{type(e).__name__}: {e}"
            log(f"[5] template skipped: {e}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=None)
    ap.add_argument("--key", default=None)
    ap.add_argument("--use-llm", action="store_true", help="force LLM-primary extraction (fallback regex)")
    ap.add_argument("--no-llm", action="store_true", help="regex only, skip LLM")
    ap.add_argument("--model", default="small-int8", help="tiny-int8 (fast) | base-int8 (better) | small-int8 (default, best CPU, required for usable Hindi/Tamil/code-mix) | medasr (google/medasr greedy, English-only medical CTC) | medasr-lm (medasr + kenlm beam search, English-only, best drug names) | medium | large-v3 (GPU if available, else CPU) | mock (instant)")
    ap.add_argument("--template", default="er_discharge", help="er_discharge | none")
    ap.add_argument(
        "--denoiser",
        choices=[item.value for item in BackendId],
        default=BackendId.NONE.value,
        help="single enhancement backend (default: none)",
    )
    ap.add_argument("--denoiser-config", default=None, help="optional JSON backend config")
    ap.add_argument("--no-validate-terms", action="store_true",
                    help="skip stage 4b semantic validation of fuzzy drug matches")
    ap.add_argument("--gen-samples", action="store_true")
    a = ap.parse_args()

    if a.gen_samples:
        gen_samples(os.path.join(_HERE, "audio_in"))
        return 0

    if not a.inp:
        print("Provide --in audio_in/<file>.wav  (or --gen-samples first)", file=sys.stderr)
        return 2

    mode = False if a.no_llm else (True if a.use_llm else "auto")
    try:
        run_pipeline(
            a.inp,
            key=a.key,
            use_llm=mode,
            model=a.model,
            template=a.template,
            denoiser=a.denoiser,
            denoiser_config=a.denoiser_config,
            validate_terms=False if a.no_validate_terms else "auto",
        )
    except ValueError as e:
        print(f"[pipeline] {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

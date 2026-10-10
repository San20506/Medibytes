"""Variant: SNR-gated denoise -> Whisper (whole script + medical clips) -> MedASR on the clips.

    python -m base_pipeline.run clip.wav --variant whisper-medasr --out runs/wm1

Whisper reads the whole recording and says *where* the medical talk is; MedASR
re-decodes only those stretches, and the usual extractor + verifier run on the
stitched MedASR text.  No second decoder pass, LLM step or discharge note.

MedASR is a CTC model and takes no text prompt, so the Whisper sentence kept with
each clip is context for locating and for the reviewer, not an input to the decode.
"""
import json
import math
import re
import time
import wave
from pathlib import Path

import numpy as np

from .audio_clean import BackendId, clean_audio
from .asr import transcribe, transcribe_medasr
from .extract import (_annotate_entity_spans, _reconcile_allergens, detect_lang_tag,
                      extract_entities, normalize_text)
from .verify import verify

VARIANT = "whisper-medasr"
DEFAULT_SNR_DB = 15.0
DENOISER = "dpdfnet2-onnx"
MAX_CLIP_S = 20.0           # MedASR chunk length; longer slices are split
_VITAL_WORDS = {"bp", "pressure", "pulse", "heart", "rate", "temp", "temperature", "spo2",
                "saturation", "saturating", "oxygen", "respiratory", "resp", "rr", "sats"}
# Medication context. Whisper garbles drug names ("amoxicin"), so a sentence that
# talks about giving, stopping or reacting to a medicine counts even when the
# extractor cannot name the drug.
_MED_CUES = {"started", "start", "advised", "prescribed", "prescribe", "given", "give", "gave",
             "administer", "administered", "dose", "doses", "dosage", "tablet", "tablets",
             "capsule", "capsules", "injection", "infusion", "iv", "intravenous", "oral",
             "medication", "medications", "medicine", "drug", "antibiotic", "antibiotics",
             "allergy", "allergic", "allergies", "mg", "mcg", "ml", "units", "stopped",
             "continue", "continued", "commenced", "considered", "reaction"}
_NUMBER_WORDS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                 "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                 "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
                 "sixty", "seventy", "eighty", "ninety", "hundred", "point", "half"}


def _word(w):
    return re.sub(r"[^a-z0-9.]", "", w.lower())


def _has_number(words):
    return any(any(c.isdigit() for c in w) or w in _NUMBER_WORDS for w in words)


def transcribe_full(wav, model="small-int8", job_id="wm"):
    """(a) Whisper over the whole recording, word timestamps included. Never mocked."""
    return transcribe(str(wav), job_id, model=model, strict=True)


def isolate_medical(whisper, pad_s=1.0, merge_gap_s=1.5, clip_s=MAX_CLIP_S):
    """(b) Time spans of Whisper's medical talk, as [{start, end, whisper_text}].

    A Whisper segment qualifies when the regex extractor finds a drug, vital or
    allergy in it, when it carries a medication cue word, or when a number sits
    beside a vital word.  Whole segments are kept, not just the hit words: the
    drug name may be the very thing Whisper got wrong.  Spans are padded and merged.
    """
    spans = []
    for seg in whisper.get("segments", []):
        text = seg.get("text", "")
        words = [_word(w) for w in text.split()]
        ent = extract_entities(text, normalize_text(text)["normalized_en"], [])
        if (ent["drugs"] or ent["vitals"] or ent["allergies"] or _MED_CUES.intersection(words)
                or (_has_number(words) and _VITAL_WORDS.intersection(words))):
            spans.append({"start": seg["start"], "end": seg["end"], "whisper_text": text})

    merged = []
    for s in sorted(spans, key=lambda s: s["start"]):
        s = {**s, "start": max(0.0, s["start"] - pad_s), "end": s["end"] + pad_s}
        if merged and s["start"] - merged[-1]["end"] <= merge_gap_s:
            merged[-1]["end"] = max(merged[-1]["end"], s["end"])
            merged[-1]["whisper_text"] += " " + s["whisper_text"]
        else:
            merged.append(s)
    out = []
    for m in merged:  # MedASR chunks at 20 s: split long spans into equal parts (no sliver left over)
        parts = max(1, math.ceil((m["end"] - m["start"]) / clip_s))
        step = (m["end"] - m["start"]) / parts
        out += [{"start": m["start"] + i * step, "end": m["start"] + (i + 1) * step,
                 "whisper_text": m["whisper_text"]} for i in range(parts)]
    return out


def _read_wav(path):
    with wave.open(str(path), "rb") as h:
        rate, n = h.getframerate(), h.getnframes()
        pcm = np.frombuffer(h.readframes(n), dtype="<i2")
    return pcm, rate


def _write_slice(pcm, rate, start, end, path):
    cut = pcm[int(start * rate):int(end * rate)]
    with wave.open(str(path), "wb") as h:
        h.setnchannels(1)
        h.setsampwidth(2)
        h.setframerate(rate)
        h.writeframes(cut.tobytes())
    return cut.size / rate


def transcribe_clips(wav, spans, clips_dir):
    """MedASR on each span, stitched in time order into one transcript.

    Zero spans -> MedASR on the whole recording, flagged, so a Whisper miss does not
    silently empty the row.
    """
    if not spans:
        res = transcribe_medasr(str(wav), "wm-full")
        res["engine"] += " (fallback: no medical clip found)"
        res["clips"] = []
        return res
    clips_dir.mkdir(parents=True, exist_ok=True)
    pcm, rate = _read_wav(wav)
    texts, segments, clips = [], [], []
    for i, span in enumerate(spans):
        path = clips_dir / f"{i:02d}.wav"
        dur = _write_slice(pcm, rate, span["start"], span["end"], path)
        res = transcribe_medasr(str(path), f"wm-{i:02d}")
        text = res["text"].strip()
        texts.append(text)
        for s in res["segments"]:
            segments.append({**s, "id": len(segments), "start": round(span["start"] + s["start"], 2),
                             "end": round(span["start"] + s["end"], 2)})
        clips.append({"clip": path.name, "start": round(span["start"], 2),
                      "end": round(span["start"] + dur, 2), "whisper_text": span["whisper_text"],
                      "medasr_text": text})
    return {"text": " ".join(t for t in texts if t), "segments": segments, "clips": clips,
            "engine": f"{res['engine']} on {len(spans)} clip(s)", "language": "en",
            "stt_provenance": {**res["stt_provenance"], "clips": len(spans)}}


def run_variant(src, arm_dir, *, snr_threshold_db=DEFAULT_SNR_DB, whisper_model="small-int8"):
    """One clip through the whole variant; same result shape as `run.run_clip`."""
    from .run import denoiser_config, render_report

    arm_dir.mkdir(parents=True, exist_ok=True)
    cleaned = arm_dir / "cleaned.wav"
    t0 = time.perf_counter()
    meta = clean_audio(src, cleaned, backend=BackendId.NONE)
    snr = meta["snr_db"]
    denoised = snr < snr_threshold_db
    if denoised:
        backend = BackendId(DENOISER)
        meta = clean_audio(src, cleaned, backend=backend, config=denoiser_config(backend))
        meta["snr_db"] = snr  # report the noise that triggered the gate, not the cleaned level
    gate = {"snr_db": snr, "threshold_db": snr_threshold_db, "denoised": denoised,
            "backend": DENOISER if denoised else "none"}
    t_clean = time.perf_counter() - t0

    t0 = time.perf_counter()
    whisper = transcribe_full(cleaned, whisper_model)
    spans = isolate_medical(whisper)
    stt = transcribe_clips(cleaned, spans, arm_dir / "clips")
    t_asr = time.perf_counter() - t0

    norm = normalize_text(stt["text"])
    for s in stt["segments"]:
        s["lang"] = detect_lang_tag(s["text"]) if len(s["text"]) < 200 else norm["lang_tag"]
    ent = extract_entities(stt["text"], norm["normalized_en"], stt["segments"])
    ent = _annotate_entity_spans(_reconcile_allergens(ent), norm["normalized_en"])
    job_id = re.sub(r"[^A-Za-z0-9._-]+", "_", src.stem)
    transcript = {"job_id": job_id, "text": stt["text"], "language": norm["lang_tag"],
                  "segments": stt["segments"], "normalized_en": norm["normalized_en"],
                  "normalizations": norm["normalizations"], "stt_engine": stt["engine"],
                  "stt_provenance": stt["stt_provenance"]}
    entities = {"job_id": job_id, **ent}
    try:
        entities["verify"] = verify(transcript, entities)
    except Exception as err:  # verification must never break a run
        entities["verify"] = {"facts": [], "summary": {"error": f"{type(err).__name__}: {err}"}}

    for name, obj in (("transcript.json", transcript), ("entities.json", entities),
                      ("clean_meta.json", meta), ("verify.json", entities["verify"]),
                      ("gate.json", gate), ("clips.json", stt["clips"]),
                      ("whisper_full.json", {"engine": whisper["engine"], "text": whisper["text"],
                                             "segments": whisper["segments"]})):
        (arm_dir / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    result = {"meta": meta, "transcript": transcript, "entities": entities, "gate": gate,
              "seconds": {"clean": round(t_clean, 2), "asr_extract": round(t_asr, 2)}}
    (arm_dir / "report.md").write_text(render_report(src, VARIANT, result), encoding="utf-8")
    return result

"""MedASR + kenlm beam-search decoder (model option `medasr-lm`).

Greedy CTC picks the best token per frame; on sound-alike drug names
("clavulic acid", "Amalify", "ergenelin") the frame-level winner is often a
near-miss the medical LM would have outscored. Shallow fusion re-ranks with
`lm_6.kenlm`, shipped in the model repo, via upstream pyctcdecode.

Two non-obvious details, both verified empirically on clip 001:
* labels are sentencepiece pieces with `▁` -> space (NOT stripped — stripping
  collides `▁ho`/`ho`, and pyctcdecode rejects duplicate alphabet entries).
* id 0 (`<epsilon>`) is the CTC blank and must be passed as `''`;
  pyctcdecode only auto-detects `<pad>`, otherwise the blank leaks literally.
"""

import contextlib
import os
import wave

import numpy as np

from stt_extract import (
    MEDASR_MODEL_ID,
    _medasr_segments,
    _model_attribute,
    _transformers_version,
    medasr_detokenize,
)

MODEL_REVISION = "ae1e4845b4b07479735d93e1e591e566435b7104"
CHUNK_S = 20.0
STRIDE_S = 2.0
BEAM_WIDTH = 8
# Tuned on in-sample clips 001/002 (C2 drug names + numbers); re-tune here, not
# on held-out audio.
ALPHA = 0.35
BETA = 1.5

_DECODER = {}
_LM_LABELS = {}


def _snapshot_dir():
    from huggingface_hub import snapshot_download

    return snapshot_download(MEDASR_MODEL_ID, revision=MODEL_REVISION)


def _labels(tokenizer, vocab_size):
    key = id(tokenizer)
    if key not in _LM_LABELS:
        pieces = [tokenizer.convert_ids_to_tokens(i) for i in range(vocab_size)]
        labels = [p.replace("▁", " ") for p in pieces]
        labels[0] = ""  # <epsilon> blank (see module docstring)
        _LM_LABELS[key] = labels
    return _LM_LABELS[key]


def _decoder(labels, lm_path):
    if lm_path not in _DECODER:
        from pyctcdecode import build_ctcdecoder

        _DECODER[lm_path] = build_ctcdecoder(labels, kenlm_model_path=lm_path)
    return _DECODER[lm_path]


def transcribe_medasr_lm(clean_wav, job_id="demo-001", device=None,
                          alpha=ALPHA, beta=BETA, beam_width=BEAM_WIDTH):
    """Beam-search decode with kenlm fusion, in the shared STT result shape."""
    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    snap = _snapshot_dir()
    proc = AutoProcessor.from_pretrained(snap, local_files_only=True)
    model = AutoModelForCTC.from_pretrained(snap, local_files_only=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    with contextlib.closing(wave.open(str(clean_wav), "rb")) as handle:
        sr = handle.getframerate()
        n = handle.getnframes()
        pcm = np.frombuffer(handle.readframes(n), dtype="<i2").astype(np.float32) / 32768.0
        duration_s = n / float(sr)
    if sr != 16000:
        raise ValueError(f"medasr-lm expects 16 kHz mono, got {sr} Hz")

    win, stride = int(CHUNK_S * sr), int(STRIDE_S * sr)
    chunks, spans = [], []
    pos = 0
    while pos < len(pcm):
        end = min(pos + win, len(pcm))
        chunks.append(pcm[pos:end])
        spans.append((pos / sr, end / sr))
        if end == len(pcm):
            break
        pos += win - stride

    feats = proc(list(chunks), sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(
            input_features=feats["input_features"].to(device),
            attention_mask=feats.get("attention_mask").to(device)
            if "attention_mask" in feats else None,
        ).logits.float().cpu().numpy()

    labels = _labels(proc.tokenizer, logits.shape[-1])
    decoder = _decoder(labels, os.path.join(snap, "lm_6.kenlm"))
    decoder.reset_params(alpha=alpha, beta=beta)
    parts = []
    for lg, (s, e) in zip(logits, spans):
        fps = lg.shape[0] / max(e - s, 1e-6)
        a = 0 if s == 0 else int(round(fps * (STRIDE_S / 2)))
        edge = e >= duration_s - 0.01
        b = lg.shape[0] if edge else lg.shape[0] - int(round(fps * (STRIDE_S / 2)))
        parts.append(decoder.decode(lg[a:b], beam_width=beam_width))
    raw_text = " ".join(parts)
    text = medasr_detokenize(raw_text)
    provenance = {
        "job_id": job_id,
        "provider": "transformers+pyctcdecode",
        "requested_model": "medasr-lm",
        "actual_model": MEDASR_MODEL_ID,
        "device": device,
        "compute_type": str(getattr(model, "dtype", torch.float32)),
        "word_timestamps": False,
        "temperature": 0.0,
        "beam_size": beam_width,
        "lm": {"kenlm": "lm_6.kenlm", "alpha": alpha, "beta": beta},
        "runtime_version": _transformers_version(),
        "model_hash": None,
        "model_snapshot": MODEL_REVISION,
        "is_mock": False,
    }
    return {"text": text, "raw_text": raw_text,
            "segments": _medasr_segments(text, duration_s),
            "engine": f"medasr-lm:{MEDASR_MODEL_ID}@{MODEL_REVISION[:8]}",
            "language": "en", "stt_provenance": provenance}

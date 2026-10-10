"""`medasr-lm` wiring in the clinical_eval harness, without weights.

Moved from base_pipeline/tests/test_medasr_lm.py: the pipeline package must not import eval code.
"""
import json

import pytest

from clinical_eval import run as eval_run


def test_harness_accepts_medasr_lm_but_keeps_it_out_of_the_default_sweep():
    assert "medasr-lm" in eval_run.ACCEPTED_DECODERS
    assert "medasr-lm" in eval_run.OPTIONAL_DECODERS
    # A 704 MB LM download must not be imposed on every future full run, nor
    # on `/api/eval/run`, which falls back to this tuple.
    assert "medasr-lm" not in eval_run.DECODERS


def _stage_one_clip(tmp_path):
    from clinical_eval.prepare import corpus_root

    root = corpus_root(tmp_path)
    root.mkdir(parents=True)
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"")
    (root / "manifest.json").write_text(json.dumps({"clips": [
        {"clip_id": "C1", "wav_mix": str(wav), "reference_text": "give 500 mg"},
    ]}), encoding="utf-8")
    return tmp_path


def test_run_routes_the_medasr_lm_name_through_to_transcribe(tmp_path, monkeypatch):
    data_root = _stage_one_clip(tmp_path)
    seen = []
    monkeypatch.setattr(eval_run, "_clean", lambda *_a, **_k: {"ok": True, "error": "",
                                                              "seconds": 0.0})

    def fake_transcribe(_wav, job_id=None, model=None, strict=None):
        seen.append((model, strict))
        return {"text": "give 500 mg", "engine": "fake", "segments": []}

    monkeypatch.setattr("base_pipeline.stt_extract.transcribe", fake_transcribe)

    summary = eval_run.run(data_root, denoisers=("none",), decoders=("medasr-lm",),
                           channel="mix")

    assert seen == [("medasr-lm", True)]
    assert "none|medasr-lm" in summary["clips"]["C1"]["arms"]


def test_run_rejects_an_unknown_decoder_instead_of_silently_using_whisper_tiny(tmp_path):
    data_root = _stage_one_clip(tmp_path)

    with pytest.raises(ValueError, match="medasr_lm"):
        eval_run.run(data_root, denoisers=("none",), decoders=("medasr_lm",),
                     channel="mix")

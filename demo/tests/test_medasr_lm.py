"""`medasr-lm`: the fallback contract and the harness wiring, without weights.

The MedASR weights plus the 704 MB kenlm LM cannot be assumed present, so
nothing here may touch `snapshot_download`. What is testable without them is
exactly what was broken: the harness could not select this decoder, and it
could not import it either — `from medasr_lm import ...` is a flat import and
`demo/` was not on `sys.path`.
"""
import json
import os
import sys
from types import ModuleType

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEMO = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_DEMO)
sys.path.insert(0, _DEMO)
sys.path.insert(0, _ROOT)

import stt_extract

from clinical_eval import run as eval_run


def _broken_medasr_lm(monkeypatch, error=RuntimeError("forced no weights")):
    """Stand in for an unloadable model, the way test_stt_contract fakes one."""
    module = ModuleType("medasr_lm")

    def transcribe_medasr_lm(*_args, **_kwargs):
        raise error

    module.transcribe_medasr_lm = transcribe_medasr_lm
    monkeypatch.setitem(sys.modules, "medasr_lm", module)


# ---- the module itself ----

def test_module_imports_without_loading_any_model():
    import medasr_lm

    assert callable(medasr_lm.transcribe_medasr_lm)
    assert medasr_lm.MODEL_REVISION and medasr_lm.BEAM_WIDTH > 1


def test_sentencepiece_labels_keep_the_word_boundary_and_blank_out_epsilon():
    import medasr_lm

    class Tokenizer:
        pieces = ["<epsilon>", "▁ho", "ho", "▁amoxicillin"]

        def convert_ids_to_tokens(self, i):
            return self.pieces[i]

    medasr_lm._LM_LABELS.clear()
    labels = medasr_lm._labels(Tokenizer(), 4)

    # id 0 must be the empty CTC blank, or pyctcdecode emits "<epsilon>".
    assert labels[0] == ""
    # Every piece is prefixed with the sentencepiece marker so pyctcdecode
    # scores it as its own "word", and the piece's own marker becomes `#`,
    # which is what `lm_6.kenlm` uses for a word boundary. Mapping the marker
    # to a space instead - which this module shipped with - left only 234 of
    # the LM's 519 tokens reachable and made fusion worse than greedy.
    assert labels[1] == "\u2581#ho" and labels[2] == "\u2581ho"
    assert labels[3] == "\u2581#amoxicillin"
    # Still distinct: pyctcdecode rejects a duplicated alphabet entry.
    assert len(set(labels)) == len(labels)


def test_restore_text_undoes_the_label_encoding():
    import medasr_lm

    # pyctcdecode emits the pieces space-separated with the marker stripped;
    # joining them and turning `#` back into a space recovers the sentence.
    assert medasr_lm._restore_text("#the #pa tient") == "the patient"
    assert (medasr_lm._restore_text("#There #is #a #mi l d #de f or m ity")
            == "There is a mild deformity")
    # the end-of-sequence token must not survive into the transcript
    assert medasr_lm._restore_text("#done</s>") == "done"



def test_labels_are_cached_per_tokenizer():
    import medasr_lm

    class Tokenizer:
        calls = 0

        def convert_ids_to_tokens(self, i):
            Tokenizer.calls += 1
            return "▁x"

    medasr_lm._LM_LABELS.clear()
    tokenizer = Tokenizer()
    first = medasr_lm._labels(tokenizer, 3)
    second = medasr_lm._labels(tokenizer, 3)

    assert first is second
    assert Tokenizer.calls == 3


# ---- the transcribe() fallback contract, matching the `medasr` path ----

def test_non_strict_medasr_lm_failure_is_an_explicit_labelled_mock(monkeypatch):
    _broken_medasr_lm(monkeypatch)

    result = stt_extract.transcribe("input.wav", model="medasr-lm", strict=False)

    assert result["text"] in stt_extract.MOCK_TEXTS.values()
    assert result["engine"] == "mock (no medasr-lm: RuntimeError)"
    assert result["stt_provenance"]["is_mock"] is True
    assert result["stt_provenance"]["requested_model"] == "medasr-lm"
    assert result["stt_provenance"]["actual_model"] == "mock"


def test_missing_module_also_degrades_rather_than_crashing(monkeypatch):
    """The real failure mode on a box without the weights or pyctcdecode."""
    monkeypatch.setitem(sys.modules, "medasr_lm", None)  # forces ImportError

    result = stt_extract.transcribe("input.wav", model="medasr-lm", strict=False)

    assert result["engine"].startswith("mock (no medasr-lm:")
    assert result["stt_provenance"]["is_mock"] is True


def test_strict_medasr_lm_failure_propagates_without_mock_leakage(monkeypatch):
    _broken_medasr_lm(monkeypatch)

    with pytest.raises(RuntimeError, match="forced no weights"):
        stt_extract.transcribe("input.wav", model="medasr-lm", strict=True)


def test_fallback_matches_the_medasr_path_it_is_modelled_on(monkeypatch):
    """Same shape as `medasr`, so the scorer cannot tell them apart by schema."""
    _broken_medasr_lm(monkeypatch)
    monkeypatch.setattr(
        stt_extract, "transcribe_medasr",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("forced no weights")))

    lm = stt_extract.transcribe("input.wav", model="medasr-lm", strict=False)
    plain = stt_extract.transcribe("input.wav", model="medasr", strict=False)

    assert lm.keys() == plain.keys()
    assert lm["stt_provenance"].keys() == plain["stt_provenance"].keys()
    assert lm["text"] == plain["text"]


# ---- the clinical_eval harness ----

def test_harness_accepts_medasr_lm_but_keeps_it_out_of_the_default_sweep():
    assert "medasr-lm" in eval_run.ACCEPTED_DECODERS
    assert "medasr-lm" in eval_run.OPTIONAL_DECODERS
    # A 704 MB LM download must not be imposed on every future full run, nor
    # on `/api/eval/run`, which falls back to this tuple.
    assert "medasr-lm" not in eval_run.DECODERS


def test_demo_dir_is_put_on_sys_path_so_the_flat_import_can_resolve(monkeypatch):
    monkeypatch.setattr(sys, "path", [
        p for p in sys.path if os.path.abspath(p) != os.path.abspath(_DEMO)])
    eval_run._ensure_demo_on_path()

    assert any(os.path.abspath(p) == os.path.abspath(_DEMO) for p in sys.path)


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

    monkeypatch.setattr("demo.stt_extract.transcribe", fake_transcribe)

    summary = eval_run.run(data_root, denoisers=("none",), decoders=("medasr-lm",),
                           channel="mix")

    assert seen == [("medasr-lm", True)]
    assert "none|medasr-lm" in summary["clips"]["C1"]["arms"]


def test_run_rejects_an_unknown_decoder_instead_of_silently_using_whisper_tiny(tmp_path):
    data_root = _stage_one_clip(tmp_path)

    with pytest.raises(ValueError, match="medasr_lm"):
        eval_run.run(data_root, denoisers=("none",), decoders=("medasr_lm",),
                     channel="mix")


# ---- the known blocker, pinned without weights ----

def test_a_real_wav_can_be_read(tmp_path, monkeypatch):
    """The WAV read used `dtype="<i2>"` - a stray `>`, not a numpy dtype - so
    every real decode died with TypeError after the full model load, which is
    why no `medasr-lm` arm had ever produced a transcript. Fixed to `"<i2"`;
    this test reaches the processor without needing the weights."""
    import wave

    import numpy as np

    import medasr_lm

    path = tmp_path / "t.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(np.zeros(16000, dtype="<i2").tobytes())

    class Sentinel(Exception):
        pass

    class FakeModel:
        dtype = "float32"

        def to(self, _device):
            return self

        def eval(self):
            return self

    class FakeProcessor:
        tokenizer = None

        def __call__(self, *_args, **_kwargs):
            raise Sentinel("reached feature extraction")

    monkeypatch.setattr(medasr_lm, "_snapshot_dir", lambda: str(tmp_path))
    import transformers

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained",
                        classmethod(lambda _cls, *_a, **_k: FakeProcessor()))
    monkeypatch.setattr(transformers.AutoModelForCTC, "from_pretrained",
                        classmethod(lambda _cls, *_a, **_k: FakeModel()))

    # With the dtype fixed, the WAV read succeeds and the call reaches the
    # faked processor, which raises Sentinel.
    with pytest.raises(Sentinel):
        medasr_lm.transcribe_medasr_lm(path)

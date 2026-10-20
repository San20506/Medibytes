from pathlib import Path

import pytest

from whisper_demo import stt

REPO = Path(__file__).resolve().parents[2]
CLIP = REPO / "fixtures" / "audio" / "luvvoice.com-20260921-LnWlf1.mp3"
CACHE = Path.home() / ".cache" / "huggingface" / "hub" / "models--Systran--faster-whisper-large-v3"


def test_confidence_flags_silence_like_segments():
    assert stt.segment_confidence(-0.2, 0.9, "Thank you.") == 0.2
    assert stt.segment_confidence(-0.2, 0.1, "thank you thank you thank you") == 0.2
    assert stt.segment_confidence(-0.2, 0.1, "BP 130 by 80") == 0.9
    assert stt.segment_confidence(None, None, "text") == 0.88


def test_unknown_language_is_refused_before_loading_anything():
    with pytest.raises(ValueError, match="unknown language"):
        stt.transcribe(None, language="fr")


@pytest.mark.skipif(not CACHE.is_dir() or not CLIP.is_file(),
                    reason="large-v3 is not in the Hugging Face cache or the fixture is missing")
def test_real_large_v3_decodes_a_clinical_clip():
    from whisper_demo import audio

    samples, _ = audio.load_audio(CLIP)
    result = stt.transcribe(samples)
    print("device:", result["stt_provenance"]["device"], result["stt_provenance"]["compute_type"])
    assert "azithromycin" in result["text"].lower()
    assert result["stt_provenance"]["is_mock"] is False
    assert result["stt_provenance"]["device"] in ("cuda", "cpu")

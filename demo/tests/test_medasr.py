"""MedASR dictation-markup handling.

MedASR does not emit punctuation characters, it emits the dictation command that
produced them. Every one of those commands is a word to the evaluation tokeniser
(`{period}` -> "period", `</s>` -> "s"), so a regression here silently inflates
WER and pollutes entity extraction rather than failing loudly.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stt_extract


def test_spoken_punctuation_becomes_real_punctuation():
    text = stt_extract.medasr_detokenize(
        "No saddle embolus {comma} lungs clear {period}"
    )

    assert text == "No saddle embolus, lungs clear."


def test_section_headers_and_end_token_are_removed():
    text = stt_extract.medasr_detokenize(
        "[EXAM TYPE] CT chest {period} [IMPRESSION] {colon} Acute PE {period}</s>"
    )

    assert text == "CT chest.: Acute PE."
    assert "[" not in text and "]" not in text and "</s>" not in text


def test_unknown_directive_is_dropped_not_spelled_out():
    text = stt_extract.medasr_detokenize("the patient {bold} denies pain {period}")

    assert text == "the patient denies pain."


def test_paragraph_directive_keeps_a_break():
    assert stt_extract.medasr_detokenize("one {new paragraph} two") == "one \n\n two"


def test_stray_bracket_from_a_truncated_directive_does_not_survive():
    assert stt_extract.medasr_detokenize("] the UN also hopes") == "the UN also hopes"


@pytest.mark.parametrize("text", ["", "   ", "</s>"])
def test_empty_and_token_only_output_detokenises_to_nothing(text):
    assert stt_extract.medasr_detokenize(text) == ""


def test_detokenised_markup_leaves_no_spurious_evaluation_tokens():
    from eval.metrics import evaluation_tokens

    raw = "[FINDINGS] {colon} Lungs clear {period} No effusion {period}</s>"

    assert evaluation_tokens(stt_extract.medasr_detokenize(raw)) == [
        "lungs", "clear", "no", "effusion",
    ]

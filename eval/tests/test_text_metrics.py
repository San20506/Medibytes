from __future__ import annotations

import pytest

from eval.metrics import (
    evaluation_tokens,
    frozen_normalize,
    levenshtein_counts,
    transcript_metrics,
)


def test_unicode_tokenization_handles_english_hindi_tamil_and_code_mix() -> None:
    text = "Café नमस्ते தமிழ் typedef और union"
    assert evaluation_tokens(text) == [
        "café",
        "नमस्ते",
        "தமிழ்",
        "typedef",
        "और",
        "union",
    ]


def test_nfc_composition_is_folded_before_tokenization() -> None:
    assert evaluation_tokens("Cafe\u0301") == ["café"]


def test_levenshtein_counts_are_explicit_and_deterministic() -> None:
    counts = levenshtein_counts(
        ["a", "b", "c"],
        ["a", "x", "c", "d"],
    )
    assert counts.as_dict() == {
        "substitutions": 1,
        "deletions": 0,
        "insertions": 1,
    }
    assert counts.edits == 2


def test_raw_and_frozen_normalized_wer_are_reported_separately() -> None:
    result = transcript_metrics("BID", "twice daily")
    assert result["raw"]["numerator"] == 2
    assert result["raw"]["denominator"] == 1
    assert result["raw"]["wer"] == 2.0
    assert result["normalized"]["numerator"] == 0
    assert result["normalized"]["wer"] == 0.0
    assert result["normalized"]["exact_match"] is True
    assert frozen_normalize("BID") == "twice daily"
    assert result["raw"]["substitutions"] == 1
    assert result["raw"]["insertions"] == 1
    assert result["raw"]["deletions"] == 0


def test_code_mix_punctuation_and_whitespace_do_not_create_tokens() -> None:
    assert evaluation_tokens("  नमस्ते,\tதமிழ்?!  ") == ["नमस्ते", "தமிழ்"]

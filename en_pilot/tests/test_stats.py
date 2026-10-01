"""Behavioural tests for the paired bootstrap and the pre-declared rule."""

from __future__ import annotations

import pytest

from en_pilot import config
from en_pilot import stats
from en_pilot.stats import StatsError, bootstrap_paired, decide
from eval.reporting import cluster_bootstrap


def _groups(values):
    return {f"sample-{index:03d}": value for index, value in enumerate(values, 1)}


def test_identical_zero_deltas_collapse_the_interval():
    result = bootstrap_paired(_groups([0.0] * config.BASE_COUNT))
    assert result["group_count"] == config.BASE_COUNT
    assert result["ci95_low"] == result["ci95_high"] == result["observed_mean_delta"] == 0.0
    # Every resample equals the observation, so the centred exceedance is total.
    assert result["p_value"] == 1.0


def test_all_negative_deltas_put_the_whole_interval_below_zero():
    result = bootstrap_paired(_groups([-0.02] * config.BASE_COUNT))
    assert result["ci95_high"] < 0.0
    assert result["observed_mean_delta"] == pytest.approx(-0.02)


def test_bootstrap_matches_the_frozen_estimator_at_its_own_group_count():
    # The reimplementation exists only because the frozen one hard-codes 15
    # groups; at 15 groups the two must be numerically identical.
    values = _groups([(-1) ** index * (index % 7) / 100.0 for index in range(15)])
    ours = bootstrap_paired(values, expected_groups=15)
    theirs = cluster_bootstrap(values)
    assert ours["ci95_low"] == theirs["ci95_low"]
    assert ours["ci95_high"] == theirs["ci95_high"]
    assert ours["observed_mean_delta"] == theirs["observed_mean_delta"]
    assert ours["p_value"] == theirs["p_value"]


@pytest.mark.parametrize("count", [1, 14, 15, 39, 41])
def test_group_count_is_this_study_not_the_frozen_fifteen(count):
    with pytest.raises(StatsError, match="40 base groups"):
        bootstrap_paired(_groups([0.0] * count))


def test_non_finite_deltas_are_refused():
    values = _groups([0.0] * config.BASE_COUNT)
    values["sample-001"] = float("nan")
    with pytest.raises(StatsError, match="finite"):
        bootstrap_paired(values)


def test_adopt_requires_proof_of_no_harm_and_at_most_two_points():
    assert decide(-0.01, 0.0) == "adopt"
    assert decide(-0.02, 0.0) == "adopt"
    assert decide(-0.03, 0.0) == "reject"
    assert decide(-0.01, 0.01) == "reject"
    assert decide(-0.03, -0.01) == "reject"
    assert decide(0.0, 0.0) == "adopt"


def test_sentence_aggregation_collapses_repeated_content():
    from en_pilot.stats import aggregate_by_sentence

    deltas = {"sample-001": -0.01, "sample-002": -0.03, "sample-003": 0.01}
    sentence_of = {"sample-001": "1510", "sample-002": "1510", "sample-003": "1620"}
    collapsed = aggregate_by_sentence(deltas, sentence_of)
    assert collapsed == {"1510": pytest.approx(-0.02), "1620": pytest.approx(0.01)}
    # One cluster fewer than bases: resampling bases would treat the two
    # speakers of sentence 1510 as independent evidence.
    assert len(collapsed) == 2 < len(deltas)


def test_sentence_aggregation_refuses_an_unknown_base():
    from en_pilot.stats import aggregate_by_sentence

    with pytest.raises(StatsError, match="no sentence identity"):
        aggregate_by_sentence({"sample-001": -0.01}, {})


def test_zero_db_conditions_are_not_mistaken_for_clean_floor_rows():
    """`snr_target_db == 0.0` is the hardest noisy band, not an absent target."""
    rows = [
        {"decoder_model": "base", "backend": "none", "speech": "s1",
         "snr_target_db": 0.0, "wer": 0.4},
        {"decoder_model": "base", "backend": "none", "speech": "s1",
         "snr_target_db": 5.0, "wer": 0.2},
        {"decoder_model": "base", "backend": "cand", "speech": "s1",
         "snr_target_db": 0.0, "wer": 0.8},
        {"decoder_model": "base", "backend": "cand", "speech": "s1",
         "snr_target_db": 5.0, "wer": 0.2},
        {"decoder_model": "base", "backend": "none", "speech": "s1",
         "snr_target_db": "", "wer": 0.0},
    ]

    deltas = stats.per_base_deltas(rows, backend="cand", decoder="base", metric="wer")

    # mean(0.8, 0.2) - mean(0.4, 0.2) = 0.5 - 0.3; dropping the 0 dB pair gives 0.0.
    assert deltas["s1"] == pytest.approx(0.2)
    assert stats.per_base_values(
        rows, backend="none", decoder="base", metric="wer"
    )["s1"] == pytest.approx(0.3)

"""
Risk bands and queue ordering.

These two conventions were previously duplicated across the API, the batch
scorer and the capacity analysis, and had drifted apart in both: the API and the
Power BI feed banded a score of exactly 0.70 differently, and the batch scorer
ranked on a rounded score while the capacity analysis ranked on the raw one.

Both defects were invisible without tests, because each implementation was
self-consistent.
"""

import numpy as np
import pytest

from scoring import (
    ELEVATED_THRESHOLD,
    HIGH_THRESHOLD,
    RISK_BANDS,
    rank_order,
    risk_band,
    risk_bands,
)


# --- banding -----------------------------------------------------------------

@pytest.mark.parametrize("probability,expected", [
    (0.0, "low"),
    (0.2999, "low"),
    (0.30, "elevated"),      # boundary the two implementations disagreed on
    (0.50, "elevated"),
    (0.6999, "elevated"),
    (0.70, "high"),          # boundary the two implementations disagreed on
    (0.99, "high"),
    (1.0, "high"),
])
def test_risk_band_boundaries_are_inclusive_lower_bounds(probability, expected):
    assert risk_band(probability) == expected


def test_vectorised_banding_matches_the_scalar_version():
    """The batch path and the single-score path must never disagree."""
    probabilities = np.linspace(0.0, 1.0, 501)
    assert list(risk_bands(probabilities)) == [risk_band(p) for p in probabilities]


def test_vectorised_banding_agrees_on_exact_boundaries():
    edges = np.array([ELEVATED_THRESHOLD, HIGH_THRESHOLD])
    assert list(risk_bands(edges)) == ["elevated", "high"]


def test_every_band_produced_is_a_declared_band():
    probabilities = np.linspace(0.0, 1.0, 101)
    assert set(risk_bands(probabilities)) <= set(RISK_BANDS)


def test_banding_is_monotone():
    """A higher score can never land in a lower band."""
    order = {b: i for i, b in enumerate(RISK_BANDS)}
    probabilities = np.linspace(0.0, 1.0, 200)
    ranks = [order[risk_band(p)] for p in probabilities]
    assert ranks == sorted(ranks)


def test_empty_input_bands_cleanly():
    assert len(risk_bands([])) == 0


# --- queue ordering ----------------------------------------------------------

def test_queue_is_ordered_by_score_descending():
    scores = np.array([0.1, 0.9, 0.5])
    amounts = np.array([10.0, 20.0, 30.0])
    assert list(rank_order(scores, amounts)) == [1, 2, 0]


def test_ties_are_broken_by_amount_descending():
    """At equal risk, the larger exposure is reviewed first."""
    scores = np.array([0.5, 0.5, 0.5])
    amounts = np.array([10.0, 300.0, 50.0])
    assert list(rank_order(scores, amounts)) == [1, 2, 0]


def test_score_outranks_amount():
    """A large amount must not jump ahead of a genuinely higher score."""
    scores = np.array([0.2, 0.8])
    amounts = np.array([1_000_000.0, 1.0])
    assert list(rank_order(scores, amounts)) == [1, 0]


def test_ordering_is_a_permutation():
    rng = np.random.default_rng(0)
    scores = rng.random(500)
    amounts = rng.random(500) * 1000
    order = rank_order(scores, amounts)
    assert sorted(order.tolist()) == list(range(500))


def test_ordering_is_deterministic_under_heavy_ties():
    """
    The failure this project documents: with many identical scores, an
    unspecified tie-break makes "the top K" implementation-dependent.
    """
    rng = np.random.default_rng(1)
    scores = np.round(rng.random(2_000), 2)   # forces many exact ties
    amounts = rng.random(2_000) * 1_000_000
    first = rank_order(scores, amounts)
    for _ in range(5):
        assert np.array_equal(rank_order(scores, amounts), first)


def test_ordering_is_stable_against_input_permutation():
    """
    Shuffling the input rows must not change WHICH transactions make the queue,
    only their positions in the input array.
    """
    rng = np.random.default_rng(2)
    scores = np.round(rng.random(400), 2)
    amounts = np.round(rng.random(400) * 1000, 2)

    top_original = {(scores[i], amounts[i]) for i in rank_order(scores, amounts)[:50]}

    shuffle = rng.permutation(400)
    s2, a2 = scores[shuffle], amounts[shuffle]
    top_shuffled = {(s2[i], a2[i]) for i in rank_order(s2, a2)[:50]}

    assert top_original == top_shuffled


def test_rounding_before_ranking_changes_the_queue():
    """
    Documents the bug that was fixed: ranking on a rounded score invents ties
    the model never produced, and the amount tie-break then reorders them.
    """
    scores = np.array([0.50000004, 0.50000001])
    amounts = np.array([1.0, 999.0])

    raw = rank_order(scores, amounts)
    rounded = rank_order(np.round(scores, 6), amounts)

    assert list(raw) == [0, 1]        # the genuinely higher score wins
    assert list(rounded) == [1, 0]    # rounding hands it to the larger amount
    assert not np.array_equal(raw, rounded)


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        rank_order(np.array([0.1, 0.2]), np.array([1.0]))


def test_empty_queue_is_allowed():
    assert len(rank_order(np.array([]), np.array([]))) == 0

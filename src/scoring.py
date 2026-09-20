"""
Operational scoring conventions, defined once.

Two decisions turn a raw probability into something a fraud team acts on:

  1. Which risk band a score falls in.
  2. How the review queue is ordered when scores tie.

Both were previously implemented separately in the API, the batch scorer, and
the capacity analysis, and all three had drifted:

  * The API banded on `p >= 0.7`, while the Power BI feed used a pandas `cut`
    with right-closed bins, so `p == 0.7` was "high" in one and "elevated" in
    the other. The same transaction got two different bands depending on which
    surface an analyst was looking at.

  * The capacity analysis ranked on raw scores; the batch scorer rounded to six
    decimal places first and ranked on the rounded value, which merges scores
    that differ in the seventh decimal into a tie and hands them to the
    amount tie-break. That is precisely the ambiguity the project set out to
    document, reproduced inside the project.

Anything that bands or ranks now goes through this module.
"""

import numpy as np

# Band edges are LOWER bounds, inclusive: a score of exactly 0.30 is elevated
# and exactly 0.70 is high. Stated explicitly because the boundary cases are
# where the two previous implementations disagreed.
ELEVATED_THRESHOLD = 0.30
HIGH_THRESHOLD = 0.70

RISK_BANDS = ("low", "elevated", "high")


def risk_band(probability: float) -> str:
    """Band a single fraud probability. Inclusive lower bounds."""
    if probability >= HIGH_THRESHOLD:
        return "high"
    if probability >= ELEVATED_THRESHOLD:
        return "elevated"
    return "low"


def risk_bands(probabilities) -> np.ndarray:
    """Vectorised `risk_band`, with identical boundary behaviour."""
    p = np.asarray(probabilities, dtype=float)
    return np.where(
        p >= HIGH_THRESHOLD, "high",
        np.where(p >= ELEVATED_THRESHOLD, "elevated", "low"),
    )


def rank_order(scores, amounts) -> np.ndarray:
    """
    Indices that order a review queue, most-risky first.

    The model assigns many transactions identical scores, so "the top K" is
    undefined unless the tie-break is stated. Ties go to the larger exposure:
    at equal risk, review the bigger transaction first.

    Ranking uses the UNROUNDED score. Rounding before sorting would invent
    ties that the model did not produce, which shifts the reported catch rate
    with no change to the model at all.

    `np.lexsort` takes its last key as primary, hence the ordering below.
    """
    scores = np.asarray(scores, dtype=float)
    amounts = np.asarray(amounts, dtype=float)
    if scores.shape != amounts.shape:
        raise ValueError(
            f"scores and amounts must be the same length, "
            f"got {scores.shape} and {amounts.shape}")
    return np.lexsort((-amounts, -scores))

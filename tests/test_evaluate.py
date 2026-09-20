"""
Metrics.

The project's argument is carried entirely by these numbers, so they are checked
against hand-computed cases rather than against themselves.
"""

import numpy as np
import pytest

from evaluate import best_f1_threshold, metrics_table, score_metrics


def test_confusion_counts_are_correct():
    y_true = np.array([1, 1, 0, 0, 0])
    y_score = np.array([0.9, 0.2, 0.8, 0.1, 0.1])
    m = score_metrics(y_true, y_score, threshold=0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 2)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)


def test_threshold_is_an_inclusive_lower_bound():
    """A score exactly at the threshold counts as a positive prediction."""
    y_true = np.array([1, 0])
    y_score = np.array([0.5, 0.4])
    assert score_metrics(y_true, y_score, threshold=0.5)["tp"] == 1


def test_perfect_separation_scores_one():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.01, 0.02, 0.98, 0.99])
    m = score_metrics(y_true, y_score, threshold=0.5)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)


def test_pr_auc_of_a_random_scorer_approaches_the_base_rate():
    """
    The no-skill floor the README quotes. A scorer carrying no information
    should land near the positive class prevalence, not near 0.5.
    """
    rng = np.random.default_rng(0)
    y_true = (rng.random(20_000) < 0.02).astype(int)
    y_score = rng.random(20_000)
    m = score_metrics(y_true, y_score, threshold=0.5)
    assert m["pr_auc"] == pytest.approx(y_true.mean(), abs=0.01)


def test_predicting_all_negative_yields_zero_not_an_error():
    """The degenerate case the project exists to warn about."""
    y_true = np.array([1, 0, 0, 0])
    y_score = np.array([0.1, 0.1, 0.1, 0.1])
    m = score_metrics(y_true, y_score, threshold=0.9)
    assert m["tp"] == 0
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_accuracy_is_never_reported():
    """
    Deliberate omission, stated in the module docstring and the README: at this
    class balance accuracy carries no information.
    """
    m = score_metrics(np.array([1, 0]), np.array([0.9, 0.1]), threshold=0.5)
    assert "accuracy" not in m


def test_hard_label_rules_report_no_ranking_metrics():
    """A rule emits 0/1, so PR-AUC and a threshold are meaningless for it."""
    y_true = np.array([1, 1, 0, 0])
    rule = np.array([1, 0, 0, 0])
    m = score_metrics(y_true, rule, name="rule", is_hard_label=True)
    assert m["threshold"] is None
    assert "pr_auc" not in m
    assert m["tp"] == 1
    assert m["fp"] == 0
    assert m["precision"] == 1.0


def test_best_f1_threshold_finds_the_separating_cut():
    y_true = np.array([0, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    thr = best_f1_threshold(y_true, y_score)
    predicted = (y_score >= thr).astype(int)
    assert list(predicted) == [0, 0, 0, 1, 1]


def test_best_f1_threshold_beats_the_default_half():
    """If the chosen threshold were not better than 0.5, the search is pointless."""
    rng = np.random.default_rng(3)
    y_true = (rng.random(4_000) < 0.05).astype(int)
    y_score = np.clip(rng.normal(0.3, 0.15, 4_000) + y_true * 0.25, 0, 1)

    thr = best_f1_threshold(y_true, y_score)
    tuned = score_metrics(y_true, y_score, threshold=thr)["f1"]
    default = score_metrics(y_true, y_score, threshold=0.5)["f1"]
    assert tuned >= default


def test_best_f1_threshold_survives_a_degenerate_scorer():
    y_true = np.array([1, 0, 1, 0])
    y_score = np.zeros(4)
    assert isinstance(best_f1_threshold(y_true, y_score), float)


def test_metrics_table_renders_rule_rows_without_pr_auc():
    rows = [
        score_metrics(np.array([1, 0]), np.array([1, 0]),
                      name="rule", is_hard_label=True),
        score_metrics(np.array([1, 0]), np.array([0.9, 0.1]),
                      threshold=0.5, name="model"),
    ]
    table = metrics_table(rows)
    assert "| rule |" in table
    assert "| model |" in table
    # The rule row has no PR-AUC and must render a placeholder, not crash.
    assert table.count("\n") == len(rows) + 1
    assert " - " in table

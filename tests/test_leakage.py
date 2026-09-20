"""
The leakage guards.

This project's claim is that Track D is free of the three artifacts it names.
That claim currently rests on a list of strings in features.py being correct and
staying correct - one careless addition to TRACK_D_FEATURES silently invalidates
every number in the README.

These tests make the claim executable.
"""

import pytest

from features import (
    TRACK_A_FEATURES,
    TRACK_B_FEATURES,
    TRACK_C_FEATURES,
    TRACK_D_FEATURES,
    add_base_features,
    apply_dest_frequency,
    fit_dest_frequency,
    subsample_majority,
    temporal_split_step,
)

# Columns that encode one of the three measured artifacts. Any of these
# appearing in Track D means the honest number is not honest.
ORIGIN_DRAIN_COLUMNS = {
    "oldbalanceOrg", "newbalanceOrig", "errorBalanceOrig",
    "balance_ratio", "is_full_drain", "orig_emptied",
}
DEST_BALANCE_LEAK_COLUMNS = {"newbalanceDest", "errorBalanceDest"}
TIMING_ARTIFACT_COLUMNS = {"hour_of_day"}


def test_track_d_carries_no_origin_drain_signature():
    assert not (set(TRACK_D_FEATURES) & ORIGIN_DRAIN_COLUMNS)


def test_track_d_carries_no_destination_balance_leak():
    assert not (set(TRACK_D_FEATURES) & DEST_BALANCE_LEAK_COLUMNS)


def test_track_d_carries_no_timing_artifact():
    assert not (set(TRACK_D_FEATURES) & TIMING_ARTIFACT_COLUMNS)


def test_track_c_removes_only_the_destination_leak():
    """Track C is Track B minus the destination-balance columns, nothing else."""
    assert set(TRACK_B_FEATURES) - set(TRACK_C_FEATURES) == DEST_BALANCE_LEAK_COLUMNS


def test_track_b_removes_only_the_origin_drain():
    assert set(TRACK_A_FEATURES) - set(TRACK_B_FEATURES) == ORIGIN_DRAIN_COLUMNS


def test_track_d_removes_only_the_timing_artifact():
    assert set(TRACK_C_FEATURES) - set(TRACK_D_FEATURES) == TIMING_ARTIFACT_COLUMNS


def test_tracks_are_strictly_nested():
    """
    The whole design depends on nesting: the PR-AUC gap between two tracks only
    prices one artifact if the feature sets differ by exactly that artifact.
    """
    assert set(TRACK_D_FEATURES) < set(TRACK_C_FEATURES)
    assert set(TRACK_C_FEATURES) < set(TRACK_B_FEATURES)
    assert set(TRACK_B_FEATURES) < set(TRACK_A_FEATURES)


def test_no_track_contains_the_label_or_its_proxy():
    for features in (TRACK_A_FEATURES, TRACK_B_FEATURES,
                     TRACK_C_FEATURES, TRACK_D_FEATURES):
        assert "isFraud" not in features
        assert "isFlaggedFraud" not in features


def test_feature_lists_have_no_duplicates():
    for features in (TRACK_A_FEATURES, TRACK_B_FEATURES,
                     TRACK_C_FEATURES, TRACK_D_FEATURES):
        assert len(features) == len(set(features))


# --- the artifacts are real, and the guards are not vacuous -----------------

def test_the_drain_signature_is_present_in_the_fixture(raw_frame):
    """
    Guard on the guards: if synthetic fraud did not carry the drain signature,
    the leakage tests above would pass against data that cannot leak.
    """
    df = add_base_features(raw_frame)
    fraud = df[df.isFraud == 1]
    legit = df[df.isFraud == 0]
    assert fraud["is_full_drain"].mean() > 0.95
    assert legit["is_full_drain"].mean() < 0.05


def test_the_destination_leak_is_present_in_the_fixture(raw_frame):
    transfers = raw_frame[raw_frame.type == "TRANSFER"]
    fraud = transfers[transfers.isFraud == 1]
    legit = transfers[transfers.isFraud == 0]
    assert (fraud["newbalanceDest"] == 0).mean() > 0.95
    assert (legit["newbalanceDest"] == 0).mean() < 0.05


# --- temporal integrity ------------------------------------------------------

def test_split_produces_disjoint_windows(built):
    train_df, test_df, split_step = built
    assert train_df["step"].max() <= split_step
    assert test_df["step"].min() > split_step
    assert not set(train_df["step"]) & set(test_df["step"])


def test_split_lands_near_the_requested_row_fraction(raw_frame):
    """
    The split is on rows, not on the step range - the docstring's whole point.
    Volume in the fixture swings by hour, so a range split would miss badly.
    """
    df = raw_frame[raw_frame.type.isin(("TRANSFER", "CASH_OUT"))]
    step = temporal_split_step(df, train_frac=0.70)
    achieved = (df["step"] <= step).mean()
    assert 0.68 <= achieved <= 0.75


def test_split_is_monotone_in_the_requested_fraction(raw_frame):
    df = raw_frame[raw_frame.type.isin(("TRANSFER", "CASH_OUT"))]
    steps = [temporal_split_step(df, f) for f in (0.3, 0.5, 0.7, 0.9)]
    assert steps == sorted(steps)


def test_no_test_row_precedes_any_train_row(built):
    train_df, test_df, _ = built
    assert train_df["step"].max() < test_df["step"].min()


# --- destination-frequency features are fitted on training only -------------

def test_dest_frequency_counts_only_the_training_window(built):
    train_df, test_df, _ = built
    freq = fit_dest_frequency(train_df)
    # No account may be credited with more appearances than it had in training.
    train_counts = train_df.groupby("nameDest").size()
    assert (freq == train_counts).all()


def test_unseen_destination_accounts_score_zero(built):
    """
    A brand-new account is exactly what a production scorer sees, and it must
    map to 0 rather than to anything derived from the test window.
    """
    train_df, test_df, _ = built
    freq = fit_dest_frequency(train_df)
    applied = apply_dest_frequency(test_df, freq)
    unseen = ~applied["nameDest"].isin(freq.index)
    if unseen.any():
        assert (applied.loc[unseen, "dest_txn_count"] == 0).all()
        assert (applied.loc[unseen, "dest_is_frequent"] == 0).all()


def test_dest_frequency_never_reflects_test_window_volume(built):
    """
    The leak this guards: counting an account across the whole dataset tells
    the model how often it gets used in the future.
    """
    train_df, test_df, _ = built
    freq = fit_dest_frequency(train_df)
    applied = apply_dest_frequency(test_df, freq)
    full_counts = (
        applied.groupby("nameDest").size()
        + train_df.groupby("nameDest").size().reindex(
            applied["nameDest"].unique()).fillna(0)
    )
    per_row_full = applied["nameDest"].map(full_counts)
    # If frequencies had been fitted on everything, at least some test row would
    # carry a count above its training-only value.
    assert (applied["dest_txn_count"] <= per_row_full).all()
    assert (applied["dest_txn_count"] < per_row_full).any()


def test_dest_is_frequent_matches_its_threshold(built):
    train_df, test_df, _ = built
    applied = apply_dest_frequency(test_df, fit_dest_frequency(train_df))
    expected = (applied["dest_txn_count"] >= 6).astype("int8")
    assert applied["dest_is_frequent"].equals(expected)


# --- subsampling touches training only ---------------------------------------

def test_subsample_keeps_every_fraud_row(built):
    train_df, _, _ = built
    out = subsample_majority(train_df, ratio=20)
    assert int(out.isFraud.sum()) == int(train_df.isFraud.sum())


def test_subsample_respects_the_requested_ratio(built):
    train_df, _, _ = built
    out = subsample_majority(train_df, ratio=20)
    n_fraud = int(out.isFraud.sum())
    n_legit = int((out.isFraud == 0).sum())
    assert n_legit <= n_fraud * 20


def test_subsample_is_deterministic(built):
    train_df, _, _ = built
    a = subsample_majority(train_df, ratio=20, seed=7)
    b = subsample_majority(train_df, ratio=20, seed=7)
    assert a.index.tolist() == b.index.tolist()


def test_subsample_draws_only_from_the_frame_it_is_given(built):
    """Subsampling must never be able to pull a test row into training."""
    train_df, test_df, _ = built
    out = subsample_majority(train_df, ratio=20)
    assert set(out.index) <= set(train_df.index)
    assert not set(out.index) & set(test_df.index)


@pytest.mark.parametrize("ratio", [1, 5, 50])
def test_subsample_ratio_is_honoured_across_settings(built, ratio):
    train_df, _, _ = built
    out = subsample_majority(train_df, ratio=ratio)
    n_fraud = int(out.isFraud.sum())
    assert int((out.isFraud == 0).sum()) <= n_fraud * ratio


# --- the threshold-selection window ------------------------------------------

def test_validation_window_is_carved_from_training_only(built):
    """
    The decision threshold is chosen here. If this window overlapped test, the
    threshold would be an oracle - the exact defect this split was added to fix.
    """
    from features import temporal_validation_split

    train_df, test_df, _ = built
    fit_df, val_df = temporal_validation_split(train_df)

    assert set(fit_df.index) <= set(train_df.index)
    assert set(val_df.index) <= set(train_df.index)
    assert not set(val_df.index) & set(test_df.index)
    assert val_df["step"].max() < test_df["step"].min()


def test_validation_window_follows_the_fit_window_in_time(built):
    from features import temporal_validation_split

    train_df, _, _ = built
    fit_df, val_df = temporal_validation_split(train_df)
    assert fit_df["step"].max() < val_df["step"].min()


def test_validation_window_partitions_the_training_window(built):
    from features import temporal_validation_split

    train_df, _, _ = built
    fit_df, val_df = temporal_validation_split(train_df)
    assert len(fit_df) + len(val_df) == len(train_df)
    assert not set(fit_df.index) & set(val_df.index)


def test_validation_window_lands_near_the_requested_fraction(built):
    from features import temporal_validation_split

    train_df, _, _ = built
    _, val_df = temporal_validation_split(train_df, val_frac=0.25)
    assert 0.15 <= len(val_df) / len(train_df) <= 0.35


def test_validation_window_contains_fraud(built):
    """A threshold cannot be chosen on a window with no positives."""
    from features import temporal_validation_split

    train_df, _, _ = built
    _, val_df = temporal_validation_split(train_df)
    assert val_df["isFraud"].sum() > 0

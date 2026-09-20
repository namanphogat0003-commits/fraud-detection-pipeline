"""
Feature derivation.

The three artifacts are *defined* by these columns, so the arithmetic behind
them is checked against hand-worked examples. A sign error in `errorBalanceOrig`
would not crash anything - it would just quietly change what Track A means.
"""

import numpy as np
import pandas as pd
import pytest

from features import (
    FRAUD_TYPES,
    TRACK_A_FEATURES,
    TRACK_D_FEATURES,
    add_base_features,
    build_features,
)


def _row(**overrides):
    base = {
        "step": 25,
        "type": "TRANSFER",
        "amount": 1_000.0,
        "oldbalanceOrg": 5_000.0,
        "newbalanceOrig": 4_000.0,
        "oldbalanceDest": 2_000.0,
        "newbalanceDest": 3_000.0,
        "isFraud": 0,
        "isFlaggedFraud": 0,
        "nameDest": "C1",
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_hour_of_day_wraps_the_step_counter():
    """`step` is hours since simulation start, so hour 25 is 1am on day two."""
    assert add_base_features(_row(step=25))["hour_of_day"].iloc[0] == 1
    assert add_base_features(_row(step=24))["hour_of_day"].iloc[0] == 0
    assert add_base_features(_row(step=23))["hour_of_day"].iloc[0] == 23


def test_log_amount_is_log1p_and_handles_zero():
    out = add_base_features(_row(amount=0.0))
    assert out["log_amount"].iloc[0] == 0.0
    out = add_base_features(_row(amount=999.0))
    assert out["log_amount"].iloc[0] == pytest.approx(np.log1p(999.0))


def test_is_transfer_flags_only_transfers():
    assert add_base_features(_row(type="TRANSFER"))["is_transfer"].iloc[0] == 1
    assert add_base_features(_row(type="CASH_OUT"))["is_transfer"].iloc[0] == 0


def test_origin_reconciliation_gap():
    """oldbalanceOrg - amount - newbalanceOrig, and it balances when consistent."""
    out = add_base_features(
        _row(oldbalanceOrg=5_000.0, amount=1_000.0, newbalanceOrig=4_000.0))
    assert out["errorBalanceOrig"].iloc[0] == pytest.approx(0.0)

    out = add_base_features(
        _row(oldbalanceOrg=5_000.0, amount=1_000.0, newbalanceOrig=0.0))
    assert out["errorBalanceOrig"].iloc[0] == pytest.approx(4_000.0)


def test_destination_reconciliation_gap():
    """oldbalanceDest + amount - newbalanceDest; nonzero when the mule is stiffed."""
    out = add_base_features(
        _row(oldbalanceDest=2_000.0, amount=1_000.0, newbalanceDest=3_000.0))
    assert out["errorBalanceDest"].iloc[0] == pytest.approx(0.0)

    out = add_base_features(
        _row(oldbalanceDest=0.0, amount=1_000.0, newbalanceDest=0.0))
    assert out["errorBalanceDest"].iloc[0] == pytest.approx(1_000.0)


def test_dest_was_empty_flag():
    assert add_base_features(_row(oldbalanceDest=0.0))["dest_was_empty"].iloc[0] == 1
    assert add_base_features(_row(oldbalanceDest=0.01))["dest_was_empty"].iloc[0] == 0


# --- the leaking features ----------------------------------------------------

def test_full_drain_flag_fires_when_the_account_is_emptied():
    """`amount == oldbalanceOrg` - the single largest leak in the dataset."""
    out = add_base_features(_row(amount=5_000.0, oldbalanceOrg=5_000.0))
    assert out["is_full_drain"].iloc[0] == 1


def test_full_drain_flag_tolerates_sub_cent_rounding_only():
    assert add_base_features(
        _row(amount=5_000.005, oldbalanceOrg=5_000.0))["is_full_drain"].iloc[0] == 1
    assert add_base_features(
        _row(amount=5_000.5, oldbalanceOrg=5_000.0))["is_full_drain"].iloc[0] == 0


def test_balance_ratio_reaches_one_on_a_full_drain():
    """Fraud sits at exactly 1.0; the +1 denominator keeps a zero balance safe."""
    out = add_base_features(_row(amount=100_000.0, oldbalanceOrg=100_000.0))
    assert out["balance_ratio"].iloc[0] == pytest.approx(1.0, abs=1e-4)


def test_balance_ratio_survives_a_zero_opening_balance():
    out = add_base_features(_row(amount=500.0, oldbalanceOrg=0.0))
    assert np.isfinite(out["balance_ratio"].iloc[0])


def test_orig_emptied_flag():
    assert add_base_features(_row(newbalanceOrig=0.0))["orig_emptied"].iloc[0] == 1
    assert add_base_features(_row(newbalanceOrig=1.0))["orig_emptied"].iloc[0] == 0


def test_add_base_features_does_not_mutate_its_input():
    df = _row()
    before = df.copy()
    add_base_features(df)
    pd.testing.assert_frame_equal(df, before)


# --- the assembled pipeline --------------------------------------------------

def test_build_features_produces_every_declared_column(synthetic_db):
    train_df, test_df, _ = build_features(synthetic_db)
    for frame in (train_df, test_df):
        for feature in TRACK_A_FEATURES:
            assert feature in frame.columns


def test_build_features_keeps_only_the_fraud_bearing_types(synthetic_db):
    """PaySim only carries fraud in TRANSFER and CASH_OUT rows."""
    train_df, test_df, _ = build_features(synthetic_db)
    for frame in (train_df, test_df):
        assert set(frame["type"].unique()) <= set(FRAUD_TYPES)


def test_build_features_leaves_no_missing_values_in_the_feature_matrix(synthetic_db):
    train_df, test_df, _ = build_features(synthetic_db)
    for frame in (train_df, test_df):
        assert not frame[TRACK_A_FEATURES].isna().any().any()


def test_both_windows_contain_fraud(synthetic_db):
    """Otherwise every metric downstream is undefined rather than merely bad."""
    train_df, test_df, _ = build_features(synthetic_db)
    assert train_df["isFraud"].sum() > 0
    assert test_df["isFraud"].sum() > 0


def test_test_window_keeps_its_natural_class_balance(synthetic_db):
    """
    The test set must never be rebalanced - the README's central method note.
    Fraud should stay a small minority of it.
    """
    _, test_df, _ = build_features(synthetic_db)
    assert 0 < test_df["isFraud"].mean() < 0.2


def test_build_features_is_deterministic(synthetic_db):
    a_train, a_test, a_step = build_features(synthetic_db)
    b_train, b_test, b_step = build_features(synthetic_db)
    assert a_step == b_step
    pd.testing.assert_frame_equal(
        a_train[TRACK_D_FEATURES], b_train[TRACK_D_FEATURES])
    pd.testing.assert_frame_equal(
        a_test[TRACK_D_FEATURES], b_test[TRACK_D_FEATURES])

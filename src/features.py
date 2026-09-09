"""
Week 2 - Feature engineering for the PaySim fraud models.

Four nested feature sets are defined here, and the distance between them is the
whole point of this project. Each track removes one artifact of how PaySim
generates data, measured directly rather than assumed:

  TRACK_A  everything. Includes the full-balance-drain signature
           (amount == oldbalanceOrg), which alone identifies 97.7% of fraud with
           zero false positives. A model given this is reading the label.

  TRACK_B  drops the origin-side drain. Still leaks: newbalanceDest == 0 covers
           99.29% of fraudulent TRANSFERs vs 0.21% of legitimate ones.

  TRACK_C  drops the destination-balance leak too. Keeps hour-of-day.

  TRACK_D  drops hour-of-day as well, on the grounds that its predictive power
           comes from the simulator running fraud at a constant hourly rate
           against a 1,259x day/night swing in legitimate volume.

Track D is the honest floor. Track A is what a PaySim notebook that skips this
analysis will report.

Temporal-leakage rule: the destination-account frequency features are fitted on
the training window only and then applied to the test window. Counting across
the whole dataset would let the model see how often an account gets used in the
future, which it could not know at scoring time.
"""

import numpy as np
import pandas as pd

FRAUD_TYPES = ("TRANSFER", "CASH_OUT")

# The four feature sets are strictly nested. Each step removes one measured
# simulator artifact, so the drop in score between tracks is the price of that
# artifact - which is the actual result this project reports.

# Track D - the pessimistic floor. Only features with no known artifact.
TRACK_D_FEATURES = [
    "log_amount",
    "is_transfer",
    "oldbalanceDest",
    "dest_was_empty",
    "dest_txn_count",
    "dest_is_frequent",
]

# Track C - adds hour-of-day. Defensible, but fragile: PaySim generates fraud at
# a near-constant rate around the clock (274-375/hour) while legitimate volume
# swings 1,259x across the day, so a night-time transaction is suspicious mostly
# because real users are asleep. Real deployment traffic would not look like this.
TRACK_C_FEATURES = TRACK_D_FEATURES + ["hour_of_day"]

# Track B - adds the destination balance. This LEAKS: for TRANSFER rows, 99.29%
# of fraud has newbalanceDest == 0 against 0.21% of legitimate, because the
# simulator never credits the mule account. errorBalanceDest is derived from the
# same column and carries the same leak.
TRACK_B_FEATURES = TRACK_C_FEATURES + ["newbalanceDest", "errorBalanceDest"]

# Track A - adds the origin-side drain signature, the largest leak of all:
# amount == oldbalanceOrg identifies 97.7% of fraud with zero false positives.
# Dropping only `is_full_drain` would not be enough, since a model holding both
# `amount` and `oldbalanceOrg` can learn the equality itself.
TRACK_A_FEATURES = TRACK_B_FEATURES + [
    "oldbalanceOrg",
    "newbalanceOrig",
    "errorBalanceOrig",
    "balance_ratio",
    "is_full_drain",
    "orig_emptied",
]


def load_transactions(db_path: str) -> pd.DataFrame:
    """Read the two fraud-bearing transaction types out of SQLite."""
    import sqlite3

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        f"""
        SELECT step, type, amount, nameDest,
               oldbalanceOrg, newbalanceOrig,
               oldbalanceDest, newbalanceDest,
               isFraud, isFlaggedFraud
        FROM transactions
        WHERE type IN {FRAUD_TYPES}
        """,
        conn,
    )
    conn.close()
    return df


def temporal_split_step(df: pd.DataFrame, train_frac: float = 0.70) -> int:
    """
    Find the `step` that puts ~train_frac of ROWS in the training window.

    Splitting on the midpoint of the step *range* would be wrong here:
    transaction volume collapses in later steps while fraud stays roughly
    constant, so a range split leaves the test window with ~10x the training
    fraud rate.
    """
    counts = df.groupby("step").size().sort_index()
    cumulative = counts.cumsum() / len(df)
    return int(cumulative[cumulative >= train_frac].index[0])


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Time. `step` is hours since simulation start.
    df["hour_of_day"] = df["step"] % 24

    # Amount is heavily right-skewed; log1p keeps zeros valid.
    df["log_amount"] = np.log1p(df["amount"])

    df["is_transfer"] = (df["type"] == "TRANSFER").astype("int8")

    # Reconciliation gaps. Destination side behaves like real-world intuition;
    # origin side is inverted by the simulator (see Week 1, finding 5).
    df["errorBalanceOrig"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]

    df["dest_was_empty"] = (df["oldbalanceDest"] == 0).astype("int8")

    # --- the leaking features (Track A only) ---
    # Share of the sender's opening balance moved. Fraud sits at exactly 1.0.
    df["balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1.0)
    df["is_full_drain"] = (
        (df["oldbalanceOrg"] - df["amount"]).abs() < 0.01
    ).astype("int8")
    df["orig_emptied"] = (df["newbalanceOrig"] == 0).astype("int8")

    return df


def fit_dest_frequency(train_df: pd.DataFrame) -> pd.Series:
    """Count destination-account appearances in the TRAINING window only."""
    return train_df.groupby("nameDest").size()


def apply_dest_frequency(df: pd.DataFrame, freq: pd.Series) -> pd.DataFrame:
    """
    Map training-window frequencies onto any window.

    Accounts never seen in training get 0 - which is itself meaningful, and is
    exactly what a production scorer would know about a brand-new account.
    """
    df = df.copy()
    df["dest_txn_count"] = df["nameDest"].map(freq).fillna(0).astype("int32")
    df["dest_is_frequent"] = (df["dest_txn_count"] >= 6).astype("int8")
    return df


def build_features(db_path: str, train_frac: float = 0.70):
    """
    Returns (train_df, test_df, split_step) with all features attached.
    """
    df = load_transactions(db_path)
    df = add_base_features(df)

    split_step = temporal_split_step(df, train_frac)
    train_df = df[df["step"] <= split_step]
    test_df = df[df["step"] > split_step]

    freq = fit_dest_frequency(train_df)
    train_df = apply_dest_frequency(train_df, freq)
    test_df = apply_dest_frequency(test_df, freq)

    return train_df, test_df, split_step


def subsample_majority(df: pd.DataFrame, ratio: int = 20, seed: int = 42) -> pd.DataFrame:
    """
    Keep every fraud row and `ratio` legitimate rows per fraud, for TRAINING only.

    The test set is never subsampled - metrics must be measured against the real
    class balance or they are meaningless.
    """
    fraud = df[df["isFraud"] == 1]
    legit = df[df["isFraud"] == 0]
    n_legit = min(len(legit), len(fraud) * ratio)
    legit_sample = legit.sample(n=n_legit, random_state=seed)
    out = pd.concat([fraud, legit_sample]).sample(frac=1.0, random_state=seed)
    return out

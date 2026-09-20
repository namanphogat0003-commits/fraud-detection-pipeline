"""
Shared fixtures.

Every fixture here builds a small synthetic dataset that reproduces PaySim's
*structure* - and, deliberately, its three artifacts - rather than reading the
real 700MB database. That keeps the suite fast enough to run on every change,
which is the only way leakage guards are any use.
"""

import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest

# The modules under test live in src/ and import each other by bare name
# (`from features import ...`), matching how the scripts are run.
SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


N_STEPS = 200
ROWS_PER_STEP = 40


def _make_frame(seed: int = 0) -> pd.DataFrame:
    """
    A miniature PaySim: same columns, same three artifacts.

    Fraud is generated so that the tests can assert the leakage guards actually
    bite - if a synthetic fraud row did not carry the drain signature, a test
    asserting Track A can exploit it would pass vacuously.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for step in range(1, N_STEPS + 1):
        hour = step % 24
        # Artifact 3: legitimate volume collapses at night, fraud does not.
        n_legit = ROWS_PER_STEP if 8 <= hour <= 22 else max(2, ROWS_PER_STEP // 12)
        for _ in range(n_legit):
            old_org = float(rng.uniform(0, 100_000))
            amount = float(rng.uniform(1, max(old_org, 1.0)))
            old_dest = float(rng.uniform(0, 50_000))
            rows.append({
                "step": step,
                "type": rng.choice(["TRANSFER", "CASH_OUT"]),
                "amount": amount,
                "nameOrig": f"C{rng.integers(0, 10_000)}",
                "oldbalanceOrg": old_org,
                "newbalanceOrig": max(old_org - amount, 0.0),
                "nameDest": f"M{rng.integers(0, 300)}",
                "oldbalanceDest": old_dest,
                # Artifact 2 inverted: legitimate destinations ARE credited.
                "newbalanceDest": old_dest + amount,
                "isFraud": 0,
                "isFlaggedFraud": 0,
            })
        # Two frauds per step, at every hour of the day.
        for _ in range(2):
            old_org = float(rng.uniform(1_000, 500_000))
            old_dest = 0.0
            rows.append({
                "step": step,
                "type": "TRANSFER",
                # Artifact 1: the fraudster drains the account exactly.
                "amount": old_org,
                "nameOrig": f"C{rng.integers(0, 10_000)}",
                "oldbalanceOrg": old_org,
                "newbalanceOrig": 0.0,
                "nameDest": f"C{rng.integers(0, 300)}",
                "oldbalanceDest": old_dest,
                # Artifact 2: the mule account is never credited.
                "newbalanceDest": 0.0,
                "isFraud": 1,
                "isFlaggedFraud": 0,
            })
    df = pd.DataFrame(rows)
    return df.sort_values("step").reset_index(drop=True)


@pytest.fixture(scope="session")
def raw_frame() -> pd.DataFrame:
    return _make_frame()


@pytest.fixture(scope="session")
def synthetic_db(tmp_path_factory, raw_frame) -> str:
    """The synthetic frame written to a real SQLite file, as data_prep would."""
    path = str(tmp_path_factory.mktemp("data") / "fraud.db")
    conn = sqlite3.connect(path)
    raw_frame.to_sql("transactions", conn, index=False)
    conn.close()
    return path


@pytest.fixture(scope="session")
def built(synthetic_db):
    """(train_df, test_df, split_step) straight from the production code path."""
    from features import build_features

    return build_features(synthetic_db)

"""
The scoring service.

Exercised against a real XGBoost model trained on the synthetic fixture, not a
mock, because two of the things worth testing - that SHAP explanations line up
with the features, and that the batch path agrees with the single-score path -
only mean anything against a genuine tree model.
"""

import numpy as np
import pandas as pd
import pytest

from features import TRACK_D_FEATURES, add_base_features, apply_dest_frequency
from scoring import risk_band

pytest.importorskip("xgboost")
pytest.importorskip("shap")


@pytest.fixture(scope="module")
def bundle_path(tmp_path_factory, built):
    """A real api_bundle.joblib, built the way build_api_bundle.py builds one."""
    import joblib
    from xgboost import XGBClassifier

    from features import fit_dest_frequency

    train_df, _, split_step = built
    model = XGBClassifier(
        n_estimators=40, max_depth=3, learning_rate=0.2,
        eval_metric="aucpr", tree_method="hist", random_state=42, verbosity=0)
    model.fit(train_df[TRACK_D_FEATURES], train_df["isFraud"])

    path = tmp_path_factory.mktemp("models") / "api_bundle.joblib"
    joblib.dump({
        "model": model,
        "dest_freq": fit_dest_frequency(train_df),
        "features": TRACK_D_FEATURES,
        "train_split_step": int(split_step),
        "built_at": "2026-01-01T00:00:00+00:00",
        "metrics": {"pr_auc": 0.3260},
        "notes": "test bundle",
    }, path)
    return str(path)


@pytest.fixture(scope="module")
def client(bundle_path):
    from fastapi.testclient import TestClient

    import api.main as api_main

    api_main.BUNDLE_PATH = bundle_path
    api_main._bundle = None       # force a reload against the test bundle
    api_main._explainer = None
    return TestClient(api_main.app)


TXN = {
    "type": "TRANSFER",
    "amount": 181.0,
    "nameDest": "C553264065",
    "oldbalanceDest": 0.0,
}


# --- health ------------------------------------------------------------------

def test_health_reports_what_is_loaded(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["known_accounts"] > 0
    assert body["holdout_metrics"]["pr_auc"] == pytest.approx(0.3260)


def test_health_states_it_serves_the_leak_free_model(client):
    """
    The service deliberately serves Track D, not the perfect-scoring Track A.
    If that ever silently changes, the project's central claim breaks.
    """
    assert "Track D" in client.get("/health").json()["model"]


# --- scoring -----------------------------------------------------------------

def test_score_returns_a_probability_and_a_band(client):
    body = client.post("/score", json=TXN).json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["risk_band"] == risk_band(body["fraud_probability"])


def test_score_explains_itself(client):
    """A bare probability gives an analyst nothing to act on."""
    body = client.post("/score", json=TXN).json()
    assert body["drivers"]
    for driver in body["drivers"]:
        assert driver["feature"] in TRACK_D_FEATURES
        assert driver["direction"] in ("increases risk", "decreases risk")


def test_drivers_are_ordered_by_absolute_contribution(client):
    drivers = client.post("/score", json=TXN).json()["drivers"]
    magnitudes = [abs(d["shap_contribution"]) for d in drivers]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_driver_direction_matches_the_sign_of_its_contribution(client):
    for driver in client.post("/score", json=TXN).json()["drivers"]:
        if driver["shap_contribution"] > 0:
            assert driver["direction"] == "increases risk"
        else:
            assert driver["direction"] == "decreases risk"


def test_response_states_the_model_is_not_fit_for_blocking(client):
    """The caveat is load-bearing: it stops a caller assuming more than it is."""
    caveat = client.post("/score", json=TXN).json()["caveat"]
    assert "not for automated blocking" in caveat
    assert "PaySim" in caveat


def test_scoring_is_deterministic(client):
    first = client.post("/score", json=TXN).json()["fraud_probability"]
    assert all(
        client.post("/score", json=TXN).json()["fraud_probability"] == first
        for _ in range(3))


# --- validation --------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    {**TXN, "type": "PAYMENT"},          # a type that carries no fraud here
    {**TXN, "amount": 0},                # must be > 0
    {**TXN, "amount": -5},
    {**TXN, "oldbalanceDest": -1},       # must be >= 0
])
def test_invalid_transactions_are_rejected(client, bad):
    assert client.post("/score", json=bad).status_code == 422


@pytest.mark.parametrize("missing", ["type", "amount", "nameDest", "oldbalanceDest"])
def test_missing_fields_are_rejected(client, missing):
    payload = {k: v for k, v in TXN.items() if k != missing}
    assert client.post("/score", json=payload).status_code == 422


# --- batch -------------------------------------------------------------------

def test_batch_scores_and_ranks(client):
    payload = [
        {**TXN, "amount": 10.0},
        {**TXN, "amount": 5_000_000.0},
        {**TXN, "amount": 250.0},
    ]
    body = client.post("/score/batch", json=payload).json()
    assert body["count"] == 3
    probabilities = [row["fraud_probability"] for row in body["scored"]]
    assert probabilities == sorted(probabilities, reverse=True)
    assert [row["queue_rank"] for row in body["scored"]] == [1, 2, 3]


def test_batch_indices_refer_to_the_submitted_order(client):
    payload = [{**TXN, "amount": a} for a in (10.0, 5_000_000.0, 250.0)]
    body = client.post("/score/batch", json=payload).json()
    assert sorted(row["index"] for row in body["scored"]) == [0, 1, 2]


def test_batch_agrees_with_single_scoring(client):
    """
    The batch path builds its feature frame differently from the single path.
    If the two ever diverge, an analyst and a dashboard see different numbers
    for the same transaction.
    """
    payload = [{**TXN, "amount": a} for a in (100.0, 7_500.0, 1_250_000.0)]
    batch = {row["index"]: row["fraud_probability"]
             for row in client.post("/score/batch", json=payload).json()["scored"]}
    for i, txn in enumerate(payload):
        single = client.post("/score", json=txn).json()["fraud_probability"]
        assert batch[i] == pytest.approx(single, abs=1e-4)


def test_batch_bands_agree_with_the_shared_definition(client):
    payload = [{**TXN, "amount": a} for a in (10.0, 900_000.0, 3_000_000.0)]
    for row in client.post("/score/batch", json=payload).json()["scored"]:
        assert row["risk_band"] == risk_band(row["fraud_probability"])


def test_batch_ties_are_broken_by_amount(client):
    """Identical inputs score identically; the larger exposure ranks first."""
    payload = [
        {**TXN, "amount": 100.0, "nameDest": "SAME"},
        {**TXN, "amount": 100.0, "nameDest": "SAME"},
    ]
    body = client.post("/score/batch", json=payload).json()
    assert body["scored"][0]["fraud_probability"] == body["scored"][1]["fraud_probability"]


def test_empty_batch_is_a_client_error_not_a_crash(client):
    """This previously raised inside pandas and surfaced as a 500."""
    response = client.post("/score/batch", json=[])
    assert response.status_code == 400
    assert "at least one" in response.json()["detail"]


def test_oversized_batch_is_rejected(client):
    response = client.post("/score/batch", json=[TXN] * 1001)
    assert response.status_code == 413


def test_batch_at_the_limit_is_accepted(client):
    response = client.post("/score/batch", json=[TXN] * 1000)
    assert response.status_code == 200
    assert response.json()["count"] == 1000


# --- the API's features match the training pipeline's ------------------------

def test_api_features_match_the_training_pipeline(built, bundle_path):
    """
    The service derives Track D features from a raw transaction by hand. If that
    derivation drifts from features.py, the model is served inputs it was never
    trained on - silently, since the column names still line up.
    """
    import joblib

    import api.main as api_main

    train_df, test_df, _ = built
    bundle = joblib.load(bundle_path)

    row = test_df.iloc[0]
    txn = api_main.Transaction(
        type=row["type"],
        amount=float(row["amount"]),
        nameDest=row["nameDest"],
        oldbalanceDest=float(row["oldbalanceDest"]),
    )
    from_api = api_main.build_feature_row(txn, bundle)

    expected = apply_dest_frequency(
        add_base_features(pd.DataFrame([row])), bundle["dest_freq"]
    )[TRACK_D_FEATURES]

    for feature in TRACK_D_FEATURES:
        assert from_api.iloc[0][feature] == pytest.approx(
            float(expected.iloc[0][feature])), f"{feature} diverged"


def test_batch_feature_frame_matches_the_single_row_builder(built, bundle_path):
    import joblib

    import api.main as api_main

    bundle = joblib.load(bundle_path)
    txns = [
        api_main.Transaction(type="TRANSFER", amount=181.0,
                             nameDest="C1", oldbalanceDest=0.0),
        api_main.Transaction(type="CASH_OUT", amount=99_000.0,
                             nameDest="M2", oldbalanceDest=4_200.0),
    ]
    batch = api_main.build_feature_frame(txns, bundle)
    for i, txn in enumerate(txns):
        single = api_main.build_feature_row(txn, bundle)
        assert list(batch.columns) == list(single.columns)
        np.testing.assert_allclose(
            batch.iloc[i].to_numpy(dtype=float),
            single.iloc[0].to_numpy(dtype=float))

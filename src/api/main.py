"""
Week 4 - Fraud scoring service.

Serves the Track D (leak-free) XGBoost model behind a JSON API. Every response
carries the SHAP contributions behind the score, because a fraud analyst handed a
bare probability has no basis to action it - "0.83" is not a reason to freeze a
customer's account, but "0.83, driven mostly by a large transfer into a
previously-empty destination account" is something a human can check.

Deliberately serves the HONEST model. The Track A model scores a perfect 1.0000
on this dataset and would look far better in a demo, but it does so by reading a
label that leaked into its features; deploying it would be dishonest.

Run with:
    python src/build_api_bundle.py          # once, to package the model
    uvicorn src.api.main:app --reload       # from the project root

Then open http://127.0.0.1:8000/docs for the interactive interface.
"""

import os
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BUNDLE_PATH = os.path.join("models", "api_bundle.joblib")

app = FastAPI(
    title="PaySim Fraud Scoring Service",
    description=(
        "Scores mobile-money transactions for fraud risk using the leak-free "
        "Track D model, and explains each score with SHAP contributions."
    ),
    version="1.0.0",
)

_bundle = None
_explainer = None


def get_bundle():
    """Load the model bundle once, on first use."""
    global _bundle, _explainer
    if _bundle is None:
        if not os.path.exists(BUNDLE_PATH):
            raise HTTPException(
                status_code=503,
                detail=f"{BUNDLE_PATH} not found. Run "
                       "`python src/build_api_bundle.py` first.")
        _bundle = joblib.load(BUNDLE_PATH)
        _explainer = shap.TreeExplainer(_bundle["model"])
    return _bundle, _explainer


class Transaction(BaseModel):
    """One mobile-money transaction, in the shape PaySim records them."""

    type: Literal["TRANSFER", "CASH_OUT"] = Field(
        ..., description="Only these two types carry fraud in this dataset.")
    amount: float = Field(..., gt=0, description="Transaction amount.")
    nameDest: str = Field(..., description="Destination account identifier.")
    oldbalanceDest: float = Field(
        ..., ge=0, description="Destination balance before the transaction.")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "type": "TRANSFER",
                "amount": 181.0,
                "nameDest": "C553264065",
                "oldbalanceDest": 0.0,
            }]
        }
    }


class Driver(BaseModel):
    feature: str
    value: float
    shap_contribution: float
    direction: Literal["increases risk", "decreases risk"]


class ScoreResponse(BaseModel):
    fraud_probability: float
    risk_band: Literal["low", "elevated", "high"]
    drivers: list[Driver]
    model_version: str
    caveat: str


def build_feature_row(txn: Transaction, bundle) -> pd.DataFrame:
    """Derive the Track D feature vector from a raw transaction."""
    dest_count = int(bundle["dest_freq"].get(txn.nameDest, 0))
    row = {
        "log_amount": float(np.log1p(txn.amount)),
        "is_transfer": 1 if txn.type == "TRANSFER" else 0,
        "oldbalanceDest": txn.oldbalanceDest,
        "dest_was_empty": 1 if txn.oldbalanceDest == 0 else 0,
        "dest_txn_count": dest_count,
        "dest_is_frequent": 1 if dest_count >= 6 else 0,
    }
    return pd.DataFrame([row])[bundle["features"]]


def _caveat(bundle) -> str:
    """State the model's real accuracy rather than letting a caller assume."""
    pr_auc = bundle.get("metrics", {}).get("pr_auc")
    measured = f"Holdout PR-AUC is {pr_auc:.2f}. " if pr_auc else ""
    return (
        "Trained on PaySim synthetic data, which is not representative of "
        f"production fraud. {measured}Suitable for ranking a manual review "
        "queue; not for automated blocking."
    )


@app.get("/health")
def health():
    """Liveness check plus what the service is actually serving."""
    try:
        bundle, _ = get_bundle()
    except HTTPException:
        return {"status": "model not loaded", "bundle_present": False}
    return {
        "status": "ok",
        "model": "Track D (leak-free) XGBoost",
        "built_at": bundle["built_at"],
        "trained_through_step": bundle["train_split_step"],
        "known_accounts": len(bundle["dest_freq"]),
        "holdout_metrics": bundle.get("metrics", {}),
    }


@app.post("/score", response_model=ScoreResponse)
def score(txn: Transaction):
    """Score one transaction and explain the result."""
    bundle, explainer = get_bundle()
    X = build_feature_row(txn, bundle)

    probability = float(bundle["model"].predict_proba(X)[0, 1])
    shap_values = explainer.shap_values(X)[0]

    drivers = sorted(
        (
            Driver(
                feature=f,
                value=float(X.iloc[0][f]),
                shap_contribution=round(float(v), 4),
                direction="increases risk" if v > 0 else "decreases risk",
            )
            for f, v in zip(bundle["features"], shap_values)
        ),
        key=lambda d: abs(d.shap_contribution),
        reverse=True,
    )

    band = "high" if probability >= 0.7 else "elevated" if probability >= 0.3 else "low"

    return ScoreResponse(
        fraud_probability=round(probability, 4),
        risk_band=band,
        drivers=drivers[:4],
        model_version=f"track-d-xgboost @ {bundle['built_at']}",
        caveat=_caveat(bundle),
    )


@app.post("/score/batch")
def score_batch(transactions: list[Transaction]):
    """Score up to 1000 transactions at once, ranked most-risky first."""
    if len(transactions) > 1000:
        raise HTTPException(
            status_code=413, detail="Send at most 1000 transactions per request.")
    bundle, _ = get_bundle()
    X = pd.concat([build_feature_row(t, bundle) for t in transactions],
                  ignore_index=True)
    probs = bundle["model"].predict_proba(X)[:, 1]
    ranked = sorted(
        ({"index": i, "fraud_probability": round(float(p), 4)}
         for i, p in enumerate(probs)),
        key=lambda r: r["fraud_probability"], reverse=True)
    return {"count": len(ranked), "scored": ranked}

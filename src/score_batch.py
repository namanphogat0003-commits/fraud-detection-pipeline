"""
Week 4 - Score the holdout window and emit a Power BI feed.

Produces a transaction-level table with the model's score, its rank, and the
review-queue flag, so the dashboard can answer operational questions - what the
queue caught, what it missed, and what that was worth - rather than just
restating model metrics.

Outputs:
  data/processed/scored_transactions.csv   full scored holdout (Power BI source)
  reports/dashboard_summary.csv            small aggregate, safe to commit

Run with: python src/score_batch.py [--db PATH] [--queue-size 1000]
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd

from features import build_features

DEFAULT_DB = os.path.join("data", "processed", "fraud.db")
BUNDLE_PATH = os.path.join("models", "api_bundle.joblib")
SCORED_CSV = os.path.join("data", "processed", "scored_transactions.csv")
SUMMARY_CSV = os.path.join("reports", "dashboard_summary.csv")


def main(db_path, queue_size):
    if not os.path.exists(BUNDLE_PATH):
        raise FileNotFoundError(
            f"{BUNDLE_PATH} not found - run `python src/build_api_bundle.py` first.")
    bundle = joblib.load(BUNDLE_PATH)

    print("Building the holdout window...")
    _, test_df, split_step = build_features(db_path)
    print(f"  {len(test_df):,} transactions, {test_df.isFraud.sum():,} fraud")

    print("Scoring...")
    scores = bundle["model"].predict_proba(test_df[bundle["features"]])[:, 1]

    out = pd.DataFrame({
        "step": test_df["step"].to_numpy(),
        "hour_of_day": (test_df["step"] % 24).to_numpy(),
        "day": (test_df["step"] // 24).to_numpy(),
        "type": test_df["type"].to_numpy(),
        "amount": test_df["amount"].to_numpy(),
        "nameDest": test_df["nameDest"].to_numpy(),
        "oldbalanceDest": test_df["oldbalanceDest"].to_numpy(),
        "dest_txn_count": test_df["dest_txn_count"].to_numpy(),
        "is_fraud": test_df["isFraud"].to_numpy(),
        "fraud_score": np.round(scores, 6),
    })

    # Rank 1 = highest risk. The model produces many tied scores, so the queue
    # order must specify a tie-break or "top K" is implementation-dependent -
    # different sort methods shift the reported catch rate by several percent.
    # At equal risk, the larger exposure is reviewed first.
    out = out.sort_values(
        ["fraud_score", "amount"], ascending=[False, False]
    ).reset_index(drop=True)
    out["score_rank"] = np.arange(1, len(out) + 1)
    out["in_review_queue"] = (out["score_rank"] <= queue_size).astype(int)
    out["risk_band"] = pd.cut(
        out["fraud_score"], bins=[-0.001, 0.3, 0.7, 1.0],
        labels=["low", "elevated", "high"])

    # Outcome label makes the confusion matrix a one-click chart in Power BI.
    conditions = [
        (out.is_fraud == 1) & (out.in_review_queue == 1),
        (out.is_fraud == 1) & (out.in_review_queue == 0),
        (out.is_fraud == 0) & (out.in_review_queue == 1),
    ]
    out["outcome"] = np.select(
        conditions, ["caught", "missed", "false alarm"], default="correctly ignored")

    os.makedirs(os.path.dirname(SCORED_CSV), exist_ok=True)
    out.to_csv(SCORED_CSV, index=False)
    print(f"Wrote {SCORED_CSV} ({os.path.getsize(SCORED_CSV) / 1e6:.0f} MB)")

    # --- small committable aggregate ---------------------------------------
    fraud_total = out.loc[out.is_fraud == 1, "amount"].sum()
    rows = []
    for q in (100, 500, 1_000, 5_000, 10_000, 25_000, 50_000):
        if q > len(out):
            continue
        top = out.iloc[:q]
        caught = top[top.is_fraud == 1]
        rows.append({
            "queue_size": q,
            "pct_of_volume": round(100 * q / len(out), 4),
            "frauds_caught": len(caught),
            "frauds_total": int(out.is_fraud.sum()),
            "value_recovered": round(caught["amount"].sum(), 2),
            "value_recovered_pct": round(100 * caught["amount"].sum() / fraud_total, 2),
            "precision": round(len(caught) / q, 4),
            "lift_vs_random": round(
                (caught["amount"].sum() / fraud_total) / (q / len(out)), 1),
        })
    summary = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(SUMMARY_CSV), exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Wrote {SUMMARY_CSV}")
    print()
    print(summary.to_string(index=False))

    print(f"\nOutcome counts at a {queue_size:,}-transaction queue:")
    print(out["outcome"].value_counts().to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--queue-size", type=int, default=1000)
    a = p.parse_args()
    main(a.db, a.queue_size)

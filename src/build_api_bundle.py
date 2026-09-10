"""
Week 4 - Package everything the scoring service needs into one artifact.

A saved model file is not enough to score a live transaction. Two of the Track D
features - `dest_txn_count` and `dest_is_frequent` - are derived from how often a
destination account appeared in the TRAINING window, so the service needs that
frequency map alongside the model. Recomputing it at request time would be both
slow and wrong (it would count future transactions).

This produces models/api_bundle.joblib containing the model, the frequency map,
the feature order, and the metadata the API reports about itself.

Run with: python src/build_api_bundle.py [--db PATH]
"""

import argparse
import json
import os
from datetime import datetime, timezone

import joblib

from features import TRACK_D_FEATURES, build_features, fit_dest_frequency

DEFAULT_DB = os.path.join("data", "processed", "fraud.db")
MODEL_PATH = os.path.join("models", "d_xgboost.pkl")
BUNDLE_PATH = os.path.join("models", "api_bundle.joblib")
WEEK3_JSON = os.path.join("reports", "week3_metrics.json")


def main(db_path):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} not found - run `python src/advanced_models.py` first.")

    print("Rebuilding the training-window destination frequency map...")
    train_df, _, split_step = build_features(db_path)
    dest_freq = fit_dest_frequency(train_df)
    print(f"  {len(dest_freq):,} destination accounts seen in steps 1-{split_step}")

    model = joblib.load(MODEL_PATH)

    # Carry the honest performance figures so the service can report what it is,
    # rather than leaving a caller to assume it is better than it is.
    metrics = {}
    if os.path.exists(WEEK3_JSON):
        with open(WEEK3_JSON, encoding="utf-8") as fh:
            data = json.load(fh)
        for r in data.get("results", []):
            if r.get("track", "").startswith("Track D") and "XGBoost" in r.get("model", ""):
                metrics = {k: r[k] for k in ("pr_auc", "precision", "recall", "threshold")
                           if k in r}
                break

    bundle = {
        "model": model,
        "dest_freq": dest_freq,
        "features": TRACK_D_FEATURES,
        "train_split_step": int(split_step),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": metrics,
        "notes": (
            "Track D (leak-free) XGBoost. Trained without the origin-balance drain "
            "signature, the destination-balance leak, or hour-of-day, all of which "
            "are artifacts of PaySim's data generation rather than fraud signal."
        ),
    }

    os.makedirs(os.path.dirname(BUNDLE_PATH), exist_ok=True)
    joblib.dump(bundle, BUNDLE_PATH, compress=3)
    size_mb = os.path.getsize(BUNDLE_PATH) / 1e6
    print(f"Wrote {BUNDLE_PATH} ({size_mb:.1f} MB)")
    if metrics:
        print(f"  model metrics carried: {metrics}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=DEFAULT_DB)
    main(p.parse_args().db)

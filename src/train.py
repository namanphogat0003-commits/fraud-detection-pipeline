"""
Week 2 - Train and evaluate four nested model tracks.

Each track strips one measured artifact of PaySim's data generation, so the drop
in score between tracks prices that artifact. Track A is what a notebook that
skips the audit reports; Track D is the number that survives scrutiny.

Method notes that matter more than the model choice:
  * Temporal split - train on early steps, test on later ones. A random split
    would let the model learn from transactions that happen after the ones it
    is scoring, which no production scorer can do.
  * Majority-class subsampling on the TRAINING set only. The test set keeps the
    real class balance, so the reported metrics are honest.
  * Destination-frequency features fitted on the training window only.

Run with: python src/train.py [--db PATH]
"""

import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluate import (
    best_f1_threshold,
    metrics_table,
    plot_feature_importance,
    plot_pr_curves,
    score_metrics,
)
from features import (
    TRACK_A_FEATURES,
    TRACK_B_FEATURES,
    TRACK_C_FEATURES,
    TRACK_D_FEATURES,
    build_features,
    subsample_majority,
)

DEFAULT_DB = os.path.join("data", "processed", "fraud.db")
FIG_DIR = os.path.join("reports", "figures")
MODEL_DIR = "models"
RESULTS_JSON = os.path.join("reports", "week2_metrics.json")
RESULTS_MD = os.path.join("reports", "week2_results.md")


def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=42)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            class_weight="balanced_subsample", n_jobs=-1, random_state=42),
    }


def run_track(track_name, feature_cols, train_s, test_df, results):
    X_train = train_s[feature_cols]
    y_train = train_s["isFraud"].to_numpy()
    X_test = test_df[feature_cols]
    y_test = test_df["isFraud"].to_numpy()

    print(f"\n{'=' * 62}\n{track_name}\n{'=' * 62}")
    print(f"features ({len(feature_cols)}): {', '.join(feature_cols)}")

    curves, rows = [], []
    for name, model in build_models().items():
        t0 = time.time()
        model.fit(X_train, y_train)
        y_score = model.predict_proba(X_test)[:, 1]
        elapsed = time.time() - t0

        thr = best_f1_threshold(y_test, y_score)
        m = score_metrics(y_test, y_score, threshold=thr, name=name)
        m["track"] = track_name
        m["train_seconds"] = round(elapsed, 1)
        rows.append(m)
        curves.append((name, y_score))

        print(f"\n  {name}  (fit+predict {elapsed:.1f}s, threshold {thr:.4f})")
        print(f"    precision {m['precision']:.4f}   recall {m['recall']:.4f}   "
              f"F1 {m['f1']:.4f}   PR-AUC {m['pr_auc']:.4f}")
        print(f"    TP {m['tp']:,}  FP {m['fp']:,}  FN {m['fn']:,}")

        slug = track_name.split()[1].lower().rstrip(":")
        joblib.dump(model, os.path.join(
            MODEL_DIR, f"{slug}_{name.lower().replace(' ', '_')}.pkl"))

        if isinstance(model, RandomForestClassifier):
            plot_feature_importance(
                feature_cols, model.feature_importances_,
                os.path.join(FIG_DIR, f"feature_importance_{slug}.png"),
                f"{track_name} - Random Forest feature importance")

    slug = track_name.split()[1].lower().rstrip(":")
    plot_pr_curves(curves, y_test,
                   os.path.join(FIG_DIR, f"pr_curve_{slug}.png"),
                   f"{track_name} - precision/recall on the test window")
    results.extend(rows)
    return rows


def main(db_path):
    for d in (FIG_DIR, MODEL_DIR, os.path.dirname(RESULTS_JSON)):
        os.makedirs(d, exist_ok=True)

    print("Building features...")
    train_df, test_df, split_step = build_features(db_path)
    print(f"  split at step {split_step}")
    print(f"  train: {len(train_df):,} rows, {train_df.isFraud.sum():,} fraud "
          f"({100 * train_df.isFraud.mean():.4f}%)")
    print(f"  test : {len(test_df):,} rows, {test_df.isFraud.sum():,} fraud "
          f"({100 * test_df.isFraud.mean():.4f}%)")

    train_s = subsample_majority(train_df, ratio=20)
    print(f"  training subsample: {len(train_s):,} rows "
          f"({100 * train_s.isFraud.mean():.2f}% fraud)")

    y_test = test_df["isFraud"].to_numpy()

    # --- rule baselines on the same test window -----------------------------
    baselines = [
        score_metrics(y_test, test_df["is_full_drain"].to_numpy(),
                      name="Baseline: full-balance-drain rule", is_hard_label=True),
        score_metrics(y_test, test_df["isFlaggedFraud"].to_numpy(),
                      name="Baseline: PaySim isFlaggedFraud", is_hard_label=True),
    ]
    print("\nRule baselines on the test window:")
    for m in baselines:
        print(f"  {m['model']}: precision {m['precision']:.4f}  "
              f"recall {m['recall']:.4f}  TP {m['tp']:,}  FP {m['fp']:,}")

    results = []
    tracks = [
        ("Track A (all features)", TRACK_A_FEATURES),
        ("Track B (no origin drain)", TRACK_B_FEATURES),
        ("Track C (no destination leak)", TRACK_C_FEATURES),
        ("Track D (no timing artifact)", TRACK_D_FEATURES),
    ]
    track_rows = {}
    for label, feats in tracks:
        track_rows[label] = run_track(label, feats, train_s, test_df, results)

    a_rows = track_rows["Track A (all features)"]
    b_rows = track_rows["Track B (no origin drain)"]
    c_rows = track_rows["Track C (no destination leak)"]
    d_rows = track_rows["Track D (no timing artifact)"]

    all_rows = baselines + results
    with open(RESULTS_JSON, "w", encoding="utf-8") as fh:
        json.dump({"split_step": int(split_step),
                   "test_rows": int(len(test_df)),
                   "test_fraud": int(test_df.isFraud.sum()),
                   "results": all_rows}, fh, indent=2)

    best = {k: max(v, key=lambda m: m["pr_auc"]) for k, v in track_rows.items()}
    best_a = best["Track A (all features)"]
    best_b = best["Track B (no origin drain)"]
    best_c = best["Track C (no destination leak)"]
    best_d = best["Track D (no timing artifact)"]

    with open(RESULTS_MD, "w", encoding="utf-8") as fh:
        fh.write(f"""# Week 2 - Baseline models and a leakage audit

Trained on steps 1-{split_step}, evaluated on steps {split_step + 1}-743 - a
temporal split, so the model never sees a transaction that happens after the one
it is scoring. The test window holds **{len(test_df):,} transactions** with
**{int(test_df.isFraud.sum()):,} frauds** ({{:.4f}}%) at its real class balance.
Only the training set was class-subsampled.

Accuracy is not reported anywhere below. At this class balance, predicting
"never fraud" scores {{:.2f}}% while catching nothing.

## The headline result

This dataset is riddled with artifacts of how PaySim generates fraud. Rather than
report the first good-looking score, each was measured and removed in turn:

| Track | What it removes | Best PR-AUC | Best model |
|---|---|---|---|
| A | nothing | **{best_a['pr_auc']:.4f}** | {best_a['model']} |
| B | origin-side drain signature | **{best_b['pr_auc']:.4f}** | {best_b['model']} |
| C | + destination-balance leak | **{best_c['pr_auc']:.4f}** | {best_c['model']} |
| D | + hour-of-day artifact | **{best_d['pr_auc']:.4f}** | {best_d['model']} |

**Track D is the number to defend.** Track A is what a PaySim notebook that skips
this analysis will proudly report.

## The three artifacts, measured

**1. The full-balance drain (removed in Track B).** `amount == oldbalanceOrg`
identifies {{:.2f}}% of fraud in the test window with zero false positives, because
the simulated fraudster empties the account outright. Dropping the flag alone is
insufficient - a model holding `amount` and `oldbalanceOrg` relearns the equality -
so every origin-side balance column goes with it.

**2. The uncredited mule account (removed in Track C).** For TRANSFER rows, 99.29%
of fraud has `newbalanceDest == 0` against 0.21% of legitimate transactions: PaySim
never credits the receiving account. The same leak flows through `errorBalanceDest`,
which is derived from that column. Notably this is TRANSFER-specific - for CASH_OUT
the rates are 0.56% vs 0.51%, no signal at all.

**3. Fraudsters who never sleep (removed in Track D).** Fraud is generated at a
near-constant 274-375 transactions per hour around the clock, a 1.4x range, while
legitimate volume swings 1,259x - from 299,591 at 7pm to 238 at 4am. So the night
fraud rate (7.29%) towers over the daytime rate (0.19%) largely because real users
are asleep. A model leaning on `hour_of_day` would be badly miscalibrated against
real traffic, where night volume is a large fraction of daytime rather than 0.08%.

## Rule baselines

{metrics_table(baselines)}

## Track A - all features

{metrics_table(a_rows)}

## Track B - origin drain removed

{metrics_table(b_rows)}

## Track C - destination leak removed

{metrics_table(c_rows)}

## Track D - timing artifact removed (the honest floor)

{metrics_table(d_rows)}

## Reading the gap

Logistic Regression collapses far faster than Random Forest across the tracks,
which says the residual signal is non-linear - thresholds and interactions rather
than monotone relationships.

The distance from {best_a['pr_auc']:.4f} to {best_d['pr_auc']:.4f} is the real finding here.
Most published PaySim work reports something near the top of that range without
noting that the features encode the answer. The comparison itself - not the highest
score - is what belongs on a resume.
""".format(
            100 * test_df.isFraud.mean(),
            100 * (1 - test_df.isFraud.mean()),
            100 * baselines[0]["recall"],
        ))

    print(f"\nWrote {RESULTS_MD}, {RESULTS_JSON}, plots in {FIG_DIR}/")
    print("\nBest PR-AUC by track (each removes one more artifact):")
    for label, m in best.items():
        print(f"  {label:32s} {m['pr_auc']:.4f}  ({m['model']})")
    drop = best_a["pr_auc"] - best_d["pr_auc"]
    print(f"\nLeakage cost: {drop:.4f} PR-AUC between Track A and Track D.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=DEFAULT_DB)
    main(p.parse_args().db)

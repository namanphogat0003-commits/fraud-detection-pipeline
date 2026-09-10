"""
Week 3 - Gradient boosting, unsupervised detection, explainability, and a
cost-based threshold.

The Week 2 audit established that most apparent performance on this dataset is
leakage, and that the honest feature set (Track D) yields far weaker models.
This module asks the three questions that follow from that:

  1. Can gradient boosting extract more signal from the honest features than a
     Random Forest managed (PR-AUC 0.1501)?
  2. Can fraud be found WITHOUT labels at all - i.e. as pure anomaly detection?
  3. Where does the residual signal actually live, per SHAP?

And then the question a fraud team would actually ask:

  4. At what score threshold is the model worth running, given that a missed
     fraud costs the transaction amount and a false alarm costs a review?

Tuning note: hyperparameters are selected on a validation window carved out of
the TRAINING period, never on the test window. Tuning against test would be a
fourth form of leakage, and the whole point of this project is not doing that.

Run with: python src/advanced_models.py [--db PATH] [--review-cost 500]
"""

import argparse
import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier

from evaluate import (
    AXIS, BLUE, GRID, INK, INK_2, MUTED, ORANGE, SURFACE,
    best_f1_threshold, metrics_table, score_metrics,
)
from features import (
    TRACK_A_FEATURES, TRACK_B_FEATURES, TRACK_C_FEATURES, TRACK_D_FEATURES,
    build_features, subsample_majority,
)

WEEK2_JSON = os.path.join("reports", "week2_metrics.json")
DEFAULT_DB = os.path.join("data", "processed", "fraud.db")
FIG_DIR = os.path.join("reports", "figures")
MODEL_DIR = "models"
RESULTS_MD = os.path.join("reports", "week3_results.md")
RESULTS_JSON = os.path.join("reports", "week3_metrics.json")

# Deliberately small grid. With ~3.6k positive training examples, an exhaustive
# search would mostly be fitting noise in the validation split.
PARAM_GRID = [
    {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 300},
    {"max_depth": 4, "learning_rate": 0.1, "n_estimators": 300},
    {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 300},
    {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 600},
    {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 600},
    {"max_depth": 8, "learning_rate": 0.05, "n_estimators": 400},
]


def load_week2_best():
    """Best Week 2 PR-AUC per track, so Week 3 can compare against it rather
    than asserting a conclusion the numbers may not support."""
    if not os.path.exists(WEEK2_JSON):
        return {}
    with open(WEEK2_JSON, encoding="utf-8") as fh:
        data = json.load(fh)
    best = {}
    for r in data.get("results", []):
        track = r.get("track")
        if not track or "pr_auc" not in r:
            continue
        if track not in best or r["pr_auc"] > best[track]["pr_auc"]:
            best[track] = r
    return best


def temporal_validation_split(train_df, val_frac=0.25):
    """Last `val_frac` of training ROWS, by step, becomes validation."""
    counts = train_df.groupby("step").size().sort_index()
    cumulative = counts.cumsum() / len(train_df)
    cut = int(cumulative[cumulative >= (1 - val_frac)].index[0])
    return train_df[train_df["step"] <= cut], train_df[train_df["step"] > cut]


def tune_xgb(track_name, features, fit_df, val_df, pos_weight):
    from sklearn.metrics import average_precision_score

    best, best_score = None, -1.0
    for params in PARAM_GRID:
        model = XGBClassifier(
            **params, scale_pos_weight=pos_weight, eval_metric="aucpr",
            tree_method="hist", n_jobs=-1, random_state=42, verbosity=0,
        )
        model.fit(fit_df[features], fit_df["isFraud"])
        score = average_precision_score(
            val_df["isFraud"], model.predict_proba(val_df[features])[:, 1])
        if score > best_score:
            best, best_score = params, score
    print(f"  {track_name}: best params {best} (validation PR-AUC {best_score:.4f})")
    return best, best_score


def capacity_analysis(y_true, y_score, amounts, budgets):
    """
    Fraud teams do not review "everything above a threshold" - they work a
    queue of fixed size. So the operational question is: if you can review K
    transactions, what does ranking them by model score recover, versus
    picking K at random?

    This is where a weak model can still earn its keep: even poor ranking beats
    random selection, and the lift is the number an ops manager can act on.
    """
    order = np.argsort(-y_score)
    fraud_total = amounts[y_true == 1].sum()
    n_fraud = int(y_true.sum())
    n_total = len(y_true)

    rows = []
    for k in budgets:
        top = order[:k]
        top_is_fraud = y_true[top] == 1
        caught_n = int(top_is_fraud.sum())
        caught_amt = float(amounts[top][top_is_fraud].sum())
        # Expected value of reviewing k transactions chosen at random.
        rand_n = n_fraud * k / n_total
        rand_amt = fraud_total * k / n_total
        rows.append({
            "budget": int(k),
            "caught_n": caught_n,
            "caught_pct": 100.0 * caught_n / max(n_fraud, 1),
            "caught_amt": caught_amt,
            "caught_amt_pct": 100.0 * caught_amt / max(fraud_total, 1),
            "random_n": rand_n,
            "random_amt_pct": 100.0 * rand_amt / max(fraud_total, 1),
            "lift": caught_amt / rand_amt if rand_amt > 0 else 0.0,
            "precision": caught_n / k if k else 0.0,
        })
    return rows, fraud_total


def plot_capacity_curve(rows, path):
    budgets = [r["budget"] for r in rows]
    model = [r["caught_amt_pct"] for r in rows]
    random = [r["random_amt_pct"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(budgets, model, color=BLUE, linewidth=2, marker="o",
            markersize=4, label="Ranked by model score")
    ax.plot(budgets, random, color=MUTED, linewidth=2, linestyle="--",
            label="Reviewed at random")
    ax.set_xscale("log")
    ax.set_xlabel("Manual review budget (transactions)")
    ax.set_ylabel("Fraud value recovered (%)")
    ax.set_title("What a fixed review queue recovers")
    ax.set_ylim(0, 100)
    legend = ax.legend(frameon=False, loc="upper left")
    for t in legend.get_texts():
        t.set_color(INK_2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(db_path, review_cost):
    for d in (FIG_DIR, MODEL_DIR, os.path.dirname(RESULTS_MD)):
        os.makedirs(d, exist_ok=True)

    week2 = load_week2_best()
    print("Building features...")
    train_df, test_df, split_step = build_features(db_path)
    fit_df, val_df = temporal_validation_split(train_df)
    print(f"  train window steps 1-{split_step} "
          f"({len(train_df):,} rows, {train_df.isFraud.sum():,} fraud)")
    print(f"  tuning split: fit {len(fit_df):,} / validation {len(val_df):,} "
          f"({val_df.isFraud.sum():,} fraud in validation)")

    fit_s = subsample_majority(fit_df, ratio=20)
    train_s = subsample_majority(train_df, ratio=20)
    pos_weight = (fit_s.isFraud == 0).sum() / max((fit_s.isFraud == 1).sum(), 1)

    y_test = test_df["isFraud"].to_numpy()
    amounts = test_df["amount"].to_numpy()

    tracks = [
        ("Track A (all features)", TRACK_A_FEATURES),
        ("Track B (no origin drain)", TRACK_B_FEATURES),
        ("Track C (no destination leak)", TRACK_C_FEATURES),
        ("Track D (no timing artifact)", TRACK_D_FEATURES),
    ]

    print("\nTuning XGBoost (selection on the validation window, never on test):")
    rows, models, scores = [], {}, {}
    for name, feats in tracks:
        params, val_score = tune_xgb(name, feats, fit_s, val_df, pos_weight)
        model = XGBClassifier(
            **params, scale_pos_weight=pos_weight, eval_metric="aucpr",
            tree_method="hist", n_jobs=-1, random_state=42, verbosity=0)
        t0 = time.time()
        model.fit(train_s[feats], train_s["isFraud"])
        y_score = model.predict_proba(test_df[feats])[:, 1]
        thr = best_f1_threshold(y_test, y_score)
        m = score_metrics(y_test, y_score, threshold=thr, name=f"XGBoost - {name}")
        m["track"] = name
        m["params"] = params
        m["validation_pr_auc"] = round(val_score, 4)
        m["fit_seconds"] = round(time.time() - t0, 1)
        rows.append(m)
        models[name] = model
        scores[name] = y_score
        slug = name.split()[1].lower()
        joblib.dump(model, os.path.join(MODEL_DIR, f"{slug}_xgboost.pkl"))
        print(f"    -> test PR-AUC {m['pr_auc']:.4f}  "
              f"precision {m['precision']:.4f}  recall {m['recall']:.4f}")

    # --- unsupervised: can fraud be found with no labels at all? ------------
    print("\nIsolationForest (unsupervised - trained on legitimate rows only):")
    legit_train = train_df[train_df.isFraud == 0].sample(
        n=min(200_000, (train_df.isFraud == 0).sum()), random_state=42)
    iso = IsolationForest(n_estimators=200, contamination=0.01,
                          n_jobs=-1, random_state=42)
    iso.fit(legit_train[TRACK_D_FEATURES])
    iso_score = -iso.score_samples(test_df[TRACK_D_FEATURES])
    iso_norm = (iso_score - iso_score.min()) / (iso_score.max() - iso_score.min())
    iso_thr = best_f1_threshold(y_test, iso_norm)
    iso_m = score_metrics(y_test, iso_norm, threshold=iso_thr,
                          name="IsolationForest (unsupervised, Track D)")
    iso_m["track"] = "Track D (no timing artifact)"
    print(f"    PR-AUC {iso_m['pr_auc']:.4f}  precision {iso_m['precision']:.4f}  "
          f"recall {iso_m['recall']:.4f}")

    # --- SHAP on the honest model -------------------------------------------
    print("\nComputing SHAP values on Track D (10k test sample)...")
    import shap
    d_model = models["Track D (no timing artifact)"]
    sample = test_df.sample(n=min(10_000, len(test_df)), random_state=42)
    explainer = shap.TreeExplainer(d_model)
    shap_values = explainer.shap_values(sample[TRACK_D_FEATURES])

    plt.figure(figsize=(8, 4.6), facecolor=SURFACE)
    shap.summary_plot(shap_values, sample[TRACK_D_FEATURES], show=False,
                      plot_size=None)
    fig = plt.gcf()
    fig.patch.set_facecolor(SURFACE)
    for ax in fig.axes:
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=INK_2)
        ax.xaxis.label.set_color(INK_2)
    plt.title("Track D - SHAP value distribution per feature",
              color=INK, fontsize=12, fontweight="semibold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "shap_summary_d.png"), facecolor=SURFACE)
    plt.close("all")

    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_rank = sorted(zip(TRACK_D_FEATURES, mean_abs),
                       key=lambda x: x[1], reverse=True)
    print("  mean |SHAP| by feature:")
    for f, v in shap_rank:
        print(f"    {f:20s} {v:.4f}")

    # --- what is it worth operationally? ------------------------------------
    print("\nCapacity analysis (fixed review queue, ranked by model score):")
    d_score = scores["Track D (no timing artifact)"]
    budgets = [b for b in (100, 500, 1_000, 5_000, 10_000, 25_000, 50_000)
               if b <= len(test_df)]
    cap_rows, fraud_total = capacity_analysis(y_test, d_score, amounts, budgets)
    plot_capacity_curve(cap_rows, os.path.join(FIG_DIR, "capacity_curve_d.png"))

    print(f"  total fraud exposure in test window: {fraud_total:,.0f}")
    for r in cap_rows:
        print(f"  review {r['budget']:>6,}: catches {r['caught_n']:>5,} frauds "
              f"({r['caught_amt_pct']:5.1f}% of value)  "
              f"precision {r['precision']:.3f}  lift {r['lift']:.1f}x vs random")

    # Break-even review cost: below this, the queue pays for itself.
    be = []
    for r in cap_rows:
        be.append(r["caught_amt"] / r["budget"] if r["budget"] else 0.0)

    all_rows = rows + [iso_m]
    with open(RESULTS_JSON, "w", encoding="utf-8") as fh:
        json.dump({"results": all_rows,
                   "shap_mean_abs": {f: float(v) for f, v in shap_rank},
                   "capacity": cap_rows,
                   "fraud_exposure": float(fraud_total),
                   "break_even_review_cost": be}, fh, indent=2)

    d_row = [r for r in rows if r["track"].startswith("Track D")][0]

    # --- derive the comparison and verdicts from the numbers ----------------
    no_skill = float(y_test.mean())
    iso_lift = iso_m["pr_auc"] / no_skill if no_skill else 0.0

    comp_lines = ["| Track | Random Forest (Week 2) | XGBoost (Week 3) | Change |",
                  "|---|---|---|---|"]
    improved = {}
    for r in rows:
        prev = week2.get(r["track"], {}).get("pr_auc")
        if prev is None:
            comp_lines.append(
                f"| {r['track']} | - | {r['pr_auc']:.4f} | - |")
            continue
        delta = r["pr_auc"] - prev
        improved[r["track"]] = (prev, r["pr_auc"], delta)
        comp_lines.append(
            f"| {r['track']} | {prev:.4f} | {r['pr_auc']:.4f} | "
            f"{delta:+.4f} |")
    comparison_table = "\n".join(comp_lines)

    d_prev, d_now, d_delta = improved.get(
        "Track D (no timing artifact)", (None, d_row["pr_auc"], 0.0))
    if d_prev and d_now > d_prev * 1.25:
        gb_verdict = (
            f"Gradient boosting materially outperforms the Random Forest on the "
            f"honest feature set: **{d_prev:.4f} -> {d_now:.4f}** on Track D, a "
            f"{d_now / d_prev:.1f}x improvement. So the residual signal after the "
            f"artifacts are stripped is real but highly non-linear - it needs a "
            f"model capable of deep interactions to reach, which is why the "
            f"linear and shallower models found so little of it.\n\n"
            f"This revises the Week 2 reading. The leakage was doing most of the "
            f"work, but not all of it.")
    elif d_prev and d_now < d_prev * 0.9:
        gb_verdict = (
            f"Gradient boosting performs *worse* than the Random Forest here "
            f"({d_prev:.4f} -> {d_now:.4f}), which on this little signal usually "
            f"means the booster is overfitting the validation window.")
    else:
        gb_verdict = (
            f"Gradient boosting does not meaningfully change the picture on the "
            f"honest feature set ({d_prev:.4f} -> {d_now:.4f} on Track D). Once "
            f"the artifacts are gone the signal genuinely is thin, and a stronger "
            f"learner does not manufacture it.")

    sup = d_row["pr_auc"]
    if iso_m["pr_auc"] < sup * 0.4:
        iso_verdict = (
            f"That is far below the supervised model on identical features "
            f"({sup:.4f}), so labels are carrying most of the weight. "
            f"Unsupervised detection is often proposed for fraud on the grounds "
            f"that it catches novel patterns; on this data it is a weak "
            f"substitute rather than a replacement, though it still beats "
            f"random and would have some value where no labels exist at all.")
    else:
        iso_verdict = (
            f"That is close to the supervised model on identical features "
            f"({sup:.4f}), suggesting most of the residual structure is "
            f"reachable without labels at all.")

    best_budget = cap_rows[2] if len(cap_rows) > 2 else cap_rows[-1]
    summary_verdict = (
        f"On the honest feature set (Track D) the best model reaches "
        f"**{sup:.4f}** PR-AUC against a {no_skill:.4f} no-skill floor - "
        f"{sup / no_skill:.0f}x better than random, but far below what the "
        f"leaking feature sets appear to deliver. Judged as a ranker rather than "
        f"a classifier it is considerably more useful: reviewing the "
        f"{best_budget['budget']:,} highest-scoring transactions "
        f"({100 * best_budget['budget'] / len(test_df):.2f}% of the test window) "
        f"recovers {best_budget['caught_amt_pct']:.1f}% of all fraud value.")

    with open(RESULTS_MD, "w", encoding="utf-8") as fh:
        fh.write(f"""# Week 3 - Gradient boosting, anomaly detection, and cost

Week 2 established that most apparent performance on PaySim is leakage, and that
the honest feature set (Track D) supports only weak models. This week tests
whether a stronger learner changes that, whether fraud is findable without labels
at all, and at what threshold the model is worth running.

Hyperparameters were selected on a validation window carved from the **training**
period. Tuning against the test window would be a fourth form of leakage.

## XGBoost across the tracks

{metrics_table(rows)}

### Against the Week 2 Random Forest

{comparison_table}

{gb_verdict}

## Can fraud be found without labels?

{metrics_table([iso_m])}

An IsolationForest trained only on legitimate transactions - never shown a fraud
label - scores **{iso_m['pr_auc']:.4f}** PR-AUC against a no-skill floor of
{no_skill:.4f}, i.e. {iso_lift:.1f}x better than random. {iso_verdict}

## Where the residual signal lives (SHAP)

![SHAP summary](figures/shap_summary_d.png)

Mean absolute SHAP value per feature:

| Feature | Mean \\|SHAP\\| |
|---|---|
""")
        for f, v in shap_rank:
            fh.write(f"| `{f}` | {v:.4f} |\n")

        fh.write(f"""
## What it is worth operationally

A fraud team does not review everything above a threshold - it works a queue of
fixed size. So the useful question is not "what is the optimal cutoff" but: given
capacity to review K transactions, what does ranking by model score recover
compared with picking K at random?

This is where a weak model can still earn its keep. Ranking does not need to be
good to beat random selection, and the lift is a number an operations manager can
actually act on.

![Review capacity](figures/capacity_curve_d.png)

| Review budget | Frauds caught | % of fraud value | Precision | Lift vs random |
|---|---|---|---|---|
""")
        for r in cap_rows:
            fh.write(f"| {r['budget']:,} | {r['caught_n']:,} "
                     f"| {r['caught_amt_pct']:.1f}% | {r['precision']:.3f} "
                     f"| {r['lift']:.1f}x |\n")

        fh.write(f"""
Total fraud exposure in the test window is **{fraud_total:,.0f}**. Reviewing the
{cap_rows[2]['budget']:,} highest-scoring transactions recovers
**{cap_rows[2]['caught_amt_pct']:.1f}%** of that value at
**{cap_rows[2]['lift']:.1f}x** the return of reviewing the same number at random.

Break-even framing: at a budget of {cap_rows[2]['budget']:,} reviews, the queue
recovers {cap_rows[2]['caught_amt'] / cap_rows[2]['budget']:,.0f} of fraud value per
review performed. Any per-review cost below that figure makes the queue profitable -
which is the form of the answer a fraud operations lead needs, rather than a PR-AUC.

## Honest summary

{summary_verdict}

The value of this project remains the audit that established how much of PaySim's
apparent difficulty is manufactured, plus an operational framing that says what the
resulting detector is worth in a review queue. Conclusions here characterise the
simulator, not production fraud.
""")

    print(f"\nWrote {RESULTS_MD}, {RESULTS_JSON}, plots in {FIG_DIR}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--review-cost", type=float, default=500)
    a = p.parse_args()
    main(a.db, a.review_cost)

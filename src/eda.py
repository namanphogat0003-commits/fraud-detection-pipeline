"""
Week 1 - Step 2: Exploratory Data Analysis on the PaySim dataset.

All heavy aggregation is pushed down into SQL rather than pulled into pandas,
so this runs in constant memory regardless of table size. Only small result
sets and a stratified sample are ever materialised as DataFrames.

Run with: python src/eda.py [--db PATH] [--figdir PATH] [--findings PATH]
Outputs: prints key stats, saves plots, writes reports/eda_findings.md
"""

import argparse
import os
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

DEFAULT_DB = os.path.join("data", "processed", "fraud.db")
DEFAULT_FIGDIR = os.path.join("reports", "figures")
DEFAULT_FINDINGS = os.path.join("reports", "eda_findings.md")

# --- chart tokens -----------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"      # categorical slot 1
ORANGE = "#eb6834"    # categorical slot 2
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", [BLUE, "#f0efec", "#d03b3b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "figure.dpi": 130,
})


def q(conn, sql):
    return pd.read_sql_query(sql, conn)


def main(db_path, fig_dir, findings_path):
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(os.path.dirname(findings_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)

    # 1. Class imbalance ----------------------------------------------------
    overall = q(conn, """
        SELECT COUNT(*) AS total,
               SUM(isFraud) AS fraud_count,
               ROUND(100.0 * SUM(isFraud) / COUNT(*), 4) AS fraud_pct
        FROM transactions
    """)
    total_txns = int(overall["total"].iloc[0])
    fraud_count = int(overall["fraud_count"].iloc[0])
    fraud_pct = float(overall["fraud_pct"].iloc[0])
    print(f"Total transactions : {total_txns:,}")
    print(f"Fraudulent         : {fraud_count:,} ({fraud_pct}%)")
    print(f"Imbalance ratio    : 1 fraud per {total_txns / fraud_count:,.0f} transactions\n")

    # 2. Fraud rate by transaction type -------------------------------------
    by_type = q(conn, """
        SELECT type,
               COUNT(*) AS total_txns,
               SUM(isFraud) AS fraud_txns,
               ROUND(100.0 * SUM(isFraud) / COUNT(*), 4) AS fraud_rate_pct
        FROM transactions
        GROUP BY type
        ORDER BY fraud_rate_pct DESC
    """)
    print("Fraud rate by transaction type:")
    print(by_type.to_string(index=False), "\n")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(by_type["type"], by_type["fraud_rate_pct"], color=BLUE, width=0.6)
    for bar, val in zip(bars, by_type["fraud_rate_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2f}%" if val > 0 else "0%",
                ha="center", va="bottom", fontsize=9, color=INK_2)
    ax.set_title("Fraud rate by transaction type")
    ax.set_ylabel("Share of transactions that are fraud (%)")
    ax.set_xlabel("")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(by_type["fraud_rate_pct"]) * 1.18)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fraud_rate_by_type.png"))
    plt.close(fig)

    # 3. Amount profile: fraud vs legitimate --------------------------------
    amounts = q(conn, """
        SELECT isFraud,
               COUNT(*) AS n,
               ROUND(AVG(amount), 2) AS avg_amount,
               ROUND(MAX(amount), 2) AS max_amount
        FROM transactions GROUP BY isFraud
    """)
    print("Amount profile (0 = legitimate, 1 = fraud):")
    print(amounts.to_string(index=False), "\n")

    # Stratified sample: every fraud row + a systematic 1-in-200 slice of the rest.
    sample = q(conn, """
        SELECT amount, oldbalanceOrg, newbalanceOrig,
               oldbalanceDest, newbalanceDest, isFraud
        FROM transactions WHERE isFraud = 1
        UNION ALL
        SELECT amount, oldbalanceOrg, newbalanceOrig,
               oldbalanceDest, newbalanceDest, isFraud
        FROM transactions WHERE isFraud = 0 AND rowid % 200 = 0
    """)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    data = [sample.loc[sample.isFraud == 0, "amount"].clip(lower=1),
            sample.loc[sample.isFraud == 1, "amount"].clip(lower=1)]
    # 'tick_labels' since Matplotlib 3.9; fall back for older versions.
    label_kw = ("tick_labels" if matplotlib.__version__ >= "3.9" else "labels")
    bp = ax.boxplot(data, vert=True, widths=0.45, patch_artist=True,
                    showfliers=False, **{label_kw: ["Legitimate", "Fraud"]})
    for patch, colour in zip(bp["boxes"], [BLUE, ORANGE]):
        patch.set_facecolor(colour)
        patch.set_edgecolor(colour)
        patch.set_linewidth(0)
    for element in ("whiskers", "caps"):
        for item in bp[element]:
            item.set_color(AXIS)
    for median in bp["medians"]:
        median.set_color(SURFACE)
        median.set_linewidth(2)
    ax.set_yscale("log")
    ax.set_title("Transaction amount: fraud vs legitimate (log scale)")
    ax.set_ylabel("Amount")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "amount_by_fraud.png"))
    plt.close(fig)

    # 4. Balance-reconciliation signal --------------------------------------
    mismatch = q(conn, """
        SELECT CASE WHEN ABS(oldbalanceOrg - amount - newbalanceOrig) > 0.01
                    THEN 'balance does NOT reconcile' ELSE 'balance reconciles' END AS bucket,
               COUNT(*) AS n,
               SUM(isFraud) AS fraud_txns,
               ROUND(100.0 * SUM(isFraud) / COUNT(*), 4) AS fraud_rate_pct
        FROM transactions
        GROUP BY bucket ORDER BY fraud_rate_pct DESC
    """)
    print("Balance-reconciliation signal:")
    print(mismatch.to_string(index=False), "\n")

    # 4b. The full-balance-drain pattern -------------------------------------
    drain = q(conn, """
        SELECT isFraud, COUNT(*) AS n,
               SUM(CASE WHEN ABS(amount - oldbalanceOrg) < 0.01 THEN 1 ELSE 0 END) AS full_drain,
               ROUND(100.0 * SUM(CASE WHEN ABS(amount - oldbalanceOrg) < 0.01
                                      THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct
        FROM transactions
        WHERE type IN ('TRANSFER', 'CASH_OUT')
        GROUP BY isFraud
    """)
    drain_fraud_n = int(drain.query("isFraud == 1")["full_drain"].iloc[0])
    drain_fraud_tot = int(drain.query("isFraud == 1")["n"].iloc[0])
    drain_fraud_pct = float(drain.query("isFraud == 1")["pct"].iloc[0])
    drain_legit_n = int(drain.query("isFraud == 0")["full_drain"].iloc[0])
    drain_legit_tot = int(drain.query("isFraud == 0")["n"].iloc[0])
    print("Full-balance-drain rule (amount == oldbalanceOrg), TRANSFER/CASH_OUT only:")
    print(f"  fraud: {drain_fraud_n:,}/{drain_fraud_tot:,} ({drain_fraud_pct}%)")
    print(f"  legitimate: {drain_legit_n:,}/{drain_legit_tot:,}\n")

    zero_bal = q(conn, """
        SELECT isFraud,
               ROUND(100.0 * SUM(CASE WHEN oldbalanceOrg = 0 AND newbalanceOrig = 0
                                      THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_both_zero
        FROM transactions GROUP BY isFraud
    """)
    zero_legit_pct = float(zero_bal.query("isFraud == 0")["pct_both_zero"].iloc[0])

    recon = q(conn, """
        SELECT CASE WHEN ABS(oldbalanceOrg - amount - newbalanceOrig) > 0.01
                    THEN 'no' ELSE 'yes' END AS reconciles,
               COUNT(*) AS n, SUM(isFraud) AS fraud,
               ROUND(100.0 * SUM(isFraud) / COUNT(*), 4) AS fraud_rate_pct
        FROM transactions WHERE type IN ('TRANSFER', 'CASH_OUT')
        GROUP BY reconciles
    """)
    recon_yes = float(recon.query("reconciles == 'yes'")["fraud_rate_pct"].iloc[0])
    recon_no = float(recon.query("reconciles == 'no'")["fraud_rate_pct"].iloc[0])

    dest_recon = q(conn, """
        SELECT CASE WHEN ABS(oldbalanceDest + amount - newbalanceDest) > 0.01
                    THEN 'no' ELSE 'yes' END AS reconciles,
               COUNT(*) AS n, SUM(isFraud) AS fraud,
               ROUND(100.0 * SUM(isFraud) / COUNT(*), 4) AS fraud_rate_pct
        FROM transactions WHERE type IN ('TRANSFER', 'CASH_OUT')
        GROUP BY reconciles
    """)
    dest_no = float(dest_recon.query("reconciles == 'no'")["fraud_rate_pct"].iloc[0])
    dest_yes = float(dest_recon.query("reconciles == 'yes'")["fraud_rate_pct"].iloc[0])

    # 5. PaySim's own rule - the baseline to beat ---------------------------
    flagged = q(conn, """
        SELECT isFraud, isFlaggedFraud, COUNT(*) AS n
        FROM transactions GROUP BY isFraud, isFlaggedFraud
    """)
    tp = int(flagged.query("isFraud == 1 and isFlaggedFraud == 1")["n"].sum())
    fn = int(flagged.query("isFraud == 1 and isFlaggedFraud == 0")["n"].sum())
    fp = int(flagged.query("isFraud == 0 and isFlaggedFraud == 1")["n"].sum())
    baseline_recall = 100.0 * tp / (tp + fn) if (tp + fn) else 0.0
    print("Built-in isFlaggedFraud rule (baseline to beat):")
    print(f"  caught {tp:,} of {tp + fn:,} frauds -> recall {baseline_recall:.4f}% "
          f"(false positives: {fp:,})\n")

    # 6. Timing pattern ------------------------------------------------------
    hourly = q(conn, """
        SELECT step % 24 AS hour_of_day,
               COUNT(*) AS total_txns,
               SUM(isFraud) AS fraud_txns,
               ROUND(100.0 * SUM(isFraud) / COUNT(*), 4) AS fraud_rate_pct
        FROM transactions GROUP BY hour_of_day ORDER BY hour_of_day
    """)
    peak = hourly.loc[hourly["fraud_rate_pct"].idxmax()]
    quiet_hours = hourly.query("hour_of_day < 9")
    quiet_share = 100.0 * quiet_hours["fraud_txns"].sum() / hourly["fraud_txns"].sum()
    print(f"Peak fraud-rate hour: {int(peak.hour_of_day):02d}:00 "
          f"({peak.fraud_rate_pct}% of that hour's transactions)")
    print(f"Share of all fraud occurring 00:00-08:59: {quiet_share:.1f}%\n")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hourly["hour_of_day"], hourly["fraud_rate_pct"],
            color=BLUE, linewidth=2, marker="o", markersize=4)
    ax.set_title("Fraud rate by hour of day")
    ax.set_xlabel("Hour of day (simulation clock)")
    ax.set_ylabel("Fraud rate (%)")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(axis="x", visible=False)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fraud_rate_by_hour.png"))
    plt.close(fig)

    # 7. Correlation structure ----------------------------------------------
    sample["errorBalanceOrig"] = (sample.oldbalanceOrg - sample.amount - sample.newbalanceOrig)
    sample["errorBalanceDest"] = (sample.oldbalanceDest + sample.amount - sample.newbalanceDest)
    corr_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
                 "newbalanceDest", "errorBalanceOrig", "errorBalanceDest", "isFraud"]
    corr = sample[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_cols)))
    ax.set_yticks(range(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=45, ha="right")
    ax.set_yticklabels(corr_cols)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            val = corr.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                    color=SURFACE if abs(val) > 0.55 else INK_2)
    ax.grid(False)
    ax.set_title("Correlation structure (stratified sample)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.outline.set_edgecolor(AXIS)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "correlation_heatmap.png"))
    plt.close(fig)

    fraud_corr = corr["isFraud"].drop("isFraud").sort_values(key=abs, ascending=False)
    print("Correlation with isFraud (strongest first):")
    print(fraud_corr.round(4).to_string(), "\n")

    # 8. Write findings ------------------------------------------------------
    fraud_types = by_type.query("fraud_txns > 0")["type"].tolist()
    mismatch_top = mismatch.iloc[0]
    with open(findings_path, "w", encoding="utf-8") as fh:
        fh.write(f"""# Week 1 - EDA Findings

Generated by `src/eda.py` against {total_txns:,} PaySim transactions.

## 1. The problem is severe class imbalance
Only **{fraud_count:,} of {total_txns:,}** transactions are fraudulent
(**{fraud_pct}%**) - roughly **1 in {total_txns / fraud_count:,.0f}**.
A model that predicts "never fraud" would score {100 - fraud_pct:.2f}% accuracy while
catching nothing, so accuracy is a meaningless metric here. Precision, recall and
PR-AUC are the metrics that matter.

## 2. Fraud lives in exactly two transaction types
All fraud occurs in **{' and '.join(fraud_types)}** transactions; the other types contain
zero fraud. This makes business sense - fraud drains an account by moving money out
(TRANSFER to a mule account, then CASH_OUT). Modelling can be scoped to these types,
which immediately removes ~{100 - 100.0 * by_type.query("fraud_txns > 0")["total_txns"].sum() / total_txns:.0f}% of the rows as irrelevant noise.

| Type | Transactions | Fraud | Fraud rate |
|---|---|---|---|
""")
        for _, r in by_type.iterrows():
            fh.write(f"| {r.type} | {r.total_txns:,} | {r.fraud_txns:,} | {r.fraud_rate_pct}% |\n")

        fh.write(f"""
## 3. Fraudulent transactions are far larger
Average legitimate transaction: **{amounts.query('isFraud == 0').avg_amount.iloc[0]:,.0f}**.
Average fraudulent transaction: **{amounts.query('isFraud == 1').avg_amount.iloc[0]:,.0f}**
- roughly **{amounts.query('isFraud == 1').avg_amount.iloc[0] / amounts.query('isFraud == 0').avg_amount.iloc[0]:.1f}x larger**.
Fraud drains accounts in big movements rather than small skims, so `amount` will
carry real signal.

## 4. Fraud drains the account completely - and this is near-perfectly separable
Restricting to TRANSFER and CASH_OUT, a single rule - *does this transaction move
exactly the sender's entire opening balance?* (`amount == oldbalanceOrg`) - separates
the classes almost perfectly:

| Class | Transactions | Drain the full balance | Share |
|---|---|---|---|
| Fraud | {drain_fraud_tot:,} | {drain_fraud_n:,} | **{drain_fraud_pct}%** |
| Legitimate | {drain_legit_tot:,} | {drain_legit_n:,} | **{100.0 * drain_legit_n / drain_legit_tot:.4f}%** |

That one hand-written rule achieves roughly **{drain_fraud_pct}% recall at ~100% precision**,
with no model at all.

**This is the single most important thing to know about this dataset, and it must be
stated openly rather than buried.** It is an artifact of how PaySim generates fraud:
the simulated fraudster empties the account outright, so the label is almost
deterministically recoverable from the features. The practical consequences:

- A 99%+ F1 score here is *not* evidence of modelling skill - the problem is close to
  linearly separable, and a decision stump would find it.
- Reporting "99.8% accuracy" as an achievement invites exactly the follow-up question
  that sinks the story in an interview.
- The defensible framing is the opposite one: identify the leak, quantify it, and then
  show what the model contributes **beyond** the trivial rule - specifically the
  {drain_fraud_tot - drain_fraud_n:,} frauds ({100.0 - drain_fraud_pct:.2f}%) the rule misses,
  which is where genuine modelling value lives.

## 5. The naive reconciliation check points the *opposite* way to intuition
The obvious feature - "flag transactions whose balances don't add up" - is inverted
here. Within TRANSFER/CASH_OUT, transactions whose sender balance reconciles cleanly
carry a **{recon_yes}%** fraud rate, versus **{recon_no}%** for those that don't.

The cause is two overlapping simulator artifacts: fraud drains the balance to exactly
zero (so the arithmetic reconciles perfectly), while **{zero_legit_pct}%** of legitimate
rows carry placeholder zero balances on both sides, which makes the naive check fail
for entirely innocent reasons.

The **destination**-side error behaves more like real-world intuition and is the more
honest feature of the two: when the recipient balance fails to reconcile, the fraud
rate is **{dest_no}%** versus **{dest_yes}%** when it does - a
**{dest_no / max(dest_yes, 0.0001):.1f}x** lift.

## 6. The built-in rule is effectively useless - and that is the baseline to beat
PaySim's own `isFlaggedFraud` rule caught **{tp:,} of {tp + fn:,}** frauds
(**{baseline_recall:.2f}% recall**). Any model beating that materially is a demonstrable
business improvement, which is how this should be framed on a resume - not as a
raw F1 score.

## 7. Timing carries some signal
Fraud rate peaks at **{int(peak.hour_of_day):02d}:00** ({peak.fraud_rate_pct}% of that hour's
transactions). **{quiet_share:.1f}%** of all fraud happens between 00:00-08:59, when
legitimate volume is lowest - so hour-of-day is worth engineering as a feature.

## 8. What this means for Week 2

The leakage in finding 4 reshapes the plan. Chasing a headline F1 score is no longer
the goal, because that score is trivially available and therefore worthless as
evidence. Two tracks are worth building instead:

**Track A - the honest benchmark.** Report the one-line drain rule as the baseline
({drain_fraud_pct}% recall), then show what a model adds on top of it. This is the
number that belongs on the resume, because it is the number a fraud team would
actually care about.

**Track B - the harder, more interesting problem.** Exclude the leaking feature
entirely and model the residual: can fraud be identified *without* being told the
account was emptied? That is a genuinely difficult problem on this data, the metrics
will be far more modest, and it is a much stronger interview story than a 99.9% score.

Priority features, in order of expected value:

1. `balance_ratio` = `amount / oldbalanceOrg` - the drain signature as a continuous
   feature (finding 4); hold it out entirely for the Track B model
2. `errorBalanceDest` - destination-side reconciliation gap (finding 5)
3. `type` restricted to TRANSFER / CASH_OUT (finding 2)
4. `amount`, log-transformed given the skew (finding 3)
5. `hour_of_day` derived from `step` (finding 7)
6. Velocity features - transactions per origin account within a rolling window
   (not yet examined; the most promising *non*-leaking direction)

Correlation with `isFraud`, measured on a stratified sample of every fraud row
plus a systematic 1-in-200 slice of legitimate rows. That sample is deliberately
fraud-enriched (~20% fraud vs 0.13% in the full data) so weak signal is visible at
all; read the ordering as a ranking of promising features, not as population
correlations.

| Feature | Correlation |
|---|---|
""")
        for name, val in fraud_corr.items():
            fh.write(f"| {name} | {val:+.4f} |\n")

    print(f"Plots saved to {fig_dir}/")
    print(f"Findings written to {findings_path}")
    conn.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--figdir", default=DEFAULT_FIGDIR)
    p.add_argument("--findings", default=DEFAULT_FINDINGS)
    a = p.parse_args()
    main(a.db, a.figdir, a.findings)

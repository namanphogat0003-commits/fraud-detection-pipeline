# Power BI dashboard — build guide

The dashboard's job is to answer operational questions, not to restate model metrics.
A fraud operations lead wants to know what the review queue caught, what it missed, and
what that was worth. Build it to answer those.

## Data sources

| File | Rows | Use |
|---|---|---|
| `data/processed/scored_transactions.csv` | ~818k | Main fact table — every holdout transaction with its score, rank, and outcome |
| `reports/dashboard_summary.csv` | 7 | Pre-aggregated queue performance, for the capacity chart |

Generate both with:

```bash
python src/build_api_bundle.py
python src/score_batch.py
```

Load both into Power BI via **Get Data → Text/CSV**. They don't need a relationship —
the summary table stands alone.

## Columns in the fact table

| Column | Meaning |
|---|---|
| `fraud_score` | Model probability, 0–1 |
| `score_rank` | 1 = highest risk. This is the queue order |
| `in_review_queue` | 1 if inside the review budget (default top 1,000) |
| `risk_band` | low / elevated / high |
| `outcome` | caught / missed / false alarm / correctly ignored |
| `is_fraud` | Ground truth, for measuring the queue — not available at scoring time |

## Measures to create

In **Modeling → New measure**:

```
Fraud Exposure = CALCULATE(SUM(scored_transactions[amount]), scored_transactions[is_fraud] = 1)

Value Recovered =
CALCULATE(
    SUM(scored_transactions[amount]),
    scored_transactions[is_fraud] = 1,
    scored_transactions[in_review_queue] = 1
)

Recovery Rate = DIVIDE([Value Recovered], [Fraud Exposure])

Frauds Caught =
CALCULATE(
    COUNTROWS(scored_transactions),
    scored_transactions[is_fraud] = 1,
    scored_transactions[in_review_queue] = 1
)

Queue Size = CALCULATE(COUNTROWS(scored_transactions), scored_transactions[in_review_queue] = 1)

Queue Precision = DIVIDE([Frauds Caught], [Queue Size])
```

## Page 1 — Operations

A KPI row across the top, then two charts, then the queue itself. Keep it to one screen.

**KPI cards (four across):** `Fraud Exposure`, `Value Recovered`, `Recovery Rate`
(format as percentage), `Queue Precision`. Set Value Recovered and Fraud Exposure to
display in millions so the cards stay readable.

**Line chart — what the queue recovers.** Source: `dashboard_summary`.
X = `queue_size` (set the axis to logarithmic), Y = `value_recovered_pct`. Add
`lift_vs_random` as a tooltip field. This is the single most important visual in the
project: it shows a model with a modest PR-AUC recovering most of the fraud value from a
fraction of a percent of the volume.

**Donut — outcomes.** Legend = `outcome`, Values = count of rows. Filter to
`is_fraud = 1` first, so it reads as "of all fraud, how much did we catch" rather than
being swamped by the 800k correctly-ignored transactions.

**Table — the review queue.** Filter `in_review_queue = 1`, sort by `score_rank`
ascending. Columns: `score_rank`, `fraud_score`, `type`, `amount`, `nameDest`,
`dest_txn_count`, `outcome`. This is what an analyst would actually work through.
Apply conditional formatting on `fraud_score` (a single-hue gradient, not a rainbow).

## Page 2 — Model behaviour

**Histogram — score distribution by class.** X = `fraud_score` binned, Y = count,
Legend = `is_fraud`. Set the Y axis to logarithmic, otherwise the legitimate class
flattens everything else. Shows separation and where the bands sit.

**Column chart — fraud rate by hour.** X = `hour_of_day`, Y = fraud rate. Keep this on
the page as documentation of the timing artifact from the audit, with a text box
explaining that the night-time spike is the simulator's constant fraud generation against
a 1,259x swing in legitimate volume — not a real behavioural pattern.

**Slicers:** `type` and `risk_band`, in a single row above the visuals.

## Design notes

Use one accent colour, not a palette per chart. Give visuals room to breathe rather than
filling every pixel. Turn off gridlines where the data labels already carry the value.
Title each visual with what it shows, not what it is — "What the review queue recovers"
rather than "Line chart of recovery".

## Export

Save as `dashboard/fraud_dashboard.pbix`, and export a PNG or PDF of each page into
`reports/figures/` so the dashboard is visible to anyone reading the repo on GitHub
without opening Power BI.

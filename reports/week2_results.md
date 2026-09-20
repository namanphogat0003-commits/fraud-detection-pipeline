# Week 2 - Baseline models and a leakage audit

Trained on steps 1-323, evaluated on steps 324-743 - a
temporal split, so the model never sees a transaction that happens after the one
it is scoring. The test window holds **818,514 transactions** with
**4,570 frauds** (0.5583%) at its real class balance.
Only the training set was class-subsampled.

Accuracy is not reported anywhere below. At this class balance, predicting
"never fraud" scores 99.44% while catching nothing.

## The headline result

This dataset is riddled with artifacts of how PaySim generates fraud. Rather than
report the first good-looking score, each was measured and removed in turn:

| Track | What it removes | Best PR-AUC | Best model |
|---|---|---|---|
| A | nothing | **1.0000** | Random Forest |
| B | origin-side drain signature | **0.7841** | Random Forest |
| C | + destination-balance leak | **0.4223** | Random Forest |
| D | + hour-of-day artifact | **0.1497** | Random Forest |

**Track D is the number to defend.** Track A is what a PaySim notebook that skips
this analysis will proudly report.

## The three artifacts, measured

**1. The full-balance drain (removed in Track B).** `amount == oldbalanceOrg`
identifies 97.68% of fraud in the test window with zero false positives, because
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

| Model | Precision | Recall | F1 | PR-AUC | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| Baseline: full-balance-drain rule | 1.0000 | 0.9768 | 0.9883 | - | 4,464 | 0 | 106 |
| Baseline: PaySim isFlaggedFraud | 1.0000 | 0.0028 | 0.0057 | - | 13 | 0 | 4,557 |

## Track A - all features

| Model | Precision | Recall | F1 | PR-AUC | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 1.0000 | 0.9444 | 0.9714 | 0.9981 | 4,316 | 0 | 254 |
| Random Forest | 1.0000 | 0.9985 | 0.9992 | 1.0000 | 4,563 | 0 | 7 |

## Track B - origin drain removed

| Model | Precision | Recall | F1 | PR-AUC | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.1850 | 0.4466 | 0.2616 | 0.2319 | 2,041 | 8,990 | 2,529 |
| Random Forest | 0.7366 | 0.7361 | 0.7363 | 0.7841 | 3,364 | 1,203 | 1,206 |

## Track C - destination leak removed

| Model | Precision | Recall | F1 | PR-AUC | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.4336 | 0.1429 | 0.2149 | 0.1945 | 653 | 853 | 3,917 |
| Random Forest | 0.8573 | 0.2024 | 0.3275 | 0.4223 | 925 | 154 | 3,645 |

## Track D - timing artifact removed (the honest floor)

| Model | Precision | Recall | F1 | PR-AUC | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.1711 | 0.1361 | 0.1516 | 0.0837 | 622 | 3,014 | 3,948 |
| Random Forest | 0.1618 | 0.3512 | 0.2215 | 0.1497 | 1,605 | 8,315 | 2,965 |

## Reading the gap

Logistic Regression collapses far faster than Random Forest across the tracks,
which says the residual signal is non-linear - thresholds and interactions rather
than monotone relationships.

The distance from 1.0000 to 0.1497 is the real finding here.
Most published PaySim work reports something near the top of that range without
noting that the features encode the answer. The comparison itself - not the highest
score - is what belongs on a resume.

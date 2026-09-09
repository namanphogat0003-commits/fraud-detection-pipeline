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
| B | origin-side drain signature | **0.7836** | Random Forest |
| C | + destination-balance leak | **0.4229** | Random Forest |
| D | + hour-of-day artifact | **0.1501** | Random Forest |

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
| Logistic Regression | 0.9998 | 0.9976 | 0.9987 | 0.9981 | 4,559 | 1 | 11 |
| Random Forest | 1.0000 | 0.9998 | 0.9999 | 1.0000 | 4,569 | 0 | 1 |

## Track B - origin drain removed

| Model | Precision | Recall | F1 | PR-AUC | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.2772 | 0.2534 | 0.2647 | 0.2319 | 1,158 | 3,020 | 3,412 |
| Random Forest | 0.9109 | 0.6781 | 0.7775 | 0.7836 | 3,099 | 303 | 1,471 |

## Track C - destination leak removed

| Model | Precision | Recall | F1 | PR-AUC | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.2184 | 0.2766 | 0.2441 | 0.1945 | 1,264 | 4,523 | 3,306 |
| Random Forest | 0.6209 | 0.3383 | 0.4380 | 0.4229 | 1,546 | 944 | 3,024 |

## Track D - timing artifact removed (the honest floor)

| Model | Precision | Recall | F1 | PR-AUC | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.1112 | 0.3348 | 0.1669 | 0.0837 | 1,530 | 12,229 | 3,040 |
| Random Forest | 0.1409 | 0.6425 | 0.2311 | 0.1501 | 2,936 | 17,900 | 1,634 |

## Reading the gap

Logistic Regression collapses far faster than Random Forest across the tracks,
which says the residual signal is non-linear - thresholds and interactions rather
than monotone relationships.

The distance from 1.0000 to 0.1501 is the real finding here.
Most published PaySim work reports something near the top of that range without
noting that the features encode the answer. The comparison itself - not the highest
score - is what belongs on a resume.

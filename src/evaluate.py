"""
Week 2 - Metrics and evaluation plots.

Everything here is measured on the untouched test window, at the real class
balance. Accuracy is deliberately absent: at a 0.2% fraud rate a model that
predicts "never fraud" scores 99.8%, so the metric carries no information.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

# --- chart tokens (same system as the Week 1 plots) -------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.grid": True, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "semibold", "figure.dpi": 130,
})


def score_metrics(y_true, y_score, threshold=0.5, name="", is_hard_label=False):
    """Metrics at a fixed threshold. `is_hard_label` for rule baselines."""
    y_pred = y_score.astype(int) if is_hard_label else (y_score >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "model": name, "threshold": None if is_hard_label else threshold,
        "precision": float(p), "recall": float(r), "f1": float(f1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
    if not is_hard_label:
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return out


def best_f1_threshold(y_true, y_score):
    """Threshold maximising F1 on the PR curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision/recall have one more element than thresholds
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(precision[:-1]),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    if len(f1) == 0:
        return 0.5
    return float(thresholds[int(np.argmax(f1))])


def plot_pr_curves(results, y_true, fig_path, title):
    """One chart per track: PR curve for each model, plus the no-skill floor."""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for (name, y_score), colour in zip(results, [BLUE, ORANGE]):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, color=colour, linewidth=2,
                label=f"{name} (PR-AUC {ap:.3f})")

    no_skill = y_true.mean()
    ax.axhline(no_skill, color=AXIS, linewidth=1, linestyle="--")
    ax.text(0.02, no_skill, f" no-skill floor ({no_skill:.4f})",
            va="bottom", ha="left", fontsize=8, color=MUTED)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    legend = ax.legend(frameon=False, loc="best")
    for text in legend.get_texts():
        text.set_color(INK_2)
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def plot_feature_importance(names, importances, fig_path, title):
    order = np.argsort(importances)
    names = np.array(names)[order]
    importances = np.array(importances)[order]

    fig, ax = plt.subplots(figsize=(7.5, max(3.2, 0.34 * len(names) + 1.2)))
    ax.barh(names, importances, color=BLUE, height=0.62)
    for y, val in enumerate(importances):
        ax.text(val, y, f" {val:.3f}", va="center", ha="left",
                fontsize=8.5, color=INK_2)
    ax.set_title(title)
    ax.set_xlabel("Importance (mean decrease in impurity)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(importances) * 1.18)
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def metrics_table(rows, headline="") -> str:
    """Render a metrics list as a markdown table."""
    lines = []
    if headline:
        lines.append(headline)
    lines.append("| Model | Precision | Recall | F1 | PR-AUC | TP | FP | FN |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for m in rows:
        pr_auc = f"{m['pr_auc']:.4f}" if "pr_auc" in m else "-"
        lines.append(
            f"| {m['model']} | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{m['f1']:.4f} | {pr_auc} | {m['tp']:,} | {m['fp']:,} | {m['fn']:,} |"
        )
    return "\n".join(lines)

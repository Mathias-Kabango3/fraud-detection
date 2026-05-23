"""Extended evaluation for a trained fraud detection pipeline.

Loads a pickled ``imblearn.Pipeline`` (preprocessor + SMOTE + XGBoost),
runs it on the time-based holdout split, and reports:

* Headline metrics: ROC-AUC, PR-AUC, F1, precision, recall, accuracy
* Confusion matrix at the chosen decision threshold
* Per-class classification report
* Threshold sweep (best-F1 threshold, plus a few representative cutoffs)
* Plots: ROC curve, precision-recall curve, confusion matrix heatmap

Plots are written to ``models/`` (gitignored) so artifacts don't pollute git.

Usage:
    python -m src.model.evaluate
    python -m src.model.evaluate --sample-size 50000 --threshold 0.3
    python -m src.model.evaluate --model-path models/fraud_xgb.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.data.preprocess import (
    DEFAULT_VAL_DAYS,
    get_feature_target_split,
    load_raw,
    reduce_memory,
    time_based_split,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Metric computation
# --------------------------------------------------------------------------- #

def compute_metrics(
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
) -> dict:
    """Return the headline metric set at the given decision threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def threshold_sweep(
    y_true: pd.Series,
    y_proba: np.ndarray,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Compute precision/recall/F1 at several decision thresholds.

    Adds a row for the F1-optimal threshold (found via the PR curve).
    """
    thresholds = list(thresholds or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    precision_arr, recall_arr, pr_thresholds = precision_recall_curve(y_true, y_proba)
    # f1 calculation skipping the trailing 0-threshold pair sklearn appends
    f1_arr = 2 * precision_arr[:-1] * recall_arr[:-1] / np.maximum(
        precision_arr[:-1] + recall_arr[:-1], 1e-12,
    )
    best_idx = int(np.argmax(f1_arr))
    best_threshold = float(pr_thresholds[best_idx])
    log.info("Best-F1 threshold (from PR curve): %.3f (F1=%.4f)",
             best_threshold, float(f1_arr[best_idx]))
    thresholds = sorted(set(thresholds + [best_threshold]))

    rows = [compute_metrics(y_true, y_proba, t) for t in thresholds]
    df = pd.DataFrame(rows)[["threshold", "precision", "recall", "f1", "tp", "fp", "fn", "tn"]]
    df["is_best_f1"] = (df["threshold"] - best_threshold).abs() < 1e-9
    return df


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def plot_roc(y_true: pd.Series, y_proba: np.ndarray, out_path: Path) -> None:
    """Save an ROC curve plot to ``out_path``."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.4f})", color="#cc4444")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", alpha=0.7, label="random")
    ax.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    log.info("Saved ROC plot to %s", out_path)


def plot_pr(y_true: pd.Series, y_proba: np.ndarray, out_path: Path) -> None:
    """Save a precision-recall curve plot to ``out_path``."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    baseline = float(y_true.mean())
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"PR (AP = {pr_auc:.4f})", color="#4477cc")
    ax.axhline(baseline, ls="--", color="grey", alpha=0.7, label=f"baseline = {baseline:.3f}")
    ax.set(xlabel="recall", ylabel="precision", title="Precision-recall curve")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    log.info("Saved PR plot to %s", out_path)


def plot_confusion(metrics: dict, out_path: Path) -> None:
    """Save a confusion matrix heatmap to ``out_path``."""
    cm = np.array([[metrics["tn"], metrics["fp"]],
                   [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Reds", cbar=False, ax=ax,
        xticklabels=["pred 0", "pred 1"], yticklabels=["true 0", "true 1"],
    )
    ax.set(title=f"Confusion matrix @ threshold = {metrics['threshold']:.2f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    log.info("Saved confusion-matrix plot to %s", out_path)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the evaluation run."""
    parser = argparse.ArgumentParser(description="Evaluate trained fraud model")
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to the pickled pipeline (default: <repo>/models/fraud_xgb.pkl).",
    )
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Read only first N transaction rows (matches train.py for consistency).",
    )
    parser.add_argument(
        "--val-days", type=int, default=DEFAULT_VAL_DAYS,
        help="Days to hold out for time-based validation (must match training).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Decision threshold for the confusion matrix and headline metrics.",
    )
    return parser.parse_args()


def main() -> None:
    """Run end-to-end evaluation against the holdout split."""
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )

    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    txn_csv = repo_root / os.getenv("TRAIN_TRANSACTION_CSV", "data/raw/train_transaction.csv")
    id_csv = repo_root / os.getenv("TRAIN_IDENTITY_CSV", "data/raw/train_identity.csv")
    model_dir = repo_root / os.getenv("MODEL_DIR", "models")
    model_path = Path(args.model_path) if args.model_path else (
        model_dir / os.getenv("MODEL_FILENAME", "fraud_xgb.pkl")
    )

    log.info("Loading pipeline from %s", model_path)
    pipeline = joblib.load(model_path)

    log.info("Loading data...")
    if args.sample_size:
        sample = pd.read_csv(txn_csv, nrows=args.sample_size).merge(
            pd.read_csv(id_csv), on="TransactionID", how="left",
        )
        df = reduce_memory(sample)
    else:
        df = load_raw(txn_csv, id_csv)

    _, val_df = time_based_split(df, val_days=args.val_days)
    X_val, y_val = get_feature_target_split(val_df)
    assert y_val is not None, "Target column missing in validation split"

    log.info("Scoring %d holdout rows...", len(X_val))
    y_proba = pipeline.predict_proba(X_val)[:, 1]

    headline = compute_metrics(y_val, y_proba, threshold=args.threshold)
    log.info("Headline metrics:\n%s", json.dumps(headline, indent=2))
    log.info("Classification report:\n%s",
             classification_report(y_val, (y_proba >= args.threshold).astype(int),
                                   target_names=["legit", "fraud"], zero_division=0))

    sweep = threshold_sweep(y_val, y_proba)
    log.info("Threshold sweep:\n%s", sweep.to_string(index=False))

    plot_roc(y_val, y_proba, model_dir / "eval_roc.png")
    plot_pr(y_val, y_proba, model_dir / "eval_pr.png")
    plot_confusion(headline, model_dir / "eval_confusion.png")

    summary = {"headline": headline,
               "best_f1_threshold": float(sweep.loc[sweep["is_best_f1"], "threshold"].iloc[0]),
               "best_f1": float(sweep.loc[sweep["is_best_f1"], "f1"].iloc[0])}
    summary_path = model_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s", summary_path)


if __name__ == "__main__":
    main()

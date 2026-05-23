"""Training pipeline for IEEE-CIS fraud detection.

Combines the preprocessing pipeline from ``src.data.preprocess`` with SMOTE
(applied inside each CV fold so the validation fold never sees synthetic data)
and an XGBoost classifier. Logs metrics and the fit pipeline to MLflow and
also pickles the pipeline to ``models/fraud_xgb.pkl`` for the API to load.

Usage:
    # Quick smoke run on a sample
    python -m src.model.train --sample-size 50000 --cv-folds 2

    # Full training run (no sample, 5 folds, MLflow on)
    python -m src.model.train
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.data.preprocess import (
    DEFAULT_VAL_DAYS,
    build_preprocessor,
    get_feature_target_split,
    load_raw,
    time_based_split,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DEFAULT_XGB_PARAMS: dict = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_jobs": -1,
}

DECISION_THRESHOLD: float = 0.5


# --------------------------------------------------------------------------- #
# Pipeline assembly
# --------------------------------------------------------------------------- #

def build_training_pipeline(
    xgb_params: dict,
    random_seed: int,
    smote_k_neighbors: int = 5,
) -> ImbPipeline:
    """Build the full training pipeline: preprocess → SMOTE → XGBoost.

    Using ``imblearn.Pipeline`` (not ``sklearn.Pipeline``) so SMOTE is applied
    only during ``fit`` and skipped at ``predict``/``transform`` — preventing
    the validation fold from being resampled in cross-validation.

    ``build_preprocessor()`` returns a sklearn Pipeline, but imblearn forbids
    nested Pipelines as intermediate steps, so we splat its ``.steps`` in.
    """
    preprocessor_steps = build_preprocessor().steps
    return ImbPipeline(steps=[
        *preprocessor_steps,
        ("smote", SMOTE(k_neighbors=smote_k_neighbors, random_state=random_seed)),
        ("xgb", XGBClassifier(**xgb_params, random_state=random_seed)),
    ])


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #

def cross_validate(
    pipeline: ImbPipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    random_seed: int,
) -> dict:
    """Run stratified K-fold CV and return per-fold + aggregate ROC-AUC.

    SMOTE runs inside each fold via the imblearn pipeline, so validation
    folds remain unresampled (no leakage).
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    fold_aucs: list[float] = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        t0 = time.time()
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
        pipeline.fit(X_tr, y_tr)
        proba_va = pipeline.predict_proba(X_va)[:, 1]
        auc = float(roc_auc_score(y_va, proba_va))
        fold_aucs.append(auc)
        log.info("CV fold %d/%d: AUC=%.4f (%.1fs)", fold_idx, cv_folds, auc, time.time() - t0)
    return {
        "fold_aucs": fold_aucs,
        "cv_auc_mean": float(np.mean(fold_aucs)),
        "cv_auc_std": float(np.std(fold_aucs)),
    }


# --------------------------------------------------------------------------- #
# Holdout evaluation
# --------------------------------------------------------------------------- #

def evaluate_on_holdout(
    pipeline: ImbPipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    threshold: float = DECISION_THRESHOLD,
) -> dict:
    """Compute the headline metrics on the time-based holdout."""
    proba = pipeline.predict_proba(X_val)[:, 1]
    preds = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
    return {
        "holdout_roc_auc": float(roc_auc_score(y_val, proba)),
        "holdout_pr_auc": float(average_precision_score(y_val, proba)),
        "holdout_f1": float(f1_score(y_val, preds)),
        "holdout_precision": float(precision_score(y_val, preds, zero_division=0)),
        "holdout_recall": float(recall_score(y_val, preds, zero_division=0)),
        "holdout_tn": int(tn),
        "holdout_fp": int(fp),
        "holdout_fn": int(fn),
        "holdout_tp": int(tp),
        "decision_threshold": threshold,
    }


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def save_pipeline(pipeline: ImbPipeline, output_path: Path) -> None:
    """Pickle the fit pipeline so the API can ``joblib.load`` it directly."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    log.info("Saved pipeline to %s (%.1f MB)", output_path,
             output_path.stat().st_size / 1024 ** 2)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments. All have sensible defaults from .env or constants."""
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="If set, read only the first N transaction rows (for fast iteration).",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=int(os.getenv("N_CV_FOLDS", "5")),
        help="Number of stratified K-fold splits (default: 5 or N_CV_FOLDS env var).",
    )
    parser.add_argument(
        "--val-days", type=int, default=DEFAULT_VAL_DAYS,
        help="Days to hold out for time-based validation (default: 7).",
    )
    parser.add_argument(
        "--no-mlflow", action="store_true",
        help="Skip MLflow logging (useful for local debugging).",
    )
    parser.add_argument(
        "--skip-cv", action="store_true",
        help="Skip cross-validation; fit once and evaluate on holdout only.",
    )
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    """Seed Python's random and numpy. XGBoost is seeded via its constructor."""
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    """Train the fraud detection model end-to-end."""
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )

    args = parse_args()
    random_seed = int(os.getenv("RANDOM_SEED", "42"))
    set_global_seed(random_seed)

    repo_root = Path(__file__).resolve().parents[2]
    txn_csv = repo_root / os.getenv("TRAIN_TRANSACTION_CSV", "data/raw/train_transaction.csv")
    id_csv = repo_root / os.getenv("TRAIN_IDENTITY_CSV", "data/raw/train_identity.csv")
    model_dir = repo_root / os.getenv("MODEL_DIR", "models")
    model_filename = os.getenv("MODEL_FILENAME", "fraud_xgb.pkl")
    model_version = os.getenv("MODEL_VERSION", "0.1.0")

    log.info("Loading raw data...")
    if args.sample_size:
        log.info("Using a %d-row sample for fast iteration", args.sample_size)
        sample = pd.read_csv(txn_csv, nrows=args.sample_size).merge(
            pd.read_csv(id_csv), on="TransactionID", how="left",
        )
        from src.data.preprocess import reduce_memory
        df = reduce_memory(sample)
    else:
        df = load_raw(txn_csv, id_csv)

    train_df, val_df = time_based_split(df, val_days=args.val_days)
    X_train, y_train = get_feature_target_split(train_df)
    X_val, y_val = get_feature_target_split(val_df)
    assert y_train is not None and y_val is not None, "Target column missing"

    pipeline = build_training_pipeline(DEFAULT_XGB_PARAMS, random_seed=random_seed)

    use_mlflow = not args.no_mlflow
    if use_mlflow:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud_detection"))

    run_ctx = mlflow.start_run(run_name=f"train_{int(time.time())}") if use_mlflow else _NullCtx()
    with run_ctx:
        if use_mlflow:
            mlflow.log_params({**DEFAULT_XGB_PARAMS, "cv_folds": args.cv_folds,
                               "val_days": args.val_days, "sample_size": args.sample_size or "full",
                               "model_version": model_version, "random_seed": random_seed})

        cv_metrics: dict = {}
        if not args.skip_cv:
            log.info("Running %d-fold stratified CV...", args.cv_folds)
            cv_metrics = cross_validate(pipeline, X_train, y_train, args.cv_folds, random_seed)
            log.info("CV ROC-AUC: %.4f ± %.4f", cv_metrics["cv_auc_mean"], cv_metrics["cv_auc_std"])
            if use_mlflow:
                mlflow.log_metric("cv_auc_mean", cv_metrics["cv_auc_mean"])
                mlflow.log_metric("cv_auc_std", cv_metrics["cv_auc_std"])
                for i, auc in enumerate(cv_metrics["fold_aucs"], start=1):
                    mlflow.log_metric(f"cv_fold_{i}_auc", auc)

        log.info("Fitting final pipeline on full training pool (%d rows)...", len(X_train))
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        log.info("Final fit: %.1fs", time.time() - t0)

        log.info("Evaluating on time-based holdout (%d rows)...", len(X_val))
        holdout_metrics = evaluate_on_holdout(pipeline, X_val, y_val)
        log.info("Holdout metrics:\n%s", json.dumps(holdout_metrics, indent=2))

        if use_mlflow:
            mlflow.log_metrics({k: v for k, v in holdout_metrics.items()
                                if isinstance(v, (int, float))})

        model_path = model_dir / model_filename
        save_pipeline(pipeline, model_path)
        if use_mlflow:
            mlflow.log_artifact(str(model_path), artifact_path="model_pkl")
            mlflow.sklearn.log_model(pipeline, artifact_path="sklearn_model")

    target_auc = 0.88
    auc = holdout_metrics["holdout_roc_auc"]
    log.info("=" * 60)
    log.info("FINAL holdout ROC-AUC = %.4f  (target > %.2f) — %s",
             auc, target_auc, "PASS" if auc > target_auc else "BELOW TARGET")
    log.info("=" * 60)


class _NullCtx:
    """No-op context manager used when --no-mlflow is set."""
    def __enter__(self): return self
    def __exit__(self, *a): return False


if __name__ == "__main__":
    main()

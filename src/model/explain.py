"""SHAP-based explainability for the fraud detection pipeline.

The trained ``imblearn.Pipeline`` has the shape:

    preprocess steps... -> SMOTE -> XGBClassifier

SHAP's ``TreeExplainer`` runs against the *XGBoost step only*, on
preprocessed features. So the per-prediction flow is:

    raw row  --[preprocessing only, skip SMOTE]-->  transformed row
    transformed row  --[TreeExplainer]-->  per-feature SHAP values
    SHAP values  --[top-k by |value|, signed direction]-->  API output

Public API:
    * ``build_explainer(pipeline)``           — cache once on app startup.
    * ``transform_for_xgb(pipeline, X)``      — apply preprocessor only.
    * ``explain_prediction(explainer, X, k)`` — return top-k feature impacts.

CLI usage (sanity / demo):
    python -m src.model.explain
    python -m src.model.explain --n-rows 5 --top-k 5
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
import shap
from dotenv import load_dotenv
from imblearn.pipeline import Pipeline as ImbPipeline

from src.data.preprocess import (
    DEFAULT_VAL_DAYS,
    get_feature_target_split,
    load_raw,
    reduce_memory,
    time_based_split,
)

log = logging.getLogger(__name__)


# Steps in the training pipeline that are not part of the inference preprocessor.
NON_PREPROCESSOR_STEPS: set[str] = {"smote", "xgb"}


# --------------------------------------------------------------------------- #
# Pipeline introspection
# --------------------------------------------------------------------------- #

def transform_for_xgb(pipeline: ImbPipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Apply every step before XGBoost, skipping the SMOTE sampler.

    SMOTE has no ``transform`` method (samplers act only at fit-time), so we
    walk the pipeline by name and apply ``transform`` to everything except
    the sampler and the classifier.
    """
    X_t = X
    for name, step in pipeline.steps:
        if name in NON_PREPROCESSOR_STEPS:
            continue
        X_t = step.transform(X_t)
    assert isinstance(X_t, pd.DataFrame), "Preprocessor should output a DataFrame"
    return X_t


def build_explainer(pipeline: ImbPipeline) -> shap.TreeExplainer:
    """Construct a SHAP ``TreeExplainer`` for the pipeline's XGBoost step.

    Build once and reuse — explainer construction is cheap but per-call
    SHAP values dominate request latency.
    """
    xgb_model = pipeline.named_steps["xgb"]
    return shap.TreeExplainer(xgb_model)


# --------------------------------------------------------------------------- #
# Per-prediction explanation
# --------------------------------------------------------------------------- #

def _shap_values_2d(explainer: shap.TreeExplainer, X: pd.DataFrame) -> np.ndarray:
    """Return SHAP values as a 2-D array of shape (n_samples, n_features).

    ``TreeExplainer.shap_values`` returns slightly different shapes depending
    on the model's output format (binary vs. multiclass, ``output_margin``,
    SHAP version). Normalize all of them to 2-D.
    """
    raw = explainer.shap_values(X)
    arr = np.asarray(raw)
    if arr.ndim == 3:
        # (n_samples, n_features, 2) for binary classifiers — keep the positive class.
        arr = arr[..., 1]
    return arr


def explain_prediction(
    explainer: shap.TreeExplainer,
    X_transformed: pd.DataFrame,
    top_k: int = 3,
) -> list[list[dict]]:
    """Return the top-k feature contributions per row, signed by direction.

    For one row of ``X_transformed`` the result is a list of dicts shaped like::

        [
            {"feature": "TransactionAmt", "impact": 0.34, "direction": "increases_risk"},
            {"feature": "card4_visa",     "impact": 0.21, "direction": "decreases_risk"},
            ...
        ]

    ``impact`` is the absolute SHAP value; ``direction`` is derived from its sign
    (positive SHAP → increases predicted fraud probability → ``increases_risk``).
    """
    shap_values = _shap_values_2d(explainer, X_transformed)
    feature_names = list(X_transformed.columns)
    results: list[list[dict]] = []
    for row_values in shap_values:
        abs_values = np.abs(row_values)
        top_indices = np.argsort(abs_values)[-top_k:][::-1]
        row_explanation = [
            {
                "feature": feature_names[i],
                "impact": float(abs_values[i]),
                "direction": "increases_risk" if row_values[i] > 0 else "decreases_risk",
            }
            for i in top_indices
        ]
        results.append(row_explanation)
    return results


# --------------------------------------------------------------------------- #
# Global summary plot (offline analysis, not served by the API)
# --------------------------------------------------------------------------- #

def save_global_summary(
    explainer: shap.TreeExplainer,
    X_transformed: pd.DataFrame,
    out_path: Path,
    max_display: int = 20,
) -> None:
    """Write a SHAP summary (beeswarm) plot showing the top features overall."""
    shap_values = _shap_values_2d(explainer, X_transformed)
    plt.figure(figsize=(8, 8))
    shap.summary_plot(shap_values, X_transformed, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Saved SHAP summary plot to %s", out_path)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the demo run."""
    parser = argparse.ArgumentParser(description="SHAP demo for fraud model")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to pickled pipeline (default: models/fraud_xgb.pkl).")
    parser.add_argument("--sample-size", type=int, default=20000,
                        help="Rows to read from the raw CSVs (default: 20000).")
    parser.add_argument("--val-days", type=int, default=DEFAULT_VAL_DAYS,
                        help="Days to hold out for time-based validation.")
    parser.add_argument("--n-rows", type=int, default=3,
                        help="Number of holdout rows to explain (default: 3).")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Top-K features to report per row (default: 3, matches the API).")
    parser.add_argument("--skip-summary-plot", action="store_true",
                        help="Skip the (slow) global summary plot.")
    return parser.parse_args()


def main() -> None:
    """Demo: load the pipeline, explain a few rows, and write a global summary plot."""
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

    log.info("Loading %d-row sample for demo...", args.sample_size)
    sample = pd.read_csv(txn_csv, nrows=args.sample_size).merge(
        pd.read_csv(id_csv), on="TransactionID", how="left",
    )
    df = reduce_memory(sample)
    _, val_df = time_based_split(df, val_days=args.val_days)
    X_val, _y_val = get_feature_target_split(val_df)
    demo_rows = X_val.head(args.n_rows)

    explainer = build_explainer(pipeline)
    X_transformed = transform_for_xgb(pipeline, demo_rows)
    proba = pipeline.predict_proba(demo_rows)[:, 1]
    explanations = explain_prediction(explainer, X_transformed, top_k=args.top_k)

    log.info("Per-row explanations (matches API output shape):")
    for i, (p, expl) in enumerate(zip(proba, explanations)):
        print(json.dumps({
            "row": i,
            "fraud_probability": float(p),
            "top_reasons": expl,
        }, indent=2))

    if not args.skip_summary_plot:
        summary_sample = transform_for_xgb(pipeline, X_val.head(min(2000, len(X_val))))
        save_global_summary(explainer, summary_sample, model_dir / "shap_summary.png")


if __name__ == "__main__":
    main()

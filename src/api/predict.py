"""Prediction service for the fraud detection API.

Loads the pickled training pipeline + SHAP explainer once at startup and
serves per-request predictions with top-K feature explanations.

The class is decoupled from FastAPI so it can be reused from notebooks, tests,
or batch jobs without HTTP overhead.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.api.schema import FeatureImpact, PredictionResponse, TransactionRequest
from src.model.explain import build_explainer, explain_prediction, transform_for_xgb

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configuration defaults
# --------------------------------------------------------------------------- #

DEFAULT_DECISION_THRESHOLD: float = 0.5
DEFAULT_RISK_MEDIUM_THRESHOLD: float = 0.3
DEFAULT_RISK_HIGH_THRESHOLD: float = 0.7
DEFAULT_TOP_K: int = 3

# Columns the EngineeredFeatures and AmountPerGroupRatio steps add downstream
# of the input. They appear in the ColumnTransformer's ``feature_names_in_``,
# but a caller never sends them — we must subtract them out when computing the
# expected *input* schema.
_DOWNSTREAM_ENGINEERED_COLS: frozenset[str] = frozenset({
    "log_amt", "hour_of_day", "day_index", "email_match", "amt_per_card1_mean_ratio",
})


# --------------------------------------------------------------------------- #
# Predictor
# --------------------------------------------------------------------------- #

class FraudPredictor:
    """Stateful predictor: loads the pipeline + SHAP explainer once.

    Configure thresholds via constructor args (or pass values read from .env
    at the call site). Call ``load()`` exactly once at startup, then ``predict``
    for each incoming request.
    """

    def __init__(
        self,
        model_path: Path,
        model_version: str = "0.1.0",
        decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
        risk_medium_threshold: float = DEFAULT_RISK_MEDIUM_THRESHOLD,
        risk_high_threshold: float = DEFAULT_RISK_HIGH_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        if not (0.0 <= risk_medium_threshold <= risk_high_threshold <= 1.0):
            raise ValueError("risk_medium_threshold must be <= risk_high_threshold and both in [0, 1]")
        self.model_path = Path(model_path)
        self.model_version = model_version
        self.decision_threshold = decision_threshold
        self.risk_medium_threshold = risk_medium_threshold
        self.risk_high_threshold = risk_high_threshold
        self.top_k = top_k
        self._pipeline = None
        self._explainer = None
        self._expected_input_columns: list[str] = []

    @property
    def is_loaded(self) -> bool:
        """True iff the pipeline has been loaded into memory."""
        return self._pipeline is not None and self._explainer is not None

    def load(self) -> None:
        """Read the pickled pipeline from disk and build the SHAP explainer.

        If the local model file is missing and ``HF_MODEL_REPO_ID`` is set, the
        artifact is first fetched from Hugging Face Hub. This lets local dev
        use a trained-locally pickle while Render (and other ephemeral hosts)
        pulls a versioned artifact on cold start.
        """
        if not self.model_path.exists():
            self._maybe_download_from_hf_hub()
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                f"Either train locally (`python -m src.model.train`) "
                f"or set HF_MODEL_REPO_ID to download from Hugging Face Hub."
            )

        log.info("Loading model from %s", self.model_path)
        self._pipeline = joblib.load(self.model_path)
        log.info("Building SHAP explainer...")
        self._explainer = build_explainer(self._pipeline)
        self._expected_input_columns = self._discover_expected_input_columns()
        log.info("Model loaded (version=%s, decision_threshold=%.2f, expected_input_columns=%d)",
                 self.model_version, self.decision_threshold, len(self._expected_input_columns))

    def _maybe_download_from_hf_hub(self) -> None:
        """If ``HF_MODEL_REPO_ID`` is set, download the artifact to ``model_path``.

        No-op when the env var isn't set, so local development is unaffected.
        Failures here are logged; the caller's ``FileNotFoundError`` check then
        produces the final clean error message.
        """
        repo_id = os.getenv("HF_MODEL_REPO_ID")
        if not repo_id:
            return

        filename = os.getenv("HF_MODEL_FILENAME", self.model_path.name)
        log.info("Downloading %s from Hugging Face Hub (%s)...", filename, repo_id)
        try:
            from huggingface_hub import hf_hub_download

            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(self.model_path.parent),
                token=os.getenv("HF_TOKEN"),  # only needed for private repos
            )
            # hf_hub_download returns the actual on-disk path; align with model_path.
            downloaded_path = Path(downloaded)
            if downloaded_path != self.model_path:
                downloaded_path.replace(self.model_path)
            log.info("Downloaded model to %s", self.model_path)
        except Exception:  # noqa: BLE001
            log.exception("Failed to download model from Hugging Face Hub")

    def _discover_expected_input_columns(self) -> list[str]:
        """Recover the input-time column list from the fitted ColumnTransformer.

        The ColumnTransformer sees columns *after* engineered features have been
        added upstream, so we subtract those out to get what callers must supply.
        Falls back to an empty list — callers will then send what they have.
        """
        column_transformer = self._pipeline.named_steps.get("encode_and_impute")  # type: ignore[union-attr]
        if column_transformer is None or not hasattr(column_transformer, "feature_names_in_"):
            log.warning("Could not infer expected input columns from pipeline")
            return []
        columns_at_ct = list(column_transformer.feature_names_in_)
        return [c for c in columns_at_ct if c not in _DOWNSTREAM_ENGINEERED_COLS]

    def predict(self, request: TransactionRequest) -> PredictionResponse:
        """Score a single transaction and explain the prediction."""
        if not self.is_loaded:
            raise RuntimeError("Predictor.load() must be called before predict()")

        df = self._request_to_dataframe(request)

        proba = float(self._pipeline.predict_proba(df)[:, 1][0])  # type: ignore[union-attr]

        top_reasons = self._explain(df)

        return PredictionResponse(
            transaction_id=request.transaction_id,
            fraud_probability=proba,
            prediction="FRAUD" if proba >= self.decision_threshold else "NOT_FRAUD",
            risk_level=self._risk_level(proba),
            top_reasons=top_reasons,
            model_version=self.model_version,
        )

    # ----------------------------------------------------------------------- #
    # Internals
    # ----------------------------------------------------------------------- #

    def _request_to_dataframe(self, request: TransactionRequest) -> pd.DataFrame:
        """Convert a TransactionRequest into the single-row DataFrame the pipeline expects.

        Pads with NaN for every training-time column the caller didn't send, so
        ``ColumnTransformer`` finds the schema it was fit on. Extra fields the
        caller sent but training never saw are kept too — the pipeline ignores them.
        """
        feature_dict: dict[str, Any] = request.to_feature_dict()
        df = pd.DataFrame([feature_dict])
        if self._expected_input_columns:
            for col in self._expected_input_columns:
                if col not in df.columns:
                    df[col] = np.nan
        return df

    def _explain(self, df: pd.DataFrame) -> list[FeatureImpact]:
        """Compute top-K SHAP-based feature contributions for the single-row df.

        Failures fall back to an empty list rather than failing the request — a
        scored prediction without explanations is still useful.
        """
        try:
            X_transformed = transform_for_xgb(self._pipeline, df)  # type: ignore[arg-type]
            explanations = explain_prediction(self._explainer, X_transformed, top_k=self.top_k)  # type: ignore[arg-type]
            return [FeatureImpact(**reason) for reason in explanations[0]]
        except Exception:  # noqa: BLE001 — explanations are best-effort
            log.exception("SHAP explanation failed; returning empty top_reasons")
            return []

    def _risk_level(self, proba: float) -> str:
        """Bucket a probability into LOW / MEDIUM / HIGH."""
        if proba >= self.risk_high_threshold:
            return "HIGH"
        if proba >= self.risk_medium_threshold:
            return "MEDIUM"
        return "LOW"

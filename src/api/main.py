"""FastAPI service for the fraud detection model.

Endpoints:
    * ``POST /predict`` — score one transaction; return probability + top-3 SHAP reasons.
    * ``GET /health``   — liveness/readiness with the loaded model version.
    * ``GET /docs``     — auto-generated Swagger UI (FastAPI default).

Run locally:
    uvicorn src.api.main:app --reload --port 8000

The pickled pipeline is loaded once at startup via the FastAPI lifespan
context. If the file is missing the app still starts, but ``/predict`` returns
503 and ``/health`` reports ``model_loaded=False``.
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status

from src.api.predict import (
    DEFAULT_DECISION_THRESHOLD,
    DEFAULT_RISK_HIGH_THRESHOLD,
    DEFAULT_RISK_MEDIUM_THRESHOLD,
    FraudPredictor,
)
from src.api.schema import HealthResponse, PredictionResponse, TransactionRequest


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #

load_dotenv()
logging.basicConfig(
    level=os.getenv("API_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("api")

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / os.getenv("MODEL_DIR", "models") / os.getenv("MODEL_FILENAME", "fraud_xgb.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "0.1.0")


def _float_env(name: str, default: float) -> float:
    """Read a float-valued env var with a fallback."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("Could not parse %s=%r as float; using default %.3f", name, raw, default)
        return default


PREDICTOR = FraudPredictor(
    model_path=MODEL_PATH,
    model_version=MODEL_VERSION,
    decision_threshold=_float_env("DECISION_THRESHOLD", DEFAULT_DECISION_THRESHOLD),
    risk_medium_threshold=_float_env("RISK_MEDIUM_THRESHOLD", DEFAULT_RISK_MEDIUM_THRESHOLD),
    risk_high_threshold=_float_env("RISK_HIGH_THRESHOLD", DEFAULT_RISK_HIGH_THRESHOLD),
)


# --------------------------------------------------------------------------- #
# Lifespan: load the model once at startup
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load the pipeline at startup; nothing to tear down at shutdown."""
    try:
        PREDICTOR.load()
    except FileNotFoundError as exc:
        log.error("Model not loaded: %s", exc)
        log.error("API will start but /predict will return 503 until the model is available.")
    except Exception:  # noqa: BLE001
        log.exception("Unexpected error during model load")
    yield


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Scores transactions with an XGBoost model trained on the IEEE-CIS dataset, "
        "and returns SHAP-based per-feature explanations."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Middleware: attach a per-request correlation id to every log line
# --------------------------------------------------------------------------- #

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Generate a short request id and log start/end for each request."""
    request_id = uuid.uuid4().hex[:8]
    log.info("[%s] %s %s", request_id, request.method, request.url.path)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    log.info("[%s] -> %d", request_id, response.status_code)
    return response


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    """Liveness / readiness check."""
    return HealthResponse(
        status="ok" if PREDICTOR.is_loaded else "degraded",
        model_loaded=PREDICTOR.is_loaded,
        model_version=PREDICTOR.model_version,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["scoring"],
    responses={
        503: {"description": "Model not loaded."},
        500: {"description": "Internal scoring error."},
    },
)
async def predict(request: TransactionRequest) -> PredictionResponse:
    """Score a single transaction and return probability + top-3 SHAP reasons."""
    if not PREDICTOR.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model not loaded. Train a model (`python -m src.model.train`) "
                f"so that {MODEL_PATH} exists, then restart the API."
            ),
        )

    try:
        return PREDICTOR.predict(request)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("Scoring failed for transaction_id=%s", request.transaction_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scoring failed: {type(exc).__name__}",
        )

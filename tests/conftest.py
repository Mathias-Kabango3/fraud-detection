"""Pytest fixtures for the fraud detection API.

A tiny model is trained on synthetic data once per session and saved to a
temp directory. The global ``PREDICTOR`` in ``src.api.main`` is then
monkeypatched per test to point at that fixture (or at a missing path, to
exercise the 503 branch).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.data.preprocess import (
    ID_COL,
    LOW_CARD_CATEGORICAL,
    TARGET,
    TIME_COL,
    get_feature_target_split,
    time_based_split,
)
from src.model.train import DEFAULT_XGB_PARAMS, build_training_pipeline


# --------------------------------------------------------------------------- #
# Synthetic data + fixture model
# --------------------------------------------------------------------------- #

def _make_synthetic_dataframe(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic transaction-like frame with all columns the pipeline needs.

    Class balance is roughly 50/50 so SMOTE has plenty of minority samples; this
    is a fixture, not a realistic distribution.
    """
    rng = np.random.default_rng(seed)
    seconds_per_day = 24 * 3600
    df = pd.DataFrame({
        ID_COL: range(n),
        TARGET: rng.integers(0, 2, n),
        TIME_COL: rng.integers(seconds_per_day, 30 * seconds_per_day, n),
        "TransactionAmt": rng.uniform(1.0, 500.0, n),
        "ProductCD": rng.choice(["W", "C", "R", "H", "S"], n),
        "card1": rng.integers(1000, 20000, n),
        "card2": rng.uniform(100.0, 600.0, n),
        "card3": rng.uniform(100.0, 200.0, n),
        "card4": rng.choice(["visa", "mastercard", "amex"], n),
        "card5": rng.uniform(100.0, 300.0, n),
        "card6": rng.choice(["debit", "credit"], n),
        "addr1": rng.uniform(100.0, 500.0, n),
        "addr2": rng.uniform(50.0, 100.0, n),
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "hotmail.com"], n),
        "R_emaildomain": rng.choice(["gmail.com", "yahoo.com", None], n),
        "DeviceType": rng.choice(["desktop", "mobile", None], n),
        "DeviceInfo": rng.choice(["iOS", "Windows", "Android", None], n),
    })
    # The ColumnTransformer expects all M1..M9 columns; fill with T/F/None
    for m in [c for c in LOW_CARD_CATEGORICAL if c.startswith("M")]:
        df[m] = rng.choice(["T", "F", None], n)
    # A few V columns so the model has more to chew on
    for v in [f"V{i}" for i in range(1, 11)]:
        df[v] = rng.normal(0, 1, n).astype("float32")
    return df


@pytest.fixture(scope="session")
def fixture_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train a tiny pipeline on synthetic data and return the pickle path.

    Session-scoped because training is the slow part (~1 second) and the
    model itself is stateless across tests.
    """
    df = _make_synthetic_dataframe()
    train_df, _ = time_based_split(df, val_days=2)
    X_train, y_train = get_feature_target_split(train_df)
    assert y_train is not None

    tiny_xgb_params = {**DEFAULT_XGB_PARAMS, "n_estimators": 10, "max_depth": 3, "n_jobs": 1}
    pipeline = build_training_pipeline(tiny_xgb_params, random_seed=42, smote_k_neighbors=3)
    pipeline.fit(X_train, y_train)

    path = tmp_path_factory.mktemp("model") / "test_model.pkl"
    joblib.dump(pipeline, path)
    return path


# --------------------------------------------------------------------------- #
# TestClient fixtures
# --------------------------------------------------------------------------- #

def _reset_predictor(monkeypatch: pytest.MonkeyPatch, model_path: Path) -> None:
    """Point the global PREDICTOR at the given path and clear its loaded state."""
    from src.api import main as api_main

    monkeypatch.setattr(api_main.PREDICTOR, "model_path", model_path)
    monkeypatch.setattr(api_main.PREDICTOR, "_pipeline", None)
    monkeypatch.setattr(api_main.PREDICTOR, "_explainer", None)
    monkeypatch.setattr(api_main.PREDICTOR, "_expected_input_columns", [])


@pytest.fixture
def client_with_model(
    fixture_model_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    """TestClient bound to an app that successfully loads the fixture model."""
    from src.api import main as api_main

    _reset_predictor(monkeypatch, fixture_model_path)
    with TestClient(api_main.app) as client:
        yield client


@pytest.fixture
def client_no_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    """TestClient bound to an app where the model file is missing on disk."""
    from src.api import main as api_main

    _reset_predictor(monkeypatch, tmp_path / "does_not_exist.pkl")
    with TestClient(api_main.app) as client:
        yield client


# --------------------------------------------------------------------------- #
# Request payloads
# --------------------------------------------------------------------------- #

@pytest.fixture
def valid_payload() -> dict[str, Any]:
    """A minimal request that satisfies TransactionRequest validation."""
    return {
        "transaction_id": "txn_test_001",
        "TransactionDT": 86400,
        "TransactionAmt": 100.0,
        "ProductCD": "W",
        "card1": 13926,
        "card4": "visa",
        "card6": "debit",
        "P_emaildomain": "gmail.com",
        "R_emaildomain": "gmail.com",
        "DeviceType": "desktop",
        "M1": "T",
        "M2": "T",
        "extra": {"V1": 1.0, "V2": 0.5},
    }

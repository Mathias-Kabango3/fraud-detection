"""Endpoint tests for the fraud detection API.

Each test uses a TestClient bound to one of two fixture flavors:
  * ``client_with_model``  — a tiny pipeline trained on synthetic data is loaded.
  * ``client_no_model``    — the configured model path is missing on disk.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient


# --------------------------------------------------------------------------- #
# /health
# --------------------------------------------------------------------------- #

class TestHealth:
    """Liveness/readiness endpoint."""

    def test_returns_ok_when_model_loaded(self, client_with_model: TestClient) -> None:
        response = client_with_model.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert isinstance(body["model_version"], str) and body["model_version"]

    def test_returns_degraded_when_model_missing(self, client_no_model: TestClient) -> None:
        response = client_no_model.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "degraded"
        assert body["model_loaded"] is False


# --------------------------------------------------------------------------- #
# /predict
# --------------------------------------------------------------------------- #

class TestPredict:
    """Scoring endpoint."""

    def test_happy_path_returns_valid_response_shape(
        self, client_with_model: TestClient, valid_payload: dict[str, Any]
    ) -> None:
        response = client_with_model.post("/predict", json=valid_payload)
        assert response.status_code == 200, response.text
        body = response.json()

        assert body["transaction_id"] == valid_payload["transaction_id"]
        assert 0.0 <= body["fraud_probability"] <= 1.0
        assert body["prediction"] in ("FRAUD", "NOT_FRAUD")
        assert body["risk_level"] in ("LOW", "MEDIUM", "HIGH")
        assert isinstance(body["top_reasons"], list)
        assert isinstance(body["model_version"], str)

    def test_top_reasons_have_correct_shape(
        self, client_with_model: TestClient, valid_payload: dict[str, Any]
    ) -> None:
        response = client_with_model.post("/predict", json=valid_payload)
        assert response.status_code == 200
        top_reasons = response.json()["top_reasons"]
        if not top_reasons:
            pytest.skip("SHAP returned no reasons; covered by predict.py best-effort handling.")
        assert len(top_reasons) <= 3
        for reason in top_reasons:
            assert isinstance(reason["feature"], str) and reason["feature"]
            assert isinstance(reason["impact"], float) and reason["impact"] >= 0
            assert reason["direction"] in ("increases_risk", "decreases_risk")

    def test_prediction_matches_threshold(
        self, client_with_model: TestClient, valid_payload: dict[str, Any]
    ) -> None:
        """`prediction` label must agree with the configured decision threshold."""
        from src.api import main as api_main

        response = client_with_model.post("/predict", json=valid_payload)
        body = response.json()
        threshold = api_main.PREDICTOR.decision_threshold
        expected = "FRAUD" if body["fraud_probability"] >= threshold else "NOT_FRAUD"
        assert body["prediction"] == expected

    def test_missing_required_field_returns_422(self, client_with_model: TestClient) -> None:
        # Drop TransactionAmt — required by Pydantic.
        payload = {"transaction_id": "x", "TransactionDT": 86400}
        response = client_with_model.post("/predict", json=payload)
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert any("TransactionAmt" in str(error.get("loc", "")) for error in detail)

    def test_negative_amount_returns_422(
        self, client_with_model: TestClient, valid_payload: dict[str, Any]
    ) -> None:
        # TransactionAmt has gt=0 in the schema.
        valid_payload["TransactionAmt"] = -1.0
        response = client_with_model.post("/predict", json=valid_payload)
        assert response.status_code == 422

    def test_malformed_json_returns_422(self, client_with_model: TestClient) -> None:
        response = client_with_model.post(
            "/predict",
            content="not even json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_no_model_returns_503(
        self, client_no_model: TestClient, valid_payload: dict[str, Any]
    ) -> None:
        response = client_no_model.post("/predict", json=valid_payload)
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]

    def test_extra_features_accepted(
        self, client_with_model: TestClient, valid_payload: dict[str, Any]
    ) -> None:
        """Unknown columns in `extra` must not break scoring."""
        valid_payload["extra"]["some_unknown_field"] = 1.23
        response = client_with_model.post("/predict", json=valid_payload)
        assert response.status_code == 200


# --------------------------------------------------------------------------- #
# /docs and OpenAPI surface
# --------------------------------------------------------------------------- #

class TestDocs:
    """Swagger / OpenAPI surface."""

    def test_docs_reachable(self, client_with_model: TestClient) -> None:
        assert client_with_model.get("/docs").status_code == 200

    def test_openapi_advertises_both_endpoints(self, client_with_model: TestClient) -> None:
        openapi = client_with_model.get("/openapi.json").json()
        assert "/health" in openapi["paths"]
        assert "/predict" in openapi["paths"]
        assert "get" in openapi["paths"]["/health"]
        assert "post" in openapi["paths"]["/predict"]

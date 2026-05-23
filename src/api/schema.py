"""Pydantic request and response models for the fraud detection API.

The IEEE-CIS dataset has ~430 columns, most of which are anonymized. Requiring
clients to populate every single one would be unrealistic; the trained
pipeline tolerates missing values via median imputation and XGBoost's native
NA handling. So the request schema declares:

* A small set of **essential** fields that the preprocessor reads directly
  (``TransactionDT``, ``TransactionAmt``, ``card1``, etc.).
* A free-form ``extra`` dict for the V/C/D-block features clients may have.

The example payload below renders in the Swagger UI at ``/docs``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Request
# --------------------------------------------------------------------------- #

class TransactionRequest(BaseModel):
    """A single transaction to be scored.

    ``transaction_id`` is echoed back in the response so callers can correlate
    asynchronous requests. ``TransactionDT`` and ``TransactionAmt`` are the only
    truly required features; the rest are optional but improve quality.
    """

    transaction_id: str = Field(..., min_length=1, max_length=128,
                                 description="Caller-supplied transaction identifier, echoed in the response.")

    # Required for feature engineering and the AmountPerGroupRatio step
    TransactionDT: int = Field(..., ge=0,
                                description="Seconds elapsed from the dataset reference point.")
    TransactionAmt: float = Field(..., gt=0,
                                   description="Transaction amount in USD.")

    # Product / card metadata — strongly recommended
    ProductCD: str | None = Field(None, description="Product code (e.g. 'W', 'C', 'R', 'H', 'S').")
    card1: int | None = Field(None, description="Anonymized card identifier 1.")
    card2: float | None = Field(None, description="Anonymized card metadata 2.")
    card3: float | None = Field(None, description="Anonymized card metadata 3.")
    card4: str | None = Field(None, description="Card network (visa / mastercard / discover / amex).")
    card5: float | None = Field(None, description="Anonymized card metadata 5.")
    card6: str | None = Field(None, description="Card type (debit / credit).")

    # Address
    addr1: float | None = Field(None, description="Anonymized billing address code 1.")
    addr2: float | None = Field(None, description="Anonymized billing address code 2.")

    # Email
    P_emaildomain: str | None = Field(None, description="Purchaser email domain.")
    R_emaildomain: str | None = Field(None, description="Recipient email domain.")

    # Identity (optional, only ~25% of transactions have these)
    DeviceType: str | None = Field(None, description="'desktop' or 'mobile'.")
    DeviceInfo: str | None = Field(None, description="Free-text device descriptor.")

    # Match flags — usually 'T' / 'F' / None
    M1: str | None = None
    M2: str | None = None
    M3: str | None = None
    M4: str | None = None
    M5: str | None = None
    M6: str | None = None
    M7: str | None = None
    M8: str | None = None
    M9: str | None = None

    # Anything else (V*, C*, D*, id_*, etc.) goes here as raw values.
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional features (V1..V339, C1..C14, D1..D15, id_*).",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transaction_id": "txn_001",
                "TransactionDT": 86400,
                "TransactionAmt": 125.50,
                "ProductCD": "W",
                "card1": 13926,
                "card2": 555.0,
                "card3": 150.0,
                "card4": "visa",
                "card5": 226.0,
                "card6": "debit",
                "addr1": 315.0,
                "addr2": 87.0,
                "P_emaildomain": "gmail.com",
                "R_emaildomain": "gmail.com",
                "DeviceType": "desktop",
                "M1": "T",
                "M2": "T",
                "extra": {
                    "C1": 1.0,
                    "C2": 1.0,
                    "V12": 1.0,
                    "V13": 1.0,
                }
            }
        }
    )

    def to_feature_dict(self) -> dict[str, Any]:
        """Flatten into a single ``{column: value}`` dict for the preprocessor.

        Pydantic's ``model_dump`` would include ``transaction_id``, which is not
        a model feature — we exclude it. ``extra`` is unpacked at the top level.
        """
        dumped = self.model_dump(exclude={"transaction_id", "extra"}, exclude_none=False)
        dumped.update(self.extra)
        return dumped


# --------------------------------------------------------------------------- #
# Response
# --------------------------------------------------------------------------- #

class FeatureImpact(BaseModel):
    """One row of SHAP-derived per-feature explanation."""

    feature: str = Field(..., description="Feature name from the post-preprocessing pipeline.")
    impact: float = Field(..., ge=0, description="Absolute SHAP value (magnitude only).")
    direction: Literal["increases_risk", "decreases_risk"] = Field(
        ..., description="Sign of the feature's contribution to the fraud probability."
    )


class PredictionResponse(BaseModel):
    """Scored response for a single ``/predict`` request."""

    model_config = ConfigDict(protected_namespaces=())

    transaction_id: str = Field(..., description="Echo of the request's transaction_id.")
    fraud_probability: float = Field(..., ge=0.0, le=1.0,
                                      description="Probability the transaction is fraudulent.")
    prediction: Literal["FRAUD", "NOT_FRAUD"] = Field(
        ..., description="Hard label, based on the configured decision threshold."
    )
    risk_level: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        ..., description="Bucketed risk level for downstream routing."
    )
    top_reasons: list[FeatureImpact] = Field(
        ..., description="Top features driving this prediction, by SHAP magnitude."
    )
    model_version: str = Field(..., description="Version of the model that produced this prediction.")


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #

class HealthResponse(BaseModel):
    """Liveness / readiness response for ``/health``."""

    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "degraded", "loading"] = Field(...,
        description="'ok' when model is loaded and ready; 'loading' at startup; 'degraded' on partial failure."
    )
    model_loaded: bool = Field(..., description="True iff the trained pipeline is in memory.")
    model_version: str = Field(..., description="Version string baked into the API at deploy time.")

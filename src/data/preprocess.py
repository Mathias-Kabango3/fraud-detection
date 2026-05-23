"""Preprocessing pipeline for IEEE-CIS fraud detection.

Exposes a fit-on-train sklearn ``Pipeline`` that can be combined with SMOTE
and XGBoost in the training script, and a deterministic feature-engineering
transformer that's safe to apply to single rows at inference time.

Usage (training):
    from src.data.preprocess import (
        load_raw, build_preprocessor, time_based_split, get_feature_target_split,
    )

    df = load_raw(TXN_CSV, ID_CSV)
    train_df, val_df = time_based_split(df, val_days=7)
    X_train, y_train = get_feature_target_split(train_df)
    X_val,   y_val   = get_feature_target_split(val_df)

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train, y_train)
    X_val_t   = preprocessor.transform(X_val)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

TARGET: str = "isFraud"
ID_COL: str = "TransactionID"
TIME_COL: str = "TransactionDT"
AMT_COL: str = "TransactionAmt"

# Low-cardinality categoricals get one-hot encoded.
LOW_CARD_CATEGORICAL: list[str] = [
    "ProductCD", "card4", "card6", "DeviceType",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
]

# Columns that must survive the high-missing dropper and the encoder, so the
# pipeline stays valid even if a particular fit data slice is unusually sparse.
PROTECTED_COLS: set[str] = {
    ID_COL, TARGET, TIME_COL, AMT_COL,
    "log_amt", "hour_of_day", "day_index", "email_match",
    *LOW_CARD_CATEGORICAL,
}

DEFAULT_MISSING_THRESHOLD: float = 0.95
DEFAULT_VAL_DAYS: int = 7


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns in-place to the smallest dtype that holds them.

    Pandas reads CSVs as int64/float64 by default, which is wasteful for this
    dataset. Returns the same dataframe for chaining.
    """
    start_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    for col in df.select_dtypes(include=["integer"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    end_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    log.info(
        "reduce_memory: %.0f MB -> %.0f MB (%.0f%% saved)",
        start_mb, end_mb, 100 * (1 - end_mb / start_mb),
    )
    return df


def load_raw(transaction_csv: Path | str, identity_csv: Path | str) -> pd.DataFrame:
    """Load both IEEE-CIS CSVs and left-merge them on ``TransactionID``.

    The identity table only covers a subset (~25%) of transactions; columns
    from it will be ``NaN`` for transactions without an identity record.
    """
    transaction_csv = Path(transaction_csv)
    identity_csv = Path(identity_csv)
    log.info("Loading transactions from %s", transaction_csv)
    transactions = pd.read_csv(transaction_csv)
    log.info("Loading identity from %s", identity_csv)
    identity = pd.read_csv(identity_csv)
    log.info("Left-merging on %s", ID_COL)
    merged = transactions.merge(identity, on=ID_COL, how="left")
    log.info("Merged shape: %s", merged.shape)
    return reduce_memory(merged)


# --------------------------------------------------------------------------- #
# Deterministic feature engineering (stateless transformer)
# --------------------------------------------------------------------------- #

class EngineeredFeatures(BaseEstimator, TransformerMixin):
    """Add deterministic engineered features. Stateless — safe for single rows.

    Adds:
      * ``log_amt``     — ``log1p(TransactionAmt)``, tames right-skew.
      * ``hour_of_day`` — derived from ``TransactionDT`` (seconds-since-ref).
      * ``day_index``   — integer day bucket, used for time-based splits.
      * ``email_match`` — 1 iff purchaser and recipient emails are present and equal.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "EngineeredFeatures":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["log_amt"] = np.log1p(X[AMT_COL]).astype("float32")
        X["hour_of_day"] = ((X[TIME_COL] / 3600.0) % 24).astype("float32")
        X["day_index"] = (X[TIME_COL] // (24 * 3600)).astype("int32")
        if "P_emaildomain" in X.columns and "R_emaildomain" in X.columns:
            both_present = X["P_emaildomain"].notna() & X["R_emaildomain"].notna()
            X["email_match"] = (
                both_present & (X["P_emaildomain"] == X["R_emaildomain"])
            ).astype("int8")
        else:
            X["email_match"] = np.int8(0)
        return X


# --------------------------------------------------------------------------- #
# Stateful transformers
# --------------------------------------------------------------------------- #

class HighMissingColumnDropper(BaseEstimator, TransformerMixin):
    """Drop columns whose missingness on the fit data exceeds ``threshold``.

    Columns listed in ``protect`` are kept regardless. Columns dropped on fit
    are also dropped on transform, so train/val/test see the same schema.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_MISSING_THRESHOLD,
        protect: Iterable[str] = (),
    ) -> None:
        self.threshold = threshold
        self.protect = set(protect)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "HighMissingColumnDropper":
        missing_pct = X.isna().mean()
        self.columns_to_drop_: list[str] = [
            col for col, pct in missing_pct.items()
            if pct > self.threshold and col not in self.protect
        ]
        log.info(
            "HighMissingColumnDropper: dropping %d / %d columns above %.0f%% missing",
            len(self.columns_to_drop_), X.shape[1], self.threshold * 100,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        existing = [c for c in self.columns_to_drop_ if c in X.columns]
        return X.drop(columns=existing)


class AmountPerGroupRatio(BaseEstimator, TransformerMixin):
    """Add ``amount / group_mean(amount)`` for each row.

    Group means are learned on the fit data; unseen group values at transform
    time fall back to the global mean so the ratio is always defined.
    """

    def __init__(
        self,
        amount_col: str = AMT_COL,
        group_col: str = "card1",
        output_col: str = "amt_per_card1_mean_ratio",
    ) -> None:
        self.amount_col = amount_col
        self.group_col = group_col
        self.output_col = output_col

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "AmountPerGroupRatio":
        means = X.groupby(self.group_col)[self.amount_col].mean()
        self.group_means_: dict = means.to_dict()
        self.global_mean_: float = float(X[self.amount_col].mean())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        denom = X[self.group_col].map(self.group_means_).fillna(self.global_mean_)
        X[self.output_col] = (X[self.amount_col] / denom).astype("float32")
        return X


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace categorical (object-dtype) values with their training frequency.

    All object-dtype columns are auto-detected at fit time, except those listed
    in ``exclude`` (typically the low-cardinality columns reserved for one-hot
    encoding). Unseen values at transform time map to 0.
    """

    def __init__(self, exclude: Iterable[str] = ()) -> None:
        self.exclude = set(exclude)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FrequencyEncoder":
        self.columns_: list[str] = [
            col for col in X.columns
            if X[col].dtype == "object" and col not in self.exclude
        ]
        self.frequencies_: dict[str, dict] = {
            col: X[col].value_counts(normalize=True, dropna=False).to_dict()
            for col in self.columns_
        }
        log.info("FrequencyEncoder: encoding %d object columns", len(self.columns_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns_:
            if col in X.columns:
                X[col] = X[col].map(self.frequencies_[col]).fillna(0.0).astype("float32")
        return X


# --------------------------------------------------------------------------- #
# Pipeline factory
# --------------------------------------------------------------------------- #

def build_preprocessor(
    low_card_cols: Iterable[str] | None = None,
    missing_threshold: float = DEFAULT_MISSING_THRESHOLD,
) -> Pipeline:
    """Assemble the full fit-on-train preprocessing pipeline.

    Stages, in order:
      1. ``EngineeredFeatures``        — deterministic, stateless feature adds.
      2. ``HighMissingColumnDropper``  — drop sparse columns (protected list kept).
      3. ``AmountPerGroupRatio``       — adds ``amt_per_card1_mean_ratio``.
      4. ``FrequencyEncoder``          — encode object columns except low-card.
      5. ``ColumnTransformer``         — one-hot for low-card, median impute for rest.

    Returns a ``Pipeline`` whose ``transform`` yields a numeric ``DataFrame``
    suitable for downstream SMOTE + XGBoost.
    """
    low_card_cols = list(low_card_cols or LOW_CARD_CATEGORICAL)

    low_card_branch = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    remainder_branch = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
    ])

    encoder = ColumnTransformer(
        transformers=[
            ("low_card_ohe", low_card_branch, low_card_cols),
        ],
        remainder=remainder_branch,
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    return Pipeline(steps=[
        ("engineer", EngineeredFeatures()),
        ("drop_high_missing", HighMissingColumnDropper(
            threshold=missing_threshold,
            protect=PROTECTED_COLS,
        )),
        ("amt_per_group", AmountPerGroupRatio()),
        ("freq_encode", FrequencyEncoder(exclude=low_card_cols)),
        ("encode_and_impute", encoder),
    ])


# --------------------------------------------------------------------------- #
# Splits and helpers
# --------------------------------------------------------------------------- #

def time_based_split(
    df: pd.DataFrame,
    val_days: int = DEFAULT_VAL_DAYS,
    time_col: str = TIME_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the most recent ``val_days`` of data for validation.

    Avoids the look-ahead leakage that a random K-fold would introduce on
    this time-ordered dataset.
    """
    if time_col not in df.columns:
        raise ValueError(f"{time_col!r} not found in dataframe columns")
    seconds_per_day = 24 * 3600
    cutoff_dt = df[time_col].max() - val_days * seconds_per_day
    is_train = df[time_col] <= cutoff_dt
    train_df = df.loc[is_train].copy()
    val_df = df.loc[~is_train].copy()
    log.info(
        "time_based_split: train=%d (%.1f%%), val=%d (%.1f%%), cutoff_DT=%.0f",
        len(train_df), 100 * len(train_df) / len(df),
        len(val_df), 100 * len(val_df) / len(df),
        cutoff_dt,
    )
    return train_df, val_df


def get_feature_target_split(
    df: pd.DataFrame,
    target: str = TARGET,
    id_col: str = ID_COL,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Split into feature frame and target series. ``y`` is ``None`` at inference."""
    drop = [c for c in (id_col,) if c in df.columns]
    if target in df.columns:
        y = df[target].copy()
        X = df.drop(columns=drop + [target])
    else:
        y = None
        X = df.drop(columns=drop)
    return X, y


# --------------------------------------------------------------------------- #
# Smoke test: ``python -m src.data.preprocess``
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import os
    import time as _time

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )

    repo_root = Path(__file__).resolve().parents[2]
    txn_csv = repo_root / os.getenv("TRAIN_TRANSACTION_CSV", "data/raw/train_transaction.csv")
    id_csv = repo_root / os.getenv("TRAIN_IDENTITY_CSV", "data/raw/train_identity.csv")

    log.info("Smoke test: loading a 20,000-row sample for speed")
    t0 = _time.time()
    sample = pd.read_csv(txn_csv, nrows=20_000).merge(
        pd.read_csv(id_csv), on=ID_COL, how="left",
    )
    sample = reduce_memory(sample)
    log.info("Sample loaded in %.1fs, shape=%s", _time.time() - t0, sample.shape)

    train_df, val_df = time_based_split(sample, val_days=2)
    X_train, y_train = get_feature_target_split(train_df)
    X_val, y_val = get_feature_target_split(val_df)

    preprocessor = build_preprocessor()
    log.info("Fitting preprocessor...")
    t0 = _time.time()
    X_train_t = preprocessor.fit_transform(X_train, y_train)
    log.info("Fit+transform train: %.1fs, output shape=%s", _time.time() - t0, X_train_t.shape)
    X_val_t = preprocessor.transform(X_val)
    log.info("Transform val: output shape=%s", X_val_t.shape)

    nans = int(X_train_t.isna().sum().sum())
    non_numeric = [c for c in X_train_t.columns if not pd.api.types.is_numeric_dtype(X_train_t[c])]
    log.info("Sanity: NaNs in train output = %d, non-numeric columns = %d", nans, len(non_numeric))
    assert nans == 0, "Preprocessor left NaNs in output"
    assert not non_numeric, f"Non-numeric columns remain: {non_numeric[:5]}"
    log.info("Smoke test PASSED")

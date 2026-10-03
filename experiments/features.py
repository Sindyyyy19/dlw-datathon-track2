"""Feature engineering for Track 2.

The same data transformation must be applied during both training and prediction.
This module intentionally keeps the logic pure and row-wise so it can be used in
pipelines, notebooks, and inference scripts without hidden state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL = ["merchant_category", "country", "transaction_channel"]
NUMERIC_RAW = [
    "transaction_amount",
    "transaction_hour",
    "transactions_last_24h",
    "spend_last_24h",
    "account_age",
    "new_device",
    "transactions_last_1h",
]
HIGH_RISK_CATEGORIES = {"luxury", "cash_transfer", "electronics", "travel"}
REQUIRED_COLUMNS = NUMERIC_RAW + CATEGORICAL + ["fraud"]


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        missing_msg = ", ".join(missing)
        raise ValueError(f"DataFrame is missing required columns: {missing_msg}")


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the engineered feature matrix used by the fraud model.

    The transformation is intentionally deterministic and row-wise so it can be
    called identically at fit time and inference time.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    _validate_columns(df)
    out = pd.DataFrame(index=df.index)

    # Raw numeric inputs remain as-is, but missing values are coerced to NaN.
    for column in NUMERIC_RAW:
        out[column] = _safe_numeric(df[column])

    # Missingness itself is informative, especially because the test set has more gaps.
    for column in NUMERIC_RAW + CATEGORICAL:
        out[f"{column}_missing"] = df[column].isna().astype(np.int8)

    amount = out["transaction_amount"]
    out["amount_log"] = np.log1p(amount)

    recent_average = out["spend_last_24h"] / out["transactions_last_24h"].replace(0, np.nan)
    out["amount_vs_recent_avg"] = amount / recent_average.replace(0, np.nan)
    out["amount_share_of_24h"] = amount / (out["spend_last_24h"] + amount)
    out["burst_1h_share"] = out["transactions_last_1h"] / out["transactions_last_24h"].replace(0, np.nan)

    out["is_night"] = out["transaction_hour"].isin([23, 0, 1, 2, 3, 4]).astype(np.int8)
    out["account_age_log"] = np.log1p(out["account_age"])
    out["is_new_account"] = (out["account_age"] < 90).astype(np.int8)
    out["is_foreign"] = (df["country"].notna() & (df["country"] != "SG")).astype(np.int8)
    out["high_risk_category"] = df["merchant_category"].isin(HIGH_RISK_CATEGORIES).astype(np.int8)
    out["new_device_foreign"] = out["new_device"].fillna(0).astype(int) * out["is_foreign"]
    out["new_device_high_risk"] = out["new_device"].fillna(0).astype(int) * out["high_risk_category"]

    # Convert categorical columns to pandas category dtype and keep missing as an explicit level.
    for column in CATEGORICAL:
        out[column] = df[column].fillna("missing").astype("category")

    return out


def feature_names(df: pd.DataFrame) -> list[str]:
    """Return the feature names for a given input frame."""
    return list(make_features(df.head(1)).columns)

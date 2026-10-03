"""Feature engineering for Track 2. Pure function of one row set: the same code runs at training and prediction."""
import numpy as np
import pandas as pd

CATEGORICAL = ["merchant_category", "country", "transaction_channel"]
NUMERIC_RAW = ["transaction_amount", "transaction_hour", "transactions_last_24h", "spend_last_24h",
               "account_age", "new_device", "transactions_last_1h"]
HIGH_RISK_CATEGORIES = {"luxury", "cash_transfer", "electronics", "travel"}


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in NUMERIC_RAW:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    # Was the value missing? Missing fields are themselves a weak signal and are far more common in the test set.
    for c in NUMERIC_RAW + CATEGORICAL:
        out[f"{c}_missing"] = df[c].isna().astype(int)
    amt = out["transaction_amount"]
    out["amount_log"] = np.log1p(amt)
    # How this payment compares with the account's recent behaviour.
    avg_recent = out["spend_last_24h"] / out["transactions_last_24h"].replace(0, np.nan)
    out["amount_vs_recent_avg"] = amt / avg_recent.replace(0, np.nan)
    out["amount_share_of_24h"] = amt / (out["spend_last_24h"] + amt)
    out["burst_1h_share"] = out["transactions_last_1h"] / out["transactions_last_24h"].replace(0, np.nan)
    out["is_night"] = out["transaction_hour"].isin([23, 0, 1, 2, 3, 4]).astype(int)
    out["account_age_log"] = np.log1p(out["account_age"])
    out["is_new_account"] = (out["account_age"] < 90).astype(int)
    out["is_foreign"] = (df["country"].notna() & (df["country"] != "SG")).astype(int)
    out["high_risk_category"] = df["merchant_category"].isin(HIGH_RISK_CATEGORIES).astype(int)
    out["new_device_foreign"] = out["new_device"].fillna(0).astype(int) * out["is_foreign"]
    out["new_device_high_risk"] = out["new_device"].fillna(0).astype(int) * out["high_risk_category"]
    for c in CATEGORICAL:
        out[c] = df[c].fillna("missing").astype("category")
    return out


def feature_names(df: pd.DataFrame) -> list[str]:
    return list(make_features(df.head(1)).columns)

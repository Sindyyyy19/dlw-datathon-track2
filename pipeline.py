"""DLW Datathon 2026, Track 2: end-to-end fraud detection pipeline.

One script, raw CSVs in, submission files out:

    1. load and validate the training and testing CSVs
    2. print a short EDA summary (class balance, missingness, univariate fraud rates)
    3. engineer 31 row-level features (same function at fit and predict time)
    4. compare candidate models with stratified 5-fold CV on PR-AUC (optional, --compare)
    5. get out-of-fold scores for the final blend and pick the threshold that maximises F-beta (beta = 1.5)
    6. refit on all training rows, bake the threshold in, save model.pkl as (model, feature_names)
    7. score the test set and write the three prediction CSVs plus metrics.json

Usage:
    python pipeline.py                              # writes to outputs/
    python pipeline.py --team "Our Team" --compare  # also runs the model comparison table
    python pipeline.py --out-dir submission         # overwrite the files in submission/

model.pkl holds only scikit-learn and LightGBM objects, so it loads with joblib without importing this file.
predict() applies the tuned threshold; predict_proba() returns the blended score.
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import FixedThresholdClassifier, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
TARGET = "fraud"
ID = "id"
CATEGORICAL = ["merchant_category", "country", "transaction_channel"]
NUMERIC_RAW = ["transaction_amount", "transaction_hour", "transactions_last_24h", "spend_last_24h",
               "account_age", "new_device", "transactions_last_1h"]
HIGH_RISK_CATEGORIES = {"luxury", "cash_transfer", "electronics", "travel"}
TRAIN_FILE = "Track 2 Training Dataset.csv"
TEST_FILE = "Track 2 Testing Dataset.csv"


# ---------------------------------------------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------------------------------------------
def find_data(data_dir: Path | None) -> tuple[Path, Path]:
    """Locate the two organiser CSVs: the given folder first, then data/, the script folder and the working dir."""
    here = Path(__file__).resolve().parent
    candidates = [data_dir] if data_dir else [here / "data", here, Path.cwd() / "data", Path.cwd()]
    for d in candidates:
        if d and (d / TRAIN_FILE).exists() and (d / TEST_FILE).exists():
            return d / TRAIN_FILE, d / TEST_FILE
    raise FileNotFoundError(f"Could not find '{TRAIN_FILE}' and '{TEST_FILE}' in: {[str(c) for c in candidates]}")


def load_data(data_dir: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path, test_path = find_data(data_dir)
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    expected = [ID] + NUMERIC_RAW + CATEGORICAL
    missing_cols = [c for c in expected + [TARGET] if c not in train.columns] + \
                   [c for c in expected if c not in test.columns]
    assert not missing_cols, f"Missing columns: {missing_cols}"
    assert train[ID].is_unique and test[ID].is_unique, "Duplicate ids"
    assert not set(train[ID]) & set(test[ID]), "Train and test ids overlap"
    assert set(train[TARGET].dropna().unique()) <= {0, 1}, "Target must be 0/1"
    print(f"Loaded train {train.shape} from {train_path}\nLoaded test  {test.shape} from {test_path}")
    return train, test


def eda_summary(train: pd.DataFrame, test: pd.DataFrame) -> None:
    y = train[TARGET]
    print(f"\nFraud: {int(y.sum())} of {len(y):,} ({y.mean() * 100:.2f}%). Always-legit accuracy would be "
          f"{(1 - y.mean()) * 100:.1f}%, so accuracy is not used.")
    miss = pd.DataFrame({"train_%": train[NUMERIC_RAW + CATEGORICAL].isna().mean() * 100,
                         "test_%": test[NUMERIC_RAW + CATEGORICAL].isna().mean() * 100}).round(2)
    print("\nMissing values (train vs test):\n" + miss.to_string())
    for col in CATEGORICAL + ["new_device"]:
        rates = train.groupby(col, dropna=False)[TARGET].agg(["mean", "size"]).sort_values("mean", ascending=False)
        rates["mean"] = (rates["mean"] * 100).round(2)
        print(f"\nFraud rate % by {col}:\n" + rates.rename(columns={"mean": "fraud_%", "size": "n"}).to_string())
    print("\nMedians by class:\n" + train.groupby(TARGET)[NUMERIC_RAW].median().round(1).T.to_string())


# ---------------------------------------------------------------------------------------------------------------
# 2. Features
# ---------------------------------------------------------------------------------------------------------------
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pure row-wise feature function. No statistics are learned here, so there is nothing to leak across folds."""
    out = pd.DataFrame(index=df.index)
    for c in NUMERIC_RAW:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    # Missingness is a weak signal and is far more common in the test set.
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


# ---------------------------------------------------------------------------------------------------------------
# 3. Models
# ---------------------------------------------------------------------------------------------------------------
def logistic(features: list[str], use_engineered: bool = True) -> Pipeline:
    num = [c for c in features if c not in CATEGORICAL]
    if not use_engineered:
        num = [c for c in NUMERIC_RAW if c in num]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=0.03, class_weight="balanced", max_iter=3000))])


def small_lgbm(pos_weight: float) -> lgb.LGBMClassifier:
    # Deliberately low capacity: ~280 positives per training fold, so deep trees memorise individual fraud rows.
    return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.02, num_leaves=3, min_child_samples=100,
                              subsample=0.7, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0,
                              scale_pos_weight=np.sqrt(pos_weight), random_state=SEED, verbose=-1)


def large_lgbm(pos_weight: float) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=15, scale_pos_weight=np.sqrt(pos_weight),
                              random_state=SEED, verbose=-1)


def blend(features: list[str], pos_weight: float) -> VotingClassifier:
    return VotingClassifier([("logistic", logistic(features)), ("lightgbm", small_lgbm(pos_weight))],
                            voting="soft", weights=[1, 1])


# ---------------------------------------------------------------------------------------------------------------
# 4. Evaluation helpers
# ---------------------------------------------------------------------------------------------------------------
def oof_scores(model, X, y, cv) -> np.ndarray:
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]


def best_fbeta(y, scores, beta: float) -> dict:
    prec, rec, thr = precision_recall_curve(y, scores)
    fb = (1 + beta ** 2) * prec * rec / np.clip(beta ** 2 * prec + rec, 1e-9, None)
    i = int(np.nanargmax(fb[:-1]))
    return {"threshold": float(thr[i]), "precision": float(prec[i]), "recall": float(rec[i]),
            "f1": float(2 * prec[i] * rec[i] / max(prec[i] + rec[i], 1e-9)), f"f{beta:g}": float(fb[i])}


def compare_models(X, y, features, pos_weight, cv) -> pd.DataFrame:
    candidates = {
        "Logistic, raw columns": logistic(features, use_engineered=False),
        "Logistic, engineered": logistic(features),
        "LightGBM, large (15 leaves)": large_lgbm(pos_weight),
        "LightGBM, small (3 leaves)": small_lgbm(pos_weight),
        "Soft vote: logistic + small LightGBM": blend(features, pos_weight),
    }
    rows = []
    for name, model in candidates.items():
        s = oof_scores(model, X, y, cv)
        rows.append({"model": name, "PR-AUC": average_precision_score(y, s), "ROC-AUC": roc_auc_score(y, s),
                     "best F1": best_fbeta(y, s, 1.0)["f1"]})
        print(f"  {name:<40s} PR-AUC {rows[-1]['PR-AUC']:.4f}  ROC-AUC {rows[-1]['ROC-AUC']:.4f}")
    return pd.DataFrame(rows).round(4)


# ---------------------------------------------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=None, help="folder holding the two organiser CSVs")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"), help="where model.pkl and the CSVs go")
    ap.add_argument("--team", default="TEAM_NAME", help="team name used in the prediction file names")
    ap.add_argument("--beta", type=float, default=1.5, help="F-beta used to pick the threshold (>1 favours recall)")
    ap.add_argument("--compare", action="store_true", help="also run the 5-model CV comparison")
    ap.add_argument("--no-eda", action="store_true", help="skip the EDA printout")
    args = ap.parse_args()
    t0 = time.time()
    np.random.seed(SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data
    train, test = load_data(args.data_dir)
    if not args.no_eda:
        eda_summary(train, test)

    # 2. Features
    X = make_features(train)
    y = train[TARGET].astype(int).values
    X_test = make_features(test)  # LightGBM remaps categories to the training levels itself
    features = list(X.columns)
    pos_weight = (y == 0).sum() / (y == 1).sum()
    print(f"\n{len(features)} features; negative/positive ratio {pos_weight:.1f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # 3. Optional model comparison
    if args.compare:
        print("\nModel comparison (stratified 5-fold, out-of-fold scores):")
        table = compare_models(X, y, features, pos_weight, cv)
        table.to_csv(args.out_dir / "model_comparison.csv", index=False)

    # 4. Out-of-fold scores for the final blend, threshold selection
    oof = oof_scores(blend(features, pos_weight), X, y, cv)
    chosen = best_fbeta(y, oof, args.beta)
    threshold = chosen["threshold"]
    fold_pr = [average_precision_score(y[va], oof[va]) for _, va in cv.split(X, y)]
    tn, fp, fn, tp = confusion_matrix(y, (oof >= threshold).astype(int)).ravel()
    top1 = np.argsort(-oof)[: int(len(oof) * 0.01)]
    metrics = {
        "n_train": int(len(y)), "n_fraud": int(y.sum()), "base_rate": float(y.mean()),
        "cv_pr_auc": float(average_precision_score(y, oof)), "cv_roc_auc": float(roc_auc_score(y, oof)),
        "cv_pr_auc_per_fold": [round(float(v), 4) for v in fold_pr],
        "beta": args.beta, "threshold": threshold,
        "cv_precision": chosen["precision"], "cv_recall": chosen["recall"], "cv_f1": chosen["f1"],
        "cv_confusion": {"tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn)},
        "cv_flagged_share": float((oof >= threshold).mean()),
        "cv_f1_at_0.5": float(f1_score(y, oof >= 0.5)), "cv_recall_at_0.5": float(recall_score(y, oof >= 0.5)),
        "cv_precision_at_0.5": float(precision_score(y, oof >= 0.5)),
        "cv_precision_top_1pct": float(y[top1].mean()),
    }
    print("\nThreshold trade-off on out-of-fold scores:")
    for b in [1.0, 1.5, 2.0]:
        r = best_fbeta(y, oof, b)
        print(f"  max F{b:<3g} threshold {r['threshold']:.3f}  precision {r['precision']:.3f}  "
              f"recall {r['recall']:.3f}  F1 {r['f1']:.3f}  flagged {(oof >= r['threshold']).mean() * 100:.1f}%")
    print(f"\nCV PR-AUC {metrics['cv_pr_auc']:.4f} (no-skill {y.mean():.4f}), ROC-AUC {metrics['cv_roc_auc']:.4f}")
    print(f"Chosen threshold (max F{args.beta:g}): {threshold:.3f} -> TP {tp}, FN {fn}, FP {fp}, TN {tn}")

    # 5. Final fit on all training rows with the threshold baked in
    final = FixedThresholdClassifier(blend(features, pos_weight), threshold=threshold,
                                     response_method="predict_proba").fit(X, y)
    model_path = args.out_dir / "model.pkl"
    joblib.dump((final, features), model_path)

    # 6. Score the test set from the saved file, exactly as a judge would
    model, feats = joblib.load(model_path)
    proba = model.predict_proba(X_test[feats])[:, 1]
    pred = model.predict(X_test[feats]).astype(int)
    assert (pred == (proba >= threshold).astype(int)).all()
    assert len(pred) == len(test)

    prefix = f"{args.team}_Datathon 2026_Track 2_Prediction"
    pd.DataFrame({"prediction": pred}).to_csv(args.out_dir / f"{prefix}.csv", index=False)
    pd.DataFrame({"prediction": proba.round(6)}).to_csv(args.out_dir / f"{prefix}_probabilities.csv", index=False)
    pd.DataFrame({ID: test[ID], "fraud_probability": proba.round(6), "prediction": pred}).to_csv(
        args.out_dir / f"{prefix}_with_ids.csv", index=False)
    metrics.update({"test_rows": int(len(pred)), "test_flagged": int(pred.sum()),
                    "test_flagged_share": float(pred.mean()), "runtime_seconds": round(time.time() - t0, 1)})
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))

    print(f"\nTest: flagged {pred.sum()} of {len(pred):,} ({pred.mean() * 100:.2f}%)")
    print(f"Wrote {model_path}, three '{prefix}*.csv' files and metrics.json to {args.out_dir}/ "
          f"in {metrics['runtime_seconds']}s")


if __name__ == "__main__":
    main()

"""Compare models with stratified 5-fold cross-validation on PR-AUC, F1 (at the best threshold) and recall."""
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import lightgbm as lgb
from features import make_features, CATEGORICAL

warnings.filterwarnings("ignore")
train = pd.read_csv("data/Track 2 Training Dataset.csv")
X = make_features(train); y = train["fraud"].values
num_cols = [c for c in X.columns if c not in CATEGORICAL]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
pos_weight = (y == 0).sum() / (y == 1).sum()


def best_f1(y_true, p):
    prec, rec, thr = precision_recall_curve(y_true, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    i = int(np.nanargmax(f1[:-1]))
    return f1[i], thr[i], rec[i], prec[i]


def logistic():
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.3))])


def hgb():
    pre = ColumnTransformer([("num", "passthrough", num_cols), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)])
    return Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=15, l2_regularization=1.0, class_weight="balanced", random_state=42))])


def lgbm(**kw):
    params = dict(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=2.0, scale_pos_weight=pos_weight ** 0.5, random_state=42, verbose=-1)
    params.update(kw)
    return lgb.LGBMClassifier(**params)


MODELS = {
    "Logistic regression (baseline)": logistic,
    "HistGradientBoosting (sklearn)": hgb,
    "LightGBM": lgbm,
    "LightGBM, full pos weight": lambda: lgbm(scale_pos_weight=pos_weight),
    "LightGBM, no pos weight": lambda: lgbm(scale_pos_weight=1.0),
}

rows = []
oof_store = {}
for name, make in MODELS.items():
    oof = np.zeros(len(y))
    for tr_i, va_i in skf.split(X, y):
        m = make()
        if name.startswith("LightGBM"):
            m.fit(X.iloc[tr_i], y[tr_i], categorical_feature=CATEGORICAL)
        else:
            m.fit(X.iloc[tr_i], y[tr_i])
        oof[va_i] = m.predict_proba(X.iloc[va_i])[:, 1]
    f1, thr, rec, prec = best_f1(y, oof)
    rows.append({"model": name, "PR-AUC": average_precision_score(y, oof), "ROC-AUC": roc_auc_score(y, oof),
                 "best F1": f1, "threshold": thr, "recall@bestF1": rec, "precision@bestF1": prec,
                 "F1@0.5": f1_score(y, oof >= 0.5), "recall@0.5": recall_score(y, oof >= 0.5)})
    oof_store[name] = oof
    print(f"done {name}")

res = pd.DataFrame(rows).set_index("model").round(4)
pd.set_option("display.width", 220)
print(res.to_string())
res.to_csv("cv_results.csv")
np.save("oof_lgbm.npy", oof_store["LightGBM"])

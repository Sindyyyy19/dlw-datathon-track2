"""Final model: soft-vote blend of regularised logistic regression and a small LightGBM, threshold tuned on
out-of-fold predictions for F1, packaged as one scikit-learn object so joblib can load it anywhere."""
import json, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.model_selection import FixedThresholdClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import lightgbm as lgb
from features import make_features, CATEGORICAL

warnings.filterwarnings("ignore")
train = pd.read_csv("data/Track 2 Training Dataset.csv"); test = pd.read_csv("data/Track 2 Testing Dataset.csv")
X = make_features(train); y = train["fraud"].values; X_test = make_features(test)
features = list(X.columns)
pos_weight = (y == 0).sum() / (y == 1).sum()
num_cols = [c for c in features if c not in CATEGORICAL]


def build():
    lr = Pipeline([
        ("pre", ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=0.03)),
    ])
    gbm = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.02, num_leaves=3, min_child_samples=100, subsample=0.7,
                             subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0, scale_pos_weight=pos_weight ** 0.5,
                             random_state=42, verbose=-1)
    return VotingClassifier([("logistic", lr), ("lightgbm", gbm)], voting="soft", weights=[1, 1])


# 1. Out-of-fold probabilities to pick the threshold and report honest metrics.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = cross_val_predict(build(), X, y, cv=skf, method="predict_proba")[:, 1]
prec, rec, thr = precision_recall_curve(y, oof)
f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
i = int(np.nanargmax(f1[:-1])); threshold = float(thr[i])
metrics = {
    "cv_pr_auc": float(average_precision_score(y, oof)), "cv_roc_auc": float(roc_auc_score(y, oof)),
    "threshold": threshold, "cv_f1": float(f1[i]), "cv_recall": float(rec[i]), "cv_precision": float(prec[i]),
    "cv_f1_at_0.5": float(f1_score(y, oof >= 0.5)), "cv_recall_at_0.5": float(recall_score(y, oof >= 0.5)),
    "flagged_share_at_threshold": float((oof >= threshold).mean()), "base_rate": float(y.mean()),
}
print(json.dumps(metrics, indent=1))

# 2. Fit on everything, bake the threshold in, save in the booklet's (model, feature_names) shape.
final = FixedThresholdClassifier(build(), threshold=threshold, response_method="predict_proba").fit(X, y)
joblib.dump((final, features), "model.pkl")
np.save("oof_final.npy", oof); json.dump(metrics, open("metrics.json", "w"), indent=1)

# 3. Predictions on the public test set.
loaded, feats = joblib.load("model.pkl")
proba = loaded.predict_proba(X_test[feats])[:, 1]; pred = loaded.predict(X_test[feats])
print("test flagged:", int(pred.sum()), "of", len(pred), f"({pred.mean()*100:.2f}%)", "| proba mean", round(float(proba.mean()), 4))
assert (pred == (proba >= threshold).astype(int)).all()

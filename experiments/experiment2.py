"""Round 2: regularisation sweep, feature-set ablation, and blends. Repeated stratified CV for stable estimates."""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import lightgbm as lgb
from features import make_features, CATEGORICAL, NUMERIC_RAW

warnings.filterwarnings("ignore")
train = pd.read_csv("data/Track 2 Training Dataset.csv")
X_full = make_features(train); y = train["fraud"].values
pos_weight = (y == 0).sum() / (y == 1).sum()
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=7)
splits = list(cv.split(X_full, y))

RAW_ONLY = NUMERIC_RAW + CATEGORICAL
FEATURE_SETS = {"raw": RAW_ONLY, "engineered": list(X_full.columns)}


def best_f1(y_true, p):
    prec, rec, thr = precision_recall_curve(y_true, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    i = int(np.nanargmax(f1[:-1])); return f1[i], thr[i], rec[i]


def lr(cols, C):
    num = [c for c in cols if c not in CATEGORICAL]
    pre = ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), [c for c in cols if c in CATEGORICAL])])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=C))])


def lgbm(**kw):
    p = dict(n_estimators=300, learning_rate=0.02, num_leaves=5, min_child_samples=100, subsample=0.7, subsample_freq=1,
             colsample_bytree=0.6, reg_lambda=5.0, scale_pos_weight=pos_weight ** 0.5, random_state=42, verbose=-1)
    p.update(kw); return lgb.LGBMClassifier(**p)


def run(name, make, cols, is_lgb=False):
    oofs = []
    for r in range(3):
        oof = np.zeros(len(y))
        for tr_i, va_i in splits[r * 5:(r + 1) * 5]:
            m = make(); Xa = X_full[cols]
            if is_lgb: m.fit(Xa.iloc[tr_i], y[tr_i], categorical_feature=[c for c in cols if c in CATEGORICAL])
            else: m.fit(Xa.iloc[tr_i], y[tr_i])
            oof[va_i] = m.predict_proba(Xa.iloc[va_i])[:, 1]
        oofs.append(oof)
    aps = [average_precision_score(y, o) for o in oofs]
    f1s = [best_f1(y, o)[0] for o in oofs]
    print(f"{name:48s} PR-AUC {np.mean(aps):.4f} ± {np.std(aps):.4f}   best F1 {np.mean(f1s):.4f}")
    return np.mean(oofs, axis=0)


print("--- logistic regression")
O = {}
for fs, cols in FEATURE_SETS.items():
    for C in [0.03, 0.1, 0.3, 1.0]:
        O[f"lr-{fs}-C{C}"] = run(f"LR {fs} C={C}", lambda C=C, cols=cols: lr(cols, C), cols)
print("--- lightgbm, small and regularised")
for fs, cols in FEATURE_SETS.items():
    for leaves, n in [(3, 200), (5, 300), (7, 300), (5, 600)]:
        O[f"lgb-{fs}-{leaves}-{n}"] = run(f"LGBM {fs} leaves={leaves} trees={n}", lambda l=leaves, n=n: lgbm(num_leaves=l, n_estimators=n), cols, True)
print("--- blends (rank average)")
def rank(p): return rankdata(p) / len(p)
best_lr = max([k for k in O if k.startswith("lr")], key=lambda k: average_precision_score(y, O[k]))
best_lgb = max([k for k in O if k.startswith("lgb")], key=lambda k: average_precision_score(y, O[k]))
print("best LR:", best_lr, "| best LGBM:", best_lgb)
for w in [0.3, 0.5, 0.7]:
    b = w * rank(O[best_lr]) + (1 - w) * rank(O[best_lgb])
    f1, thr, rec = best_f1(y, b)
    print(f"blend LR weight {w}: PR-AUC {average_precision_score(y, b):.4f}  best F1 {f1:.4f} (recall {rec:.3f})")

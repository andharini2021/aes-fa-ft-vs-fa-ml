"""FA-ML rerun: original (early-stop on test fold) vs fixed (inner early-stop split).
Same XGBoost config as the notebooks; device=cpu (no GPU here)."""
import json, time, numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error

D = "data/"  # folder with embeddings_*.npy, features_raw.npy, scores.npy
F = np.load(D + "features_raw.npy").astype(np.float32)
y = np.load(D + "scores.npy").astype(np.float32)
EMB = {"IndoBERT": "embeddings_indobert_hyb.npy",
       "DistilBERT": "embeddings_distilbert_hyb.npy",
       "IndoSBERT": "embeddings_sbert_hyb.npy"}
P = dict(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.5,
         colsample_bytree=0.5, booster="gbtree", reg_lambda=2.0, reg_alpha=0.1,
         gamma=0.1, min_child_weight=20, tree_method="hist", device="cpu",
         random_state=42, verbosity=0)


def qwk(t, p, lo=2, hi=10):
    t = np.asarray(t, int); p = np.round(np.clip(p, lo, hi)).astype(int)
    n = hi - lo + 1
    O = np.zeros((n, n))
    for a, b in zip(t, p): O[a - lo, b - lo] += 1
    E = np.outer(np.bincount(t - lo, minlength=n), np.bincount(p - lo, minlength=n)) / len(t)
    W = np.array([[(i - j) ** 2 / (n - 1) ** 2 for j in range(n)] for i in range(n)])
    return 1 - (W * O).sum() / (W * E).sum()


def inner_split(train_idx, y, fold_idx, frac=0.10, seed=42):
    bands = np.clip(np.asarray(y)[train_idx].astype(int), 5, 10)
    tr, es = train_test_split(train_idx, test_size=frac, random_state=seed + fold_idx, stratify=bands)
    return np.sort(tr), np.sort(es)


rows, preds = [], []
kf = KFold(10, shuffle=True, random_state=42)
for enc, f in EMB.items():
    E = np.load(D + f).astype(np.float32)
    for fe in (True, False):
        X = np.concatenate([E, F], 1) if fe else E
        for mode in ("original", "fixed"):
            for k, (trv, te) in enumerate(kf.split(X)):
                tr, es = (trv, te) if mode == "original" else inner_split(trv, y, k)
                m = xgb.XGBRegressor(**P, early_stopping_rounds=20)
                t0 = time.perf_counter()
                m.fit(X[tr], y[tr], eval_set=[(X[es], y[es])], verbose=False)
                tfit = time.perf_counter() - t0
                p = m.predict(X[te])
                rows.append(dict(model=enc, fe=fe, mode=mode, fold=k + 1,
                                 qwk=qwk(y[te], p), mse=mean_squared_error(y[te], p),
                                 best_iter=m.best_iteration, fit_s=tfit))
                preds += [dict(model=enc, fe=fe, mode=mode, fold=k + 1, idx=int(i),
                               true=float(y[i]), pred=float(v)) for i, v in zip(te, p)]
        print(enc, fe, "done", flush=True)

df = pd.DataFrame(rows); df.to_csv("faml_folds.csv", index=False)
pd.DataFrame(preds).to_csv("faml_preds.csv", index=False)
print(df.groupby(["model", "fe", "mode"])[["qwk", "mse", "best_iter"]].agg(["mean", "std"]).round(4).to_string())

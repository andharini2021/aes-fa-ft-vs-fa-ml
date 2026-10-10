exec(open("faml_from_saved_embeddings.py").read().split("rows, preds = [], []")[0])
rows=[]
kf = KFold(10, shuffle=True, random_state=42)
for enc, f in EMB.items():
    E = np.load(D + f).astype(np.float32)
    for fe in (True, False):
        X = np.concatenate([E, F], 1) if fe else E
        for seed in range(5):
            for k, (trv, te) in enumerate(kf.split(X)):
                tr, es = inner_split(trv, y, k)
                pp = dict(P, random_state=42+seed)
                m = xgb.XGBRegressor(**pp, early_stopping_rounds=20)
                m.fit(X[tr], y[tr], eval_set=[(X[es], y[es])], verbose=False)
                p1 = m.predict(X[te])
                m2 = xgb.XGBRegressor(**dict(pp, n_estimators=m.best_iteration+1))
                m2.fit(X[trv], y[trv], verbose=False)            # refit on full outer-train
                p2 = m2.predict(X[te])
                for mode,p in (("fixed",p1),("fixed_refit",p2)):
                    rows.append(dict(model=enc, fe=fe, mode=mode, seed=seed, fold=k+1,
                                     qwk=qwk(y[te],p), mse=mean_squared_error(y[te],p)))
        print(enc, fe, flush=True)
df=pd.DataFrame(rows); df.to_csv("faml_seeds.csv",index=False)
g=df.groupby(["model","fe","mode","seed"])[["qwk","mse"]].mean().reset_index()
print(g.groupby(["model","fe","mode"])[["qwk","mse"]].agg(["mean","std","min","max"]).round(4).to_string())

"""Every measurement behind README.md and findings/FINDINGS.md.

    python3 analysis/run_all.py [data_root]
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from housing import data, evaluate, pipeline  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "findings", "results")
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data")
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def emit(name, df):
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    print(f"\n### {name}\n{df.to_string(index=False)}")


def main() -> int:
    full = data.load(DATA, "it5_all")
    X_all, y = data.features_and_target(full, "Expensive")
    price = data.load(DATA, "it6_regression")["SalePrice"]
    cv = evaluate.folds(X_all, y)
    print(f"{len(X_all):,} houses | {X_all.shape[1]} features | "
          f"expensive: {y.mean():.4f} ({int(y.sum())} of {len(y)})")

    # --- 1. baselines -----------------------------------------------------
    rows = []
    for name, est in [('always "not expensive"', DummyClassifier(strategy="most_frequent")),
                      ("stratified coin", DummyClassifier(strategy="stratified", random_state=0))]:
        r = cross_validate(pipeline.model(X_all, est), X_all, y, cv=cv, scoring=SCORING)
        rows.append({"model": name,
                     **{m: round(float(r["test_" + m].mean()), 3) for m in SCORING}})
    emit("baselines", pd.DataFrame(rows))

    # --- 2. the feature ladder, at two model capacities -------------------
    rows = []
    for it, label in [("it0_2_numeric", "0-2  nine numeric"),
                      ("it3_onehot", "3    + one-hot categoricals"),
                      ("it4_ordinal", "4    + ordinal categoricals"),
                      ("it5_all", "5    all features")]:
        X, yy = data.features_and_target(data.load(DATA, it), "Expensive")
        f = evaluate.folds(X, yy)
        for cname, est in [("decision tree (depth 5)", DecisionTreeClassifier(max_depth=5, random_state=0)),
                           ("random forest", RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1))]:
            r = cross_validate(pipeline.model(X, est), X, yy, cv=f, scoring=SCORING)
            rows.append({"iteration": label, "features": X.shape[1], "model": cname,
                         **{m: round(float(r["test_" + m].mean()), 3) for m in SCORING}})
    emit("feature_ladder", pd.DataFrame(rows))

    # --- 3. error analysis -------------------------------------------------
    forest = pipeline.model(X_all, RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1))
    proba = evaluate.cv_predict_proba(forest, X_all, y, cv)
    pred = (proba >= 0.5).astype(int)
    emit("classification_scores", pd.DataFrame([evaluate.scores(y, pred)]))
    cut = float(price[y == 1].min())
    print(f"\n   the 'Expensive' label starts at ${cut:,.0f}")
    emit("error_prices", evaluate.error_prices(y, pred, price, cut))
    emit("threshold_table", evaluate.threshold_table(y, proba))

    # --- 4. classify directly, or predict the price and cut? --------------
    # Twenty folds, and the cut-off is recomputed inside each training split --
    # it is derived from the target, so taking it from the whole dataset would
    # hand the validation rows a number computed from themselves.
    rows = []
    for seed in range(4):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X_all, y):
            local_cut = price.to_numpy()[tr][y.to_numpy()[tr] == 1].min()
            a = pipeline.model(X_all, RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)) \
                .fit(X_all.iloc[tr], y.iloc[tr]).predict(X_all.iloc[te])
            b = np.exp(pipeline.model(X_all, RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1))
                       .fit(X_all.iloc[tr], np.log(price.iloc[tr])).predict(X_all.iloc[te])) >= local_cut
            rows.append({"seed": seed,
                         "A_classify": f1_score(y.iloc[te], a),
                         "B_regress_then_cut": f1_score(y.iloc[te], b.astype(int))})
    comp = pd.DataFrame(rows)
    diff = comp["B_regress_then_cut"] - comp["A_classify"]
    # Paired, because both approaches were scored on the identical folds. An
    # unpaired test would compare two means whose fold-to-fold noise is shared
    # and would miss an effect this size.
    from scipy import stats
    tstat, pval = stats.ttest_rel(comp["B_regress_then_cut"], comp["A_classify"])
    emit("classify_vs_regress", pd.DataFrame([{
        "folds": len(comp),
        "A_classify_f1": round(comp["A_classify"].mean(), 3),
        "A_sd": round(comp["A_classify"].std(), 3),
        "B_regress_then_cut_f1": round(comp["B_regress_then_cut"].mean(), 3),
        "B_sd": round(comp["B_regress_then_cut"].std(), 3),
        "paired_diff": round(diff.mean(), 3), "paired_sd": round(diff.std(), 3),
        "B_wins_folds": int((diff > 0).sum()),
        "paired_t_p_value": round(float(pval), 4)}]))

    # --- 5. phase two: the price itself -----------------------------------
    rows = []
    for name, target, back in [("trained on SalePrice", price, lambda v: v),
                               ("trained on log(SalePrice)", np.log(price), np.exp)]:
        yh = back(cross_val_predict(
            pipeline.model(X_all, RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)),
            X_all, target, cv=cv))
        rows.append({"target": name,
                     "MAE": round(float(mean_absolute_error(price, yh))),
                     "RMSE": round(float(mean_squared_error(price, yh) ** 0.5)),
                     "RMSLE": round(float(mean_squared_error(np.log(price), np.log(np.clip(yh, 1, None))) ** 0.5), 4),
                     "MAPE_pct": round(float(np.mean(np.abs(yh - price) / price) * 100), 2),
                     "within_10pct": round(float(np.mean(np.abs(yh - price) / price < 0.10) * 100), 1)})
    emit("regression_scores", pd.DataFrame(rows))
    print(f"\nWrote {len(os.listdir(OUT))} files to findings/results/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

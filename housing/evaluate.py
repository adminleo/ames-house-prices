"""Scoring that survives an imbalanced target.

14.86 % of these houses are "expensive". Accuracy is therefore a number between
0.851 and 1.0, and the bottom of that range is what you get for predicting
"no" every time. Every function here reports against that floor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def folds(X, y, n: int = 5, seed: int = 0):
    """Stratified, so every fold keeps the 14.86 % positives.

    With 217 positives in 1,460 rows, an unstratified five-way split leaves
    about 43 positives per fold and a visible chance of far fewer. The metric
    that matters is computed from that handful, so its fold-to-fold variance is
    mostly an artefact of the split rather than of the model.
    """
    return list(StratifiedKFold(n, shuffle=True, random_state=seed).split(X, y))


def scores(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    majority = max(float(np.mean(y_true)), 1 - float(np.mean(y_true)))
    accuracy = (tp + tn) / len(y_true)
    return {"precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
            "recall": round(recall_score(y_true, y_pred), 3),
            "f1": round(f1_score(y_true, y_pred), 3),
            "accuracy": round(accuracy, 3),
            "accuracy_over_constant": round(accuracy - majority, 3),
            "true_pos": int(tp), "false_pos": int(fp),
            "false_neg": int(fn), "true_neg": int(tn)}


def threshold_table(y_true, proba, recalls=(0.60, 0.70, 0.80, 0.90, 0.95)) -> pd.DataFrame:
    """What each level of recall costs in false alarms.

    The 0.5 cut-off is a default, not a decision. An agency that must not miss a
    high-value listing and one that cannot waste a valuation visit want
    different points on this curve, and neither is "the model's accuracy".
    """
    p, r, t = precision_recall_curve(y_true, proba)
    rows = []
    for target in recalls:
        i = int(np.argmin(np.abs(r[:-1] - target)))
        pred = (proba >= t[i]).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        rows.append({"threshold": round(float(t[i]), 2),
                     "precision": round(float(p[i]), 3),
                     "recall": round(float(r[i]), 3),
                     "missed_expensive": int(fn), "false_alarms": int(fp)})
    return pd.DataFrame(rows)


def error_prices(y_true, y_pred, price: pd.Series, cut: float) -> pd.DataFrame:
    """Where the mistakes sit relative to the cut-off that defined the label."""
    t = pd.DataFrame({"price": price.to_numpy(), "y": np.asarray(y_true),
                      "pred": np.asarray(y_pred)})
    kinds = {"true positive": (t.y == 1) & (t.pred == 1),
             "false negative (missed)": (t.y == 1) & (t.pred == 0),
             "false positive (false alarm)": (t.y == 0) & (t.pred == 1),
             "true negative": (t.y == 0) & (t.pred == 0)}
    return pd.DataFrame([
        {"outcome": k, "houses": int(m.sum()),
         "median_price": round(float(t.loc[m, "price"].median()), 0),
         "distance_from_cutoff": round(float(t.loc[m, "price"].median() - cut), 0)}
        for k, m in kinds.items()])


def cv_predict_proba(pipe, X, y, cv):
    return cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]

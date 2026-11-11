"""Preprocessing that lives inside the model, not beside it."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Median-impute and scale numbers; fill and one-hot the categories.

    Built as a ColumnTransformer inside a Pipeline rather than applied to the
    frame beforehand, and the reason is not tidiness. `SimpleImputer` learns a
    median. Fit it on the whole table and every validation fold is scored by a
    model that has already seen a statistic computed from its own rows. The
    score that comes back is optimistic and nothing warns you.

    `min_frequency=10` folds rare categories into one level. Ames has
    categorical values that occur twice in 1,460 rows; one-hot encoding each of
    them gives the model a column that is almost always zero and occasionally a
    perfect predictor of a single training row.

    `handle_unknown="ignore"` matters for the same reason: a category present in
    the training fold but absent from the validation fold would otherwise raise
    at predict time, which in production means a house with an unusual roof
    takes the service down.
    """
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant",
                                                   fill_value="MISSING")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                   min_frequency=10))]), categorical),
    ])


def model(X: pd.DataFrame, estimator) -> Pipeline:
    return Pipeline([("pre", preprocessor(X)), ("est", estimator)])

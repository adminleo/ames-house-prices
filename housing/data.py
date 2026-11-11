"""Load the course's iteration files.

The dataset is Ames, Iowa: 1,460 sales, 80 features. The course hands it over in
stages — nine numeric columns first, then categoricals, then everything — so the
feature ladder can be measured rather than assumed.
"""
from __future__ import annotations

import glob
import os

import pandas as pd

ITERATIONS = {
    "it0_2_numeric": "houseprices_overview",
    "it3_onehot": "houseprices_onehot",
    "it4_ordinal": "houseprices_ordinal",
    "it5_all": "houseprices_modelsel",
    "it6_regression": "houseprices_regression",
}


def _find(root: str, folder: str) -> str:
    hits = [p for p in glob.glob(os.path.join(root, folder, "**", "*.csv"), recursive=True)
            if "__MACOSX" not in p]
    if not hits:
        raise FileNotFoundError(f"no CSV under {os.path.join(root, folder)}")
    return hits[0]


def load(root: str, iteration: str) -> pd.DataFrame:
    return pd.read_csv(_find(root, ITERATIONS[iteration]))


def features_and_target(d: pd.DataFrame, target: str):
    """Split off the target and drop `Id`.

    `Id` is a row number. A tree will happily split on it and score well in
    cross-validation, because neighbouring ids were entered together and share
    a neighbourhood — an identifier that leaks through collection order.
    """
    y = d[target]
    X = d.drop(columns=[c for c in (target, "Id") if c in d.columns])
    return X, y

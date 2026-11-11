# Findings

Every figure comes from `analysis/run_all.py` and is written to
`findings/results/`. 1,460 houses, 79 features, 217 of them (14.86 %) labelled
expensive.

## 1. The floor under every accuracy number
`baselines.csv`

| model | accuracy | precision | recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| always "not expensive" | **0.851** | 0.000 | 0.000 | 0.000 | 0.500 |
| stratified coin | 0.745 | 0.145 | 0.147 | 0.146 | 0.498 |

Predicting the majority class is right 85.1 % of the time and finds nothing.
Any accuracy figure for this dataset has to be read against that floor, which is
why `evaluate.scores()` reports `accuracy_over_constant` alongside it. The best
model here reaches 94.9 %, which is **+9.8 points**.

Note that the coin is *worse* on accuracy (0.745) and better on everything else.
Accuracy and the metrics that matter move in opposite directions on an
imbalanced target, which is the whole problem in one row.

## 2. Preprocessing belongs inside the estimator
`housing/pipeline.py`, `tests/test_housing.py`

`SimpleImputer` learns a median. Fit it on the full frame before
cross-validating and every validation fold is scored by a model that has already
absorbed a statistic computed from its own rows. The leak is small with a median
and large with target-derived features, and in neither case does anything warn
you — the score simply comes back better than it should.

Everything therefore sits in a `Pipeline`, so each fold refits its own imputer,
scaler and encoder. Two further settings are deliberate:

- `min_frequency=10` folds rare categorical levels together. Ames has values
  that occur twice in 1,460 rows; one-hot encoding each gives the model a column
  that is almost always zero and occasionally identifies a single training row.
- `handle_unknown="ignore"` so a category present in training and absent from
  validation does not raise. In production that failure mode is "house with an
  unusual roof takes the service down", which a test now covers.

`Id` is dropped. It is a row number, and rows were entered in an order that
carries neighbourhood, so a tree will split on it and score well for a reason
that will not survive contact with a new house.

## 3. More features only help a model with the capacity to use them
`feature_ladder.csv`

| features | tree F1 | tree AUC | forest F1 | forest AUC |
|---|---:|---:|---:|---:|
| 9 numeric | **0.724** | 0.868 | 0.693 | 0.933 |
| 15 (+ one-hot) | 0.702 | 0.870 | 0.713 | 0.948 |
| 23 (+ ordinal) | 0.690 | 0.864 | 0.773 | 0.965 |
| 79 (all) | 0.758 | 0.849 | **0.813** | **0.986** |

Judged on the depth-5 tree, the course's middle iterations make things *worse*
(0.724 → 0.702 → 0.690) and the full set buys 0.034 F1 while AUC falls from
0.868 to 0.849. Judged on the forest, the ladder climbs monotonically and the
full set buys 0.120 F1 with AUC from 0.933 to 0.986.

The ranking between the two models inverts along the way: the tree wins on nine
features and loses badly on seventy-nine. A fixed estimator does not measure
whether features carry signal; it measures whether that estimator can reach it.

## 4. The errors are a property of the label, not of the houses
`classification_scores.csv`, `error_prices.csv`

At the default 0.5 threshold the forest gives 159 true positives, 16 false
positives, 58 false negatives, 1,227 true negatives — precision 0.909, recall
0.733.

"Expensive" begins at **$250,580**. Sorting the outcomes by actual sale price:

| outcome | houses | median price | vs the cut-off |
|---|---:|---:|---:|
| true positive | 159 | $320,000 | +$69,420 |
| false negative | 58 | $268,000 | **+$17,420** |
| false positive | 16 | $238,500 | **−$12,080** |
| true negative | 1,227 | $149,900 | −$100,680 |

Both error types sit within about $17,000 of the boundary; both correct types
sit $70,000–$100,000 away from it. Nothing here is a model that mistakes a
mansion for a bungalow. The residual difficulty of this task is the width of the
band around an arbitrary line.

## 4b. Model selection, and the cost of tuning on your test folds
`model_selection.csv`

| model | best params | same folds | nested CV | optimism |
|---|---|---:|---:|---:|
| decision tree | depth 5, leaf 1 | 0.758 | 0.731 | **+0.027** |
| k-nearest neighbours | k=3, uniform | 0.744 | 0.725 | +0.019 |
| logistic regression | C=0.3 | 0.819 | 0.819 | **+0.000** |
| random forest | max_features 0.3, leaf 1 | 0.833 | 0.826 | +0.008 |

`GridSearchCV.best_score_` is the cross-validated score of whichever
configuration won on those folds. Reporting it as the model's performance counts
the winner's luck as skill, and the bias is not uniform — it scales with the
size of the search and the variance of the estimator. Twelve configurations of a
depth-tunable tree inflate F1 by 0.027; four values of `C` on a linear model by
0.000. So the ranking itself is distorted by how hard each family was searched,
which is the argument for nesting rather than for searching less.

Under honest evaluation the gap between the best and second-best family is
**0.007 F1** — logistic regression at 0.819 against the random forest at 0.826.
On one-hot encoded Ames features this boundary is close to linear. The forest is
not meaningfully better, and the linear model's coefficients can be shown to a
client.

k-nearest neighbours comes last (0.725). After one-hot encoding the feature
space has a couple of hundred mostly-zero dimensions, and a distance metric
averaged over those is close to uninformative — the one family here whose core
assumption the preprocessing breaks.

## 5. The threshold is a business decision
`threshold_table.csv`

| threshold | precision | recall | missed | false alarms |
|---:|---:|---:|---:|---:|
| 0.65 | 0.970 | 0.594 | 88 | 4 |
| 0.56 | 0.944 | 0.700 | 65 | 9 |
| 0.43 | 0.888 | 0.802 | 43 | 22 |
| 0.30 | 0.782 | 0.894 | 23 | 54 |
| 0.22 | 0.678 | 0.949 | 11 | **98** |

0.5 is a library default, not a decision. An agency that must not miss a
high-value listing buys recall 0.949 for 98 wasted valuations; one that cannot
spare a visit takes recall 0.594 and misses 88 houses. The model does not choose
between those and neither does a single F1.

## 6. Predict the price, then apply the rule
`classify_vs_regress.csv`

Both approaches on the identical 20 folds (5-fold × 4 seeds), with the cut-off
recomputed inside each training split — it is derived from the target, so a
cut-off taken from the whole dataset would hand every validation row a number
computed from itself.

| approach | F1 | sd |
|---|---:|---:|
| A — classify directly | 0.813 | 0.064 |
| B — predict the price, then cut | **0.843** | 0.050 |

Paired difference **+0.030 ± 0.047**, B ahead in 14 of 20 folds, paired t-test
**p = 0.0098**. The test is paired because the two approaches saw the same
folds; an unpaired comparison of these means would discard the shared
fold-to-fold noise and miss an effect this size.

Modest, real, and free: B also returns the estimated price, so one training run
answers both phases of the brief. The rule generalises — when a categorical
target was manufactured by thresholding a number you still have, model the
number and apply the threshold afterwards.

## 7. The price model
`regression_scores.csv`

`SalePrice` is right-skewed (skew 1.88); the log is nearly symmetric (0.12).

| target | MAE | RMSE | RMSLE | MAPE | within 10 % |
|---|---:|---:|---:|---:|---:|
| SalePrice | **$17,675** | **$30,510** | 0.1472 | 10.34 % | 67.5 % |
| log(SalePrice) | $17,800 | $31,098 | **0.1451** | **10.11 %** | 67.0 % |

The log transform is usually presented as the fix for a skewed target. Here it
is a trade, not a fix: training on the raw price wins on the absolute metrics
(MAE, RMSE), training on the log wins on the relative ones (RMSLE, MAPE). That
is exactly what each objective optimises — squared error in dollars punishes a
$100,000 miss on a mansion, squared error in logs punishes a 20 % miss equally
wherever it happens.

Which to prefer depends on whether being wrong by $30,000 is equally bad on a
$120,000 house and a $600,000 one. For a valuation tool it is not, so the log
model is the better default despite losing on the headline MAE.

Typical error is $17,675 on a median house of roughly $163,000 — about 10 %.
Two thirds of houses land within 10 % of their price.

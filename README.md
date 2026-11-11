# Ames house prices — is this house expensive, and what is it worth?

Two supervised-learning tasks on 1,460 house sales from Ames, Iowa: first a
classifier for "expensive or not", then a model for the price itself.

The short version: the classification task is harder than it needs to be,
because its label was made by drawing a line through a continuous price. Model
the price and apply the line afterwards and you do better — measurably, on the
same folds.

## The numbers

| | |
|---|---|
| Houses | 1,460, of which **217 (14.86 %) are "expensive"** |
| Predicting "not expensive" every time | **85.1 % accuracy**, recall 0.000 |
| Random forest, all 79 features | precision 0.909 · recall 0.733 · F1 0.811 |
| …its accuracy | 94.9 %, i.e. **+9.8 points** over doing nothing |
| Predicting the price, then applying the cut-off | **F1 0.843** (p = 0.0098, 20 folds) |
| Price model | MAE **$17,675** · MAPE **10.3 %** · 67.5 % of houses within 10 % |

## Accuracy is the wrong number here, and it is the one that gets quoted

A model that answers "not expensive" to every house in Ames is right 85.1 % of
the time. It finds none of the 217 expensive houses — recall 0.000 — and on a
slide, 85 % reads like a working model. The honest figure is how far above that
floor you got, which for the best model here is **+9.8 points**, and the figures
that actually describe it are precision 0.909 and recall 0.733: when it says
expensive it is almost always right, and it still misses 58 of 217.

`evaluate.scores()` therefore returns `accuracy_over_constant` next to
`accuracy`, and a test asserts the constant predictor scores 0.0 on it.

## The mistakes are not about houses, they are about a line

"Expensive" starts at **$250,580** — the cheapest house carrying the label. Sort
the model's outcomes by what the houses actually sold for:

| outcome | houses | median price | distance from the cut-off |
|---|---:|---:|---:|
| true positive | 159 | $320,000 | +$69,420 |
| **false negative** (missed) | 58 | $268,000 | **+$17,420** |
| **false positive** (false alarm) | 16 | $238,500 | **−$12,080** |
| true negative | 1,227 | $149,900 | −$100,680 |

Every error sits within about $17,000 of the boundary, and every confident
correct answer sits far from it. The model is not confusing mansions with
bungalows; it is being asked to resolve a distinction of a few per cent in price
that someone invented by thresholding a continuous variable.

Which suggests an experiment.

## Predict the price, then apply the rule

Two approaches, scored on **the same 20 folds** (5-fold × 4 seeds), with the
cut-off recomputed inside each training split — it is derived from the target,
so taking it from the whole dataset would hand every validation row a number
computed from itself:

| approach | F1 |
|---|---:|
| A — classify "expensive" directly | 0.813 ± 0.064 |
| B — predict the price, then cut at the threshold | **0.843 ± 0.050** |

Paired difference **+0.030 ± 0.047**, B ahead in 14 of 20 folds, paired
t-test **p = 0.0098**.

It is a modest effect and it is real. It also comes free: approach B returns the
estimated price as well as the label, so the same training run answers both
phases of the brief. When a categorical target was manufactured by thresholding
a number you still have, model the number.

## Tuning a model on the folds you then score it on

Four families, each tuned by grid search, each scored two ways: once on the same
folds the search used, and once with nested cross-validation, where the search
runs inside every training split.

| model | same folds | nested CV | optimism |
|---|---:|---:|---:|
| decision tree | 0.758 | 0.731 | **+0.027** |
| k-nearest neighbours | 0.744 | 0.725 | +0.019 |
| logistic regression | 0.819 | 0.819 | **+0.000** |
| random forest | 0.833 | 0.826 | +0.008 |

`GridSearchCV.best_score_` is the score of the configuration that won on those
folds, so quoting it as the model's performance counts the winner's luck as
skill. The size of that bias tracks how hard you searched and how unstable the
estimator is: twelve configurations of a high-variance tree inflate F1 by 0.027,
four values of `C` on a linear model by nothing at all.

The other result in that table is that **logistic regression lands within 0.007
F1 of a tuned random forest** (0.819 against 0.826) under honest evaluation. On
one-hot encoded Ames features the expensive/not boundary is close to linear, so
the forest's extra capacity buys almost nothing — and the linear model's
coefficients can be read aloud to an estate agent.

k-nearest neighbours is last at 0.725. It is the one family here whose distance
metric has to survive two hundred-odd one-hot columns, and it does not.

## More features only help a model that can use them

The course adds features in stages. Measured at two model capacities:

| features | decision tree (depth 5) | random forest |
|---|---:|---:|
| 9 numeric | F1 **0.724** · AUC 0.868 | F1 0.693 · AUC 0.933 |
| + one-hot categoricals (15) | 0.702 · 0.870 | 0.713 · 0.948 |
| + ordinal categoricals (23) | 0.690 · 0.864 | 0.773 · 0.965 |
| all 79 | 0.758 · **0.849** | **0.813** · **0.986** |

Read the tree column alone and the conclusion is that seventy extra features buy
0.034 F1 and cost AUC. Read the forest column and they buy 0.120 F1 and take AUC
from 0.933 to 0.986. The ranking even inverts: on nine features the depth-5 tree
*beats* the forest, and on seventy-nine it loses by a wide margin.

"Do more features help?" has no answer without naming the model, and a feature
iteration judged against one fixed estimator will reach whichever conclusion
that estimator's capacity allows.

## Running it

```bash
python3 analysis/run_all.py          # every table, written to findings/results/
python3 -m unittest discover -s tests -v
```

The dataset is bootcamp-provided and not redistributed here. Point the script at
the folder holding the iteration directories: `python3 analysis/run_all.py /path/to/data`.

```
housing/    data.py (the iteration files, and why Id is dropped)
            pipeline.py (preprocessing inside the estimator, and why)
            evaluate.py (metrics that survive a 14.86 % positive rate)
analysis/   run_all.py — every number in this README
findings/   FINDINGS.md and the result tables
tests/      the decisions pinned
```

Requires pandas, numpy, scikit-learn and scipy.

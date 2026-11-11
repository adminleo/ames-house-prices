"""The decisions, pinned."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from housing import data, evaluate, pipeline  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


class TestScoring(unittest.TestCase):
    def test_a_constant_predictor_scores_zero_over_the_floor(self):
        """The guard that stops 0.851 being reported as a result."""
        y = np.array([0] * 851 + [1] * 149)
        s = evaluate.scores(y, np.zeros_like(y))
        self.assertEqual(s["accuracy"], 0.851)
        self.assertEqual(s["accuracy_over_constant"], 0.0)
        self.assertEqual(s["recall"], 0.0)

    def test_recall_and_precision_are_not_symmetric(self):
        y = np.array([0, 0, 0, 0, 1, 1])
        s = evaluate.scores(y, np.array([0, 0, 0, 0, 1, 0]))
        self.assertEqual(s["precision"], 1.0)   # never wrong when it says yes
        self.assertEqual(s["recall"], 0.5)      # finds half of them

    def test_threshold_table_trades_recall_for_false_alarms(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 400)
        proba = np.clip(y * 0.4 + rng.normal(0.3, 0.2, 400), 0, 1)
        t = evaluate.threshold_table(y, proba)
        self.assertTrue(t["recall"].is_monotonic_increasing)
        self.assertTrue(t["false_alarms"].is_monotonic_increasing)
        self.assertTrue(t["missed_expensive"].is_monotonic_decreasing)


@unittest.skipUnless(os.path.isdir(DATA), "course dataset not present")
class TestAgainstTheRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = data.features_and_target(data.load(DATA, "it5_all"), "Expensive")

    def test_the_target_is_imbalanced(self):
        self.assertLess(float(self.y.mean()), 0.2)

    def test_id_never_reaches_the_model(self):
        """`Id` is a row number, and row order carries neighbourhood."""
        self.assertNotIn("Id", self.X.columns)

    def test_preprocessing_is_inside_the_estimator(self):
        """Fitting an imputer on the whole frame leaks a statistic into every
        validation fold. The test asserts the structure that prevents it."""
        m = pipeline.model(self.X, None)
        self.assertEqual(m.steps[0][0], "pre")
        self.assertEqual(len(m.steps), 2)

    def test_unseen_categories_do_not_raise(self):
        from sklearn.ensemble import RandomForestClassifier
        cat = [c for c in self.X.columns
               if c not in self.X.select_dtypes(include=np.number).columns]
        self.assertTrue(cat, "expected categorical columns")
        tr = self.X.iloc[:1000]
        m = pipeline.model(self.X, RandomForestClassifier(n_estimators=10, random_state=0))
        m.fit(tr, self.y.iloc[:1000])
        unseen = self.X.iloc[1000:].copy()
        unseen[cat[0]] = "A_CATEGORY_NEVER_SEEN"
        m.predict(unseen)        # must not raise

    def test_folds_keep_the_class_balance(self):
        rates = [float(self.y.iloc[te].mean()) for _, te in evaluate.folds(self.X, self.y)]
        self.assertLess(max(rates) - min(rates), 0.02)


if __name__ == "__main__":
    unittest.main(verbosity=2)

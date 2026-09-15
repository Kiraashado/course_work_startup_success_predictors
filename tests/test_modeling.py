import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import Pool
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from modeling import (FEATURE_SETS, make_model, make_splits, point_loss,
                      prepare_features, validate_predictions)


class ModelingTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "batch_year": [2010, 2012, 2014, 2016, 2018, 2020] * 4,
            "batch_season": ["Winter", "Summer"] * 12,
            "company_age": [1., 2., 3., 4., 5., 6.] * 4,
            "team_size": [0., 2., 5., 10., 30., 100.] * 4,
            "num_founders": [1., 2., 3., 2., 1., 2.] * 4,
            "status_followup": ["acquired"] * 24,
            "id": range(24),
        })
        self.y = np.array([0, 1] * 12)

    def test_only_baseline_features_are_exposed(self):
        x = prepare_features(self.frame)
        self.assertNotIn("status_followup", x)
        self.assertNotIn("id", x)
        self.assertNotIn("team_size", x)
        self.assertEqual(x.log_team_size.iloc[0], 0)
        self.assertAlmostEqual(x.log_team_size.iloc[5], np.log1p(100))

    def test_grouped_splits_never_mix_batches(self):
        groups = np.repeat(np.arange(12), 2)
        for train, test in make_splits(self.y, groups, "grouped", 42, 3):
            self.assertFalse(set(groups[train]) & set(groups[test]))

    def test_inner_indices_do_not_include_outer_test(self):
        groups = np.repeat(np.arange(12), 2)
        for train, test in make_splits(self.y, groups, "stratified", 42, 3):
            for inner_train, inner_valid in make_splits(self.y[train], groups[train], "stratified", 1, 2):
                self.assertFalse(set(train[inner_train]) & set(test))
                self.assertFalse(set(train[inner_valid]) & set(test))

    def test_imputation_is_fitted_on_training_only(self):
        x = prepare_features(self.frame)
        model = make_model("lr_team", {"C": 1}, 42).fit(x, self.y)
        imputer = model.named_steps["features"].named_transformers_["team"].named_steps["impute"]
        before = imputer.statistics_.copy()
        changed = x.iloc[:2].copy()
        changed.loc[:, "num_founders"] = 100000
        changed.loc[:, "log_team_size"] = np.nan
        model.predict_proba(changed)
        np.testing.assert_array_equal(imputer.statistics_, before)
        self.assertAlmostEqual(before[0], x.log_team_size.median())

    def test_native_shap_additivity_for_heldout_rows(self):
        x = prepare_features(self.frame)
        model = make_model("cb_team", {"iterations": 5, "depth": 2}, 42).fit(x.iloc[:18], self.y[:18])
        pool = Pool(x.iloc[18:], cat_features=["batch_season"])
        shap = model.get_feature_importance(pool, type="ShapValues", thread_count=2)
        np.testing.assert_allclose(shap.sum(axis=1), model.predict(pool, prediction_type="RawFormulaVal"), atol=1e-8)

    def test_constant_probability_is_training_prevalence(self):
        x = prepare_features(self.frame)
        y = np.array([1] * 3 + [0] * 21)
        model = make_model("constant", {}, 42).fit(x, y)
        np.testing.assert_allclose(model.predict_proba(x.iloc[:2])[:, 1], .125)

    def test_pointwise_loss_matches_metric(self):
        p = np.linspace(.1, .9, len(self.y))
        self.assertAlmostEqual(point_loss(self.y, p).mean(), log_loss(self.y, p))

    def test_predictions_require_unique_complete_coverage(self):
        row = {"scheme": "stratified", "repeat": 0, "model": "constant", "id": 1, "probability": .1}
        with self.assertRaises(ValueError):
            validate_predictions(pd.DataFrame([row, row]), [1])
        with self.assertRaises(ValueError):
            validate_predictions(pd.DataFrame([row]), [1, 2])

    def test_ablation_changes_only_requested_feature(self):
        self.assertEqual(set(FEATURE_SETS["cb_team"]) - set(FEATURE_SETS["cb_no_size"]), {"log_team_size"})
        self.assertEqual(set(FEATURE_SETS["cb_team"]) - set(FEATURE_SETS["cb_no_founders"]), {"num_founders"})


if __name__ == "__main__":
    unittest.main()

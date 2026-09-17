import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import Pool
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from modeling import (FEATURE_SETS, aggregate_scores, make_model, make_splits, paired_comparisons, point_loss,
                      prepare_features, scores, tune_and_fit, validate_predictions)


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
        cols = FEATURE_SETS["cb_team"]
        model = make_model("cb_team", {"iterations": 5, "depth": 2}, 42).fit(x.iloc[:18][cols], self.y[:18])
        pool = Pool(x.iloc[18:][cols], cat_features=["batch_season"])
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

    def test_feature_blocks_extend_team_model_as_declared(self):
        team = set(FEATURE_SETS["lr_team"])
        self.assertEqual(set(FEATURE_SETS["lr_geo"]) - team, {"country"})
        self.assertEqual(set(FEATURE_SETS["lr_tags"]) - team, {"tag_text"})
        self.assertEqual(set(FEATURE_SETS["lr_text"]) - team, {"short_description"})
        self.assertEqual(set(FEATURE_SETS["lr_all"]) - team,
                         {"country", "tag_text", "short_description"})

    def test_balanced_model_reuses_the_full_feature_set(self):
        self.assertEqual(FEATURE_SETS["lr_all_balanced"], FEATURE_SETS["lr_all"])
        model = make_model("lr_all_balanced", {"C": 1}, 42)
        self.assertEqual(model.named_steps["model"].class_weight, "balanced")

    def test_top_selection_uses_counts_not_average_fold_ratios(self):
        # Unequal sizes and event rates make the two aggregations differ.
        first = scores([1, 0, 0, 0], [.9, .3, .2, .1])
        second = scores([1, 1, 1, 0, 0, 0], [.9, .8, .7, .6, .5, .4])
        combined = aggregate_scores(pd.DataFrame([first, second]))
        self.assertEqual(combined["precision_at_10"], 1)
        self.assertEqual(combined["recall_at_10"], .5)
        self.assertEqual(combined["lift_at_10"], 2.5)

    def test_ties_use_expected_selection_counts(self):
        result = scores([1, 0, 0, 0], [.2] * 4)
        self.assertEqual(result["selected_positive"], .25)
        self.assertEqual(result["lift_at_10"], 1)

    def test_ap_tuning_never_ranks_across_fold_probability_scales(self):
        class FixedModel:
            def fit(self, x, y):
                return self

            def predict_proba(self, x):
                return np.column_stack([1-x.p, x.p])

        x = pd.DataFrame({"p": [.4, .3, .9, .8]})
        inner = [(np.array([2, 3]), np.array([0, 1])), (np.array([0, 1]), np.array([2, 3]))]
        with patch("modeling.make_model", return_value=FixedModel()):
            _, best, _ = tune_and_fit("lr_all_balanced", x, np.array([1, 0, 1, 0]), inner, 42)
        self.assertEqual(best["inner_average_precision"], 1)

    def test_tag_encoder_preserves_punctuation_and_multiword_labels(self):
        frame = self.frame.copy()
        frame["tags"] = [["Health & Wellness", "B2B/SaaS"] for _ in range(len(frame))]
        model = make_model("lr_tags", {"C": 1}, 42).fit(prepare_features(frame), self.y)
        vocabulary = model.named_steps["features"].named_transformers_["tags"].get_feature_names_out()
        self.assertEqual(set(vocabulary), {"Health & Wellness", "B2B/SaaS"})

    def test_cluster_resampling_preserves_shared_batch_error(self):
        rows = []
        for i in range(100):
            y = i % 2
            good = i < 50
            probability = (.8 if y else .2) if good else (.2 if y else .8)
            for model, p in [("constant", .5), ("lr_all", probability)]:
                rows.append({"scheme": "grouped", "repeat": 0, "id": i,
                             "y": y, "batch": "A" if good else "B", "model": model, "probability": p})
        result = paired_comparisons(pd.DataFrame(rows), n_bootstrap=500).iloc[0]
        self.assertGreater(result.cluster_high-result.cluster_low,
                           3*(result.conditional_high-result.conditional_low))


if __name__ == "__main__":
    unittest.main()

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import config
from modeling import FEATURE_SETS, OUT, point_loss, scores, validate_predictions


@unittest.skipUnless((OUT / "results.json").exists(), "Прогнозный эксперимент ещё не выполнен")
class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pred = pd.read_csv(OUT / "predictions.csv")
        cls.outer = pd.read_csv(OUT / "outer_splits.csv")
        cls.inner = pd.read_csv(OUT / "inner_splits.csv")
        cls.cohort = pd.read_parquet(config.PREPROCESSED_DIR / "analysis_cohort.parquet")
        cls.known = cls.cohort.loc[cls.cohort.outcome_known].set_index("id")

    def test_all_models_cover_known_companies_only(self):
        validate_predictions(self.pred, self.known.index)
        self.assertEqual(len(self.pred), len(self.known) * len(FEATURE_SETS) * 3)
        for _, group in self.pred.groupby(["scheme", "repeat"]):
            self.assertEqual(set(group.model), set(FEATURE_SETS))
        expected = self.known.success_binary.loc[self.pred.id].to_numpy(dtype=int)
        np.testing.assert_array_equal(self.pred.y, expected)

    def test_persisted_inner_splits_never_use_outer_test(self):
        all_ids = set(self.known.index)
        for keys, test in self.outer.groupby(["scheme", "repeat", "fold"]):
            scheme, repeat, fold = keys
            inner = self.inner.loc[self.inner.scheme.eq(scheme) & self.inner.repeat.eq(repeat) & self.inner.fold.eq(fold)]
            self.assertFalse(inner.id.duplicated().any())
            self.assertEqual(set(inner.id), all_ids - set(test.id))
            if scheme == "grouped":
                train_batches = set(self.known.loc[inner.id, "batch"])
                self.assertFalse(train_batches & set(test.batch))
                for _, valid in inner.groupby("inner_fold"):
                    train_ids = set(inner.id) - set(valid.id)
                    self.assertFalse(set(self.known.loc[list(train_ids), "batch"]) & set(self.known.loc[valid.id, "batch"]))

    def test_metrics_reproduce_from_saved_predictions(self):
        table = pd.read_csv(OUT / "metrics_by_repeat.csv").set_index(["scheme", "repeat", "model"])
        for keys, group in self.pred.groupby(["scheme", "repeat", "model"]):
            for metric, value in scores(group.y, group.probability).items():
                self.assertAlmostEqual(value, table.loc[keys, metric], places=12)

    def test_shap_matches_saved_probabilities(self):
        shap = pd.read_csv(OUT / "shap_oof.csv").set_index("id")
        self.assertEqual(set(shap.index), set(self.known.index))
        self.assertTrue(shap.index.is_unique)
        raw = shap[FEATURE_SETS["cb_team"]].sum(axis=1) + shap.expected_value
        np.testing.assert_allclose(raw, shap.raw_prediction, atol=1e-8)
        pred = self.pred.loc[self.pred.scheme.eq("stratified") & self.pred.repeat.eq(0) & self.pred.model.eq("cb_team")].set_index("id")
        np.testing.assert_allclose(1/(1+np.exp(-raw)), pred.loc[shap.index, "probability"], atol=1e-8)
        np.testing.assert_array_equal(shap.fold, pred.loc[shap.index, "fold"])

    def test_selected_parameters_minimize_inner_loss(self):
        tuning = pd.read_csv(OUT / "tuning.csv")
        for keys, trials in tuning.groupby(["scheme", "repeat", "fold", "model"]):
            selected = trials.loc[trials.selected]
            self.assertEqual(len(selected), 1)
            if keys[-1] != "constant":
                self.assertAlmostEqual(selected.inner_log_loss.iloc[0], trials.inner_log_loss.min())


if __name__ == "__main__":
    unittest.main()

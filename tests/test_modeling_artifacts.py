import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import config
from modeling import FEATURE_SETS, OUT, RANKING_MODELS, SCHEME_SEEDS, scores, validate_predictions


@unittest.skipUnless((OUT / "results.json").exists(), "Прогнозный эксперимент ещё не выполнен")
class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pred = pd.read_csv(OUT / "predictions.csv", float_precision="round_trip")
        cls.outer = pd.read_csv(OUT / "outer_splits.csv")
        cls.inner = pd.read_csv(OUT / "inner_splits.csv")
        cls.cohort = pd.read_parquet(config.PREPROCESSED_DIR / "analysis_cohort.parquet")
        cls.known = cls.cohort.loc[cls.cohort.outcome_known].set_index("id")

    def test_all_models_cover_known_companies_only(self):
        validate_predictions(self.pred, self.known.index)
        self.assertEqual(len(self.pred), len(self.known) * len(FEATURE_SETS) * sum(map(len, SCHEME_SEEDS.values())))
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
        table = pd.read_csv(OUT / "metrics_by_fold.csv").set_index(["scheme", "repeat", "fold", "model"])
        for keys, group in self.pred.groupby(["scheme", "repeat", "fold", "model"]):
            for metric, value in scores(group.y, group.probability).items():
                self.assertAlmostEqual(value, table.loc[keys, metric], places=12)

    def test_repeat_metrics_are_fold_weighted(self):
        fold = pd.read_csv(OUT / "metrics_by_fold.csv")
        repeat = pd.read_csv(OUT / "metrics_by_repeat.csv").set_index(["scheme", "repeat", "model"])
        for keys, group in fold.groupby(["scheme", "repeat", "model"]):
            expected = np.average(group.roc_auc, weights=group.n)
            self.assertAlmostEqual(expected, repeat.loc[keys, "roc_auc"])

    def test_shap_matches_saved_probabilities(self):
        shap = pd.read_csv(OUT / "shap_oof.csv").set_index("id")
        self.assertEqual(set(shap.index), set(self.known.index))
        self.assertTrue(shap.index.is_unique)
        raw = shap[FEATURE_SETS["cb_team"]].sum(axis=1) + shap.expected_value
        np.testing.assert_allclose(raw, shap.raw_prediction, atol=1e-8)
        pred = self.pred.loc[self.pred.scheme.eq("grouped") & self.pred.repeat.eq(0) & self.pred.model.eq("cb_team")].set_index("id")
        np.testing.assert_allclose(1/(1+np.exp(-raw)), pred.loc[shap.index, "probability"], atol=1e-8)
        np.testing.assert_array_equal(shap.fold, pred.loc[shap.index, "fold"])

    def test_selected_parameters_minimize_inner_loss(self):
        tuning = pd.read_csv(OUT / "tuning.csv")
        for keys, trials in tuning.groupby(["scheme", "repeat", "fold", "model"]):
            selected = trials.loc[trials.selected]
            self.assertEqual(len(selected), 1)
            if keys[-1] != "constant":
                if keys[-1] in RANKING_MODELS:
                    self.assertAlmostEqual(selected.inner_average_precision.iloc[0],
                                           trials.inner_average_precision.max())
                else:
                    self.assertAlmostEqual(selected.inner_log_loss.iloc[0], trials.inner_log_loss.min())

    def test_grouped_partitions_are_substantially_different(self):
        from sklearn.metrics import adjusted_rand_score
        wide = self.outer[self.outer.scheme.eq("grouped")].pivot(index="id", columns="repeat", values="fold")
        for a in wide.columns:
            for b in wide.columns:
                if a < b:
                    self.assertLess(adjusted_rand_score(wide[a], wide[b]), .8)

    def test_selection_metric_definitions_hold_after_aggregation(self):
        repeat = pd.read_csv(OUT / "metrics_by_repeat.csv")
        np.testing.assert_allclose(repeat.precision_at_10, repeat.selected_positive/repeat.selected_n)
        np.testing.assert_allclose(repeat.recall_at_10, repeat.selected_positive/repeat.positive)
        np.testing.assert_allclose(repeat.lift_at_10, repeat.precision_at_10/repeat.prevalence)

    def test_sensitivity_exclusions_and_outcomes(self):
        for experiment, exclude in [("without_ik12", self.known.batch.eq("IK12")),
                                    ("acquisitions_only", self.known.status_followup.eq("public"))]:
            pred = pd.read_csv(OUT / experiment / "predictions.csv", float_precision="round_trip")
            expected = self.known.loc[~exclude]
            validate_predictions(pred, expected.index)
            np.testing.assert_array_equal(pred.y, expected.success_binary.loc[pred.id].to_numpy(dtype=int))
            outer = pd.read_csv(OUT / experiment / "outer_splits.csv")
            for _, part in outer.groupby(["repeat", "fold"]):
                self.assertFalse(set(part.batch) & set(expected.loc[~expected.index.isin(part.id), "batch"]))

    def test_artifacts_match_current_code_and_complete_suite(self):
        import hashlib
        protocol = json.loads((OUT / "protocol.json").read_text())
        suite = json.loads((OUT / "suite_complete.json").read_text())
        code = Path(__file__).resolve().parents[1] / "src" / "modeling.py"
        self.assertEqual(protocol["code_sha256"], hashlib.sha256(code.read_bytes()).hexdigest())
        self.assertEqual(suite["code_sha256"], protocol["code_sha256"])
        self.assertGreater(suite["completed_at"], protocol["created_at"])


if __name__ == "__main__":
    unittest.main()

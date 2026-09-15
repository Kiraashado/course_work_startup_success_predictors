import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_preprocessing import build_cohort, clean_snapshot, consistency_report, parse_batch, parse_tags
from eda_analysis import describe, cohort_group, cohort_diagnostics


def raw_snapshot():
    return pd.DataFrame({
        "company_id": [1, 2, 3, 4, 5, 6],
        "company_name": list("abcdef"),
        "status": ["Active"] * 5 + ["Acquired"],
        "batch": ["W22", "S21", "W20", "IK12", "W23", "W10"],
        "team_size": [0, 10, 100, None, 3, 1],
        "num_founders": [0, 2, 3, 1, 2, 1],
        "year_founded": [2022, 2021, 2019, 2012, 2023, 2010],
        "country": ["CA", "US", "GB", None, "IN", "US"],
        "tags": ["[]", "['AI', 'AI']", "['B2B']", "[]", "[]", "[]"],
        "short_description": ["a", "bb", "ccc", None, "dd", "ee"],
    })


class CohortTests(unittest.TestCase):
    def test_exact_duplicates_removed(self):
        raw = raw_snapshot()
        clean, report = clean_snapshot(pd.concat([raw, raw.iloc[:2]]), "yc_2023_jul")
        self.assertEqual(len(clean), 6)
        self.assertEqual(report["removed_exact_duplicates"], 2)

    def test_conflicting_duplicates_rejected(self):
        raw = raw_snapshot()
        other = raw.iloc[:1].copy()
        other["team_size"] = 999
        with self.assertRaisesRegex(ValueError, "несовместимые"):
            clean_snapshot(pd.concat([raw, other]), "yc_2023_jul")

    def test_missing_identifier_rejected(self):
        raw = raw_snapshot()
        raw.loc[0, "company_id"] = np.nan
        with self.assertRaises(ValueError):
            clean_snapshot(raw, "yc_2023_jul")

    def test_seasons_and_special_batch(self):
        self.assertEqual(parse_batch("Spring 2026"), (2026, "Spring", True))
        self.assertEqual(parse_batch("Summer 2026"), (2026, "Summer", True))
        self.assertEqual(parse_batch("F25"), (2025, "Fall", True))
        self.assertEqual(parse_batch("IK12"), (2012, None, False))
        self.assertEqual(parse_batch(None), (None, None, False))

    def test_invalid_future_values_not_used(self):
        raw = raw_snapshot()
        raw.loc[0, "batch"] = "W27"
        raw.loc[0, "year_founded"] = 2024
        clean, _ = clean_snapshot(raw, "yc_2023_jul")
        self.assertTrue(pd.isna(clean.loc[0, "batch_year"]))
        self.assertTrue(pd.isna(clean.loc[0, "company_age"]))

    def test_zero_founders_missing_zero_team_retained(self):
        clean, _ = clean_snapshot(raw_snapshot(), "yc_2023_jul")
        self.assertTrue(pd.isna(clean.loc[0, "num_founders"]))
        self.assertEqual(clean.loc[0, "team_size"], 0)
        self.assertEqual(clean.loc[0, "country"], "CA")
        self.assertEqual(clean.loc[0, "company_age"], 1)

    def test_tags_are_deduplicated_without_alias_merging(self):
        self.assertEqual(parse_tags("['AI', 'AI', 'Artificial Intelligence']"), ["AI", "Artificial Intelligence"])
        self.assertEqual(parse_tags(np.array(["B2B", "AI"])), ["AI", "B2B"])
        self.assertEqual(parse_tags(np.nan), [])

    def test_active_and_inactive_are_negative_missing_is_not(self):
        baseline, _ = clean_snapshot(raw_snapshot(), "yc_2023_jul")
        followup = pd.DataFrame({"id": [1,2,3,4,99], "status": ["active","inactive","acquired","public","active"]})
        cohort = build_cohort(baseline, followup).set_index("id")
        self.assertEqual(cohort.index.tolist(), [1,2,3,4,5])
        self.assertEqual(cohort.loc[[1,2,3,4], "success_binary"].tolist(), [0,0,1,1])
        self.assertTrue(pd.isna(cohort.loc[5, "success_binary"]))
        self.assertFalse(cohort.loc[5, "followup_found"])

    def test_unrecognized_matched_status_is_unknown(self):
        baseline, _ = clean_snapshot(raw_snapshot(), "yc_2023_jul")
        cohort = build_cohort(baseline, pd.DataFrame({"id":[1],"status":["unknown"]}))
        self.assertTrue(cohort.loc[0,"followup_found"])
        self.assertFalse(cohort.loc[0,"outcome_known"])
        self.assertTrue(pd.isna(cohort.loc[0,"success_binary"]))

    def test_no_late_feature_completion(self):
        baseline, _ = clean_snapshot(raw_snapshot(), "yc_2023_jul")
        followup = pd.DataFrame({"id":[4],"status":["active"],"team_size":[999],"industry":["AI"]})
        cohort = build_cohort(baseline, followup).set_index("id")
        self.assertTrue(pd.isna(cohort.loc[4,"team_size"]))
        self.assertTrue(pd.isna(cohort.loc[4,"industry"]))
        self.assertEqual(cohort.loc[1,"team_size"], 0)

    def test_february_is_audit_not_population(self):
        july, _ = clean_snapshot(raw_snapshot(), "yc_2023_jul")
        feb = july.copy()
        feb.loc[0,"id"] = 999
        report = consistency_report(feb, july)
        self.assertEqual(report["only_february"], 1)
        self.assertFalse(report["used_to_fill_july"])
        self.assertNotIn(999, build_cohort(july, feb).id.tolist())

    def test_numerical_summary_missing_count(self):
        result = describe(pd.Series([1.0, 3.0, np.nan]))
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["missing"], 1)
        self.assertEqual(result["median"], 2)

    def test_special_group_is_separate(self):
        july, _ = clean_snapshot(raw_snapshot(), "yc_2023_jul")
        self.assertEqual(cohort_group(july.iloc[3]), "IK12")
        self.assertEqual(cohort_group(july.iloc[0]), "2021-2023")

    def test_completeness_diagnostics_keep_unknown_outcomes(self):
        july, _ = clean_snapshot(raw_snapshot(), "yc_2023_jul")
        cohort = build_cohort(july, pd.DataFrame({"id": [2], "status": ["active"]}))
        result = cohort_diagnostics(cohort)
        self.assertEqual(result["complete_cases"], 3)
        self.assertEqual(result["by_outcome_availability"]["known"]["n"], 1)
        self.assertEqual(result["by_outcome_availability"]["unknown"]["n"], 4)
        self.assertEqual(result["by_outcome_availability"]["unknown"]["complete_cases"], 2)


if __name__ == "__main__":
    unittest.main()

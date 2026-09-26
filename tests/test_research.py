import tempfile
import unittest
from pathlib import Path

import pandas as pd

from stem_projects.research import _input_type, analyze, load_studies

SOURCE = Path(__file__).resolve().parents[1] / "keratoconus_analysis/data/kc_studies.csv"


class ResearchTests(unittest.TestCase):
    def test_original_sheet_counts_and_missing_data(self):
        frame = load_studies(SOURCE)
        report = analyze(frame)
        self.assertEqual(report["study_rows"], 28)
        self.assertEqual(report["exact_accuracy_rows"], 20)
        self.assertEqual(report["metric_availability"]["sensitivity"]["exact"], 22)
        self.assertEqual(report["metric_availability"]["specificity"]["exact"], 22)
        self.assertEqual(len(report["excluded_accuracy_rows"]), 8)
        self.assertIn({"study_id": "KC16", "accuracy_raw": ">95.5",
                       "reason": "bound, not an exact value"}, report["excluded_accuracy_rows"])
        self.assertEqual(report["severity_accuracy"]["status"], "not computed")
        self.assertEqual([x["study_id"] for x in report["coding_review_flags"]], ["KC03"])

    def test_original_values_preserved_and_typo_mapped_only_in_analysis(self):
        frame = load_studies(SOURCE)
        self.assertEqual(frame.loc[frame.study_id == "KC16", "accuracy_raw"].iat[0], ">95.5")
        self.assertEqual(_input_type("TMS-4 Corneal Tomography and Topoagraphy"), "both")

    def test_invalid_percent_fails(self):
        frame = pd.read_csv(SOURCE, dtype=str, keep_default_na=False)
        frame.loc[0, "accuracy_raw"] = "101"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "percentage"):
                load_studies(path)


if __name__ == "__main__":
    unittest.main()

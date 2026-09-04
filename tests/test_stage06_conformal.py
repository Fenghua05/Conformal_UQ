import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conformal_uq.conformal import (
    binary_geometry_categories,
    evaluate_split_cp,
    exact_conformal_quantile,
    nonconformity_scores,
    prediction_sets,
    select_nested_calibration_subsets,
)
from conformal_uq.metrics import binary_predictive_metrics, global_coverage_sanity, wilson_interval


class Stage06ConformalTests(unittest.TestCase):
    def setUp(self) -> None:
        # Protocol label 0 is deliberately minority here; numerical label is not semantics.
        self.calibration_ids = tuple(f"cal:{index:03d}" for index in range(300))
        self.calibration_y = np.array([1] * 200 + [0] * 100, dtype=np.int8)
        p1 = np.concatenate((np.linspace(0.62, 0.99, 200), np.linspace(0.01, 0.38, 100)))
        self.calibration_proba = np.column_stack((1.0 - p1, p1))
        self.test_y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int8)
        test_p1 = np.array([0.10, 0.90, 0.45, 0.55, 0.20, 0.80])
        self.test_proba = np.column_stack((1.0 - test_p1, test_p1))
        self.subsets = select_nested_calibration_subsets(
            self.calibration_ids, self.calibration_y, protocol_version="v1.0",
            dataset_id="stage06_unit", base_seed=104729, minority_label=0,
        )

    def test_exact_rank_qc_boundary_and_score_mapping(self) -> None:
        scores10 = np.linspace(0.0, 0.9, 10)
        q10 = exact_conformal_quantile(scores10, alpha=0.1)
        self.assertEqual(q10.rank, 10)
        self.assertEqual(q10.threshold, max(scores10))
        self.assertEqual(exact_conformal_quantile(np.arange(20, dtype=float), 0.1).rank, 19)
        scores = nonconformity_scores(np.array([0, 1]), np.array([[0.8, 0.2], [0.3, 0.7]]))
        np.testing.assert_allclose(scores, [0.2, 0.3])

    def test_fixed_majority_nested_minority_and_shared_cp_subset_identity(self) -> None:
        self.assertEqual(len(self.subsets.majority_ids), 200)
        self.assertEqual(len(self.subsets.minority_ids_by_m[100]), 100)
        self.assertTrue(set(self.subsets.minority_ids_by_m[10]) < set(self.subsets.minority_ids_by_m[20]))
        self.assertTrue(set(self.subsets.minority_ids_by_m[20]) < set(self.subsets.minority_ids_by_m[50]))
        self.assertTrue(set(self.subsets.minority_ids_by_m[50]) < set(self.subsets.minority_ids_by_m[100]))
        global_result = evaluate_split_cp(self.calibration_ids, self.calibration_y, self.calibration_proba, self.test_y, self.test_proba, self.subsets, m_minority=20, cp_method="global_split_cp")
        conditional_result = evaluate_split_cp(self.calibration_ids, self.calibration_y, self.calibration_proba, self.test_y, self.test_proba, self.subsets, m_minority=20, cp_method="class_conditional_cp")
        self.assertEqual(global_result["subset_hash"], conditional_result["subset_hash"])
        self.assertEqual(global_result["n_cal_total"], 220)
        self.assertEqual(conditional_result["rank_minority"], 19)
        self.assertEqual(conditional_result["rank_majority"], 181)

    def test_binary_geometry_matches_actual_sets_for_empty_and_doubleton_cases(self) -> None:
        probabilities = np.column_stack((1.0 - np.array([0.10, 0.30, 0.50, 0.70, 0.90]), np.array([0.10, 0.30, 0.50, 0.70, 0.90])))
        empty_actual = prediction_sets(probabilities, (0.25, 0.25)).sum(axis=1)
        empty_geometry = binary_geometry_categories(probabilities[:, 1], 0.25, 0.25)
        self.assertEqual(empty_geometry.tolist(), ["singleton", "empty", "empty", "empty", "singleton"])
        self.assertEqual(np.where(empty_actual == 0, "empty", "singleton").tolist(), empty_geometry.tolist())
        double_actual = prediction_sets(probabilities, (0.75, 0.75)).sum(axis=1)
        double_geometry = binary_geometry_categories(probabilities[:, 1], 0.75, 0.75)
        self.assertEqual(double_geometry.tolist(), ["singleton", "doubleton", "doubleton", "doubleton", "singleton"])
        self.assertEqual(np.where(double_actual == 2, "doubleton", "singleton").tolist(), double_geometry.tolist())
        equality_probability = np.array([[0.6, 0.4]])
        self.assertEqual(binary_geometry_categories(equality_probability[:, 1], 0.4, 0.6).tolist(), ["doubleton"])
        self.assertEqual(int(prediction_sets(equality_probability, (0.4, 0.6)).sum()), 2)

    def test_metrics_wilson_decomposition_and_predictive_invariance(self) -> None:
        base = binary_predictive_metrics(self.test_y, self.test_proba)
        grid = [
            evaluate_split_cp(self.calibration_ids, self.calibration_y, self.calibration_proba, self.test_y, self.test_proba, self.subsets, m_minority=m, cp_method=method)
            for m in (10, 20, 50, 100) for method in ("global_split_cp", "class_conditional_cp")
        ]
        self.assertTrue(all(abs(row["empty_rate"] + row["singleton_rate"] + row["doubleton_rate"] - 1.0) < 1e-12 for row in grid))
        self.assertTrue(all(base == binary_predictive_metrics(self.test_y, self.test_proba) for _ in grid))
        low, high = wilson_interval(9, 10)
        self.assertLess(low, 0.9)
        self.assertGreater(high, 0.9)
        sanity = global_coverage_sanity(9, 10)
        self.assertTrue(sanity["nominal_within_wilson"])
        self.assertEqual(sanity["observed_coverage"], 0.9)
        self.assertTrue(all(0.0 <= row["coverage_overall"] <= 1.0 for row in grid))


if __name__ == "__main__":
    unittest.main()

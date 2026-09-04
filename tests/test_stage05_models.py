import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conformal_uq.data import BinaryTable
from conformal_uq.metrics import binary_predictive_metrics
from conformal_uq.models import fit_predict_locked_pipeline
from conformal_uq.split import make_stratified_split


class Stage05ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = 80
        self.table = BinaryTable(
            dataset_id="stage05_unit",
            features=pd.DataFrame({
                "numeric": [float(i) for i in range(rows)],
                "category": ["a" if i % 3 else "b" for i in range(rows)],
            }),
            labels=pd.Series([0] * 48 + [1] * 32, dtype="int8"),
            sample_ids=tuple(f"stage05_unit:{i:08d}" for i in range(rows)),
            raw_path=Path("unit.arff"), raw_sha256="b" * 64,
            target_column="target", label_mapping={"no": 0, "yes": 1},
        )
        self.split = make_stratified_split(self.table, 104729)

    def test_lr_outputs_aligned_binary_probabilities_and_metrics(self) -> None:
        result = fit_predict_locked_pipeline(
            self.table, self.split, "logistic_regression", {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 2000, "class_weight": None}, 7,
        )
        self.assertEqual(result.class_labels, (0, 1))
        self.assertEqual(result.calibration_probabilities.shape, (len(self.split.ids.calibration_pool), 2))
        self.assertEqual(result.test_probabilities.shape, (len(self.split.ids.test), 2))
        self.assertTrue(np.allclose(result.calibration_probabilities.sum(axis=1), 1.0))
        self.assertTrue(np.all((result.test_probabilities >= 0) & (result.test_probabilities <= 1)))
        metrics = binary_predictive_metrics(result.test_y, result.test_probabilities)
        self.assertGreaterEqual(metrics["auroc"], 0.0)
        self.assertLessEqual(metrics["auprc"], 1.0)

    def test_xgboost_is_deterministic_from_derived_seed(self) -> None:
        hp = {"objective": "binary:logistic", "eval_metric": "logloss", "n_estimators": 4, "max_depth": 2, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "reg_lambda": 1, "reg_alpha": 0, "scale_pos_weight": 1, "tree_method": "hist", "n_jobs": 1, "early_stopping": False}
        one = fit_predict_locked_pipeline(self.table, self.split, "xgboost", hp, 13)
        two = fit_predict_locked_pipeline(self.table, self.split, "xgboost", hp, 13)
        np.testing.assert_allclose(one.test_probabilities, two.test_probabilities)
        self.assertEqual(one.model_hash, two.model_hash)


if __name__ == "__main__":
    unittest.main()

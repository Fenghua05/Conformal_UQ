import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conformal_uq.data import BinaryTable, audit_table
from conformal_uq.preprocessing import TrainOnlyPreprocessor
from conformal_uq.split import make_stratified_split, split_feasibility


def _table() -> BinaryTable:
    rows = 40
    return BinaryTable(
        dataset_id="unit_dataset",
        features=pd.DataFrame(
            {
                "numeric": [float(i) if i not in {3, 19} else np.nan for i in range(rows)],
                "category": ["seen" if i % 2 else "other" for i in range(rows)],
                "customer_id": [f"row-{i:03d}" for i in range(rows)],
            }
        ),
        labels=pd.Series([0] * 24 + [1] * 16, dtype="int8"),
        sample_ids=tuple(f"unit_dataset:{i:08d}" for i in range(rows)),
        raw_path=Path("unit.arff"),
        raw_sha256="a" * 64,
        target_column="target",
        label_mapping={"negative": 0, "positive": 1},
    )


class Stage04SplitTests(unittest.TestCase):
    def test_split_is_stratified_disjoint_and_repeatable(self) -> None:
        table = _table()
        first = make_stratified_split(table, base_seed=104729)
        second = make_stratified_split(table, base_seed=104729)
        self.assertEqual(first.ids, second.ids)
        self.assertEqual(first.split_hash, second.split_hash)
        self.assertFalse(set(first.ids.train) & set(first.ids.calibration_pool))
        self.assertFalse(set(first.ids.train) & set(first.ids.test))
        self.assertFalse(set(first.ids.calibration_pool) & set(first.ids.test))
        self.assertEqual(len(first.ids.train), 24)
        self.assertEqual(len(first.ids.calibration_pool), 8)
        self.assertEqual(len(first.ids.test), 8)
        self.assertEqual(first.class_counts["calibration_pool"]["minority"], 3)
        self.assertEqual(first.class_counts["test"]["minority"], 3)

    def test_feasibility_reports_frozen_minimums_without_changing_split(self) -> None:
        report = split_feasibility(_table(), base_seed=104729)
        self.assertFalse(report["pass"])
        self.assertEqual(report["requirements"], {"calibration_minority": 100, "calibration_majority": 200, "test_minority": 75})


class Stage04PreprocessingTests(unittest.TestCase):
    def test_train_only_fit_ignores_unknown_category_and_records_scope(self) -> None:
        table = _table()
        split = make_stratified_split(table, base_seed=104729)
        processor = TrainOnlyPreprocessor("logistic_regression")
        processor.fit(table, split)
        self.assertEqual({entry["fit_scope"] for entry in processor.fit_audit}, {"train"})
        self.assertEqual(processor.fit_audit[0]["fit_ids_hash"], processor.train_ids_hash)
        cal = processor.transform(table, split.ids.calibration_pool, partition="calibration_pool")
        self.assertEqual(cal.shape[0], len(split.ids.calibration_pool))
        held_out = table.features.copy()
        held_out.loc[table.row_positions(split.ids.test)[0], "category"] = "unseen-at-fit"
        transformed = processor.transform_features(held_out.iloc[table.row_positions(split.ids.test)], split.ids.test, partition="test")
        self.assertEqual(transformed.shape[0], len(split.ids.test))

    def test_fit_refuses_non_train_ids_and_transform_never_accepts_labels(self) -> None:
        table = _table()
        split = make_stratified_split(table, base_seed=104729)
        processor = TrainOnlyPreprocessor("xgboost")
        with self.assertRaises(ValueError):
            processor.fit(table, split, fit_ids=split.ids.calibration_pool)
        with self.assertRaises(TypeError):
            processor.transform(table, split.ids.test, partition="test", labels=table.labels)


class Stage04AuditTests(unittest.TestCase):
    def test_audit_exposes_identifier_and_missingness_without_mutating_raw_table(self) -> None:
        table = _table()
        original = table.features.copy(deep=True)
        report = audit_table(table)
        self.assertIn("customer_id", report["identifier_like_columns"])
        self.assertEqual(report["missing_values"], 2)
        pd.testing.assert_frame_equal(table.features, original)


if __name__ == "__main__":
    unittest.main()

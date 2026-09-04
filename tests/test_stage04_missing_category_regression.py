import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conformal_uq.data import BinaryTable
from conformal_uq.preprocessing import TrainOnlyPreprocessor
from conformal_uq.split import make_stratified_split


class MissingCategoricalRegressionTest(unittest.TestCase):
    def test_pandas_na_categorical_values_are_accepted_by_train_only_imputer(self) -> None:
        table = BinaryTable(
            dataset_id="missing_category",
            features=pd.DataFrame({"category": [pd.NA, "seen", "other", "seen", "other"] * 8}),
            labels=pd.Series([0] * 24 + [1] * 16, dtype="int8"),
            sample_ids=tuple(f"missing_category:{index:08d}" for index in range(40)),
            raw_path=Path("missing-category.arff"), raw_sha256="b" * 64,
            target_column="target", label_mapping={"negative": 0, "positive": 1},
        )
        split = make_stratified_split(table, 104729)
        TrainOnlyPreprocessor("logistic_regression").fit(table, split)


if __name__ == "__main__":
    unittest.main()

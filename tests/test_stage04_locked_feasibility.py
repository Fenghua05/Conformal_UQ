import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conformal_uq.data import load_locked_dataset
from conformal_uq.split import make_stratified_split, split_feasibility_from_split


class LockedDatasetSplitRegressionTests(unittest.TestCase):
    def test_all_locked_dataset_seed_splits_are_repeatable_and_feasible(self) -> None:
        config = yaml.safe_load((ROOT / "configs" / "stage03_base_v1.0.yaml").read_text(encoding="utf-8"))
        for dataset_id in config["datasets"]["primary_ids"]:
            table = load_locked_dataset(ROOT, dataset_id)
            for base_seed in config["experiment"]["seeds"]:
                first = make_stratified_split(table, base_seed)
                second = make_stratified_split(table, base_seed)
                self.assertEqual(first.ids, second.ids, f"{dataset_id}, seed {base_seed}")
                self.assertEqual(first.split_hash, second.split_hash, f"{dataset_id}, seed {base_seed}")
                self.assertTrue(split_feasibility_from_split(first)["pass"], f"{dataset_id}, seed {base_seed}")


if __name__ == "__main__":
    unittest.main()

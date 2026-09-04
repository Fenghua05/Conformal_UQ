import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud" / "tabpfn_stage05b"))

from stage05b_common import checked_lock, expected_units_from_lock


class Stage05BFormalLockTests(unittest.TestCase):
    def test_approved_formal_lock_has_exact_controlled_inputs_and_twenty_units(self) -> None:
        lock, config_sha256, code_sha256 = checked_lock(
            ROOT,
            ROOT / "configs/stage05b_tabpfn_v1.0.yaml",
            ROOT / "decisions/pilot_decision_stage07_v1.0.json",
        )
        self.assertEqual(lock["pilot_dataset_ids"], ["openml_3_kr_vs_kp", "openml_24_mushroom"])
        self.assertEqual(len(expected_units_from_lock(lock)), 20)
        self.assertEqual(len(config_sha256), 64)
        self.assertEqual(len(code_sha256), 64)
        self.assertEqual(lock["runtime_budget"]["maximum_wall_clock_hours"], 6)
        self.assertEqual(lock["runtime_budget"]["maximum_cloud_storage_gb"], 30)


if __name__ == "__main__":
    unittest.main()

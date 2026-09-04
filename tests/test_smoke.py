import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conformal_uq.toy import make_toy_contract, toy_results_record
from conformal_uq.results_schema import validate_results_record


class SmokeTests(unittest.TestCase):
    def test_toy_data_is_disjoint_and_not_research(self) -> None:
        table, split = make_toy_contract()
        self.assertEqual(len(table.sample_ids), 12)
        self.assertFalse(set(split.train) & set(split.calibration_pool))
        self.assertFalse(set(split.train) & set(split.test))
        self.assertFalse(set(split.calibration_pool) & set(split.test))
        record = toy_results_record("a" * 64, "b" * 64, "toy")
        self.assertEqual(record["status"], "TOY_ONLY_NOT_RESEARCH_RESULT")
        self.assertEqual(validate_results_record(record), [])

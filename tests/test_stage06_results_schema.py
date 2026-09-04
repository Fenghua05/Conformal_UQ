import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conformal_uq.results_schema import validate_results_record, validate_results_records
from conformal_uq.toy import toy_results_record


class Stage06ResultsSchemaTests(unittest.TestCase):
    def test_complete_provenance_wilson_and_unique_key_contract(self) -> None:
        global_row = toy_results_record("a" * 64, "b" * 64, "stage06-toy")
        self.assertEqual(validate_results_record(global_row), [])
        duplicate = copy.deepcopy(global_row)
        self.assertEqual(validate_results_records([global_row, duplicate]), ["row:1:duplicate:unique_key"])
        changed = copy.deepcopy(global_row)
        changed["coverage_minority_wilson_low"] = 0.0
        self.assertIn("invalid:wilson_minority", validate_results_record(changed))

    def test_class_conditional_threshold_fields_are_coherent(self) -> None:
        row = toy_results_record("a" * 64, "b" * 64, "stage06-toy")
        row.update({
            "cp_method": "class_conditional_cp", "q_global": None, "rank_global": None,
            "q_minority": 0.4, "q_majority": 0.2, "threshold_gap": 0.2, "threshold_sum": 0.6,
            "rank_minority": 10, "rank_majority": 181,
        })
        self.assertEqual(validate_results_record(row), [])
        row["threshold_sum"] = 0.7
        self.assertIn("invalid:threshold_geometry_fields", validate_results_record(row))

    def test_json_schema_declares_new_required_contract_fields(self) -> None:
        schema = json.loads((ROOT / "configs" / "results_long.schema.json").read_text(encoding="utf-8"))
        names = {column["name"] for column in schema["columns"]}
        self.assertEqual(schema["schema_id"], "conformal-uq-results-long-v1.1.0")
        self.assertTrue({"coverage_minority_wilson_low", "prediction_cache_hash", "label_mapping_hash", "rank_minority"}.issubset(names))


if __name__ == "__main__":
    unittest.main()

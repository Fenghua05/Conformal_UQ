import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class RegistryReconciliationTests(unittest.TestCase):
    def test_reconciliation_accepts_interleaved_candidate_catalogue(self) -> None:
        from reconcile_stage02_registry_v1_0_1 import PRIMARY_IDS, REPLACEMENT_IDS, reconcile_registry

        source = json.loads((ROOT / "artifacts/stage02/dataset_registry_v1.0.json").read_text(encoding="utf-8"))
        self.assertNotEqual(tuple(record["dataset_id"] for record in source["records"][:8]), PRIMARY_IDS)
        revised = reconcile_registry(source)
        by_id = {record["dataset_id"]: record for record in revised["records"]}
        self.assertEqual(revised["status"], "LOCKED_BY_USER_CONFIRMATION")
        self.assertTrue(all(by_id[dataset_id]["selection_status"] == "LOCKED_BY_USER_CONFIRMATION" for dataset_id in PRIMARY_IDS))
        self.assertTrue(all(by_id[dataset_id]["selection_status"] == "LOCKED_ORDERED_REPLACEMENT" for dataset_id in REPLACEMENT_IDS))

"""Contract tests for the frozen formal run manifest and the 1,920-cell experiment.

All tests run locally WITHOUT importing TabPFN or fitting any model.  The final
class verifies the real immutable formal run after
``src/run_formal_experiment_v1_1.py`` has executed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conformal_uq.config import FROZEN_DATASETS, FROZEN_SEEDS

RUNNER = ROOT / "src" / "run_formal_experiment_v1_1.py"
MANIFEST = ROOT / "configs" / "formal_run_manifest_v1.1.yaml"
RECEIPT = ROOT / "decisions" / "D08-004_FORMAL_RUN_GO_RECEIPT.json"
AUDIT = ROOT / "artifacts" / "stage08_v11" / "20260831T085012Z_pilot_independent_audit" / "independent_audit.json"
INTAKE_AUDIT = ROOT / "artifacts" / "stage08_v11_cloud" / "cache_intake_20260831T081631Z" / "intake_audit.json"
PILOT_RUN = ROOT / "artifacts" / "stage07_v1.1" / "20260831T083430Z_stage07-v11-pilot_32b7e4728b8d"
V10_PILOT_RUN = ROOT / "artifacts" / "stage07" / "20260830T161214Z_stage07-pilot_32b7e4728b8d"
RUNS_ROOT = ROOT / "artifacts" / "runs"

MODELS = ("logistic_regression", "xgboost", "tabpfn")
CP_METHODS = ("global_split_cp", "class_conditional_cp")
M_MINORITY = (10, 20, 50, 100)


def load_runner():
    spec = importlib.util.spec_from_file_location("stage09_formal_runner_for_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage09_formal_runner_for_test"] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FormalAuthorizationTests(unittest.TestCase):
    """D08-004 and the frozen manifest must bind each other and the audit."""

    def test_d08_004_receipt_records_the_user_go_with_bounded_scope(self) -> None:
        receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "APPROVED_FOR_FORMAL_MANIFEST_FREEZE_AND_1920_CELL_EXPERIMENT")
        self.assertEqual(receipt["explicit_go"], "go")
        self.assertEqual(receipt["approver"], "user")
        self.assertEqual(receipt["protocol_version"], "v1.1")
        self.assertTrue(receipt["formal_run_manifest_authorized"])
        self.assertTrue(receipt["full_experiment_authorized"])
        self.assertEqual(receipt["expected_formal_cells"], 1920)
        basis = receipt["authorization_basis"]
        self.assertEqual(basis["stage08_v11_independent_audit_sha256"], sha256_file(AUDIT))
        self.assertEqual(basis["stage08_v11_audit_verdict"], "PASS")
        self.assertEqual(basis["gate_recommendation_reviewed_by_user"], "CONDITIONAL-GO")
        self.assertEqual(receipt["formal_run_manifest"]["path"], "configs/formal_run_manifest_v1.1.yaml")

    def test_frozen_manifest_binds_every_controlled_input(self) -> None:
        import yaml

        manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["artifact_status"], "FROZEN_FORMAL_RUN_MANIFEST")
        self.assertEqual(manifest["grid"]["expected_cells"], 1920)
        self.assertEqual(manifest["registry"]["locked_primary_ids"], FROZEN_DATASETS)
        self.assertEqual(manifest["seeds"], FROZEN_SEEDS)
        self.assertEqual(manifest["authorization"]["d08_004_receipt_sha256"], sha256_file(RECEIPT))
        expected_bindings = {
            "protocol_sha256": "protocols/protocol_v1.1.md",
            "dataset_lock_sha256": "protocols/dataset_lock_v1.0.md",
            "split_lock_sha256": "configs/stage04_splits_v1.1.yaml",
            "local_cache_lock_sha256": "configs/stage05_lr_xgboost_v1.1.yaml",
            "tabpfn_cache_lock_sha256": "configs/stage05b_tabpfn_v1.1.yaml",
            "pilot_decision_sha256": "decisions/pilot_decision_stage07_v1.1.json",
            "environment_sha256": "environment/environment_lock_v1.0.json",
            "results_schema_sha256": "configs/results_long.schema.json",
            "stage08_v11_audit_sha256": str(AUDIT.relative_to(ROOT)).replace("\\", "/"),
        }
        for key, relative in expected_bindings.items():
            self.assertEqual(manifest["input_hashes"][key], sha256_file(ROOT / relative), f"binding drifted: {key}")
        lineage = manifest["cache_lineage"]
        intake = json.loads(INTAKE_AUDIT.read_text(encoding="utf-8"))
        self.assertEqual(intake["verdict"], "PASS")
        self.assertEqual(lineage["intake_audit_sha256"], sha256_file(INTAKE_AUDIT))
        self.assertEqual(lineage["local_config_hash"], intake["lineage"]["local_config_hash"])
        self.assertEqual(lineage["local_cache_time_code_hash"], intake["lineage"]["local_cache_time_code_hash"])
        self.assertEqual(lineage["tabpfn_config_hash"], intake["lineage"]["tabpfn_config_hash"])
        self.assertEqual(lineage["tabpfn_cache_time_code_hash"], intake["lineage"]["tabpfn_cache_time_code_hash"])
        self.assertEqual(lineage["complete_cache_units"], 240)
        gate = manifest["execution_gate"]
        self.assertTrue(gate["d08_004_go_receipt_required_before_execution"])
        self.assertFalse(gate["model_fitting_allowed"])
        self.assertFalse(gate["cloud_execution_allowed"])
        self.assertFalse(gate["cache_regeneration_allowed"])
        self.assertFalse(gate["protocol_change_allowed"])


class FormalRunnerContractTests(unittest.TestCase):
    """The formal runner gates and grid, before any cell exists."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load_runner()

    def test_runner_module_imports_without_tabpfn(self) -> None:
        self.assertNotIn("tabpfn", sys.modules)

    def test_expected_formal_grid_is_exactly_1920(self) -> None:
        cells = self.runner.expected_formal_cells()
        self.assertEqual(len(cells), 1920)
        self.assertEqual(len(set(cells)), 1920)
        self.assertEqual({cell[0] for cell in cells}, set(FROZEN_DATASETS))
        self.assertEqual({cell[2] for cell in cells}, set(MODELS))
        self.assertEqual({cell[3] for cell in cells}, set(CP_METHODS))
        self.assertEqual({cell[4] for cell in cells}, set(M_MINORITY))

    def test_receipt_gate_requires_the_canonical_d08_004(self) -> None:
        with self.assertRaises(Exception):
            self.runner.load_validated_go_receipt(ROOT, ROOT / "decisions" / "D08-003_V11_CACHE_AND_PILOT_BUDGET_RECEIPT.json")
        receipt = self.runner.load_validated_go_receipt(ROOT, RECEIPT)
        self.assertEqual(receipt["explicit_go"], "go")
        self.assertEqual(receipt["expected_formal_cells"], 1920)

    def test_manifest_gate_requires_the_canonical_frozen_manifest(self) -> None:
        with self.assertRaises(Exception):
            self.runner.load_validated_formal_manifest(ROOT, ROOT / "configs" / "stage05b_tabpfn_v1.1.yaml")
        manifest = self.runner.load_validated_formal_manifest(ROOT, MANIFEST)
        self.assertEqual(manifest["grid"]["expected_cells"], 1920)

    def test_lineage_resolution_matches_the_manifest_and_current_locks(self) -> None:
        lineage = self.runner.resolve_formal_lineage(ROOT)
        manifest_lineage = self.runner.load_validated_formal_manifest(ROOT, MANIFEST)["cache_lineage"]
        key_map = {
            "local_config_hash": "local_config_hash",
            "local_code_hash": "local_cache_time_code_hash",
            "tabpfn_config_hash": "tabpfn_config_hash",
            "tabpfn_code_hash": "tabpfn_cache_time_code_hash",
            "environment_hash": "environment_hash",
        }
        for key, manifest_key in key_map.items():
            self.assertEqual(lineage[key], manifest_lineage[manifest_key], f"lineage key {key} drifted")

    def test_config_hash_is_deterministic_over_manifest_and_receipt(self) -> None:
        digest = hashlib.sha256()
        digest.update(MANIFEST.read_bytes())
        digest.update(RECEIPT.read_bytes())
        self.assertEqual(self.runner.formal_config_hash(ROOT), digest.hexdigest())

    def test_write_cell_once_never_overwrites(self) -> None:
        directory = ROOT / "tmp" / f"formal_write_once_{uuid.uuid4().hex}"
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        directory.mkdir(parents=True)
        cell = directory / "cell.json"
        self.assertTrue(self.runner.write_cell_once(cell, {"value": 1}))
        self.assertFalse(self.runner.write_cell_once(cell, {"value": 2}))
        self.assertEqual(json.loads(cell.read_text(encoding="utf-8"))["value"], 1)

    def test_expected_cache_provenance_templates_per_family(self) -> None:
        lineage = self.runner.resolve_formal_lineage(ROOT)
        table = SimpleNamespace(raw_sha256="d" * 64, label_mapping={"negative": 0, "positive": 1})
        split = SimpleNamespace(split_hash="e" * 64)
        for model, cfg_key, code_key, lock_key in (
            ("logistic_regression", "local_config_hash", "local_code_hash", "local_lock_sha256"),
            ("xgboost", "local_config_hash", "local_code_hash", "local_lock_sha256"),
            ("tabpfn", "tabpfn_config_hash", "tabpfn_code_hash", "tabpfn_lock_sha256"),
        ):
            provenance = self.runner.expected_cache_provenance(lineage, table, split, 104729, model)
            self.assertEqual(provenance["protocol_version"], "v1.1")
            self.assertEqual(provenance["model_name"], model)
            self.assertEqual(provenance["config_hash"], lineage[cfg_key])
            self.assertEqual(provenance["code_hash"], lineage[code_key])
            self.assertEqual(provenance["local_cache_lock_sha256"], lineage[lock_key])
            self.assertEqual(provenance["d08_003_cache_lock_sha256"], lineage["tabpfn_lock_sha256"])


class FormalRunRealEvidenceTests(unittest.TestCase):
    """Verify the immutable 1,920-cell formal run after it exists."""

    @classmethod
    def setUpClass(cls) -> None:
        candidates = sorted(RUNS_ROOT.glob("*_stage09-formal_*/run_status.json"))
        if not candidates:
            raise AssertionError("No formal run exists yet; run src/run_formal_experiment_v1_1.py first.")
        cls.status_path = candidates[-1]
        cls.run_root = cls.status_path.parent
        cls.status = json.loads(cls.status_path.read_text(encoding="utf-8"))
        cls.records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((cls.run_root / "cells").glob("*.json"))]

    def test_run_status_passes_with_exactly_1920_cells(self) -> None:
        self.assertEqual(self.status["status"], "PASS")
        self.assertEqual(self.status["verified_cells"], 1920)
        self.assertEqual(self.status["expected_cells"], 1920)
        self.assertEqual(len(self.records), 1920)
        keys = {(row["dataset_id"], row["seed"], row["model"], row["cp_method"], row["m_minority"]) for row in self.records}
        self.assertEqual(len(keys), 1920)
        self.assertEqual({row["protocol_version"] for row in self.records}, {"v1.1"})

    def test_results_long_has_1920_unique_rows(self) -> None:
        import pandas as pd

        frame = pd.read_parquet(self.run_root / "results_long.parquet")
        self.assertEqual(len(frame), 1920)
        self.assertEqual(len(frame.drop_duplicates(subset=["dataset_id", "seed", "model", "cp_method", "m_minority"])), 1920)
        csv = pd.read_csv(self.run_root / "results_long.csv")
        self.assertEqual(len(csv), 1920)
        self.assertEqual(set(frame["dataset_id"]), set(FROZEN_DATASETS))

    def test_full_recomputation_qc_passes(self) -> None:
        qc = json.loads((self.run_root / "qc.json").read_text(encoding="utf-8"))
        self.assertEqual(qc["status"], "PASS")
        self.assertEqual(qc["protocol_version"], "v1.1")
        self.assertEqual(qc["validated_cells"], 1920)
        for check in ("m10_max_score_rank", "m20_19th_order_statistic_rank", "nested_subset", "cp_subset_identity", "q_threshold_sum_and_geometry"):
            self.assertTrue(qc["checks"][check], f"QC check failed: {check}")

    def test_run_manifest_binds_the_frozen_manifest_and_receipt(self) -> None:
        run_manifest = json.loads((self.run_root / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(run_manifest["formal_run_manifest_sha256"], sha256_file(MANIFEST))
        self.assertEqual(run_manifest["d08_004_receipt_sha256"], sha256_file(RECEIPT))
        self.assertEqual(run_manifest["expected_cells"], 1920)
        self.assertEqual(run_manifest["protocol_version"], "v1.1")

    def test_no_v10_mixing_and_no_new_caches(self) -> None:
        cache_root = ROOT / "artifacts" / "caches" / "v1.1"
        manifests = list(cache_root.rglob("manifest.json"))
        self.assertEqual(len(manifests), 240)
        for record in self.records:
            self.assertEqual(record["protocol_version"], "v1.1")
        import pandas as pd

        self.assertEqual(len(pd.read_parquet(V10_PILOT_RUN / "results_long.parquet")), 480)
        self.assertEqual(len(list((V10_PILOT_RUN / "cells").glob("*.json"))), 480)
        self.assertEqual(len(pd.read_parquet(PILOT_RUN / "results_long.parquet")), 480)

    def test_diagnostic_outputs_are_diagnostic_only(self) -> None:
        qc = json.loads((self.run_root / "qc.json").read_text(encoding="utf-8"))
        self.assertIn("figures", qc)
        self.assertGreaterEqual(len(list((self.run_root / "figures").glob("*.png"))), 4)


if __name__ == "__main__":
    unittest.main()

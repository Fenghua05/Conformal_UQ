"""Contract tests for the fixed 480-cell v1.1 pilot (Stage 08 / Task 7).

All tests run locally WITHOUT importing TabPFN or fitting any model.  The
final class verifies the real immutable pilot run after
``src/run_stage07_pilot_v1_1.py`` has executed.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conformal_uq.config import FROZEN_DATASETS, FROZEN_SEEDS

RUNNER = ROOT / "src" / "run_stage07_pilot_v1_1.py"
DECISION = ROOT / "decisions" / "pilot_decision_stage07_v1.1.json"
REGISTRY = ROOT / "artifacts" / "stage02" / "dataset_registry_v1.0.1.json"
DATASET_LOCK = ROOT / "protocols" / "dataset_lock_v1.0.md"
PROTOCOL_V11 = ROOT / "protocols" / "protocol_v1.1.md"
SPLIT_LOCK = ROOT / "configs" / "stage04_splits_v1.1.yaml"
LOCAL_LOCK = ROOT / "configs" / "stage05_lr_xgboost_v1.1.yaml"
TABPFN_LOCK = ROOT / "configs" / "stage05b_tabpfn_v1.1.yaml"
RECEIPT = ROOT / "decisions" / "D08-003_V11_CACHE_AND_PILOT_BUDGET_RECEIPT.json"
INTAKE_ROOT = ROOT / "artifacts" / "stage08_v11_cloud"
PILOT_ROOT = ROOT / "artifacts" / "stage07_v1.1"
V10_PILOT_RUN = ROOT / "artifacts" / "stage07" / "20260830T161214Z_stage07-pilot_32b7e4728b8d"

MODELS = ("logistic_regression", "xgboost", "tabpfn")
CP_METHODS = ("global_split_cp", "class_conditional_cp")
M_MINORITY = (10, 20, 50, 100)


def load_runner():
    spec = importlib.util.spec_from_file_location("stage07_v11_pilot_runner_for_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage07_v11_pilot_runner_for_test"] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class V11PilotDecisionContractTests(unittest.TestCase):
    """The v1.1 pilot decision must be version-specific and outcome-blind."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.decision = json.loads(DECISION.read_text(encoding="utf-8"))
        cls.registry = json.loads(REGISTRY.read_text(encoding="utf-8"))

    def registry_ratio(self, dataset_id: str) -> float:
        record = next(item for item in self.registry["records"] if item["dataset_id"] == dataset_id)
        counts = record["raw_class_counts"]
        return counts[record["minority_original_label"]] / record["n_rows"]

    def test_decision_is_version_specific_with_matching_hashes(self) -> None:
        self.assertEqual(self.decision["protocol_version"], "v1.1")
        self.assertEqual(self.decision["status"], "APPROVED_PRE_OUTCOME_PILOT_DECISION")
        self.assertEqual(self.decision["registry_sha256"], sha256_file(REGISTRY))
        self.assertEqual(self.decision["dataset_lock_sha256"], sha256_file(DATASET_LOCK))
        self.assertEqual(self.decision["protocol_sha256"], sha256_file(PROTOCOL_V11))
        binding = self.decision["v11_lineage_binding"]
        self.assertEqual(binding["split_lock_sha256"], sha256_file(SPLIT_LOCK))
        self.assertEqual(binding["local_cache_lock_sha256"], sha256_file(LOCAL_LOCK))
        self.assertEqual(binding["tabpfn_cache_lock_sha256"], sha256_file(TABPFN_LOCK))
        self.assertEqual(binding["d08_003_receipt_sha256"], sha256_file(RECEIPT))

    def test_pilot_pair_is_mechanically_derived_from_registry_facts(self) -> None:
        inputs = self.decision["outcome_blind_inputs"]
        ratios = {dataset_id: self.registry_ratio(dataset_id) for dataset_id in FROZEN_DATASETS}
        for dataset_id, ratio in ratios.items():
            self.assertAlmostEqual(inputs["registry_minority_ratios"][dataset_id], ratio, places=12)
        moderate = sorted(dataset_id for dataset_id, ratio in ratios.items() if 0.10 <= ratio <= 0.30)
        severe = sorted(dataset_id for dataset_id, ratio in ratios.items() if 0.02 <= ratio < 0.10)
        self.assertEqual(sorted(inputs["moderate_imbalance_ids"]), moderate)
        self.assertEqual(inputs["severe_but_feasible_ids"], severe)
        self.assertEqual(inputs["severe_but_feasible_ids"], [])
        self.assertTrue(inputs["fallback_applied"])
        ranked = sorted(
            (record for record in self.registry["records"] if record.get("proposed_role") == "primary" and record.get("all_seed_feasible") is True),
            key=lambda record: record["frozen_selection_rank"],
        )
        self.assertEqual(inputs["eligible_primary_rank_order"], [record["dataset_id"] for record in ranked])
        self.assertEqual(self.decision["pilot_dataset_ids"], [record["dataset_id"] for record in ranked[:2]])

    def test_decision_prohibits_v10_outcome_inputs(self) -> None:
        prohibited = self.decision["outcome_blind_inputs"]["prohibited_inputs"]
        self.assertTrue(any("v1.0" in item for item in prohibited))
        text = DECISION.read_text(encoding="utf-8")
        for banned in ("coverage_overall", "auroc", "auprc", "q_global", "singleton_rate"):
            self.assertNotIn(banned, text)
        # The decision may reference v1.0 only as prohibited history, never as a selector.
        self.assertNotIn("v1.0 pilot result", text.replace("v1.0 pilot results or v1.0 Stage 08 audit findings", ""))


class V11PilotRunnerContractTests(unittest.TestCase):
    """The v1.1 pilot runner contract before any cell exists."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load_runner()

    def test_runner_module_imports_without_tabpfn(self) -> None:
        self.assertNotIn("tabpfn", sys.modules)

    def test_expected_pilot_grid_is_exactly_480(self) -> None:
        cells = self.runner.expected_pilot_cells_v11(
            ("openml_3_kr_vs_kp", "openml_24_mushroom"), tuple(FROZEN_SEEDS),
        )
        self.assertEqual(len(cells), 480)
        self.assertEqual(len(set(cells)), 480)
        self.assertEqual({cell[2] for cell in cells}, set(MODELS))
        self.assertEqual({cell[3] for cell in cells}, set(CP_METHODS))
        self.assertEqual({cell[4] for cell in cells}, set(M_MINORITY))

    def test_intake_gate_requires_a_pass_240_cache_audit(self) -> None:
        empty_root = ROOT / "tmp" / f"stage07_v11_gate_test_{uuid.uuid4().hex}"
        self.addCleanup(shutil.rmtree, empty_root, ignore_errors=True)
        empty_root.mkdir(parents=True)
        with self.assertRaises(Exception):
            self.runner.require_intake_pass(empty_root)
        audit = self.runner.require_intake_pass(ROOT)
        self.assertEqual(audit["verdict"], "PASS")
        self.assertEqual(audit["valid_units"], 240)
        self.assertEqual(audit["model_counts"], {"logistic_regression": 80, "xgboost": 80, "tabpfn": 80})

    def test_lineage_resolution_binds_to_the_intake_audit_and_current_locks(self) -> None:
        lineage = self.runner.resolve_v11_cache_lineage(ROOT)
        audit = self.runner.require_intake_pass(ROOT)
        audit_lineage = audit["lineage"]
        self.assertEqual(lineage["local_config_hash"], audit_lineage["local_config_hash"])
        self.assertEqual(lineage["local_code_hash"], audit_lineage["local_cache_time_code_hash"])
        self.assertEqual(lineage["tabpfn_config_hash"], audit_lineage["tabpfn_config_hash"])
        self.assertEqual(lineage["tabpfn_code_hash"], audit_lineage["tabpfn_cache_time_code_hash"])
        self.assertEqual(lineage["split_lock_sha256"], sha256_file(SPLIT_LOCK))
        self.assertEqual(lineage["local_lock_sha256"], sha256_file(LOCAL_LOCK))
        self.assertEqual(lineage["tabpfn_lock_sha256"], sha256_file(TABPFN_LOCK))
        self.assertEqual(lineage["d08_003_cache_lock_sha256"], sha256_file(TABPFN_LOCK))

    def test_expected_cache_provenance_templates_match_the_audited_families(self) -> None:
        lineage = self.runner.resolve_v11_cache_lineage(ROOT)
        table = SimpleNamespace(raw_sha256="d" * 64, label_mapping={"negative": 0, "positive": 1})
        split = SimpleNamespace(split_hash="e" * 64)
        local = self.runner.expected_cache_provenance(lineage, table, split, 104729, "logistic_regression")
        tabpfn = self.runner.expected_cache_provenance(lineage, table, split, 104729, "tabpfn")
        for provenance, config_key, code_key, lock_key in (
            (local, "local_config_hash", "local_code_hash", "local_lock_sha256"),
            (tabpfn, "tabpfn_config_hash", "tabpfn_code_hash", "tabpfn_lock_sha256"),
        ):
            self.assertEqual(provenance["protocol_version"], "v1.1")
            self.assertEqual(provenance["config_hash"], lineage[config_key])
            self.assertEqual(provenance["code_hash"], lineage[code_key])
            self.assertEqual(provenance["local_cache_lock_sha256"], lineage[lock_key])
            self.assertEqual(provenance["split_lock_sha256"], lineage["split_lock_sha256"])
            self.assertEqual(provenance["d08_003_cache_lock_sha256"], lineage["tabpfn_lock_sha256"])
            self.assertEqual(provenance["class_labels"], [0, 1])

    def test_write_cell_once_never_overwrites_an_existing_cell(self) -> None:
        directory = ROOT / "tmp" / f"stage07_v11_write_once_{uuid.uuid4().hex}"
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        directory.mkdir(parents=True)
        cell = directory / "cell.json"
        self.assertTrue(self.runner.write_cell_once(cell, {"value": 1}))
        self.assertFalse(self.runner.write_cell_once(cell, {"value": 2}))
        self.assertEqual(json.loads(cell.read_text(encoding="utf-8"))["value"], 1)

    def test_pilot_decision_validation_rejects_the_v10_decision(self) -> None:
        with self.assertRaises(Exception):
            self.runner.load_and_validate_v11_pilot_decision(ROOT, ROOT / "decisions" / "pilot_decision_stage07_v1.0.json")
        decision = self.runner.load_and_validate_v11_pilot_decision(ROOT, DECISION)
        self.assertEqual(decision["protocol_version"], "v1.1")
        self.assertEqual(decision["pilot_dataset_ids"], ["openml_3_kr_vs_kp", "openml_24_mushroom"])


class V11PilotRegressionTests(unittest.TestCase):
    """The two Stage 07 bug fixes must remain in force for the v1.1 pilot."""

    def test_wilson_endpoint_clipping_is_preserved(self) -> None:
        from conformal_uq.metrics import wilson_interval

        low, high = wilson_interval(783, 783)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)
        low, high = wilson_interval(0, 783)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)

    def test_stored_probability_geometry_is_preserved(self) -> None:
        from conformal_uq.conformal import binary_geometry_categories

        categories = binary_geometry_categories(
            [[0.999987006187439, 0.000013013937859795988]],
            0.000012993812561035156,
            0.000012993812561035156,
        )
        self.assertEqual(categories.tolist(), ["singleton"])


class V11PilotRealEvidenceTests(unittest.TestCase):
    """Verify the immutable 480-cell v1.1 pilot run after it exists."""

    @classmethod
    def setUpClass(cls) -> None:
        candidates = sorted(PILOT_ROOT.glob("*/run_status.json"))
        if not candidates:
            raise AssertionError("No v1.1 pilot run exists yet; run src/run_stage07_pilot_v1_1.py first.")
        cls.status_path = candidates[-1]
        cls.run_root = cls.status_path.parent
        cls.status = json.loads(cls.status_path.read_text(encoding="utf-8"))
        cls.records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((cls.run_root / "cells").glob("*.json"))]

    def test_run_status_passes_with_exactly_480_cells(self) -> None:
        self.assertEqual(self.status["status"], "PASS")
        self.assertEqual(self.status["verified_cells"], 480)
        self.assertEqual(self.status["expected_cells"], 480)
        self.assertEqual(len(self.records), 480)
        keys = {(row["dataset_id"], row["seed"], row["model"], row["cp_method"], row["m_minority"]) for row in self.records}
        self.assertEqual(len(keys), 480)
        self.assertEqual({row["protocol_version"] for row in self.records}, {"v1.1"})

    def test_results_long_matches_cells_with_unique_keys(self) -> None:
        import pandas as pd

        frame = pd.read_parquet(self.run_root / "results_long.parquet")
        self.assertEqual(len(frame), 480)
        self.assertEqual(len(frame.drop_duplicates(subset=["dataset_id", "seed", "model", "cp_method", "m_minority"])), 480)
        csv = pd.read_csv(self.run_root / "results_long.csv")
        self.assertEqual(len(csv), 480)

    def test_pilot_qc_passes_with_v11_lineage(self) -> None:
        qc = json.loads((self.run_root / "pilot_qc.json").read_text(encoding="utf-8"))
        self.assertEqual(qc["status"], "PASS")
        self.assertTrue(qc["checks"]["cp_subset_identity"])
        self.assertTrue(qc["checks"]["m10_max_score_rank"])
        self.assertTrue(qc["checks"]["m20_19th_order_statistic_rank"])
        self.assertTrue(qc["checks"]["nested_subset"])
        self.assertTrue(qc["checks"]["q_threshold_sum_and_geometry"])
        self.assertEqual(qc["protocol_version"], "v1.1")

    def test_global_and_class_conditional_share_identical_subsets(self) -> None:
        groups: dict[tuple[str, int, str, int], set[str]] = {}
        for row in self.records:
            groups.setdefault((row["dataset_id"], row["seed"], row["model"], row["m_minority"]), set()).add(row["subset_hash"])
        self.assertEqual(len(groups), 2 * 10 * 3 * 4)
        self.assertTrue(all(len(hashes) == 1 for hashes in groups.values()))

    def test_exact_ranks_decomposition_and_wilson_bounds(self) -> None:
        for row in self.records:
            if row["cp_method"] == "class_conditional_cp":
                if row["m_minority"] == 10:
                    self.assertEqual(row["rank_minority"], 10)
                if row["m_minority"] == 20:
                    self.assertEqual(row["rank_minority"], 19)
            decomposition = row["singleton_rate"] + row["empty_rate"] + row["doubleton_rate"]
            self.assertAlmostEqual(decomposition, 1.0, places=12)
            for scope, total in (("overall", row["n_test"]), ("minority", row["n_test_minority"]), ("majority", row["n_test_majority"])):
                self.assertGreaterEqual(row[f"coverage_{scope}_wilson_low"], 0.0)
                self.assertLessEqual(row[f"coverage_{scope}_wilson_high"], 1.0)
                self.assertLessEqual(row[f"coverage_{scope}_wilson_low"], row[f"coverage_{scope}_wilson_high"])

    def test_auroc_auprc_invariance_for_representative_cells(self) -> None:
        from conformal_uq.data import load_locked_dataset
        from conformal_uq.metrics import binary_predictive_metrics
        from conformal_uq.prediction_cache import read_valid_cache
        from conformal_uq.split import make_stratified_split

        runner = load_runner()
        lineage = runner.resolve_v11_cache_lineage(ROOT)
        registry_path = ROOT / "artifacts" / "stage02" / "dataset_registry_v1.0.1.json"
        sampled = [row for row in self.records if row["seed"] == 104729 and row["m_minority"] == 10]
        self.assertGreaterEqual(len(sampled), 12)
        for row in sampled:
            table = load_locked_dataset(ROOT, row["dataset_id"], registry_path=registry_path)
            split = make_stratified_split(table, row["seed"], protocol_version="v1.1")
            provenance = runner.expected_cache_provenance(lineage, table, split, row["seed"], row["model"])
            cache_dir = runner.cache_directory(ROOT, lineage, row["dataset_id"], row["seed"], row["model"])
            cached = read_valid_cache(
                cache_dir, provenance,
                {"calibration_pool": split.ids.calibration_pool, "test": split.ids.test},
                {
                    "calibration_pool": table.subset_labels(split.ids.calibration_pool).to_numpy(dtype="int8", copy=True),
                    "test": table.subset_labels(split.ids.test).to_numpy(dtype="int8", copy=True),
                },
            )
            recomputed = binary_predictive_metrics(cached["labels"]["test"], cached["probabilities"]["test"])
            self.assertAlmostEqual(row["auroc"], recomputed["auroc"], places=12)
            self.assertAlmostEqual(row["auprc"], recomputed["auprc"], places=12)
            self.assertEqual(row["prediction_cache_hash"], cached["manifest"]["cache_sha256"])

    def test_no_formal_outputs_and_v10_pilot_untouched(self) -> None:
        forbidden = [path for path in PILOT_ROOT.rglob("*") if path.is_file() and path.name in {"formal_run_manifest.json", "results_formal.parquet"}]
        self.assertEqual(forbidden, [])
        self.assertEqual(len(list((V10_PILOT_RUN / "cells").glob("*.json"))), 480)
        import pandas as pd

        self.assertEqual(len(pd.read_parquet(V10_PILOT_RUN / "results_long.parquet")), 480)
        v10_versions = {row["protocol_version"] for row in self.records}
        self.assertEqual(v10_versions, {"v1.1"})


if __name__ == "__main__":
    unittest.main()

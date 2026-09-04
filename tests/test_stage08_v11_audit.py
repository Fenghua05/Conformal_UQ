"""Contract tests for the fresh independent Stage 08 v1.1 pilot audit.

All tests run locally WITHOUT importing TabPFN, fitting a model, or creating
any cache/CP/pilot/formal output.  The final class verifies the real audit
evidence after ``src/audit_stage08_v11_pilot.py`` has audited the v1.1 pilot.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

AUDITOR = ROOT / "src" / "audit_stage08_v11_pilot.py"
AUDIT_ROOT = ROOT / "artifacts" / "stage08_v11"
PILOT_ROOT = ROOT / "artifacts" / "stage07_v1.1"
V10_PILOT_RUN = ROOT / "artifacts" / "stage07" / "20260830T161214Z_stage07-pilot_32b7e4728b8d"
AUDIT_GLOB = "*_pilot_independent_audit/independent_audit.json"


def load_auditor():
    spec = importlib.util.spec_from_file_location("stage08_v11_pilot_auditor_for_test", AUDITOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage08_v11_pilot_auditor_for_test"] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class V11AuditMethodTests(unittest.TestCase):
    """The audit method must be independent of the pilot implementation."""

    @classmethod
    def setUpClass(cls) -> None:
        before = set(sys.modules)
        cls.auditor = load_auditor()
        cls.newly_imported = sorted(set(sys.modules) - before)

    def test_module_imports_no_tabpfn_and_no_project_helpers(self) -> None:
        for name in self.newly_imported:
            self.assertFalse(name == "tabpfn" or name.startswith("tabpfn."), f"audit must not import {name}")
            self.assertFalse(name.startswith("conformal_uq"), f"audit must not import {name}")
            self.assertFalse("run_stage07" in name or "stage07_qc" in name, f"audit must not import {name}")

    def test_v11_seed_derivation_matches_the_identity_module(self) -> None:
        from conformal_uq.identity import derive_seed

        for dataset in ("openml_3_kr_vs_kp", "openml_24_mushroom"):
            for base_seed in (104729, 552721):
                for purpose in ("calibration_majority_subset", "calibration_minority_nested_subset"):
                    self.assertEqual(
                        self.auditor.seed(dataset, base_seed, purpose),
                        derive_seed("v1.1", dataset, base_seed, purpose)[1],
                    )

    def test_order_statistic_and_wilson_match_shared_modules(self) -> None:
        import numpy as np
        from conformal_uq.conformal import exact_conformal_quantile
        from conformal_uq.metrics import wilson_interval

        rng = np.random.default_rng(0)
        scores = rng.uniform(size=211)
        expected = exact_conformal_quantile(scores)
        rank, threshold = self.auditor.q(scores)
        self.assertEqual(rank, expected.rank)
        self.assertEqual(threshold, expected.threshold)
        for k, n in ((0, 783), (783, 783), (7, 29), (120, 640)):
            self.assertEqual(self.auditor.wilson(k, n), wilson_interval(k, n))

    def test_representative_spec_spans_the_full_pilot_grid(self) -> None:
        spec = self.auditor.representative_spec(("dataset_a", "dataset_b"))
        self.assertEqual(len(spec), 36)
        self.assertEqual(len(set(spec)), 36)
        self.assertEqual({item[0] for item in spec}, {"dataset_a", "dataset_b"})
        self.assertEqual({item[2] for item in spec}, {"logistic_regression", "xgboost", "tabpfn"})
        self.assertEqual({item[3] for item in spec}, {"global_split_cp", "class_conditional_cp"})
        self.assertEqual({item[4] for item in spec}, {10, 20, 50, 100})
        self.assertTrue(all(item[1] in {104729, 130363, 262147, 374209, 481517, 552721} for item in spec))


class V11AuditRealEvidenceTests(unittest.TestCase):
    """Verify the real independent audit after it has run."""

    @classmethod
    def setUpClass(cls) -> None:
        candidates = sorted(AUDIT_ROOT.glob(AUDIT_GLOB))
        if not candidates:
            raise AssertionError(
                "No Stage 08 v1.1 pilot independent audit exists yet under "
                f"{AUDIT_ROOT}; run src/audit_stage08_v11_pilot.py first."
            )
        cls.audit_path = candidates[-1]
        cls.audit = json.loads(cls.audit_path.read_text(encoding="utf-8"))

    def test_verdict_pass_with_complete_cell_integrity(self) -> None:
        integrity = self.audit["cell_integrity"]
        self.assertEqual(self.audit["verdict"], "PASS")
        self.assertEqual(integrity["cell_json_count"], 480)
        self.assertEqual(integrity["expected_count"], 480)
        self.assertEqual(integrity["unique_keys"], 480)
        self.assertEqual(integrity["duplicate_keys"], [])
        self.assertEqual(integrity["missing_keys"], [])
        self.assertEqual(integrity["extra_keys"], [])
        self.assertTrue(integrity["parquet_key_set_matches_cells"])
        self.assertTrue(integrity["csv_key_set_matches_cells"])

    def test_cache_lineage_covers_60_units_in_two_authorized_trees(self) -> None:
        lineage = self.audit["cache_lineage"]
        self.assertEqual(lineage["base_cache_units"], 60)
        self.assertEqual(lineage["errors"], [])
        self.assertEqual(sorted(entry["base_cache_units"] for entry in lineage["lineages"]), [20, 20, 20])
        trees = {(entry["config_hash"], entry["code_hash"]) for entry in lineage["lineages"]}
        self.assertEqual(len(trees), 2)
        self.assertEqual({entry["model"] for entry in lineage["lineages"]}, {"logistic_regression", "xgboost", "tabpfn"})
        self.assertEqual(len({entry["environment_hash"] for entry in lineage["lineages"]}), 1)

    def test_all_cell_invariants_and_metric_invariance(self) -> None:
        invariants = self.audit["all_cell_invariants"]
        self.assertEqual(invariants["errors"], [])
        self.assertEqual(invariants["predictive_metric_invariance_errors"], [])
        self.assertEqual(invariants["cp_subset_identity_errors"], [])
        self.assertEqual(invariants["set_decomposition_errors"], [])

    def test_representative_recomputation_has_zero_mismatches(self) -> None:
        recomputation = self.audit["representative_recomputation"]
        self.assertEqual(recomputation["n_cells"], 36)
        self.assertEqual(recomputation["errors"], [])
        self.assertTrue(all(item["geometry_matches_sets"] for item in recomputation["evidence"]))

    def test_v11_specific_lineage_checks(self) -> None:
        checks = self.audit["v11_lineage_checks"]
        self.assertEqual(checks["protocol_versions"], ["v1.1"])
        self.assertTrue(checks["split_hash_matches_locked_v11_manifests"])
        self.assertTrue(checks["pilot_decision_hash_bindings_match"])
        self.assertTrue(checks["cell_lineages_match_intake_audit"])
        self.assertTrue(checks["v10_pilot_untouched"])
        self.assertEqual(checks["v10_pilot_cell_count"], 480)
        self.assertEqual(checks["v10_pilot_parquet_rows"], 480)
        inventory = checks["full_run_cache_inventory"]
        self.assertEqual(inventory["complete_units"], 240)
        self.assertEqual(inventory["per_model"], {"logistic_regression": 80, "xgboost": 80, "tabpfn": 80})

    def test_cloud_budget_evidence_is_within_the_d08_003_limits(self) -> None:
        evidence = self.audit["cloud_budget_evidence"]
        self.assertTrue(evidence["found"])
        self.assertEqual(evidence["status"], "PASS")
        self.assertEqual(evidence["completed_units"], 80)
        self.assertTrue(evidence["within_budget"])
        self.assertLessEqual(evidence["elapsed_seconds"], 12 * 3600)
        self.assertLessEqual(evidence["produced_bytes"], 50 * 1024 ** 3)
        self.assertEqual(evidence["runtime_evidence"]["gpu_name"], "NVIDIA GeForce RTX 4090")
        self.assertEqual(evidence["runtime_evidence"]["tabpfn_version"], "8.5.0")

    def test_wilson_diagnostic_is_reproduced_as_diagnostic_only(self) -> None:
        sanity = self.audit["global_coverage_sanity"]
        self.assertEqual(sanity["nominal"], 0.9)
        self.assertTrue(sanity["matches_pilot_qc"])
        self.assertIn("diagnostic", sanity["interpretation"].lower())
        groups = {(item["model"], item["m_minority"]): item["flagged_cells"] for item in sanity["groups"]}
        self.assertEqual(len(groups), 12)

    def test_stale_artifact_map_keeps_v10_historical_only(self) -> None:
        stale = self.audit["stale_artifact_map"]
        self.assertEqual(stale["v1.0_splits_caches_pilot_audits"], "historical_only_not_reusable_as_v1.1_evidence")
        self.assertEqual(stale["v1.1_splits"], "current_and_locked")
        self.assertEqual(stale["v1.1_local_lr_xgboost_caches"], "current_and_audited")
        self.assertEqual(stale["v1.1_tabpfn_caches"], "current_and_audited")
        self.assertEqual(stale["v1.1_pilot_run"], "current_and_independently_audited")
        self.assertEqual(stale["stale_v1.1_artifacts"], [])

    def test_checklist_has_ten_items_with_explicit_verdicts(self) -> None:
        checklist = self.audit["checklist"]
        self.assertEqual(len(checklist), 10)
        self.assertEqual([item["item"] for item in checklist], list(range(1, 11)))
        for item in checklist:
            self.assertIn(item["verdict"], {"PASS", "PASS WITH DIAGNOSTIC FLAG", "FAIL", "NOT_RUN"})
            self.assertTrue(item["check"])
            self.assertTrue(item["evidence"])

    def test_gate_recommendation_is_conditional_go_and_respects_authorization(self) -> None:
        gate = self.audit["gate_recommendation"]
        self.assertEqual(gate["recommendation"], "CONDITIONAL-GO")
        self.assertTrue(gate["technical_readiness"]["pilot_independent_recomputation_pass"])
        self.assertTrue(gate["technical_readiness"]["full_run_base_caches_available"])
        self.assertTrue(gate["technical_readiness"]["tabpfn_full_context_route_proven"])
        self.assertFalse(gate["formal_run_authorized"])
        self.assertFalse(gate["formal_run_manifest_frozen"])
        self.assertTrue(gate["explicit_user_go_required"])
        self.assertTrue(gate["remaining_gates"])


if __name__ == "__main__":
    unittest.main()

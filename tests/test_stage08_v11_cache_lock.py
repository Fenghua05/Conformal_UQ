"""Preparation-only checks for the separately versioned Stage 08 v1.1 locks.

These tests intentionally exercise configuration parsing and path derivation only.
They must not access real data, fit a model, write a split/cache, import TabPFN,
or initiate cloud work.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPLIT_LOCK = ROOT / "configs" / "stage04_splits_v1.1.yaml"
REPAIR_SPLIT_LOCK = ROOT / "configs" / "stage04_splits_v1.1.1.yaml"
LOCAL_LOCK = ROOT / "configs" / "stage05_lr_xgboost_v1.1.yaml"
TABPFN_LOCK = ROOT / "configs" / "stage05b_tabpfn_v1.1.yaml"
DATASETS = [
    "openml_3_kr_vs_kp", "openml_24_mushroom", "openml_1486_nomao",
    "openml_1489_phoneme", "openml_1590_adult", "openml_4534_phishingwebsite",
    "openml_23512_higgs", "openml_23517_numerai28_6",
]
SEEDS = [104729, 130363, 155921, 196613, 262147, 318281, 374209, 419893, 481517, 552721]
CHECKPOINT = "/root/autodl-fs/tabpfn-model-cache/tabpfn-v3-classifier-v3_default.ckpt"
CHECKPOINT_SHA256 = "d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988"


def _load_runner(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "src" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class Stage08V11CacheLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stage04 = _load_runner("run_stage04_prepare.py", "stage08_v11_stage04_runner")
        cls.stage05 = _load_runner("run_stage05_predict.py", "stage08_v11_stage05_runner")
        cls.split_audit = _load_runner("audit_stage08_v11_split_regeneration.py", "stage08_v11_split_audit")
        from conformal_uq import stage08_authorization
        cls.authorization = stage08_authorization

    def load_yaml(self, path: Path) -> dict:
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_locks_pin_the_v11_registry_seeds_and_isolated_roots(self) -> None:
        split_lock = self.load_yaml(SPLIT_LOCK)
        local_lock = self.load_yaml(LOCAL_LOCK)
        tabpfn_lock = self.load_yaml(TABPFN_LOCK)
        for lock in (split_lock, local_lock, tabpfn_lock):
            self.assertEqual(lock["protocol_version"], "v1.1")
            self.assertEqual(lock["registry"]["locked_primary_ids"], DATASETS)
            self.assertEqual(lock["seeds"], SEEDS)
            self.assertFalse(lock["execution_gate"]["formal_run_manifest_authorized"])
            self.assertFalse(lock["execution_gate"]["full_experiment_authorized"])
            self.assertFalse(lock["execution_gate"]["split_regeneration_authorized"])
            self.assertFalse(lock["execution_gate"]["local_cache_generation_authorized"])
            self.assertFalse(lock["execution_gate"]["tabpfn_cache_generation_authorized"])
            self.assertFalse(lock["execution_gate"]["pilot_authorized"])
            self.assertFalse(lock["output_contract"]["conformal_prediction_allowed"])
            self.assertFalse(lock["output_contract"]["pilot_allowed"])
            self.assertFalse(lock["output_contract"]["formal_outputs_allowed"])
        self.assertEqual(split_lock["paths"]["split_root"], "artifacts/splits/v1.1")
        self.assertEqual(split_lock["paths"]["stage04_evidence_root"], "artifacts/stage08_v11/stage04_split_preparation")
        self.assertEqual(local_lock["paths"]["split_root"], "artifacts/splits/v1.1")
        self.assertEqual(local_lock["paths"]["cache_root"], "artifacts/caches/v1.1")
        self.assertEqual(tabpfn_lock["paths"]["split_root"], "artifacts/splits/v1.1")
        self.assertEqual(tabpfn_lock["paths"]["cache_root"], "artifacts/caches/v1.1")

    def test_tabpfn_lock_pins_full_context_runtime_and_guards(self) -> None:
        lock = self.load_yaml(TABPFN_LOCK)
        runtime = lock["runtime"]
        self.assertEqual(runtime["provider"], "AutoDL")
        self.assertEqual(runtime["os"], "Ubuntu 22.04")
        self.assertEqual(runtime["gpu_name"], "NVIDIA GeForce RTX 4090")
        self.assertEqual(runtime["gpu_memory_gb"], 24)
        self.assertEqual(runtime["tabpfn_version"], "8.5.0")
        self.assertEqual(runtime["checkpoint_path"], CHECKPOINT)
        self.assertEqual(runtime["checkpoint_sha256"], CHECKPOINT_SHA256)
        self.assertEqual(runtime["train_partition_contract"], "full_fixed_train_partition")
        self.assertTrue(runtime["no_truncation_or_subsampling"])
        self.assertFalse(runtime["ignore_pretraining_limits"])
        self.assertEqual(lock["safety_limits"], {"max_train_rows": 100000, "max_transformed_features": 2000})

    def test_explicit_lock_loading_only_validates_and_derives_v11_paths(self) -> None:
        split_lock = self.stage04.load_split_lock(ROOT, SPLIT_LOCK)
        split_path = self.stage04.split_manifest_path(ROOT, split_lock, DATASETS[0], SEEDS[0])
        self.assertEqual(split_path, ROOT / "artifacts/splits/v1.1" / DATASETS[0] / f"seed-{SEEDS[0]}.json")
        split_snapshot = (split_path.exists(), split_path.stat().st_mtime_ns if split_path.exists() else None)
        outputs = self.stage04.stage04_v11_output_paths(ROOT, split_lock)
        self.assertEqual(outputs["stage_root"], ROOT / "artifacts/stage08_v11/stage04_split_preparation")
        self.assertEqual(outputs["summary_path"], ROOT / "artifacts/stage08_v11/stage04_split_preparation/stage04_summary_v1.1.json")

        base, local_lock, config_hash = self.stage05.load_stage05_lock(ROOT, LOCAL_LOCK)
        self.assertEqual(base["protocol"]["protocol_version_for_seed_derivation"], "v1.1")
        self.assertEqual(local_lock["protocol_version"], "v1.1")
        self.assertEqual(len(config_hash), 64)
        self.assertEqual(
            self.stage05.locked_split_manifest_path(ROOT, local_lock, DATASETS[0], SEEDS[0]),
            ROOT / "artifacts/splits/v1.1" / DATASETS[0] / f"seed-{SEEDS[0]}.json",
        )
        self.assertEqual(
            self.stage05.locked_cache_root(ROOT, local_lock),
            ROOT / "artifacts/caches/v1.1",
        )
        self.assertEqual(
            (split_path.exists(), split_path.stat().st_mtime_ns if split_path.exists() else None),
            split_snapshot,
            "Lock parsing and path derivation must not mutate an already-authorized split artifact.",
        )

    def test_v11_locks_do_not_reference_v10_output_roots(self) -> None:
        for path in (SPLIT_LOCK, LOCAL_LOCK, TABPFN_LOCK):
            payload = self.load_yaml(path)
            rendered = yaml.safe_dump(payload, sort_keys=True)
            self.assertNotIn("artifacts/splits/v1.0", rendered)
            self.assertNotIn("artifacts/caches/v1.0", rendered)

    def test_valid_receipt_builds_v11_execution_plans_without_data_or_output_work(self) -> None:
        with patch.object(self.stage04, "_write_immutable_json") as stage04_writer, patch.object(
            self.stage05, "write_prediction_cache"
        ) as stage05_writer, patch.object(self.stage05, "fit_predict_locked_pipeline") as model_fit:
            stage04_plan = self.stage04.build_stage04_v11_execution_plan(ROOT, SPLIT_LOCK)
            stage05_plan = self.stage05.build_stage05_v11_execution_plan(ROOT, LOCAL_LOCK)
        self.assertEqual(stage04_plan["unit_count"], 80)
        self.assertEqual(stage05_plan["unit_count"], 160)
        self.assertEqual(stage04_plan["split_root"], ROOT / "artifacts/splits/v1.1")
        self.assertEqual(stage05_plan["cache_root"], ROOT / "artifacts/caches/v1.1")
        self.assertFalse(stage04_writer.called)
        self.assertFalse(stage05_writer.called)
        self.assertFalse(model_fit.called)

    def test_receipt_binding_mutation_is_rejected_by_production_validator(self) -> None:
        receipt = json.loads((ROOT / "decisions" / "D08-003_V11_CACHE_AND_PILOT_BUDGET_RECEIPT.json").read_text(encoding="utf-8"))
        final_lock = self.load_yaml(TABPFN_LOCK)
        mutated = deepcopy(receipt)
        mutated["cache_lock_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash binding"):
            self.authorization.validate_d08_003_receipt(mutated, TABPFN_LOCK, final_lock)

    def test_production_validator_rejects_final_lock_split_linkage_registry_seed_or_derivation_mutation(self) -> None:
        receipt = json.loads((ROOT / "decisions" / "D08-003_V11_CACHE_AND_PILOT_BUDGET_RECEIPT.json").read_text(encoding="utf-8"))
        original = self.load_yaml(TABPFN_LOCK)
        mutations = (
            ("split_lock_path", "configs/stage04_splits_v1.0.yaml"),
            ("registry", {"locked_primary_ids": DATASETS[:-1]}),
            ("seeds", SEEDS[:-1]),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                mutated = deepcopy(original)
                mutated[field] = value
                with self.assertRaises(ValueError):
                    self.authorization.validate_d08_003_receipt(receipt, TABPFN_LOCK, mutated)
        mutated_split = self.load_yaml(SPLIT_LOCK)
        mutated_split["protocol"]["protocol_version_for_seed_derivation"] = "v1.0"
        with patch.object(self.authorization.yaml, "safe_load", return_value=mutated_split):
            with self.assertRaisesRegex(ValueError, "seed derivation"):
                self.authorization.validate_d08_003_receipt(receipt, TABPFN_LOCK, original)

    def test_invalid_receipt_rejects_before_stage04_data_or_output_work(self) -> None:
        with patch.object(self.stage04, "load_d08_003_authorization", side_effect=ValueError("bad receipt binding")), patch.object(
            self.stage04, "load_locked_dataset"
        ) as load_data, patch.object(self.stage04, "_write_immutable_json") as writer, patch.object(
            sys, "argv", ["run_stage04_prepare.py", "--root", str(ROOT), "--lock", str(SPLIT_LOCK)]
        ):
            with self.assertRaisesRegex(ValueError, "bad receipt binding"):
                self.stage04.main()
        self.assertFalse(load_data.called)
        self.assertFalse(writer.called)

    def test_invalid_receipt_rejects_before_stage05_data_model_or_cache_work(self) -> None:
        with patch.object(self.stage05, "load_d08_003_authorization", side_effect=ValueError("bad receipt binding")), patch.object(
            self.stage05, "load_locked_dataset"
        ) as load_data, patch.object(self.stage05, "fit_predict_locked_pipeline") as model_fit, patch.object(
            self.stage05, "write_prediction_cache"
        ) as cache_writer, patch.object(sys, "argv", [
            "run_stage05_predict.py", "--root", str(ROOT), "--mode", "formal", "--lock", str(LOCAL_LOCK)
        ]):
            with self.assertRaisesRegex(ValueError, "bad receipt binding"):
                self.stage05.main()
        self.assertFalse(load_data.called)
        self.assertFalse(model_fit.called)
        self.assertFalse(cache_writer.called)

    def test_bad_stage04_v11_lock_rejects_in_main_before_data_or_output_work(self) -> None:
        original = self.load_yaml(SPLIT_LOCK)
        mutations = (
            ("protocol_version", "v1.0"),
            ("paths.split_root", "artifacts/splits/v1.0"),
            ("seeds", SEEDS[:-1]),
            ("execution_gate.split_regeneration_authorized", True),
            ("execution_gate.local_cache_generation_authorized", True),
            ("execution_gate.tabpfn_cache_generation_authorized", True),
            ("execution_gate.pilot_authorized", True),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                mutated = deepcopy(original)
                if field == "paths.split_root":
                    mutated["paths"]["split_root"] = value
                elif field.startswith("execution_gate."):
                    mutated["execution_gate"][field.rsplit(".", 1)[1]] = value
                else:
                    mutated[field] = value
                with patch.object(self.stage04, "_read_split_lock_mapping", return_value=mutated), patch.object(
                    self.stage04, "load_locked_dataset"
                ) as load_data, patch.object(self.stage04, "_write_immutable_json") as writer, patch.object(
                    sys, "argv", ["run_stage04_prepare.py", "--root", str(ROOT), "--lock", str(SPLIT_LOCK)]
                ):
                    with self.assertRaises(ValueError):
                        self.stage04.main()
                self.assertFalse(load_data.called)
                self.assertFalse(writer.called)

    def test_bad_stage05_v11_lock_rejects_in_main_before_data_model_cache_or_run_dir_work(self) -> None:
        original = self.load_yaml(LOCAL_LOCK)
        split_lock = self.load_yaml(SPLIT_LOCK)
        mutations = (
            ("protocol_version", "v1.0"),
            ("paths.split_root", "artifacts/splits/v1.0"),
            ("seeds", SEEDS[:-1]),
            ("execution_gate.split_regeneration_authorized", True),
            ("execution_gate.local_cache_generation_authorized", True),
            ("execution_gate.tabpfn_cache_generation_authorized", True),
            ("execution_gate.pilot_authorized", True),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                mutated = deepcopy(original)
                if field == "paths.split_root":
                    mutated["paths"]["split_root"] = value
                elif field.startswith("execution_gate."):
                    mutated["execution_gate"][field.rsplit(".", 1)[1]] = value
                else:
                    mutated[field] = value
                with patch.object(self.stage05, "_read_stage05_lock_mapping", side_effect=[mutated, split_lock]), patch.object(
                    self.stage05, "load_locked_dataset"
                ) as load_data, patch.object(self.stage05, "fit_predict_locked_pipeline") as model_fit, patch.object(
                    self.stage05, "write_prediction_cache"
                ) as cache_writer, patch.object(self.stage05, "create_immutable_run_dir") as run_dir, patch.object(
                    sys, "argv", ["run_stage05_predict.py", "--root", str(ROOT), "--mode", "formal", "--lock", str(LOCAL_LOCK)]
                ):
                    with self.assertRaises(ValueError):
                        self.stage05.main()
                self.assertFalse(load_data.called)
                self.assertFalse(model_fit.called)
                self.assertFalse(cache_writer.called)
                self.assertFalse(run_dir.called)

    def test_builders_defensively_reject_mutated_non_authorizing_gate_flags(self) -> None:
        split_lock = self.stage04.load_split_lock(ROOT, SPLIT_LOCK)
        base, local_lock, local_hash = self.stage05.load_stage05_lock(ROOT, LOCAL_LOCK)
        for key in (
            "split_regeneration_authorized",
            "local_cache_generation_authorized",
            "tabpfn_cache_generation_authorized",
            "pilot_authorized",
        ):
            with self.subTest(stage="stage04", gate=key):
                mutated = deepcopy(split_lock)
                mutated["execution_gate"][key] = True
                with patch.object(self.stage04, "load_split_lock", return_value=mutated):
                    with self.assertRaisesRegex(ValueError, key):
                        self.stage04.build_stage04_v11_execution_plan(ROOT, SPLIT_LOCK)
            with self.subTest(stage="stage05", gate=key):
                mutated = deepcopy(local_lock)
                mutated["execution_gate"][key] = True
                with patch.object(self.stage05, "load_stage05_lock", return_value=(base, mutated, local_hash)):
                    with self.assertRaisesRegex(ValueError, key):
                        self.stage05.build_stage05_v11_execution_plan(ROOT, LOCAL_LOCK)

    def test_v11_split_manifests_are_complete_versioned_and_independently_recomputable(self) -> None:
        """Require the authorized 8 x 10 v1.1 split lineage without generating it.

        This is intentionally a read-only contract test: it must fail before the
        receipt-gated Stage 04 command materializes the new v1.1 split root.
        """
        from conformal_uq.identity import derive_seed, sha256_text

        split_root = ROOT / "artifacts" / "splits" / "v1.1"
        expected_keys = {(dataset_id, base_seed) for dataset_id in DATASETS for base_seed in SEEDS}
        manifests = list(split_root.glob("*/seed-*.json")) if split_root.exists() else []
        observed_keys = set()
        for path in manifests:
            dataset_id = path.parent.name
            try:
                base_seed = int(path.stem.removeprefix("seed-"))
            except ValueError as exc:
                self.fail(f"Invalid v1.1 split-manifest filename: {path}: {exc}")
            observed_keys.add((dataset_id, base_seed))
        self.assertEqual(observed_keys, expected_keys)
        self.assertEqual(len(manifests), 80)

        registry = json.loads((ROOT / "artifacts" / "stage02" / "dataset_registry_v1.0.1.json").read_text(encoding="utf-8"))
        raw_hashes = {record["dataset_id"]: record["source"]["raw_sha256"] for record in registry["records"]}
        expected_fractions = {"train": 0.6, "calibration_pool": 0.2, "test": 0.2}
        for dataset_id, base_seed in sorted(expected_keys):
            with self.subTest(dataset_id=dataset_id, base_seed=base_seed):
                path = split_root / dataset_id / f"seed-{base_seed}.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(payload["artifact_type"], "stage04_stratified_split")
                self.assertEqual(payload["dataset_id"], dataset_id)
                self.assertEqual(payload["base_seed"], base_seed)
                self.assertEqual(payload["raw_sha256"], raw_hashes[dataset_id])
                self.assertEqual(payload["fractions"], expected_fractions)

                ids = payload["split_ids"]
                train, calibration, test = set(ids["train"]), set(ids["calibration_pool"]), set(ids["test"])
                self.assertTrue(train and calibration and test)
                self.assertFalse(train & calibration)
                self.assertFalse(train & test)
                self.assertFalse(calibration & test)

                hash_input = {
                    "dataset_id": dataset_id,
                    "base_seed": base_seed,
                    "fractions": expected_fractions,
                    "split_ids": {
                        "train": ids["train"],
                        "calibration_pool": ids["calibration_pool"],
                        "test": ids["test"],
                    },
                }
                expected_split_hash = sha256_text(json.dumps(hash_input, sort_keys=True, separators=(",", ":")))
                self.assertEqual(payload["split_hash"], expected_split_hash)

                for purpose in ("stratified_test_split", "stratified_calibration_split"):
                    canonical, derived_seed = derive_seed("v1.1", dataset_id, base_seed, purpose)
                    self.assertEqual(payload["seed_provenance"][purpose], {
                        "canonical_input": canonical,
                        "derived_seed": derived_seed,
                    })
                    v10_canonical, v10_seed = derive_seed("v1.0", dataset_id, base_seed, purpose)
                    self.assertNotEqual((canonical, derived_seed), (v10_canonical, v10_seed))

                v10_path = ROOT / "artifacts" / "splits" / "v1.0" / dataset_id / f"seed-{base_seed}.json"
                self.assertTrue(v10_path.exists(), f"Missing immutable v1.0 comparator: {v10_path}")
                v10_payload = json.loads(v10_path.read_text(encoding="utf-8"))
                self.assertNotEqual(
                    payload["split_hash"], v10_payload["split_hash"],
                    "v1.1 seed derivation differs from v1.0, so this controlled split hash must not reuse v1.0.",
                )

    def test_stage04_immutable_json_writer_hashes_exact_lf_bytes(self) -> None:
        payload = {"artifact": "stage04-writer-regression", "nested": {"count": 1}}
        path = ROOT / "tmp" / "stage08_v11_writer_hash_regression" / "stage04_lf_bytes.json"
        observed_hash = self.stage04._write_immutable_json(path, payload, byte_stable_lf=True)
        raw = path.read_bytes()
        self.assertEqual(observed_hash, hashlib.sha256(raw).hexdigest())
        self.assertNotIn(b"\r\n", raw)
        self.assertTrue(raw.endswith(b"\n"))

    def test_stage04_default_immutable_json_writer_retains_historical_v10_text_bytes(self) -> None:
        payload = {"artifact": "stage04-v10-historical-writer-regression", "nested": {"count": 1}}
        path = ROOT / "tmp" / "stage08_v11_writer_hash_regression" / "stage04_v10_historical_text_bytes.json"
        observed_hash = self.stage04._write_immutable_json(path, payload)
        raw = path.read_bytes()
        expected_canonical_hash = hashlib.sha256(self.stage04._canonical(payload).encode("utf-8")).hexdigest()
        self.assertEqual(observed_hash, expected_canonical_hash)
        self.assertIn(b"\r\n", raw)
        self.assertNotEqual(observed_hash, hashlib.sha256(raw).hexdigest())

    def test_independent_split_audit_writer_hashes_exact_lf_bytes(self) -> None:
        payload = {"artifact": "split-audit-writer-regression", "nested": {"count": 1}}
        path = ROOT / "tmp" / "stage08_v11_writer_hash_regression" / "audit_lf_bytes.json"
        observed_hash = self.split_audit._immutable_write(path, payload)
        raw = path.read_bytes()
        self.assertEqual(observed_hash, hashlib.sha256(raw).hexdigest())
        self.assertNotIn(b"\r\n", raw)
        self.assertTrue(raw.endswith(b"\n"))

    def test_v11_byte_hash_repair_lock_is_receipt_gated_and_uses_new_evidence_root(self) -> None:
        repair_lock = self.load_yaml(REPAIR_SPLIT_LOCK)
        self.assertEqual(repair_lock["protocol_version"], "v1.1")
        self.assertEqual(repair_lock["registry"]["locked_primary_ids"], DATASETS)
        self.assertEqual(repair_lock["seeds"], SEEDS)
        self.assertEqual(repair_lock["paths"]["split_root"], "artifacts/splits/v1.1")
        self.assertEqual(repair_lock["paths"]["stage04_evidence_root"], "artifacts/stage08_v11/stage04_split_preparation_v1.1.1")
        self.assertEqual(repair_lock["output_identifiers"]["version"], "v1.1.1")
        self.assertTrue(repair_lock["execution_gate"]["d08_003_numeric_cache_budget_receipt_required_before_execution"])
        self.assertFalse(repair_lock["execution_gate"]["local_cache_generation_authorized"])
        self.assertFalse(repair_lock["execution_gate"]["tabpfn_cache_generation_authorized"])
        self.assertFalse(repair_lock["execution_gate"]["pilot_authorized"])
        self.assertFalse(repair_lock["execution_gate"]["formal_run_manifest_authorized"])
        self.assertFalse(repair_lock["execution_gate"]["full_experiment_authorized"])
        plan = self.stage04.build_stage04_v11_execution_plan(ROOT, REPAIR_SPLIT_LOCK)
        self.assertEqual(plan["unit_count"], 80)
        self.assertEqual(plan["output_paths"]["stage_root"], ROOT / "artifacts/stage08_v11/stage04_split_preparation_v1.1.1")
        self.assertEqual(plan["authorization"]["receipt"]["artifact_id"], "D08-003_stage08_v11_cache_and_pilot_budget")

    def test_v11_byte_hash_repair_lock_rejects_any_non_metadata_or_cross_profile_change(self) -> None:
        repair_lock = self.load_yaml(REPAIR_SPLIT_LOCK)
        mutations = (
            ("protocol_version", "v1.0"),
            ("protocol.protocol_version_for_seed_derivation", "v1.0"),
            ("registry.locked_primary_ids", DATASETS[:-1]),
            ("seeds", SEEDS[:-1]),
            ("split_fractions.train", 0.5),
            ("paths.split_root", "artifacts/splits/v1.0"),
            ("paths.stage04_evidence_root", "artifacts/stage08_v11/stage04_split_preparation"),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                mutated = deepcopy(repair_lock)
                target = mutated
                keys = field.split(".")
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                with patch.object(self.stage04, "_read_split_lock_mapping", return_value=mutated):
                    with self.assertRaises(ValueError):
                        self.stage04.load_split_lock(ROOT, REPAIR_SPLIT_LOCK)

    def test_v11_lock_filename_must_match_its_exact_approved_profile(self) -> None:
        original_lock = self.load_yaml(SPLIT_LOCK)
        repair_lock = self.load_yaml(REPAIR_SPLIT_LOCK)
        for lock_path, wrong_profile in ((SPLIT_LOCK, repair_lock), (REPAIR_SPLIT_LOCK, original_lock)):
            with self.subTest(lock_path=lock_path.name, wrong_version=wrong_profile["output_identifiers"]["version"]):
                with patch.object(self.stage04, "_read_split_lock_mapping", return_value=wrong_profile):
                    with self.assertRaises(ValueError):
                        self.stage04.load_split_lock(ROOT, lock_path)

    def test_independent_split_audit_can_bind_the_v111_repair_lock_and_evidence(self) -> None:
        payload = self.split_audit.audit(ROOT, split_lock_relative_path=Path("configs/stage04_splits_v1.1.1.yaml"))
        self.assertEqual(payload["version"], "v1.1.1")
        self.assertEqual(payload["controlled_inputs"]["split_lock_path"], "configs/stage04_splits_v1.1.1.yaml")
        self.assertEqual(payload["stage04_repair_evidence"]["status"], "PASS")


if __name__ == "__main__":
    unittest.main()

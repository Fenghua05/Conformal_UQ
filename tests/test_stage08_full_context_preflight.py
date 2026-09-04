import hashlib
import importlib.util
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "stage08_tabpfn_full_context_preflight_v1.1.yaml"
RUNNER = ROOT / "cloud" / "tabpfn_stage08" / "00_full_context_preflight.py"
BUILDER = ROOT / "cloud" / "tabpfn_stage08" / "01_build_upload_bundle.py"
BUDGET_RECEIPT = ROOT / "decisions" / "D08-002_CLOUD_PREFLIGHT_BUDGET_RECEIPT.json"


class Stage08FullContextPreflightConfigTests(unittest.TestCase):
    def load(self):
        return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    def test_config_has_exact_approved_scope_and_runtime(self):
        config = self.load()
        self.assertEqual(config["artifact_status"], "APPROVED_PACKAGE_PREPARATION_ONLY")
        self.assertEqual(config["protocol_version"], "v1.1")
        self.assertEqual(config["runtime"]["device"], "cuda")
        self.assertEqual(config["runtime"]["tabpfn_version"], "8.5.0")
        self.assertEqual(config["runtime"]["checkpoint_sha256"], "d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988")
        self.assertFalse(config["runtime"]["ignore_pretraining_limits"])
        self.assertEqual(config["safety_limits"], {"max_train_rows": 100000, "max_transformed_features": 2000})
        self.assertEqual(config["units"], [
            {"dataset_id": "openml_23512_higgs", "seed": 104729},
            {"dataset_id": "openml_23517_numerai28_6", "seed": 104729},
            {"dataset_id": "openml_1590_adult", "seed": 104729},
        ])
        self.assertEqual(config["repeat_unit"], {"dataset_id": "openml_23512_higgs", "seed": 104729, "max_abs_probability_difference": 1e-10})

    def test_config_forbids_cache_cp_and_formal_outputs(self):
        config = self.load()
        self.assertEqual(config["output_contract"]["allowed_artifacts"], ["preflight_manifest.json", "events.jsonl", "failure_records/"])
        self.assertEqual(set(config["output_contract"]["prohibited_artifacts"]), {"predictions.npz", "stage05_base_prediction_cache manifest.json", "results_long.parquet", "results_long.csv", "CP cell JSON", "figures"})
        self.assertTrue(config["execution_gate"]["numeric_cloud_budget_required_before_execution"])
        self.assertFalse(config["execution_gate"]["formal_run_manifest_authorized"])

    def test_config_hash_pins_approval_and_protocol(self):
        config = self.load()
        for key, relative in {"protocol_sha256": "protocols/protocol_v1.1.md", "approval_sha256": "decisions/D08-001_APPROVAL_RECEIPT.md"}.items():
            self.assertEqual(config["input_hashes"][key], hashlib.sha256((ROOT / relative).read_bytes()).hexdigest())


class Stage08FullContextPreflightRunnerTests(unittest.TestCase):
    @staticmethod
    def module():
        spec = importlib.util.spec_from_file_location("stage08_preflight", RUNNER)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module

    def test_runner_imports_without_tabpfn_and_enforces_local_guards(self):
        module = self.module()
        config = module.load_and_validate_config(ROOT, CONFIG)
        self.assertEqual(module.validate_matrix_shape((100000, 2000), config["safety_limits"]), (100000, 2000))
        with self.assertRaisesRegex(ValueError, "train rows"):
            module.validate_matrix_shape((100001, 10), config["safety_limits"])
        with self.assertRaisesRegex(ValueError, "features"):
            module.validate_matrix_shape((100, 2001), config["safety_limits"])
        self.assertGreaterEqual(module.assert_budget_not_exhausted(0.0, {"maximum_wall_clock_hours": 10**9}), 0.0)
        with self.assertRaisesRegex(TimeoutError, "budget exhausted"):
            module.assert_budget_not_exhausted(0.0, {"maximum_wall_clock_hours": 1e-12})

    def test_runner_requires_a_matching_budget_receipt_before_model_import(self):
        module = self.module()
        config = module.load_and_validate_config(ROOT, CONFIG)
        with self.assertRaisesRegex(FileNotFoundError, "budget receipt"):
            module.load_budget_receipt(ROOT / "decisions" / "MISSING_STAGE08_BUDGET.json", config)

    def test_approved_budget_receipt_is_bound_to_the_current_config(self):
        module = self.module()
        config = module.load_and_validate_config(ROOT, CONFIG)
        config["_config_path"] = str(CONFIG.resolve())
        receipt = module.load_budget_receipt(BUDGET_RECEIPT, config)
        self.assertEqual(receipt["maximum_wall_clock_hours"], 12)
        self.assertEqual(receipt["maximum_cloud_storage_gb"], 50)


class Stage08FullContextPreflightBundleTests(unittest.TestCase):
    @staticmethod
    def module():
        spec = importlib.util.spec_from_file_location("stage08_preflight_builder", BUILDER)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module

    def test_bundle_controls_every_hash_checked_by_the_runner(self):
        module = self.module()
        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        controlled = set(module.controlled_common_inputs(config))
        self.assertIn(Path("environment/environment_lock_v1.0.json"), controlled)


if __name__ == "__main__":
    unittest.main()

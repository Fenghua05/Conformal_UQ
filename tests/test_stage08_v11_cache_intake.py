"""Contract tests for the independent Stage 08 v1.1 240-cache intake audit.

All tests run locally WITHOUT importing TabPFN, fitting a model, or creating
any cache/CP/pilot/formal output.  The final class verifies the real intake
evidence after ``src/audit_stage08_v11_cache_intake.py`` has audited the
user-returned TabPFN cache archive.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conformal_uq.config import FROZEN_DATASETS, FROZEN_SEEDS

AUDITOR = ROOT / "src" / "audit_stage08_v11_cache_intake.py"
SPLIT_LOCK = ROOT / "configs" / "stage04_splits_v1.1.yaml"
LOCAL_LOCK = ROOT / "configs" / "stage05_lr_xgboost_v1.1.yaml"
TABPFN_LOCK = ROOT / "configs" / "stage05b_tabpfn_v1.1.yaml"
ENV_LOCK = ROOT / "environment" / "environment_lock_v1.0.json"
RECEIPT = ROOT / "decisions" / "D08-003_V11_CACHE_AND_PILOT_BUDGET_RECEIPT.json"
CACHE_ROOT = ROOT / "artifacts" / "caches" / "v1.1"
INTAKE_GLOB = "cache_intake_*/intake_audit.json"

KNOWN_TABPFN_CFG = "cee5c7d7da780885942a924b66276e7256469a013c9b5a0db98ba39249daa893"
KNOWN_TABPFN_CODE = "8be59da84b507ba06778a020b2cb54bc187326421acd0f68428f71367079f9c8"
KNOWN_LOCAL_CFG = "40f29139c9db63b2118c0efb28daa37940065a33dc52ec607b3e16bea0b786f9"


def load_auditor():
    spec = importlib.util.spec_from_file_location("stage08_v11_cache_intake_auditor_for_test", AUDITOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage08_v11_cache_intake_auditor_for_test"] = module
    spec.loader.exec_module(module)
    return module


def sha256_bytes(*chunks: bytes) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


class V11CacheIntakeUnitScopeTests(unittest.TestCase):
    """The intake must cover exactly 240 keys: 80 per model."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.auditor = load_auditor()

    def test_expected_keys_are_exactly_240_with_80_per_model(self) -> None:
        keys = self.auditor.expected_intake_unit_keys()
        self.assertEqual(len(keys), 240)
        self.assertEqual(sum(key[2] == "tabpfn" for key in keys), 80)
        self.assertEqual(sum(key[2] == "logistic_regression" for key in keys), 80)
        self.assertEqual(sum(key[2] == "xgboost" for key in keys), 80)
        self.assertEqual(
            keys,
            {(dataset, seed, model) for dataset in FROZEN_DATASETS for seed in FROZEN_SEEDS for model in ("logistic_regression", "xgboost", "tabpfn")},
        )

    def test_recomputed_hashes_match_the_bound_lineage(self) -> None:
        self.assertEqual(self.auditor.recompute_tabpfn_config_hash(ROOT), KNOWN_TABPFN_CFG)
        self.assertEqual(self.auditor.recompute_local_config_hash(ROOT), KNOWN_LOCAL_CFG)
        self.assertEqual(self.auditor.ENVIRONMENT_HASH, hashlib.sha256(ENV_LOCK.read_bytes()).hexdigest())
        # Cache-time code hash is bound to the immutable upload receipt, never
        # to the current tree (the auditor itself lives under src/).
        self.assertEqual(self.auditor.CACHE_TIME_TABPFN_CONFIG_HASH, KNOWN_TABPFN_CFG)
        self.assertEqual(self.auditor.CACHE_TIME_TABPFN_CODE_HASH, KNOWN_TABPFN_CODE)
        self.assertTrue(self.auditor.UPLOAD_RECEIPT_PATH.startswith("artifacts/stage08_v11_transfer/cache_upload_authorized_"))

    def test_tabpfn_provenance_carries_the_full_v11_cloud_lineage(self) -> None:
        table = SimpleNamespace(raw_sha256="d" * 64, label_mapping={"negative": 0, "positive": 1})
        split = SimpleNamespace(split_hash="e" * 64)
        provenance = self.auditor.expected_tabpfn_cache_provenance(
            ROOT, table, split, 104729,
            config_hash=KNOWN_TABPFN_CFG, code_hash=KNOWN_TABPFN_CODE,
        )
        self.assertEqual(provenance["protocol_version"], "v1.1")
        self.assertEqual(provenance["model_name"], "tabpfn")
        self.assertEqual(provenance["config_hash"], KNOWN_TABPFN_CFG)
        self.assertEqual(provenance["split_lock_sha256"], hashlib.sha256(SPLIT_LOCK.read_bytes()).hexdigest())
        self.assertEqual(provenance["d08_003_cache_lock_sha256"], hashlib.sha256(TABPFN_LOCK.read_bytes()).hexdigest())
        self.assertEqual(provenance["local_cache_lock_sha256"], hashlib.sha256(TABPFN_LOCK.read_bytes()).hexdigest())

    def test_local_provenance_carries_the_local_lock_lineage(self) -> None:
        table = SimpleNamespace(raw_sha256="d" * 64, label_mapping={"negative": 0, "positive": 1})
        split = SimpleNamespace(split_hash="e" * 64)
        provenance = self.auditor.expected_local_cache_provenance(
            ROOT, table, split, 104729, "logistic_regression",
            config_hash=KNOWN_LOCAL_CFG, code_hash="b" * 64,
        )
        self.assertEqual(provenance["protocol_version"], "v1.1")
        self.assertEqual(provenance["local_cache_lock_sha256"], hashlib.sha256(LOCAL_LOCK.read_bytes()).hexdigest())
        self.assertEqual(provenance["split_lock_sha256"], hashlib.sha256(SPLIT_LOCK.read_bytes()).hexdigest())
        self.assertEqual(provenance["d08_003_cache_lock_sha256"], hashlib.sha256(TABPFN_LOCK.read_bytes()).hexdigest())


class V11CacheIntakeValidationTests(unittest.TestCase):
    """Pure validators for receipt, summary, members, and the combined tree."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.auditor = load_auditor()

    def test_return_receipt_validation(self) -> None:
        receipt = {
            "artifact_id": "stage08_v11_tabpfn_cache_return_archive",
            "archive": "stage08_v11_tabpfn_cache_return.tar.gz",
            "archive_sha256": "a" * 64, "archive_bytes": 10,
            "inventory_sha256": "b" * 64, "source_run": "artifacts/stage08_v11_cloud/cache_run_x",
            "verified_units": 80, "config_hash": KNOWN_TABPFN_CFG, "code_hash": KNOWN_TABPFN_CODE,
            "d08_003_cache_lock_sha256": hashlib.sha256(TABPFN_LOCK.read_bytes()).hexdigest(),
            "cp_evaluated": False, "pilot_outputs": False, "formal_run_manifest_created": False,
        }
        errors = self.auditor.validate_return_receipt(
            receipt, archive_sha256="a" * 64, archive_bytes=10,
            config_hash=KNOWN_TABPFN_CFG, code_hash=KNOWN_TABPFN_CODE,
            inventory_sha256="b" * 64, final_lock_sha256=hashlib.sha256(TABPFN_LOCK.read_bytes()).hexdigest(),
        )
        self.assertEqual(errors, [])
        broken = [
            {**receipt, "verified_units": 79},
            {**receipt, "archive_sha256": "c" * 64},
            {**receipt, "config_hash": "f" * 64},
            {**receipt, "cp_evaluated": True},
            {**receipt, "formal_run_manifest_created": True},
        ]
        for payload in broken:
            self.assertNotEqual(
                self.auditor.validate_return_receipt(
                    payload, archive_sha256="a" * 64, archive_bytes=10,
                    config_hash=KNOWN_TABPFN_CFG, code_hash=KNOWN_TABPFN_CODE,
                    inventory_sha256="b" * 64, final_lock_sha256=hashlib.sha256(TABPFN_LOCK.read_bytes()).hexdigest(),
                ),
                [],
            )

    def test_generator_summary_validation(self) -> None:
        summary = {
            "status": "PASS", "protocol_version": "v1.1", "scope": "D08_003_V11_TABPFN_PROBABILITY_CACHES_ONLY",
            "expected_units": 80, "completed_units": 80,
            "config_hash": KNOWN_TABPFN_CFG, "code_hash": KNOWN_TABPFN_CODE,
            "environment_hash": hashlib.sha256(ENV_LOCK.read_bytes()).hexdigest(),
            "split_lock_sha256": hashlib.sha256(SPLIT_LOCK.read_bytes()).hexdigest(),
            "tabpfn_cache_lock_sha256": hashlib.sha256(TABPFN_LOCK.read_bytes()).hexdigest(),
            "d08_003_cache_lock_sha256": hashlib.sha256(TABPFN_LOCK.read_bytes()).hexdigest(),
            "budget": {"maximum_wall_clock_hours": 12, "maximum_cloud_storage_gb": 50, "elapsed_seconds": 100.0, "produced_bytes": 1024},
            "runtime_evidence": {"cuda_available": True, "gpu_name": "NVIDIA GeForce RTX 4090", "tabpfn_version": "8.5.0", "checkpoint_sha256": "d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988"},
            "failures": [],
            "cp_evaluated": False, "pilot_outputs": False, "formal_run_manifest_created": False, "full_experiment_executed": False,
        }
        budget = {"maximum_wall_clock_hours": 12, "maximum_cloud_storage_gb": 50}
        self.assertEqual(self.auditor.validate_generator_summary(summary, config_hash=KNOWN_TABPFN_CFG, code_hash=KNOWN_TABPFN_CODE, budget=budget), [])
        broken = [
            {**summary, "completed_units": 79},
            {**summary, "status": "FAIL"},
            {**summary, "cp_evaluated": True},
            {**summary, "budget": {**summary["budget"], "elapsed_seconds": 12 * 3600 + 1}},
            {**summary, "budget": {**summary["budget"], "produced_bytes": 50 * 1024 ** 3 + 1}},
            {**summary, "runtime_evidence": {**summary["runtime_evidence"], "gpu_name": "other gpu"}},
        ]
        for payload in broken:
            self.assertNotEqual(self.auditor.validate_generator_summary(payload, config_hash=KNOWN_TABPFN_CFG, code_hash=KNOWN_TABPFN_CODE, budget=budget), [])

    def test_prohibited_return_member_scan(self) -> None:
        prohibited = [
            "s/results_long.parquet", "s/figures/f.png", "s/formal_run_manifest.json",
            "s/cred.json", "s/token.txt", "s/model.ckpt",
            "s/data/raw.arff",
            "s/artifacts/caches/v1.0/cfg-x/d/seed-1/tabpfn/manifest.json",
            "s/artifacts/splits/v1.0/openml_3_kr_vs_kp/seed-104729.json",
            "s/artifacts/caches/v1.1/cfg-foreign/code-foreign/d/seed-1/tabpfn/manifest.json",
        ]
        for member in prohibited:
            self.assertTrue(self.auditor.scan_prohibited_return_members([member]), f"scan must flag {member}")
        allowed = [
            "stage08_v11_tabpfn_cache_return/artifacts/caches/v1.1/cfg-cee5c7d7da78/code-8be59da84b50/openml_3_kr_vs_kp/seed-104729/tabpfn/predictions.npz",
            "stage08_v11_tabpfn_cache_return/artifacts/caches/v1.1/cfg-cee5c7d7da78/code-8be59da84b50/openml_3_kr_vs_kp/seed-104729/tabpfn/manifest.json",
            "stage08_v11_tabpfn_cache_return/configs/stage05b_tabpfn_v1.1.yaml",
            "stage08_v11_tabpfn_cache_return/artifacts/stage08_v11_cloud/cache_run_x/summary.json",
            "stage08_v11_tabpfn_cache_return/return_inventory.json",
        ]
        self.assertEqual(self.auditor.scan_prohibited_return_members(allowed), [])

    def test_combined_cache_entry_validation(self) -> None:
        local_cfg, local_code = "a" * 64, "b" * 64
        tabpfn_cfg, tabpfn_code = "c" * 64, "d" * 64
        lc, lco = f"cfg-{local_cfg[:12]}", f"code-{local_code[:12]}"
        tc, tco = f"cfg-{tabpfn_cfg[:12]}", f"code-{tabpfn_code[:12]}"
        expected = {("dataset_a", 1, "logistic_regression"), ("dataset_a", 1, "xgboost"), ("dataset_a", 1, "tabpfn")}
        entries = set()
        for cfg, code, model in ((lc, lco, "logistic_regression"), (lc, lco, "xgboost"), (tc, tco, "tabpfn")):
            unit = PurePosixPath(cfg, code, "dataset_a", "seed-1", model)
            entries.update({unit.parents[2], unit.parents[1], unit.parents[0], unit, unit / "manifest.json", unit / "predictions.npz"})
            entries.update({PurePosixPath(cfg), PurePosixPath(cfg, code)})
        self.assertEqual(self.auditor.validate_combined_cache_relative_entries(entries, local_cfg, local_code, tabpfn_cfg, tabpfn_code, expected), [])
        errors = self.auditor.validate_combined_cache_relative_entries(
            entries | {PurePosixPath("cfg-foreign/code-x/dataset_a/seed-1/tabpfn/manifest.json")},
            local_cfg, local_code, tabpfn_cfg, tabpfn_code, expected,
        )
        self.assertTrue(any("foreign" in error for error in errors))
        errors = self.auditor.validate_combined_cache_relative_entries(
            entries - {PurePosixPath(tc, tco, "dataset_a", "seed-1", "tabpfn", "predictions.npz")},
            local_cfg, local_code, tabpfn_cfg, tabpfn_code, expected,
        )
        self.assertTrue(any("incomplete" in error for error in errors))

    def test_install_rejects_existing_targets_and_copies_once(self) -> None:
        unit_dir = PurePosixPath(f"cfg-{KNOWN_TABPFN_CFG[:12]}", f"code-{KNOWN_TABPFN_CODE[:12]}", "dataset_a", "seed-1", "tabpfn")
        source_root = ROOT / "tmp" / f"stage08_v11_intake_install_{uuid.uuid4().hex}"
        cache_root = source_root / "cache_root"
        extracted = source_root / "extracted"
        self.addCleanup(shutil.rmtree, source_root, ignore_errors=True)
        for name in ("manifest.json", "predictions.npz"):
            path = extracted / Path(*unit_dir.parts) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"payload-{name}".encode("utf-8"))
        installed = self.auditor.install_tabpfn_cache_units(
            extracted, cache_root, [("dataset_a", 1)], KNOWN_TABPFN_CFG, KNOWN_TABPFN_CODE,
        )
        self.assertEqual(len(installed), 1)
        for name in ("manifest.json", "predictions.npz"):
            self.assertEqual((cache_root / Path(*unit_dir.parts) / name).read_bytes(), f"payload-{name}".encode("utf-8"))
        with self.assertRaises(FileExistsError):
            self.auditor.install_tabpfn_cache_units(extracted, cache_root, [("dataset_a", 1)], KNOWN_TABPFN_CFG, KNOWN_TABPFN_CODE)


class V11CacheIntakeRealEvidenceTests(unittest.TestCase):
    """Verify the real intake audit after it has run on the returned archive."""

    @classmethod
    def setUpClass(cls) -> None:
        candidates = sorted((ROOT / "artifacts" / "stage08_v11_cloud").glob(INTAKE_GLOB))
        if not candidates:
            raise AssertionError("No cache intake audit exists yet; run src/audit_stage08_v11_cache_intake.py first.")
        cls.audit_path = candidates[-1]
        cls.audit = json.loads(cls.audit_path.read_text(encoding="utf-8"))

    def test_intake_audit_passes_with_240_units(self) -> None:
        self.assertEqual(self.audit["verdict"], "PASS")
        self.assertEqual(self.audit["expected_units"], 240)
        self.assertEqual(self.audit["valid_units"], 240)
        self.assertEqual(self.audit["model_counts"], {"logistic_regression": 80, "xgboost": 80, "tabpfn": 80})
        self.assertFalse(self.audit["cp_evaluated"])
        self.assertFalse(self.audit["pilot_outputs"])
        self.assertFalse(self.audit["formal_run_manifest_created"])

    def test_installed_tabpfn_tree_is_complete_and_not_mixed(self) -> None:
        auditor = load_auditor()
        entries = {PurePosixPath(path.relative_to(CACHE_ROOT).as_posix()) for path in CACHE_ROOT.rglob("*")}
        keys = auditor.expected_intake_unit_keys()
        errors = auditor.validate_combined_cache_relative_entries(
            entries, KNOWN_LOCAL_CFG, self.audit["lineage"]["local_cache_time_code_hash"], KNOWN_TABPFN_CFG, KNOWN_TABPFN_CODE, keys,
        )
        self.assertEqual(errors, [])
        manifests = list(CACHE_ROOT.rglob("manifest.json"))
        self.assertEqual(len(manifests), 240)
        for manifest_path in manifests:
            provenance = json.loads(manifest_path.read_text(encoding="utf-8"))["provenance"]
            self.assertEqual(provenance["protocol_version"], "v1.1")


if __name__ == "__main__":
    unittest.main()

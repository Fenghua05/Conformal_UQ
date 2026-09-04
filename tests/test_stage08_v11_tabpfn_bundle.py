"""Contract tests for the user-operated 80-unit v1.1 TabPFN cache package.

Everything here runs on the local machine WITHOUT importing TabPFN, accessing
a GPU, fitting a model, or creating any cache/CP/pilot/formal output.  The
final class verifies the immutable credential-free upload archive after it
has been built by ``04_build_v11_cache_upload_bundle.py``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import tarfile
import tempfile
import unittest
import uuid
from pathlib import Path, PurePosixPath
from unittest.mock import patch

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conformal_uq.config import FROZEN_DATASETS, FROZEN_SEEDS

RUNNER = ROOT / "cloud" / "tabpfn_stage08" / "02_generate_v11_tabpfn_caches.py"
PACKER = ROOT / "cloud" / "tabpfn_stage08" / "03_verify_and_pack_v11_caches.py"
BUILDER = ROOT / "cloud" / "tabpfn_stage08" / "04_build_v11_cache_upload_bundle.py"
FINAL_LOCK = ROOT / "configs" / "stage05b_tabpfn_v1.1.yaml"
SPLIT_LOCK = ROOT / "configs" / "stage04_splits_v1.1.yaml"
RECEIPT = ROOT / "decisions" / "D08-003_V11_CACHE_AND_PILOT_BUDGET_RECEIPT.json"
ENVIRONMENT_LOCK = ROOT / "environment" / "environment_lock_v1.0.json"
REGISTRY = ROOT / "artifacts" / "stage02" / "dataset_registry_v1.0.1.json"
TRANSFER_ROOT = ROOT / "artifacts" / "stage08_v11_transfer"
ARCHIVE_NAME = "stage08_v11_tabpfn_cache_upload.tar.gz"


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_bytes(*chunks: bytes) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


class V11TabPFNCacheScopeTests(unittest.TestCase):
    """The final lock and D08-003 receipt must bound exactly eighty units."""

    def test_final_lock_pins_the_exact_v11_tabpfn_cache_contract(self) -> None:
        lock = yaml.safe_load(FINAL_LOCK.read_text(encoding="utf-8"))
        self.assertEqual(lock["protocol_version"], "v1.1")
        self.assertEqual(lock["registry"]["locked_primary_ids"], FROZEN_DATASETS)
        self.assertEqual(lock["seeds"], FROZEN_SEEDS)
        self.assertEqual(lock["split_lock_path"], "configs/stage04_splits_v1.1.yaml")
        self.assertEqual(lock["paths"]["split_root"], "artifacts/splits/v1.1")
        self.assertEqual(lock["paths"]["cache_root"], "artifacts/caches/v1.1")
        runtime = lock["runtime"]
        self.assertEqual(runtime["device"], "cuda")
        self.assertEqual(runtime["tabpfn_version"], "8.5.0")
        self.assertEqual(runtime["checkpoint_path"], "/root/autodl-fs/tabpfn-model-cache/tabpfn-v3-classifier-v3_default.ckpt")
        self.assertEqual(runtime["checkpoint_sha256"], "d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988")
        self.assertFalse(runtime["ignore_pretraining_limits"])
        self.assertTrue(runtime["no_truncation_or_subsampling"])
        self.assertEqual(runtime["train_partition_contract"], "full_fixed_train_partition")
        self.assertEqual(runtime["preprocessing_contract"], "stage04_train_only_unscaled_onehot_dense")
        self.assertEqual(lock["safety_limits"], {"max_train_rows": 100000, "max_transformed_features": 2000})
        scope = lock["authorization"]["cache_only_scope"]
        self.assertEqual(scope["authorized_tabpfn_units"], 80)
        self.assertEqual(scope["datasets"], 8)
        self.assertEqual(scope["seeds_per_dataset"], 10)
        self.assertTrue(scope["probability_cache_only"])
        self.assertFalse(scope["conformal_prediction_allowed"])
        self.assertFalse(scope["pilot_output_allowed"])
        self.assertFalse(scope["formal_output_allowed"])
        for key in ("conformal_prediction_allowed", "pilot_allowed", "formal_outputs_allowed"):
            self.assertFalse(lock["output_contract"][key])
        self.assertFalse(lock["execution_gate"]["pilot_authorized"])
        self.assertFalse(lock["execution_gate"]["formal_run_manifest_authorized"])
        self.assertFalse(lock["execution_gate"]["full_experiment_authorized"])

    def test_d08_003_receipt_bounds_the_cache_budget_and_final_lock(self) -> None:
        receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "APPROVED_FOR_STAGE08_V11_CACHE_AND_PILOT_ONLY")
        self.assertEqual(receipt["maximum_wall_clock_hours"], 12)
        self.assertEqual(receipt["maximum_cloud_storage_gb"], 50)
        self.assertEqual(receipt["authorized_tabpfn_units"], 80)
        self.assertFalse(receipt["formal_run_manifest_authorized"])
        self.assertFalse(receipt["full_experiment_authorized"])
        self.assertEqual(receipt["cache_lock_sha256"], hashlib.sha256(FINAL_LOCK.read_bytes()).hexdigest())


class V11TabPFNCacheRunnerContractTests(unittest.TestCase):
    """The cloud generator must be importable and locked without TabPFN."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load_script_module("stage08_v11_tabpfn_cache_runner_for_test", RUNNER)

    def test_runner_module_imports_without_tabpfn(self) -> None:
        self.assertNotIn("tabpfn", sys.modules)

    def test_expected_units_are_exactly_the_frozen_eight_by_ten(self) -> None:
        lock = yaml.safe_load(FINAL_LOCK.read_text(encoding="utf-8"))
        units = self.runner.expected_v11_tabpfn_units(lock)
        self.assertEqual(len(units), 80)
        self.assertEqual(set(units), {(dataset, seed) for dataset in FROZEN_DATASETS for seed in FROZEN_SEEDS})
        self.assertEqual(units, tuple((dataset, seed) for dataset in FROZEN_DATASETS for seed in FROZEN_SEEDS))
        with self.assertRaisesRegex(self.runner.V11TabPFNCacheLockError, "80"):
            self.runner.expected_v11_tabpfn_units({**lock, "seeds": FROZEN_SEEDS[:9]})

    def test_lock_validation_accepts_only_the_canonical_final_lock(self) -> None:
        lock = self.runner.load_and_validate_v11_cache_lock(ROOT, FINAL_LOCK)
        self.assertEqual(lock["protocol_version"], "v1.1")
        with self.assertRaisesRegex(self.runner.V11TabPFNCacheLockError, "canonical"):
            self.runner.load_and_validate_v11_cache_lock(ROOT, ROOT / "configs" / "stage05_lr_xgboost_v1.1.yaml")

    def test_receipt_gate_requires_the_canonical_d08_003_receipt(self) -> None:
        with self.assertRaisesRegex(self.runner.V11TabPFNCacheLockError, "canonical"):
            self.runner.load_validated_d08_003_authorization(ROOT, ROOT / "decisions" / "D08-002_CLOUD_PREFLIGHT_BUDGET_RECEIPT.json")
        authorization = self.runner.load_validated_d08_003_authorization(ROOT, RECEIPT)
        self.assertEqual(authorization["receipt"]["maximum_wall_clock_hours"], 12)
        self.assertEqual(authorization["receipt"]["maximum_cloud_storage_gb"], 50)
        self.assertEqual(authorization["final_lock_sha256"], hashlib.sha256(FINAL_LOCK.read_bytes()).hexdigest())

    def test_config_hash_is_independently_recomputable(self) -> None:
        lock = yaml.safe_load(FINAL_LOCK.read_text(encoding="utf-8"))
        interim = sha256_bytes(SPLIT_LOCK.read_bytes(), FINAL_LOCK.read_bytes())
        expected = sha256_bytes(bytes.fromhex(interim), bytes.fromhex(sha256_bytes(FINAL_LOCK.read_bytes())))
        self.assertEqual(self.runner.v11_tabpfn_config_hash(ROOT, lock), expected)

    def test_code_hash_covers_src_and_the_stage08_cloud_tree(self) -> None:
        entries = sorted(
            str(path.relative_to(ROOT)).replace("\\", "/")
            for directory in (ROOT / "src", ROOT / "cloud" / "tabpfn_stage08")
            for path in directory.rglob("*.py")
        )
        digest = hashlib.sha256()
        for relative in entries:
            digest.update(relative.encode("utf-8"))
            digest.update((ROOT / relative).read_bytes())
        self.assertEqual(self.runner.v11_tabpfn_code_hash(ROOT), digest.hexdigest())

    def test_full_context_matrix_guards(self) -> None:
        limits = {"max_train_rows": 100000, "max_transformed_features": 2000}
        self.assertEqual(self.runner.validate_matrix_shape((100000, 2000), limits), (100000, 2000))
        with self.assertRaisesRegex(ValueError, "train rows"):
            self.runner.validate_matrix_shape((100001, 10), limits)
        with self.assertRaisesRegex(ValueError, "features"):
            self.runner.validate_matrix_shape((100, 2001), limits)

    def test_aligned_probabilities_guard(self) -> None:
        import numpy as np

        aligned = self.runner.aligned_probabilities([1, 0], [[0.7, 0.3], [0.1, 0.9]])
        self.assertTrue(np.allclose(aligned, [[0.3, 0.7], [0.9, 0.1]]))
        with self.assertRaisesRegex(ValueError, "row-sum|probabilities"):
            self.runner.aligned_probabilities([0, 1], [[0.6, 0.5], [0.2, 0.8]])

    def test_budget_guards(self) -> None:
        import time

        self.assertGreaterEqual(self.runner.assert_budget_not_exhausted(time.perf_counter(), {"maximum_wall_clock_hours": 10 ** 9}), 0.0)
        with self.assertRaisesRegex(TimeoutError, "budget exhausted"):
            self.runner.assert_budget_not_exhausted(0.0, {"maximum_wall_clock_hours": 1e-12})
        with tempfile.TemporaryDirectory() as temporary:
            probe = Path(temporary) / "probe.bin"
            probe.write_bytes(b"x" * 1024)
            with self.assertRaisesRegex(self.runner.V11TabPFNCacheLockError, "storage"):
                self.runner.assert_storage_budget({"maximum_cloud_storage_gb": 0.5 / (1024 ** 2)}, probe.parent)
            self.assertEqual(self.runner.assert_storage_budget({"maximum_cloud_storage_gb": 50}, probe.parent), 1024)

    def test_cache_provenance_carries_the_full_v11_lock_lineage(self) -> None:
        from conformal_uq.prediction_cache import _required_provenance

        table = type("Table", (), {"raw_sha256": "d" * 64, "label_mapping": {"negative": 0, "positive": 1}})()
        split = type("Split", (), {"split_hash": "e" * 64})()
        provenance = self.runner.v11_tabpfn_cache_provenance(
            config_hash="a" * 64, code_hash="b" * 64, environment_hash="c" * 64,
            table=table, split=split, base_seed=104729,
            split_lock_sha256=hashlib.sha256(SPLIT_LOCK.read_bytes()).hexdigest(),
            tabpfn_cache_lock_sha256=hashlib.sha256(FINAL_LOCK.read_bytes()).hexdigest(),
        )
        self.assertEqual(provenance["protocol_version"], "v1.1")
        self.assertEqual(provenance["model_name"], "tabpfn")
        self.assertEqual(provenance["split_lock_sha256"], hashlib.sha256(SPLIT_LOCK.read_bytes()).hexdigest())
        self.assertEqual(provenance["d08_003_cache_lock_sha256"], hashlib.sha256(FINAL_LOCK.read_bytes()).hexdigest())
        self.assertEqual(provenance["local_cache_lock_sha256"], hashlib.sha256(FINAL_LOCK.read_bytes()).hexdigest())
        self.assertEqual(_required_provenance(provenance), provenance)

    def test_failure_records_are_unique_and_exclusive_within_the_same_second(self) -> None:
        root = ROOT / "tmp" / f"stage08_v11_tabpfn_failure_test_{uuid.uuid4().hex}"
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        scope = {"dataset_id": "dataset_a", "seed": 1, "model": "tabpfn"}
        first = self.runner.immutable_failure(root, run_identifier="run_one", config_hash="a" * 64, scope=scope, exception=ValueError("first"), retry_count=0)
        second = self.runner.immutable_failure(root, run_identifier="run_one", config_hash="a" * 64, scope=scope, exception=ValueError("second"), retry_count=0)
        self.assertNotEqual(first, second)
        self.assertTrue(first.is_file() and second.is_file())
        self.assertEqual(json.loads(first.read_text(encoding="utf-8"))["exception"], "first")
        self.assertEqual(json.loads(second.read_text(encoding="utf-8"))["exception"], "second")

    def test_duplicate_cache_scope_is_rejected_before_any_side_effect(self) -> None:
        root = ROOT / "tmp" / f"stage08_v11_tabpfn_scope_lock_test_{uuid.uuid4().hex}"
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        with self.runner.exclusive_v11_tabpfn_cache_run_lock(root, "a" * 64, "b" * 64):
            with self.assertRaisesRegex(self.runner.V11TabPFNCacheRunLockError, "already held"):
                with self.runner.exclusive_v11_tabpfn_cache_run_lock(root, "a" * 64, "b" * 64):
                    pass
        # The owner releases its own token, so a later run may acquire the scope.
        with self.runner.exclusive_v11_tabpfn_cache_run_lock(root, "a" * 64, "b" * 64):
            pass

    def test_valid_cache_reuse_never_fits_tabpfn(self) -> None:
        import numpy as np
        from types import SimpleNamespace

        cached = {"model_hash": "f" * 64, "manifest": {"metrics": {"calibration_pool": {"auroc": 0.5}, "test": {"auroc": 0.5}}}}
        labels = SimpleNamespace(to_numpy=lambda **_kwargs: np.asarray([0, 1], dtype=np.int8))
        table = SimpleNamespace(dataset_id="dataset_a", raw_sha256="1" * 64, label_mapping={"n": 0, "p": 1}, subset_labels=lambda _ids: labels)
        split = SimpleNamespace(split_hash="2" * 64, ids=SimpleNamespace(calibration_pool=("a",), test=("b",)))
        with patch.object(self.runner, "read_valid_cache", return_value=cached) as cache_reader, patch.object(
            self.runner, "full_context_tabpfn_fit_predict"
        ) as model_fit, patch.object(self.runner, "write_prediction_cache") as cache_writer:
            result = self.runner.reuse_or_generate_v11_tabpfn_cache(
                root=ROOT, lock=yaml.safe_load(FINAL_LOCK.read_text(encoding="utf-8")),
                config_hash="a" * 64, code_hash="b" * 64, environment_hash="c" * 64,
                split_lock_sha256="d" * 64, tabpfn_cache_lock_sha256="e" * 64,
                table=table, split=split,
                base_seed=1, cache_root=ROOT / "tmp" / "unused_cache_root", event_path=None, run_identifier="run", retry_event=None,
            )
        self.assertTrue(cache_reader.called)
        self.assertEqual(result["action"], "reused_validated_v11_tabpfn_cache")
        self.assertFalse(model_fit.called)
        self.assertFalse(cache_writer.called)


class V11TabPFNCachePackerContractTests(unittest.TestCase):
    """The cloud packer must verify exactly eighty complete cache units."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.packer = load_script_module("stage08_v11_tabpfn_cache_packer_for_test", PACKER)

    def test_packer_module_imports_without_tabpfn(self) -> None:
        self.assertNotIn("tabpfn", sys.modules)

    def test_cache_entry_validation_rejects_foreign_extra_and_incomplete_units(self) -> None:
        config_hash, code_hash = "a" * 64, "b" * 64
        cfg, code = f"cfg-{config_hash[:12]}", f"code-{code_hash[:12]}"
        expected = {("dataset_a", 1), ("dataset_a", 2)}
        good = {
            PurePosixPath(cfg), PurePosixPath(cfg, code),
            PurePosixPath(cfg, code, "dataset_a", "seed-1", "tabpfn"),
            PurePosixPath(cfg, code, "dataset_a", "seed-1", "tabpfn", "manifest.json"),
            PurePosixPath(cfg, code, "dataset_a", "seed-1", "tabpfn", "predictions.npz"),
            PurePosixPath(cfg, code, "dataset_a", "seed-2", "tabpfn"),
            PurePosixPath(cfg, code, "dataset_a", "seed-2", "tabpfn", "manifest.json"),
            PurePosixPath(cfg, code, "dataset_a", "seed-2", "tabpfn", "predictions.npz"),
        }
        self.assertEqual(self.packer.validate_exact_v11_tabpfn_cache_relative_entries(good, config_hash, code_hash, expected), [])
        errors = self.packer.validate_exact_v11_tabpfn_cache_relative_entries(
            good | {PurePosixPath("cfg-foreign/code-foreign/dataset_a/seed-1/tabpfn/manifest.json"), PurePosixPath(cfg, code, "dataset_a", "seed-3", "tabpfn", "predictions.npz")},
            config_hash, code_hash, expected,
        )
        self.assertTrue(any("foreign" in error for error in errors))
        self.assertTrue(any("incomplete" in error for error in errors))
        incomplete = good - {PurePosixPath(cfg, code, "dataset_a", "seed-2", "tabpfn", "predictions.npz")}
        errors = self.packer.validate_exact_v11_tabpfn_cache_relative_entries(incomplete, config_hash, code_hash, expected)
        self.assertTrue(any("incomplete" in error for error in errors))
        self.assertTrue(any("count mismatch" in error for error in errors))

    def test_summary_gate_requires_a_complete_eighty_unit_pass_run(self) -> None:
        summary = {
            "status": "PASS", "protocol_version": "v1.1", "expected_units": 80,
            "scope": "D08_003_V11_TABPFN_PROBABILITY_CACHES_ONLY",
            "completed_units": 80, "config_hash": "a" * 64, "code_hash": "b" * 64,
            "environment_hash": "c" * 64, "cp_evaluated": False, "pilot_outputs": False,
            "formal_run_manifest_created": False, "full_experiment_executed": False,
        }
        self.assertIsNone(self.packer.validate_generator_summary(summary, config_hash="a" * 64, code_hash="b" * 64, environment_hash="c" * 64))
        with self.assertRaisesRegex(ValueError, "80|eighty"):
            self.packer.validate_generator_summary({**summary, "completed_units": 79}, config_hash="a" * 64, code_hash="b" * 64, environment_hash="c" * 64)
        with self.assertRaisesRegex(ValueError, "status PASS"):
            self.packer.validate_generator_summary({**summary, "status": "FAIL"}, config_hash="a" * 64, code_hash="b" * 64, environment_hash="c" * 64)


class V11CacheUploadBundleBuilderContractTests(unittest.TestCase):
    """The upload bundle must control every input the cloud runner validates."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = load_script_module("stage08_v11_tabpfn_cache_builder_for_test", BUILDER)

    def test_builder_controls_every_runner_validated_input(self) -> None:
        lock = yaml.safe_load(FINAL_LOCK.read_text(encoding="utf-8"))
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        controlled = set(self.builder.controlled_upload_inputs(lock, registry))
        required = {
            Path("src"),
            Path("cloud/tabpfn_stage08"),
            Path("cloud/tabpfn_stage05b/requirements-tabpfn.lock"),
            Path("configs/stage05b_tabpfn_v1.1.yaml"),
            Path("configs/stage04_splits_v1.1.yaml"),
            Path("decisions/D08-003_V11_CACHE_AND_PILOT_BUDGET_RECEIPT.json"),
            Path("protocols/protocol_v1.1.md"),
            Path("protocols/dataset_lock_v1.0.md"),
            Path("environment/environment_lock_v1.0.json"),
            Path("artifacts/stage02/dataset_registry_v1.0.1.json"),
        }
        for dataset in FROZEN_DATASETS:
            record = next(item for item in registry["records"] if item["dataset_id"] == dataset)
            required.add(Path(str(record["source"]["raw_local_path"]).replace("\\", "/")))
            for seed in FROZEN_SEEDS:
                required.add(Path("artifacts/splits/v1.1") / dataset / f"seed-{seed}.json")
        self.assertTrue(required.issubset(controlled))
        for relative in sorted(controlled):
            self.assertTrue((ROOT / relative).exists(), f"controlled input is absent: {relative}")

    def test_prohibited_member_scan_rejects_caches_credentials_and_formal_outputs(self) -> None:
        prohibited = [
            "artifacts/caches/v1.0/cfg-x/dataset/seed-1/tabpfn/predictions.npz",
            "artifacts/caches/v1.1/cfg-x/dataset/seed-1/tabpfn/manifest.json",
            "credentials.json",
            "auth_token.txt",
            "model.ckpt",
            "results_long.parquet",
            "formal_run_manifest.json",
            "figures/figure_1.png",
            "artifacts/splits/v1.0/openml_3_kr_vs_kp/seed-104729.json",
        ]
        for member in prohibited:
            self.assertTrue(self.builder.scan_prohibited_members([member]), f"scan must flag {member}")
        allowed = [
            "src/conformal_uq/split.py",
            "cloud/tabpfn_stage08/02_generate_v11_tabpfn_caches.py",
            "artifacts/splits/v1.1/openml_3_kr_vs_kp/seed-104729.json",
            "data/stage02_raw/openml/3_3.arff",
            "configs/stage05b_tabpfn_v1.1.yaml",
            "decisions/D08-003_V11_CACHE_AND_PILOT_BUDGET_RECEIPT.json",
        ]
        self.assertEqual(self.builder.scan_prohibited_members(allowed), [])

    def test_builder_refuses_an_existing_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            existing = Path(temporary)
            with self.assertRaises(FileExistsError):
                self.builder.require_absent_directory(existing)
            fresh = existing / "fresh_package_dir"
            self.assertIsNone(self.builder.require_absent_directory(fresh))


class V11TabPFNCacheUploadArchiveTests(unittest.TestCase):
    """Verify the immutable, credential-free upload archive after it is built."""

    @classmethod
    def setUpClass(cls) -> None:
        candidates = sorted(TRANSFER_ROOT.glob(f"*/{ARCHIVE_NAME}"))
        if not candidates:
            raise AssertionError(
                "No v1.1 TabPFN cache upload archive exists yet under "
                f"{TRANSFER_ROOT}; build it with 04_build_v11_cache_upload_bundle.py first."
            )
        cls.archive_path = candidates[-1]
        cls.package_dir = cls.archive_path.parent
        with tarfile.open(cls.archive_path, "r:gz") as archive:
            cls.members = {member.name: member for member in archive.getmembers()}
            inventory_name = next(name for name in cls.members if name.endswith("/upload_inventory.json"))
            cls.inventory = json.loads(archive.extractfile(inventory_name).read().decode("utf-8"))

    def test_package_receipt_matches_the_archive_on_disk(self) -> None:
        receipt = json.loads((self.package_dir / "archive_receipt.json").read_text(encoding="utf-8"))
        self.assertTrue(receipt["cloud_execution_authorized"])
        self.assertEqual(receipt["archive_sha256"], hashlib.sha256(self.archive_path.read_bytes()).hexdigest())
        self.assertEqual(receipt["archive_bytes"], self.archive_path.stat().st_size)

    def test_every_inventory_file_matches_its_archive_member(self) -> None:
        stage_prefix = self.inventory["stage_root"] + "/"
        with tarfile.open(self.archive_path, "r:gz") as archive:
            self.assertGreater(len(self.inventory["files"]), 0)
            for record in self.inventory["files"]:
                name = stage_prefix + record["path"]
                self.assertIn(name, self.members, f"inventory file missing from archive: {record['path']}")
                member = self.members[name]
                self.assertTrue(member.isfile())
                self.assertEqual(member.size, record["bytes"])
                payload = archive.extractfile(member).read()
                self.assertEqual(hashlib.sha256(payload).hexdigest(), record["sha256"])

    def test_receipt_appears_exactly_once_and_controls_are_present(self) -> None:
        stage_prefix = self.inventory["stage_root"] + "/"
        receipt_members = [name for name in self.members if name.endswith("D08-003_V11_CACHE_AND_PILOT_BUDGET_RECEIPT.json")]
        self.assertEqual(len(receipt_members), 1)
        required_suffixes = [
            "configs/stage05b_tabpfn_v1.1.yaml",
            "configs/stage04_splits_v1.1.yaml",
            "environment/environment_lock_v1.0.json",
            "artifacts/stage02/dataset_registry_v1.0.1.json",
            "cloud/tabpfn_stage05b/requirements-tabpfn.lock",
            "cloud/tabpfn_stage08/02_generate_v11_tabpfn_caches.py",
            "cloud/tabpfn_stage08/03_verify_and_pack_v11_caches.py",
            "src/conformal_uq/stage08_authorization.py",
        ]
        for suffix in required_suffixes:
            self.assertIn(stage_prefix + suffix, self.members, f"archive is missing {suffix}")
        for dataset in FROZEN_DATASETS:
            for seed in FROZEN_SEEDS:
                self.assertIn(stage_prefix + f"artifacts/splits/v1.1/{dataset}/seed-{seed}.json", self.members)
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        for dataset in FROZEN_DATASETS:
            record = next(item for item in registry["records"] if item["dataset_id"] == dataset)
            raw = str(record["source"]["raw_local_path"]).replace("\\", "/")
            self.assertIn(stage_prefix + raw, self.members)

    def test_archive_carries_no_prohibited_members(self) -> None:
        builder = load_script_module("stage08_v11_tabpfn_cache_builder_for_scan", BUILDER)
        names = [name for name, member in self.members.items() if member.isfile()]
        flagged = builder.scan_prohibited_members(names)
        self.assertEqual(flagged, [], f"prohibited archive members: {flagged}")

    def test_recorded_bundle_hashes_are_recomputable_from_the_archive(self) -> None:
        stage_prefix = self.inventory["stage_root"] + "/"
        with tarfile.open(self.archive_path, "r:gz") as archive:

            def read(relative: str) -> bytes:
                return archive.extractfile(self.members[stage_prefix + relative]).read()

            split_bytes = read("configs/stage04_splits_v1.1.yaml")
            lock_bytes = read("configs/stage05b_tabpfn_v1.1.yaml")
            interim = sha256_bytes(split_bytes, lock_bytes)
            config_hash = sha256_bytes(bytes.fromhex(interim), bytes.fromhex(sha256_bytes(lock_bytes)))
            digest = hashlib.sha256()
            # Platform-independent ordering: one flat list sorted by POSIX relative string.
            code_members = sorted(
                name for name, member in self.members.items()
                if member.isfile() and name.endswith(".py")
                and (name.startswith(stage_prefix + "src/") or name.startswith(stage_prefix + "cloud/tabpfn_stage08/"))
            )
            for name in code_members:
                digest.update(name[len(stage_prefix):].encode("utf-8"))
                digest.update(archive.extractfile(self.members[name]).read())
        self.assertEqual(config_hash, self.inventory["bundle_config_sha256"])
        self.assertEqual(digest.hexdigest(), self.inventory["bundle_code_sha256"])
        self.assertEqual(self.inventory["authorized_tabpfn_units"], 80)
        self.assertFalse(self.inventory["formal_run_manifest_authorized"])
        self.assertFalse(self.inventory["full_experiment_authorized"])


if __name__ == "__main__":
    unittest.main()

"""Read-only contract for the receipt-authorized v1.1 local cache set.

The test deliberately fails until the full, exact 160-unit local cache scope
has been generated.  It never fits a model, imports TabPFN, or writes an
artifact.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import numpy as np
import yaml
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_runner():
    spec = importlib.util.spec_from_file_location("stage08_v11_local_runner_for_test", ROOT / "src" / "run_stage05_predict.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_auditor():
    spec = importlib.util.spec_from_file_location("stage08_v11_local_auditor_for_test", ROOT / "src" / "audit_stage08_v11_local_cache_lineage.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage08V11LocalCacheLineageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = _load_runner()
        cls.auditor = _load_auditor()
        cls.local_lock_path = ROOT / "configs" / "stage05_lr_xgboost_v1.1.yaml"
        cls.split_lock_path = ROOT / "configs" / "stage04_splits_v1.1.yaml"
        cls.final_lock_path = ROOT / "configs" / "stage05b_tabpfn_v1.1.yaml"

    def temporary_test_root(self, prefix: str) -> Path:
        path = ROOT / "tmp" / f"{prefix}_{uuid.uuid4().hex}"
        self.addCleanup(shutil.rmtree, path, ignore_errors=True)
        return path

    def test_exact_v11_local_cache_lineage_and_no_disallowed_outputs(self) -> None:
        """Require every authorized LR/XGBoost unit and its complete lineage."""
        from conformal_uq.metrics import binary_predictive_metrics
        from conformal_uq.paths import cache_path
        from conformal_uq.prediction_cache import PARTITIONS, read_valid_cache
        from conformal_uq.provenance import sha256_path

        plan = self.runner.build_stage05_v11_execution_plan(ROOT, self.local_lock_path)
        lock = plan["lock"]
        expected_keys = {
            (dataset_id, seed, model)
            for dataset_id in lock["registry"]["locked_primary_ids"]
            for seed in lock["seeds"]
            for model in lock["authorized_local_models"]
        }
        cache_root = plan["cache_root"]
        # Scope discovery to the authorized local LR/XGBoost cfg/code tree.
        # The cloud TabPFN tree installed by the Task 6 intake audit under the
        # same v1.1 root is validated independently by the combined-tree checks
        # in tests/test_stage08_v11_cache_intake.py.
        expected_code_hash = self.auditor.cache_time_source_hash(ROOT, plan["config_hash"])
        local_tree = f"cfg-{plan['config_hash'][:12]}/code-{expected_code_hash[:12]}"
        manifests = list(cache_root.glob(f"{local_tree}/*/seed-*/*/manifest.json")) if cache_root.exists() else []
        observed_keys = {
            (path.parents[2].name, int(path.parents[1].name.removeprefix("seed-")), path.parent.name)
            for path in manifests
        }
        self.assertEqual(observed_keys, expected_keys)
        self.assertEqual(len(manifests), 160)
        self.assertEqual(sum(key[2] == "logistic_regression" for key in observed_keys), 80)
        self.assertEqual(sum(key[2] == "xgboost" for key in observed_keys), 80)

        expected_local_lock_hash = sha256_path(self.local_lock_path)
        expected_split_lock_hash = sha256_path(self.split_lock_path)
        expected_final_lock_hash = sha256_path(self.final_lock_path)
        expected_environment_hash = sha256_path(ROOT / "environment" / "environment_lock_v1.0.json")
        tables: dict[str, object] = {}
        for dataset_id, seed, model in sorted(expected_keys):
            with self.subTest(dataset_id=dataset_id, seed=seed, model=model):
                if dataset_id not in tables:
                    tables[dataset_id] = self.runner.load_locked_dataset(ROOT, dataset_id)
                table = tables[dataset_id]
                split = self.runner._locked_v11_split(ROOT, lock, table, seed)
                expected_ids = {"calibration_pool": split.ids.calibration_pool, "test": split.ids.test}
                expected_y = {
                    partition: table.subset_labels(ids).to_numpy(dtype="int8", copy=True)
                    for partition, ids in expected_ids.items()
                }
                destination = cache_path(cache_root, plan["config_hash"], expected_code_hash, dataset_id, seed, model)
                provenance = {
                    "config_hash": plan["config_hash"],
                    "code_hash": expected_code_hash,
                    "environment_hash": expected_environment_hash,
                    "dataset_hash": table.raw_sha256,
                    "split_hash": split.split_hash,
                    "model_name": model,
                    "base_seed": seed,
                    "label_mapping": table.label_mapping,
                    "class_labels": [0, 1],
                    "protocol_version": "v1.1",
                    "local_cache_lock_sha256": expected_local_lock_hash,
                    "split_lock_sha256": expected_split_lock_hash,
                    "d08_003_cache_lock_sha256": expected_final_lock_hash,
                }
                cached = read_valid_cache(destination, provenance, expected_ids, expected_y)
                self.assertEqual(cached["manifest"]["format_version"], "v1.1.0")
                self.assertEqual(cached["manifest"]["provenance"], provenance)
                self.assertEqual(cached["manifest"]["model_hash"].__len__(), 64)
                for partition in PARTITIONS:
                    probabilities = cached["probabilities"][partition]
                    self.assertTrue(np.isfinite(probabilities).all())
                    self.assertTrue(((probabilities >= 0.0) & (probabilities <= 1.0)).all())
                    self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6))
                    self.assertEqual(binary_predictive_metrics(cached["labels"][partition], probabilities), cached["manifest"]["metrics"][partition])

        # v1.1 local caching may not create CP/pilot/formal material inside the
        # cache tree, and no formal output may exist anywhere in the v1.1 scope.
        # (The D08-003-authorized 480-cell v1.1 pilot run under
        # artifacts/stage07_v1.1 is validated by its own contract tests.)
        forbidden_names = {"results_long.parquet", "results_long.csv", "formal_run_manifest.json"}
        forbidden = [path for path in cache_root.rglob("*") if path.is_file() and path.name in forbidden_names]
        self.assertEqual(forbidden, [])
        pilot_root = ROOT / "artifacts" / "stage07_v1.1"
        if pilot_root.exists():
            formal = [path for path in pilot_root.rglob("*") if path.is_file() and path.name == "formal_run_manifest.json"]
            self.assertEqual(formal, [])
        self.assertFalse((ROOT / "results" / "runs" / "v1.1").exists())

    def test_second_same_scope_run_rejects_before_data_model_cache_or_run_work(self) -> None:
        """A held v1.1 scope lock must reject a duplicate runner before side effects."""
        plan = self.runner.build_stage05_v11_execution_plan(ROOT, self.local_lock_path)
        with self.runner.exclusive_v11_local_cache_run_lock(ROOT, plan), patch.object(
            self.runner, "load_locked_dataset"
        ) as load_data, patch.object(self.runner, "fit_predict_locked_pipeline") as model_fit, patch.object(
            self.runner, "write_prediction_cache"
        ) as cache_writer, patch.object(self.runner, "create_immutable_run_dir") as run_dir, patch.object(
            sys, "argv", ["run_stage05_predict.py", "--root", str(ROOT), "--mode", "formal", "--lock", str(self.local_lock_path)]
        ):
            with self.assertRaisesRegex(self.runner.V11LocalCacheRunLockError, "already held"):
                self.runner.main()
        self.assertFalse(load_data.called)
        self.assertFalse(model_fit.called)
        self.assertFalse(cache_writer.called)
        self.assertFalse(run_dir.called)

    def test_auditor_uses_real_cache_layout_and_cache_time_source_hash(self) -> None:
        """Audit must not substitute its post-run source hash for cache provenance."""
        cache_dir = ROOT / "artifacts" / "caches" / "v1.1" / "cfg-40f29139c9db" / "code-cb25b48d6b1f" / "openml_3_kr_vs_kp" / "seed-104729" / "logistic_regression"
        self.assertEqual(
            self.auditor.cache_key_from_directory(cache_dir),
            ("openml_3_kr_vs_kp", 104729, "logistic_regression"),
        )
        with patch.object(self.auditor, "_code_hash", return_value="f" * 64):
            cache_time_hash = self.auditor.cache_time_source_hash(ROOT, "40f29139c9db63b2118c0efb28daa37940065a33dc52ec607b3e16bea0b786f9")
        self.assertEqual(cache_time_hash, "cb25b48d6b1f005f6de7bb13eb9f9dad8e789fbe7b1529b96851742b85a4eea6")

    def test_v11_run_identifier_is_unique_with_a_prior_immutable_run_directory(self) -> None:
        """A valid-cache resume must not collide with a prior same-second run directory."""
        run_root = self.temporary_test_root("stage08_v11_run_id_regression")
        with patch.object(self.runner, "run_id", return_value="20260831T000000Z_stage08-v11-local-cache_40f29139c9db"):
            prior_immutable_run = self.runner.v11_local_cache_run_identifier("40f29139c9db" + "0" * 52)
            prior_directory = self.runner.create_immutable_run_dir(run_root, prior_immutable_run)
            resumed_run = self.runner.v11_local_cache_run_identifier("40f29139c9db" + "0" * 52)
            resumed_directory = self.runner.create_immutable_run_dir(run_root, resumed_run)
        self.assertNotEqual(prior_immutable_run, resumed_run)
        self.assertTrue(prior_directory.is_dir())
        self.assertTrue(resumed_directory.is_dir())
        self.assertNotEqual(prior_directory, resumed_directory)
        cached = {"model_hash": "a" * 64, "manifest": {"metrics": {"calibration_pool": {"auroc": 0.5}, "test": {"auroc": 0.5}}}}
        with patch.object(self.runner, "read_valid_cache", return_value=cached) as cache_reader, patch.object(
            self.runner, "fit_predict_locked_pipeline"
        ) as model_fit, patch.object(self.runner, "write_prediction_cache") as cache_writer:
            model_hash, metrics, action = self.runner.reuse_or_write_v11_cache(
                Path("unused"), {}, {}, {}, None, None, "logistic_regression", {}, 1
            )
        self.assertTrue(cache_reader.called)
        self.assertEqual((model_hash, metrics, action), ("a" * 64, cached["manifest"]["metrics"], "reused_validated_v11_cache"))
        self.assertFalse(model_fit.called)
        self.assertFalse(cache_writer.called)

    def test_v11_same_second_failure_records_are_unique_and_immutable(self) -> None:
        """Retries of one v1.1 scope must never overwrite an earlier failure record."""
        root = self.temporary_test_root("stage08_v11_failure_regression")
        scope = {"dataset_id": "dataset_a", "seed": 1, "model": "logistic_regression"}
        fixed_now = self.runner.datetime.now(self.runner.timezone.utc)
        class FrozenDateTime:
            @staticmethod
            def now(_timezone):
                return fixed_now
        with patch.object(self.runner, "datetime", FrozenDateTime):
            first = self.runner._failure(root, scope, ValueError("first"), "20260831T000000Z_stage08-v11-local-cache_hash_aaaaaaaaaaaa", "c" * 64, v11_scope=True)
            second = self.runner._failure(root, scope, ValueError("second"), "20260831T000000Z_stage08-v11-local-cache_hash_bbbbbbbbbbbb", "c" * 64, v11_scope=True)
        self.assertNotEqual(first, second)
        self.assertTrue(first.is_file())
        self.assertTrue(second.is_file())
        self.assertEqual(json.loads(first.read_text(encoding="utf-8"))["exception"], "first")
        self.assertEqual(json.loads(second.read_text(encoding="utf-8"))["exception"], "second")

    def test_v11_runtime_failure_call_uses_v11_scope_evidence(self) -> None:
        """The real v1.1 loop must request the unique exclusive failure path."""
        run_root = self.temporary_test_root("stage08_v11_failure_call_regression")
        run_root.mkdir(parents=True)
        labels = SimpleNamespace(to_numpy=lambda **_kwargs: np.asarray([0, 1], dtype=np.int8))
        table = SimpleNamespace(raw_sha256="d" * 64, label_mapping={"negative": 0, "positive": 1}, subset_labels=lambda _ids: labels)
        split = SimpleNamespace(ids=SimpleNamespace(calibration_pool=("cal-0", "cal-1"), test=("test-0", "test-1")), split_hash="e" * 64)
        plan = {
            "base": {"datasets": {"primary_ids": ["dataset_a"]}, "experiment": {"seeds": [1]}, "protocol": {"protocol_version_for_seed_derivation": "v1.1"}},
            "lock": {"authorized_local_models": ["logistic_regression"], "model_hyperparameters": {"logistic_regression": {} }},
            "config_hash": "f" * 64,
            "authorization": {"final_lock_sha256": "a" * 64},
            "local_cache_lock_sha256": "b" * 64,
            "split_lock_sha256": "c" * 64,
            "cache_root": run_root / "cache-root",
            "unit_count": 1,
        }
        with patch.object(self.runner, "v11_local_cache_run_identifier", return_value="runtime-test"), patch.object(
            self.runner, "create_immutable_run_dir", return_value=run_root
        ), patch.object(self.runner, "write_event"), patch.object(self.runner, "load_locked_dataset", return_value=table), patch.object(
            self.runner, "_locked_v11_split", return_value=split
        ), patch.object(self.runner, "derive_seed", return_value=("canonical", 1)), patch.object(
            self.runner, "reuse_or_write_v11_cache", side_effect=ValueError("forced failure")
        ), patch.object(self.runner, "_failure", return_value=ROOT / "artifacts" / "failures" / "mock.json") as failure_writer:
            self.assertEqual(self.runner._run_stage05_v11_local_cache(ROOT, plan), 2)
        self.assertTrue(failure_writer.called)
        self.assertIs(failure_writer.call_args.kwargs["v11_scope"], True)

    def test_auditor_rejects_foreign_tree_and_incomplete_cache_directory(self) -> None:
        """Only one exact cfg/code cache tree with complete unit pairs is admissible."""
        config_hash, code_hash = "a" * 64, "b" * 64
        expected = {("dataset_a", 1, "logistic_regression")}
        entries = {
            PurePosixPath(f"cfg-{config_hash[:12]}/code-{code_hash[:12]}/dataset_a/seed-1/logistic_regression/manifest.json"),
            PurePosixPath(f"cfg-{config_hash[:12]}/code-{code_hash[:12]}/dataset_a/seed-1/logistic_regression/predictions.npz"),
            PurePosixPath("cfg-foreign/code-foreign/dataset_a/seed-1/xgboost/manifest.json"),
            PurePosixPath(f"cfg-{config_hash[:12]}/code-{code_hash[:12]}/dataset_a/seed-2/xgboost/predictions.npz"),
        }
        errors = self.auditor.validate_exact_v11_cache_relative_entries(entries, config_hash, code_hash, expected)
        self.assertTrue(any("foreign" in error for error in errors))
        self.assertTrue(any("incomplete" in error for error in errors))


if __name__ == "__main__":
    unittest.main()

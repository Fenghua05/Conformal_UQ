"""Static D08-003 authorization checks; no data, model, cloud, or output access."""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from conformal_uq.stage08_authorization import sha256_file as production_sha256_file
from conformal_uq.stage08_authorization import validate_d08_003_receipt

RECEIPT_PATH = ROOT / "decisions" / "D08-003_V11_CACHE_AND_PILOT_BUDGET_RECEIPT.json"
CACHE_LOCK_PATH = ROOT / "configs" / "stage05b_tabpfn_v1.1.yaml"


def sha256_file(path: Path) -> str:
    return production_sha256_file(path)


def validate_receipt(receipt: dict, cache_lock_sha256: str) -> None:
    if cache_lock_sha256 != sha256_file(CACHE_LOCK_PATH):
        raise ValueError("D08-003 cache-lock hash binding mismatch")
    validate_d08_003_receipt(receipt, CACHE_LOCK_PATH, yaml.safe_load(CACHE_LOCK_PATH.read_text(encoding="utf-8")))


class D08003ReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
        cls.cache_lock_sha256 = sha256_file(CACHE_LOCK_PATH)

    def test_receipt_has_exact_scope_limits_and_hash_binding(self) -> None:
        validate_receipt(self.receipt, self.cache_lock_sha256)
        self.assertEqual(self.receipt["maximum_wall_clock_hours"], 12)
        self.assertEqual(self.receipt["maximum_cloud_storage_gb"], 50)
        self.assertIn("80 TabPFN", self.receipt["scope"])
        self.assertIn("160 local LR/XGBoost", self.receipt["scope"])
        self.assertIn("480-cell v1.1 pilot", self.receipt["scope"])
        prohibitions = " ".join(self.receipt["prohibitions"])
        self.assertIn("formal-run manifest", prohibitions)
        self.assertIn("eight-dataset formal experiment", prohibitions)
        self.assertNotIn("D08-002", json.dumps(self.receipt, sort_keys=True))

    def test_rejects_missing_or_non_positive_numeric_limits(self) -> None:
        for field, value in (
            ("maximum_wall_clock_hours", None),
            ("maximum_wall_clock_hours", 0),
            ("maximum_cloud_storage_gb", None),
            ("maximum_cloud_storage_gb", -1),
        ):
            with self.subTest(field=field, value=value):
                mutated = deepcopy(self.receipt)
                if value is None:
                    mutated.pop(field)
                else:
                    mutated[field] = value
                with self.assertRaisesRegex(ValueError, field):
                    validate_receipt(mutated, self.cache_lock_sha256)

    def test_rejects_missing_or_wrong_mandatory_status(self) -> None:
        for value in (None, "APPROVED_FOR_STAGE08_V11_FULL_EXPERIMENT"):
            with self.subTest(value=value):
                mutated = deepcopy(self.receipt)
                if value is None:
                    mutated.pop("status")
                else:
                    mutated["status"] = value
                with self.assertRaisesRegex(ValueError, "status"):
                    validate_receipt(mutated, self.cache_lock_sha256)

    def test_rejects_missing_or_wrong_unit_counts_protocol_or_formal_authority(self) -> None:
        cases = (
            ("authorized_tabpfn_units", 79),
            ("authorized_local_lr_xgboost_units", 159),
            ("authorized_cache_intake_units", 239),
            ("authorized_pilot_cells", 479),
            ("protocol_version", "v1.0"),
            ("formal_run_manifest_authorized", True),
            ("full_experiment_authorized", True),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value):
                mutated = deepcopy(self.receipt)
                mutated[field] = value
                with self.assertRaises(ValueError):
                    validate_receipt(mutated, self.cache_lock_sha256)

        for field in (
            "authorized_tabpfn_units",
            "authorized_local_lr_xgboost_units",
            "authorized_cache_intake_units",
            "authorized_pilot_cells",
        ):
            with self.subTest(missing_field=field):
                mutated = deepcopy(self.receipt)
                mutated.pop(field)
                with self.assertRaisesRegex(ValueError, field):
                    validate_receipt(mutated, self.cache_lock_sha256)

        for field in ("formal_run_manifest_authorized", "full_experiment_authorized"):
            with self.subTest(missing_field=field):
                mutated = deepcopy(self.receipt)
                mutated.pop(field)
                with self.assertRaises(ValueError):
                    validate_receipt(mutated, self.cache_lock_sha256)

    def test_rejects_missing_or_wrong_cache_lock_and_binding_fields(self) -> None:
        for field, value in (
            ("cache_lock_path", None),
            ("cache_lock_path", "configs/stage05b_tabpfn_v1.0.yaml"),
            ("cache_lock_sha256", None),
            ("cache_lock_sha256", "0" * 64),
        ):
            with self.subTest(field=field, value=value):
                mutated = deepcopy(self.receipt)
                if value is None:
                    mutated.pop(field)
                else:
                    mutated[field] = value
                with self.assertRaises(ValueError):
                    validate_receipt(mutated, self.cache_lock_sha256)

        for field, value in (
            ("receipt_to_lock_binding", None),
            ("receipt_path", None),
            ("cache_lock_path", "incorrect-lock-path"),
            ("cache_lock_sha256", "0" * 64),
        ):
            with self.subTest(binding_field=field, value=value):
                mutated = deepcopy(self.receipt)
                if field == "receipt_to_lock_binding":
                    mutated.pop(field)
                elif value is None:
                    mutated["receipt_to_lock_binding"].pop(field)
                else:
                    mutated["receipt_to_lock_binding"][field] = value
                with self.assertRaises(ValueError):
                    validate_receipt(mutated, self.cache_lock_sha256)

    def test_rejects_missing_or_mutated_runtime_device_and_checkpoint_fields(self) -> None:
        for field, value in (
            ("device", None),
            ("device", "cpu"),
            ("checkpoint_path", None),
            ("checkpoint_path", "/different/checkpoint.ckpt"),
            ("checkpoint_sha256", None),
            ("checkpoint_sha256", "0" * 64),
        ):
            with self.subTest(field=field, value=value):
                mutated = deepcopy(self.receipt)
                if value is None:
                    mutated["immutable_runtime_inputs"].pop(field)
                else:
                    mutated["immutable_runtime_inputs"][field] = value
                with self.assertRaises(ValueError):
                    validate_receipt(mutated, self.cache_lock_sha256)


if __name__ == "__main__":
    unittest.main()

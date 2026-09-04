import json
import sys
import unittest
import uuid
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conformal_uq.config import FROZEN_SEEDS
from conformal_uq.tabpfn_cloud import CloudLockError, align_protocol_probabilities, expected_tabpfn_units, load_approved_stage05b_lock

sys.path.insert(0, str(ROOT / "cloud" / "tabpfn_stage05b"))
from stage05b_common import validate_unit_inventory


class TabPFNCloudContractTests(unittest.TestCase):
    def _write_lock(self, root: Path, *, device: str = "cuda") -> tuple[Path, Path]:
        decision = root / "pilot_decision.json"
        decision.write_text(json.dumps({"status": "APPROVED_PRE_OUTCOME_PILOT_DECISION", "protocol_version": "v1.0", "pilot_dataset_ids": ["dataset_a", "dataset_b"]}), encoding="utf-8")
        lock = root / "stage05b.yaml"
        lock.write_text(yaml.safe_dump({"artifact_status": "APPROVED_FOR_STAGE05B", "protocol_version": "v1.0", "pilot_dataset_ids": ["dataset_a", "dataset_b"], "seeds": FROZEN_SEEDS, "runtime": {"device": device, "tabpfn_version": "9.9.9", "checkpoint_path": "/models/classifier.ckpt", "checkpoint_sha256": "a" * 64, "context_limit": 1000, "constructor_kwargs": {"n_estimators": 8}, "preprocessing_contract": "stage04_train_only_unscaled_onehot_dense", "ignore_pretraining_limits": False}}), encoding="utf-8")
        return lock, decision

    def test_lock_requires_approved_pilot_decision_and_cuda(self) -> None:
        root = ROOT / "artifacts" / f"stage05b_test_lock_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        lock, decision = self._write_lock(root)
        self.assertEqual(load_approved_stage05b_lock(lock, decision)["runtime"]["device"], "cuda")
        lock, decision = self._write_lock(root, device="cpu")
        with self.assertRaisesRegex(CloudLockError, "CUDA"):
            load_approved_stage05b_lock(lock, decision)

    def test_probability_alignment_reorders_tabpfn_classes(self) -> None:
        actual = align_protocol_probabilities([1, 0], np.array([[0.9, 0.1], [0.2, 0.8]]))
        np.testing.assert_allclose(actual, np.array([[0.1, 0.9], [0.8, 0.2]]))

    def test_expected_grid_is_exactly_twenty_unique_units(self) -> None:
        units = expected_tabpfn_units(["dataset_a", "dataset_b"], FROZEN_SEEDS)
        self.assertEqual(len(units), 20)
        self.assertEqual(len(set(units)), 20)

    def test_inventory_rejects_missing_or_extra_units(self) -> None:
        expected = expected_tabpfn_units(["dataset_a", "dataset_b"], FROZEN_SEEDS)
        with self.assertRaisesRegex(CloudLockError, "exactly 20"):
            validate_unit_inventory(list(expected[:-1]), expected)


if __name__ == "__main__":
    unittest.main()

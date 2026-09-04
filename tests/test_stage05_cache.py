import sys
import unittest
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conformal_uq.prediction_cache import read_valid_cache, write_prediction_cache


class Stage05CacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ids = {"calibration_pool": ("d:0001", "d:0002"), "test": ("d:0003", "d:0004")}
        self.y = {"calibration_pool": np.array([0, 1], dtype=np.int8), "test": np.array([0, 1], dtype=np.int8)}
        self.probabilities = {
            "calibration_pool": np.array([[0.8, 0.2], [0.1, 0.9]]),
            "test": np.array([[0.7, 0.3], [0.2, 0.8]]),
        }
        self.provenance = {"config_hash": "a" * 64, "code_hash": "b" * 64, "environment_hash": "c" * 64, "dataset_hash": "d" * 64, "split_hash": "e" * 64, "model_name": "logistic_regression", "base_seed": 104729, "label_mapping": {"negative": 0, "positive": 1}, "class_labels": [0, 1]}

    def test_cache_round_trip_and_safe_reuse(self) -> None:
        cache_dir = ROOT / "artifacts" / f"stage05_test_cache_{uuid.uuid4().hex}"
        manifest = write_prediction_cache(cache_dir, self.provenance, self.ids, self.y, self.probabilities, "f" * 64, {"test": {"auroc": 1.0, "auprc": 1.0}})
        loaded = read_valid_cache(cache_dir, self.provenance, self.ids, self.y)
        self.assertEqual(loaded["model_hash"], "f" * 64)
        self.assertEqual(manifest["qc_status"], "PASS")
        np.testing.assert_allclose(loaded["probabilities"]["test"], self.probabilities["test"])

    def test_cache_rejects_probability_or_order_tampering(self) -> None:
        cache_dir = ROOT / "artifacts" / f"stage05_test_cache_{uuid.uuid4().hex}"
        write_prediction_cache(cache_dir, self.provenance, self.ids, self.y, self.probabilities, "f" * 64, {})
        tampered = dict(self.ids)
        tampered["test"] = tuple(reversed(tampered["test"]))
        with self.assertRaises(ValueError):
            read_valid_cache(cache_dir, self.provenance, tampered, self.y)


if __name__ == "__main__":
    unittest.main()

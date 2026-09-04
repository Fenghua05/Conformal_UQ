import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conformal_uq.config import ConfigError, load_config, validate_config
from conformal_uq.toy import toy_results_record
from conformal_uq.results_schema import validate_results_record


class ConfigTests(unittest.TestCase):
    def test_frozen_values_and_toy_results_contract(self) -> None:
        config = load_config(ROOT / "configs" / "stage03_base_v1.0.yaml")
        self.assertEqual(config["experiment"]["alpha"], 0.1)
        self.assertEqual(config["experiment"]["m_minority"], [10, 20, 50, 100])
        self.assertEqual(config["experiment"]["m_majority"], 200)
        self.assertEqual(len(config["experiment"]["seeds"]), 10)
        self.assertEqual(validate_results_record(toy_results_record("a" * 64, "b" * 64, "toy")), [])

    def test_rejects_protocol_change(self) -> None:
        config = load_config(ROOT / "configs" / "stage03_base_v1.0.yaml")
        config["experiment"]["alpha"] = 0.2
        with self.assertRaises(ConfigError):
            validate_config(config)

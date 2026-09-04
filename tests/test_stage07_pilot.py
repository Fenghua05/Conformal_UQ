import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "cloud" / "tabpfn_stage05b")]

from stage05b_common import cache_provenance, checked_lock, expected_units_from_lock, locked_table_and_split
from conformal_uq.prediction_cache import read_valid_cache
from conformal_uq.provenance import sha256_path


class Stage07InputTests(unittest.TestCase):
    def test_imported_tabpfn_cache_inventory_is_exact_and_valid(self) -> None:
        lock, config_hash, _guard_code_hash = checked_lock(
            ROOT,
            ROOT / "configs/stage05b_tabpfn_v1.0.yaml",
            ROOT / "decisions/pilot_decision_stage07_v1.0.json",
        )
        expected = expected_units_from_lock(lock)
        actual = []
        for dataset_id, seed, model in expected:
            table, split = locked_table_and_split(ROOT, dataset_id, seed, registry_path=ROOT / lock["registry_path"])
            candidates = list((ROOT / "artifacts" / "caches").glob(f"cfg-{config_hash[:12]}/code-*/{dataset_id}/seed-{seed}/{model}"))
            self.assertEqual(len(candidates), 1)
            cache_dir = candidates[0]
            provenance = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))["provenance"]
            self.assertEqual(provenance["config_hash"], config_hash)
            read_valid_cache(
                cache_dir,
                provenance,
                {"calibration_pool": split.ids.calibration_pool, "test": split.ids.test},
                {"calibration_pool": table.subset_labels(split.ids.calibration_pool).to_numpy(dtype="int8"), "test": table.subset_labels(split.ids.test).to_numpy(dtype="int8")},
            )
            actual.append((dataset_id, seed, model))
        self.assertEqual(actual, list(expected))


class Stage07RunnerContractTests(unittest.TestCase):
    def test_expected_pilot_grid_is_exact(self) -> None:
        from run_stage07_pilot import expected_pilot_cells

        cells = expected_pilot_cells(
            ("openml_3_kr_vs_kp", "openml_24_mushroom"),
            (104729, 130363, 155921, 196613, 262147, 318281, 374209, 419893, 481517, 552721),
        )
        self.assertEqual(len(cells), 480)
        self.assertEqual(len(set(cells)), 480)
        self.assertEqual({cell[2] for cell in cells}, {"logistic_regression", "xgboost", "tabpfn"})


class Stage07QCContractTests(unittest.TestCase):
    def test_qc_declares_exactly_four_diagnostic_figure_types(self) -> None:
        from conformal_uq.stage07_qc import DIAGNOSTIC_FIGURES

        self.assertEqual(
            DIAGNOSTIC_FIGURES,
            ("overall_coverage", "classwise_coverage", "set_decomposition", "threshold_geometry"),
        )


class Stage07RegressionTests(unittest.TestCase):
    def test_wilson_and_geometry_handle_float_probability_rows(self) -> None:
        from conformal_uq.conformal import binary_geometry_categories
        from conformal_uq.metrics import wilson_interval

        self.assertLessEqual(wilson_interval(783, 783)[1], 1.0)
        categories = binary_geometry_categories(
            [[0.999987006187439, 0.000013013937859795988]],
            0.000012993812561035156,
            0.000012993812561035156,
        )
        self.assertEqual(categories.tolist(), ["singleton"])


if __name__ == "__main__":
    unittest.main()

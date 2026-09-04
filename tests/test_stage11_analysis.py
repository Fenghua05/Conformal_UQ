"""Stage 11 analysis contract tests.

Unit tests run on synthetic complete grids; frozen-data tests are skipped when
the frozen results are not present. The frozen tests never write anywhere.
"""
import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import run_stage11_analysis as s11  # noqa: E402

RESULTS_LONG = ROOT / "results/results_long.parquet"
STAGE10_EVIDENCE = ROOT / s11.STAGE10_EVIDENCE_REL


def synthetic_grid() -> pd.DataFrame:
    """Complete synthetic grid 8 datasets x 10 seeds x 3 models x 2 cp x 4 m with
    a deterministic metric formula, so every paired effect is hand-checkable."""
    rows = []
    for di, d in enumerate(s11.DATASETS):
        for si, s in enumerate(s11.SEEDS):
            for mi, mdl in enumerate(s11.MODELS):
                for ci, cp in enumerate(s11.CP_METHODS):
                    for m in s11.M_VALUES:
                        base = di + 0.1 * si + 0.01 * mi + 0.0001 * m
                        rows.append({
                            "dataset_id": d, "seed": s, "model": mdl,
                            "cp_method": cp, "m_minority": m,
                            "singleton_rate": base,
                            "average_set_size": base,
                            "coverage_minority": base,
                            "coverage_majority": base,
                            "coverage_disparity": base,
                            "empty_rate": base,
                            "doubleton_rate": base,
                            "threshold_sum": base + (0.001 if cp == "class_conditional_cp" else 0.0),
                            "threshold_gap": base,
                            "q_minority": base,
                            "q_majority": base,
                            "q_global": base,
                            "coverage_overall": base,
                            "auroc": base,
                            "auprc": base,
                        })
    return pd.DataFrame(rows)


class SeedConventionTests(unittest.TestCase):
    def test_d01_seed_matches_stage10_recorded_value(self) -> None:
        # Stage 10 evidence records seed_uint32 = 3463499878 for this endpoint.
        expected = int(hashlib.sha256(
            b"v1.1|EIGHT_DATASET|0|d08_bootstrap_rq1a_cc_m100_minus_m50_singleton_rate"
        ).hexdigest()[:8], 16)
        self.assertEqual(expected, 3463499878)
        self.assertEqual(s11.d01_seed("rq1a_cc_m100_minus_m50_singleton_rate"),
                         3463499878)

    def test_d08_ci_deterministic_and_matches_manual_replication(self) -> None:
        rng = np.random.default_rng(7)
        d = rng.normal(size=8)
        eff1, lo1, hi1 = s11.d08_effect_ci(d, "unit_test_endpoint")
        eff2, lo2, hi2 = s11.d08_effect_ci(d, "unit_test_endpoint")
        self.assertEqual((eff1, lo1, hi1), (eff2, lo2, hi2))
        r = np.random.default_rng(s11.d01_seed("unit_test_endpoint"))
        idx = r.choice(8, size=(20000, 8), replace=True)
        boots = np.median(d[idx], axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        self.assertEqual((lo1, hi1), (float(lo), float(hi)))
        self.assertEqual(eff1, float(np.median(d)))

    def test_across_seed_bootstrap_deterministic(self) -> None:
        v = np.arange(10, dtype=float)
        lo1, hi1 = s11.across_seed_bootstrap_ci(v, "tag1")
        lo2, hi2 = s11.across_seed_bootstrap_ci(v, "tag1")
        self.assertEqual((lo1, hi1), (lo2, hi2))
        lo3, _ = s11.across_seed_bootstrap_ci(v, "tag2")
        self.assertNotEqual((lo1, hi1), (lo3, hi2))


class InferenceTests(unittest.TestCase):
    def test_wilcoxon_exact_all_positive_n8(self) -> None:
        p, n_zero = s11.wilcoxon_exact(np.array([0.01, 0.02, 0.03, 0.04,
                                                 0.05, 0.06, 0.07, 0.08]))
        self.assertEqual(n_zero, 0)
        self.assertAlmostEqual(p, 2.0 / 256.0, places=12)

    def test_wilcoxon_zero_discarding(self) -> None:
        p, n_zero = s11.wilcoxon_exact(np.array([0.0, 0.0, 0.01, 0.02,
                                                 -0.03, 0.04, 0.05, 0.06]))
        self.assertEqual(n_zero, 2)
        self.assertFalse(np.isnan(p))
        # all-zero effects: p NA and all zeros reported
        p0, n0 = s11.wilcoxon_exact(np.zeros(8))
        self.assertEqual(n0, 8)
        self.assertTrue(np.isnan(p0))

    def test_holm_known_values(self) -> None:
        adj = s11.holm_adjust([0.01, 0.04, 0.03])
        self.assertEqual(adj, [0.03, 0.06, 0.06])
        self.assertEqual(s11.holm_adjust([]), [])
        self.assertEqual(s11.holm_adjust([0.9]), [0.9])

    def test_holm_monotone_and_clipped(self) -> None:
        adj = s11.holm_adjust([0.0001, 0.0002, 0.5, 0.9])
        self.assertAlmostEqual(adj[0], 0.0004, places=15)
        self.assertAlmostEqual(adj[1], 0.0006, places=15)
        self.assertEqual(adj[2], 1.0)
        self.assertEqual(adj[3], 1.0)


class AggregationUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.piv = synthetic_grid().sort_values(s11.KEY_COLS).set_index(s11.KEY_COLS)

    def test_paired_effects_unit_is_dataset_not_seed(self) -> None:
        d, dsets, per_pipeline = s11.paired_dataset_effects(
            self.piv, "average_set_size", "class_conditional_cp", 100,
            "class_conditional_cp", 50)
        self.assertEqual(len(d), 8)
        self.assertEqual(len(dsets), 8)
        self.assertEqual(dsets, s11.DATASETS)
        # contrast value is identical (0.0001*50) for every seed x pipeline pair,
        # so every d_j must equal exactly 0.005 — and there are 8 effects, not 80.
        self.assertTrue(np.allclose(d, 0.0001 * 50))
        for ds in dsets:
            for mdl in s11.MODELS:
                self.assertAlmostEqual(per_pipeline[ds][mdl], 0.005, places=15)

    def test_b_contrast_hand_checked(self) -> None:
        d, dsets, _ = s11.paired_dataset_effects(
            self.piv, "threshold_sum", "class_conditional_cp", 50,
            "global_split_cp", 50)
        # threshold_sum formula adds 0.001 exactly for CC rows -> d_j = 0.001
        self.assertTrue(np.allclose(d, 0.001))

    def test_pipeline_pair_effects_unit(self) -> None:
        d, dsets = s11.pipeline_pair_dataset_effects(
            self.piv, "singleton_rate", "class_conditional_cp", 50,
            "xgboost", "logistic_regression")
        self.assertEqual(len(d), 8)
        # model term 0.01*mi: xgboost(mi=1) - lr(mi=0) = 0.01 for every seed
        self.assertTrue(np.allclose(d, 0.01))


class EndpointFlagTests(unittest.TestCase):
    def test_inference_status_flags(self) -> None:
        d = np.array([0.01, -0.02, 0.03, 0.04, -0.05, 0.06, 0.07, 0.08])
        rec_c = s11.endpoint_record("rq3c_unit", "singleton_rate", "cmp", "C_exploratory",
                                    d, s11.DATASETS[:8], None, False,
                                    "exploratory_not_preregistered")
        rec_a = s11.endpoint_record("rq1a_unit", "singleton_rate", "cmp", "A_confirmatory",
                                    d, s11.DATASETS[:8], None, True,
                                    "confirmatory_preregistered")
        self.assertFalse(rec_c["preregistered_endpoint"])
        self.assertEqual(rec_c["inference_status"], "exploratory_not_preregistered")
        self.assertTrue(rec_a["preregistered_endpoint"])
        self.assertEqual(rec_a["inference_status"], "confirmatory_preregistered")

    def test_family_definitions(self) -> None:
        self.assertEqual(len(s11.FAMILY_A), 2)
        self.assertEqual(len(s11.FAMILY_B), 10)
        self.assertEqual(len(s11.FAMILY_A_SECONDARY), 6)
        # C: 3 pairs x 2 m x (7 global + 8 cc metrics)
        self.assertEqual(len(s11.FAMILY_C), 90)
        # confirmatory families only contain the preregistered Stage 10 tags
        stage10_tags = {
            "rq1a_cc_m100_minus_m50_singleton_rate",
            "rq1a_cc_m100_minus_m50_average_set_size",
        } | {f"rq2b_cc_minus_global_m{m}_{k}" for m in (50, 100)
             for k in ["coverage_minority", "coverage_majority", "coverage_disparity",
                       "singleton_rate", "average_set_size"]}
        self.assertEqual({t for t, *_ in s11.FAMILY_A + s11.FAMILY_B}, stage10_tags)


class ManifestClosureTests(unittest.TestCase):
    def _build_manifest(self, payload_v1: bytes, self_sha: str, self_bytes: int) -> bytes:
        entry = (',\r\n    "results/results_manifest.json": {\r\n'
                 f'      "sha256": "{self_sha}",\r\n      "bytes": {self_bytes}\r\n    }}')
        return payload_v1 + entry.encode("utf-8")

    def test_closure_accepts_v1_plus_self_entry(self) -> None:
        v1 = b'{"a": 1,\r\n  "outputs": {\r\n    "x": 2\r\n  }\r\n}'
        closure = s11.verify_manifest_self_closure(
            self._build_manifest(v1, s11.sha256_bytes(v1), len(v1)))
        self.assertTrue(closure["closure_ok"])
        self.assertEqual(closure["v1_sha256_reconstructed"], s11.sha256_bytes(v1))

    def test_closure_detects_tampering(self) -> None:
        v1 = b'{"a": 1,\r\n  "outputs": {\r\n    "x": 2\r\n  }\r\n}'
        tampered = v1.replace(b'"a": 1', b'"a": 999')
        closure = s11.verify_manifest_self_closure(
            self._build_manifest(tampered, s11.sha256_bytes(v1), len(v1)))
        self.assertFalse(closure["closure_ok"])

    def test_closure_detects_wrong_recorded_hash(self) -> None:
        v1 = b'{"a": 1,\r\n  "outputs": {\r\n    "x": 2\r\n  }\r\n}'
        closure = s11.verify_manifest_self_closure(
            self._build_manifest(v1, "0" * 64, len(v1)))
        self.assertFalse(closure["closure_ok"])


class DescriptiveTests(unittest.TestCase):
    def test_seed_summary_known_values(self) -> None:
        s = s11.seed_summary(np.array([1.0, 2.0, 3.0, 4.0]))
        self.assertEqual(s["n"], 4)
        self.assertAlmostEqual(s["mean"], 2.5)
        self.assertAlmostEqual(s["median"], 2.5)
        self.assertAlmostEqual(s["sd_ddof1"], np.std([1, 2, 3, 4], ddof=1), places=15)
        self.assertAlmostEqual(s["iqr"], 1.5, places=15)

    def test_descriptives_run_on_synthetic_grid_and_flag_exploratory(self) -> None:
        # stub the bootstrap to keep the unit test fast and deterministic
        orig = s11.across_seed_bootstrap_ci
        s11.across_seed_bootstrap_ci = (
            lambda v, tag, n_boot=0: (float(np.mean(v) - 1.0), float(np.mean(v) + 1.0)))
        try:
            rows, tvar = s11.run_descriptives(synthetic_grid())
        finally:
            s11.across_seed_bootstrap_ci = orig
        # 96 CC cells x 14 metrics + 96 Global cells x 11 metrics
        self.assertEqual(len(rows), 96 * 14 + 96 * 11)
        # threshold variability: 96 CC cells x 3 metrics + 96 Global cells x 1
        self.assertEqual(len(tvar), 96 * 3 + 96 * 1)
        self.assertTrue(all(r["uncertainty_type"] == "across_seed_bootstrap_exploratory"
                            for r in rows))


@unittest.skipUnless(RESULTS_LONG.exists() and STAGE10_EVIDENCE.exists(),
                     "frozen results or Stage 10 evidence not present")
class FrozenReproductionTests(unittest.TestCase):
    def test_stage10_endpoint_reproduced_exactly(self) -> None:
        df = pd.read_parquet(RESULTS_LONG)
        piv = df.sort_values(s11.KEY_COLS).set_index(s11.KEY_COLS)
        ev = json.loads(STAGE10_EVIDENCE.read_text(encoding="utf-8"))
        ref = {e["endpoint_tag"]: e
               for e in ev["d08_confirmatory_endpoints_report_only"]}
        for tag, metric, a_sel, b_sel in s11.FAMILY_A + s11.FAMILY_B:
            d, dsets, _ = s11.paired_dataset_effects(
                piv, metric, a_sel[0], a_sel[1], b_sel[0], b_sel[1])
            eff, lo, hi = s11.d08_effect_ci(d, tag)
            r = ref[tag]
            self.assertEqual(eff, r["effect_median_dj"], tag)
            self.assertEqual(lo, r["ci95_low"], tag)
            self.assertEqual(hi, r["ci95_high"], tag)
            self.assertEqual(int((d > 0).sum()), r["direction_count"]["positive"], tag)
            self.assertEqual(int((d < 0).sum()), r["direction_count"]["negative"], tag)

    def test_frozen_grid_shape(self) -> None:
        df = pd.read_parquet(RESULTS_LONG)
        self.assertEqual(len(df), 1920)
        keys = df[s11.KEY_COLS].drop_duplicates()
        self.assertEqual(len(keys), 1920)


if __name__ == "__main__":
    unittest.main()

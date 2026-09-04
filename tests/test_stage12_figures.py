"""Stage 12 contract tests: figures/tables must be reproducible from frozen data.

Unit tests on a synthetic contract-shaped grid; frozen-data tests are skipped
when the frozen results or the Stage 12 outputs are absent. Frozen-data tests
never write anywhere.
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

import run_stage12_figures as s12  # noqa: E402
import stage12_style as st  # noqa: E402

RESULTS_LONG = ROOT / "results/results_long.parquet"
STAGE12_ROOT = ROOT / "artifacts/stage12"
FIG_DIR = ROOT / "manuscript/figures"
TABLE_DIR = ROOT / "manuscript/tables"


def latest_run_dir() -> Path | None:
    if not STAGE12_ROOT.exists():
        return None
    runs = sorted(p for p in STAGE12_ROOT.iterdir() if p.is_dir())
    return runs[-1] if runs else None


def synthetic_grid() -> pd.DataFrame:
    """Complete 8x10x3x2x4 grid with a deterministic per-cell formula."""
    rows = []
    for di, d in enumerate(st.DATASETS):
        for si, s in enumerate(st.SEEDS):
            for mi, mdl in enumerate(st.MODELS):
                for cp in st.CP_METHODS:
                    for m in st.M_VALUES:
                        base = di + 0.01 * si + 0.001 * mi + 0.0001 * m
                        rows.append({
                            "dataset_id": d, "seed": s, "model": mdl,
                            "cp_method": cp, "m_minority": m,
                            "singleton_rate": base, "coverage_minority": base,
                            "q_minority": base, "q_majority": base,
                            "q_global": base, "threshold_sum": base,
                            "auroc": base, "auprc": base, "status": "PASS",
                            "alpha": 0.1, "m_majority": 200,
                            "protocol_version": "v1.1", "doubleton_rate": base,
                            "coverage_majority": base, "coverage_overall": base,
                            "threshold_gap": base, "average_set_size": base,
                        })
    return pd.DataFrame(rows)


class TestAggregationContract(unittest.TestCase):
    def test_dataset_level_is_unit_of_inference(self):
        """Dataset value = mean over 3 models x 10 seeds (30 cells); not 80 seed units."""
        df = synthetic_grid()
        for cp in st.CP_METHODS:
            for m in st.M_VALUES:
                s = s12.dataset_level_series(df, "singleton_rate", cp, m)
                self.assertEqual(list(s.index), st.DATASETS)
                for di, d in enumerate(st.DATASETS):
                    expected = di + 0.01 * np.mean(np.arange(len(st.SEEDS))) + 0.001 * np.mean(np.arange(len(st.MODELS))) + 0.0001 * m
                    self.assertAlmostEqual(float(s[d]), expected, places=12)

    def test_dataset_level_series_requires_complete_groups(self):
        df = synthetic_grid()
        incomplete = df[df["model"] != "xgboost"]  # 20 cells per group, contract requires 30
        with self.assertRaises(SystemExit):
            s12.dataset_level_series(incomplete, "singleton_rate", "global_split_cp", 10)


@unittest.skipUnless(RESULTS_LONG.exists(), "frozen results not present")
class TestFrozenOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_dir = latest_run_dir()
        cls.have_outputs = cls.run_dir is not None and FIG_DIR.exists() and TABLE_DIR.exists()
        if cls.have_outputs:
            cls.df = pd.read_parquet(RESULTS_LONG)

    def test_status_and_qa_verdict_pass(self):
        if not self.have_outputs:
            self.skipTest("stage12 outputs not present")
        status = json.loads((self.run_dir / "stage12_status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "PASS")
        qa = json.loads((self.run_dir / "stage12_qa_evidence.json").read_text(encoding="utf-8"))
        self.assertEqual(qa["verdict"], "PASS")
        self.assertTrue(all(c["status"] == "PASS" for c in qa["checks"]))

    def test_fig_source_data_reproduces_from_frozen_parquet(self):
        if not self.have_outputs:
            self.skipTest("stage12 outputs not present")
        for n in (1, 2, 3):
            spec = s12.FIG_SPECS[n]
            f = FIG_DIR / spec["dir"] / f"{spec['stem']}_source_data.csv"
            src = pd.read_csv(f)
            self.assertEqual(len(src), 8 * len(spec["cps"]) * 4)
            for _, r in src.iterrows():
                sub = self.df[(self.df["dataset_id"] == r["dataset_id"])
                              & (self.df["cp_method"] == r["cp_method"])
                              & (self.df["m_minority"] == r["m_minority"])]
                self.assertEqual(len(sub), 30)
                recomputed = float(sub[spec["metric"]].sum() / 30)
                self.assertLessEqual(abs(recomputed - float(r["value"])), 1e-12)

    def test_fig4_source_matches_stage11_table(self):
        if not self.have_outputs:
            self.skipTest("stage12 outputs not present")
        f4 = pd.read_csv(FIG_DIR / "figure4_threshold_variability/figure4_source_data.csv")
        s11 = pd.read_csv(ROOT / "results/stats/stage11_threshold_variability.csv")
        s11 = s11[s11["threshold_metric"] == "q_minority"]
        mg = f4.merge(s11, on=["dataset_id", "model", "m_minority"], suffixes=("_f4", "_s11"))
        self.assertEqual(len(mg), 96)
        for col in ("across_seed_sd", "across_seed_iqr", "across_seed_mean"):
            self.assertLessEqual(float((mg[f"{col}_f4"] - mg[f"{col}_s11"]).abs().max()), 1e-12)

    def test_table2_invariance_and_means(self):
        if not self.have_outputs:
            self.skipTest("stage12 outputs not present")
        t2 = pd.read_csv(TABLE_DIR / "table2_base_predictive_performance.csv")
        for ds in st.DATASETS:
            for model in st.MODELS:
                sub = self.df[(self.df["dataset_id"] == ds) & (self.df["model"] == model)]
                for metric in ("auroc", "auprc"):
                    spread = float(sub.groupby("seed")[metric].agg(lambda v: v.max() - v.min()).max())
                    self.assertLessEqual(spread, 1e-12, f"{ds}/{model}/{metric} not invariant across CP x m")
                    mu = float(t2[(t2["dataset_id"] == ds) & (t2["model"] == model)
                                  & (t2["metric"] == metric)]["mean_across_seeds"].iloc[0])
                    self.assertLessEqual(abs(mu - float(sub[metric].mean())), 1e-12)

    def test_table1_matches_registry_and_split_manifests(self):
        if not self.have_outputs:
            self.skipTest("stage12 outputs not present")
        t1 = pd.read_csv(TABLE_DIR / "table1_dataset_characteristics.csv")
        registry = s12.load_registry()
        splits = s12.load_split_manifests()
        for ds in st.DATASETS:
            r = registry[ds]
            row = t1[t1["dataset_id"] == ds].iloc[0]
            self.assertAlmostEqual(
                float(row["minority_ratio"]),
                r["raw_class_counts"][r["minority_original_label"]] / r["n_rows"], places=12)
            for split, split_key in (("train", "train"), ("cal_pool", "calibration_pool"), ("test", "test")):
                for cls in ("majority", "minority"):
                    vals = [splits[ds][seed]["class_counts"][split_key][cls] for seed in st.SEEDS]
                    self.assertEqual(float(row[f"{split}_{cls}_mean"]), float(np.mean(vals)))
                    self.assertEqual(int(row[f"{split}_{cls}_min"]), int(np.min(vals)))
                    self.assertEqual(int(row[f"{split}_{cls}_max"]), int(np.max(vals)))

    def test_table3_by_dataset_reproduces_from_frozen_parquet(self):
        if not self.have_outputs:
            self.skipTest("stage12 outputs not present")
        t3 = pd.read_csv(TABLE_DIR / "table3_main_cp_results_by_dataset.csv")
        self.assertEqual(len(t3), 32)
        for _, r in t3.iterrows():
            sub = self.df[(self.df["dataset_id"] == r["dataset_id"])
                          & (self.df["cp_method"] == r["cp_method"])
                          & (self.df["m_minority"] == r["m_minority"])]
            self.assertEqual(len(sub), 30)
            self.assertLessEqual(abs(float(sub["singleton_rate"].mean()) - float(r["singleton_rate"])), 1e-12)
            self.assertLessEqual(abs(float(sub["coverage_minority"].mean()) - float(r["coverage_minority"])), 1e-12)

    def test_manifest_hashes_match_live_outputs(self):
        if not self.have_outputs:
            self.skipTest("stage12 outputs not present")
        manifest = json.loads((self.run_dir / "stage12_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "PASS")
        for rec in manifest["outputs"]:
            p = Path(rec["path"])
            self.assertTrue(p.exists(), f"missing output {p}")
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertEqual(h, rec["sha256"], f"hash mismatch for {p}")

    def test_mapping_and_captions_reference_existing_files(self):
        if not self.have_outputs:
            self.skipTest("stage12 outputs not present")
        mapping = json.loads((FIG_DIR / "figure_number_mapping.json").read_text(encoding="utf-8"))
        self.assertEqual([f["figure_number"] for f in mapping["figures"]], [1, 2, 3, 4])
        self.assertEqual([t["table_number"] for t in mapping["tables"]], [1, 2, 3])
        for f in mapping["figures"]:
            for p in list(f["files"].values()) + f["source_data"]:
                self.assertTrue(Path(p).exists(), f"missing {p}")
            self.assertIn("Source data", f["caption"])
        self.assertIn("n=8", mapping["shared_caption_preamble"])
        self.assertIn("n = 8", mapping["figures"][0]["caption"])    # Fig. 1
        self.assertIn("m = 10", mapping["figures"][0]["caption"])   # Fig. 1
        self.assertIn("not overlaid", mapping["figures"][3]["caption"])  # Fig. 4 Beta-variance rule
        self.assertIn("Beta", mapping["figures"][3]["caption"])
        for t in mapping["tables"]:
            for p in t["files"].values():
                self.assertTrue(Path(p).exists(), f"missing {p}")
        captions = (FIG_DIR / "captions.md").read_text(encoding="utf-8")
        for token in ("boundary", "near-boundary", "main comparison", "Wilson", "Beta"):
            self.assertIn(token, captions)
        self.assertIn("never pooled", captions)

    def test_boundary_annotation_present_in_figures(self):
        if not self.have_outputs:
            self.skipTest("stage12 outputs not present")
        for n in (1, 2, 3):
            svg = (FIG_DIR / s12.FIG_SPECS[n]["dir"] / f"{s12.FIG_SPECS[n]['stem']}.svg").read_text(encoding="utf-8")
            for token in ("boundary", "near-boundary", "main comparison"):
                self.assertIn(token, svg, f"figure {n} missing boundary annotation token {token}")
        svg4 = (FIG_DIR / "figure4_threshold_variability/figure4_threshold_variability.svg").read_text(encoding="utf-8")
        self.assertIn("near-boundary", svg4)


if __name__ == "__main__":
    unittest.main()

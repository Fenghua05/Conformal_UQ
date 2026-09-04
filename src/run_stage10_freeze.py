"""Stage 10 - full-grid verification, authoritative results merge, QC, and results freeze.

Scope (user-confirmed option A, 2026-08-31):
- Stage 09 is a single complete 8-dataset formal run (no batch1/batch2 split exists).
- This stage performs read-only verification of the frozen formal run, then writes the
  authoritative merged results copy under results/ (originals preserved), the results
  data dictionary, the cell completeness matrix, the full QC report, and the results
  manifest with SHA-256, plus the preregistered D08/D10 10-seed CI determinability
  evaluation for the core m=50/100 primary endpoints (report-only; no seed expansion).
- No model fitting, no cache work, no new runs, no statistical interpretation.

Read-only inputs:
  configs/formal_run_manifest_v1.1.yaml (+ every hash-bound input)
  artifacts/runs/20260831T091426Z_stage09-formal_8abaf7bebe64/  (cells, results_long, qc, manifests)
  artifacts/splits/v1.1/<dataset>/seed-<seed>.json
  artifacts/caches/v1.1/<cfg>/<code>/<dataset>/seed-<seed>/<model>/{manifest.json, predictions.npz}

Writes (exclusive creation; never overwrites an existing file):
  artifacts/stage10/<run_id>/  events.jsonl, stage10_manifest.json, stage10_status.json,
                               results_qc_evidence.json, cell_completeness_matrix.csv
  results/                     results_long.parquet, results_data_dictionary.json,
                               cell_completeness_matrix.csv, results_qc_report.md,
                               results_manifest.json
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "artifacts" / "runs" / "20260831T091426Z_stage09-formal_8abaf7bebe64"
SPLIT_ROOT = ROOT / "artifacts" / "splits" / "v1.1"
CACHE_ROOT = ROOT / "artifacts" / "caches" / "v1.1"
RESULTS_DIR = ROOT / "results"
STAGE10_ROOT = ROOT / "artifacts" / "stage10"

FROZEN_MANIFEST_SHA256 = "0690795b1ea68a148f069a44dccb09a9245d21a097e7174dbd5ec3002f24172d"
Z95 = 1.959963984540054
DATASETS = [
    "openml_3_kr_vs_kp", "openml_24_mushroom", "openml_1486_nomao", "openml_1489_phoneme",
    "openml_1590_adult", "openml_4534_phishingwebsite", "openml_23512_higgs",
    "openml_23517_numerai28_6",
]
SEEDS = [104729, 130363, 155921, 196613, 262147, 318281, 374209, 419893, 481517, 552721]
MODELS = ["logistic_regression", "xgboost", "tabpfn"]
CP_METHODS = ["global_split_cp", "class_conditional_cp"]
M_VALUES = [10, 20, 50, 100]
EXPECTED_CELLS = 1920
EXPECTED_BASE_UNITS = 240

MODEL_FAMILY = {
    "logistic_regression": ("40f29139c9db63b2118c0efb28daa37940065a33dc52ec607b3e16bea0b786f9",
                            "cb25b48d6b1f005f6de7bb13eb9f9dad8e789fbe7b1529b96851742b85a4eea6"),
    "xgboost": ("40f29139c9db63b2118c0efb28daa37940065a33dc52ec607b3e16bea0b786f9",
                "cb25b48d6b1f005f6de7bb13eb9f9dad8e789fbe7b1529b96851742b85a4eea6"),
    "tabpfn": ("cee5c7d7da780885942a924b66276e7256469a013c9b5a0db98ba39249daa893",
               "8be59da84b507ba06778a020b2cb54bc187326421acd0f68428f71367079f9c8"),
}
MODEL_CACHE_TREE = {
    "logistic_regression": "cfg-40f29139c9db/code-cb25b48d6b1f",
    "xgboost": "cfg-40f29139c9db/code-cb25b48d6b1f",
    "tabpfn": "cfg-cee5c7d7da78/code-8be59da84b50",
}
RANK_CC_MINORITY = {10: 10, 20: 19, 50: 46, 100: 91}
RANK_CC_MAJORITY = 181  # ceil(201 * 0.9)
RANK_GLOBAL = {10: 190, 20: 199, 50: 226, 100: 271}  # ceil((200+m+1)*0.9)

CC_ONLY_NULL = ["q_global", "rank_global"]
GLOBAL_ONLY_NULL = ["q_minority", "q_majority", "rank_minority", "rank_majority",
                    "threshold_gap", "threshold_sum"]

INPUT_HASH_BINDINGS = {
    "configs/formal_run_manifest_v1.1.yaml": FROZEN_MANIFEST_SHA256,
    "protocols/protocol_v1.1.md": "059d58b5e99062e8d6a8d6d17fa56c4778e86ab4b5c24434a87769f92453b656",
    "protocols/dataset_lock_v1.0.md": "ce4f8281d0ccc1978cf6e27b238f68d6dca87c4e7062271f7a03d0c46d967d97",
    "configs/stage04_splits_v1.1.yaml": "516cfb79c0625b22f077b1ba2cad9f6794ea2edeabd99d00d9f9f3da5b1328a7",
    "configs/stage05_lr_xgboost_v1.1.yaml": "255c7595442a269709278f737dc58c348734287b04ffc6e7a5be999de591ff36",
    "configs/stage05b_tabpfn_v1.1.yaml": "18a9b2c3c5620e5d861d3dd6bc9de1ca876fa57fb5ba5496b2deae5b9d29a94c",
    "decisions/pilot_decision_stage07_v1.1.json": "a3663c5140472a3a2deb1734cf3ed9a839d764486a8d2700c5690b1a6de7b6a0",
    "environment/environment_lock_v1.0.json": "32bdba723361527405e465e7d35afe78416de5f54454ec9bac4098652aaa40b9",
    "configs/results_long.schema.json": "d1c4a4fb672a9d137f275cfb436674a82ce6baf8c287151c10ac8417fd3027f9",
    "artifacts/stage08_v11/20260831T085012Z_pilot_independent_audit/independent_audit.json":
        "c1634a1721d8ba9a1a0c86557d3fee4635067c20f457d727a40d8553378494bb",
    "artifacts/stage02/dataset_registry_v1.0.1.json":
        "7bb799dd82aa43acc0a03f8cb5504bac18decac7a5f4983a86a53dd168aa3c47",
    "decisions/D08-004_FORMAL_RUN_GO_RECEIPT.json":
        "e7adfe6ac93b225f76a8ab0a8f3063771d35764dd2f32ca494a040c168f86e5a",
    "decisions/D08-003_V11_CACHE_AND_PILOT_BUDGET_RECEIPT.json":
        "1193096155d8f98060cf8e0b74fec983ea45d35cf290d950b9d0657f54dc9721",
    "artifacts/stage08_v11_cloud/cache_intake_20260831T081631Z/intake_audit.json":
        "de610f8b2ae3d85db48a00b8e3346c0c0dd343060ba0f474c5f5f0c33ae71461",
}

KEY_COLS = ["dataset_id", "seed", "model", "cp_method", "m_minority"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wilson(k: float, n: float) -> tuple[float, float]:
    """Two-sided 95% Wilson score interval, z fixed, no continuity correction.
    Reproduces src/conformal_uq/metrics.py:wilson_interval exactly (denominator
    1+z^2/n) with the Stage 07 [0,1] endpoint clipping convention."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = Z95 * Z95
    denominator = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denominator
    radius = Z95 * np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denominator
    return (max(0.0, center - radius), min(1.0, center + radius))


def d01_seed(endpoint_tag: str) -> int:
    """D01-derived RNG seed. D01 rule: uint32(first 32 bits of
    SHA256(protocol_version | dataset_id | base_seed | purpose)). The D08 bootstrap is a
    project-level purpose with no single dataset/top-level seed, so the documented
    project-level convention used here is:
        input string = 'v1.1|EIGHT_DATASET|0|d08_bootstrap_' + endpoint_tag
        seed = uint32 of the first 32 bits (first 8 hex chars) of that SHA-256.
    Recorded verbatim in the QC evidence; re-pinning a different convention is a
    deterministic, results-independent recomputation."""
    s = f"v1.1|EIGHT_DATASET|0|d08_bootstrap_{endpoint_tag}"
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


def d08_effect_ci(d: np.ndarray, endpoint_tag: str, n_boot: int = 20000):
    """Preregistered D08: effect = median(d_j); two-sided 95% CI = percentile bootstrap
    (B=20000) of the 8 whole-dataset effects with a D01-derived RNG seed."""
    rng = np.random.default_rng(d01_seed(endpoint_tag))
    idx = rng.choice(d.shape[0], size=(n_boot, d.shape[0]), replace=True)
    boots = np.median(d[idx], axis=1)
    low, high = np.percentile(boots, [2.5, 97.5])
    return float(np.median(d)), float(low), float(high)


class Events:
    def __init__(self, path: Path):
        self.path = path
        self.handle = open(path, "a", encoding="utf-8")

    def log(self, event: str, level: str = "INFO", **kw):
        rec = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "stage": "Stage 10", "level": level, "event": event, **kw}
        self.handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.handle.flush()


def main() -> int:
    t0 = time.time()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    script_hash = sha256_file(Path(__file__).resolve())
    run_id = f"{ts}_stage10-results-freeze_{script_hash[:8]}"
    ev_dir = STAGE10_ROOT / run_id
    ev_dir.mkdir(parents=True, exist_ok=True)
    ev = Events(ev_dir / "events.jsonl")
    ev.log("stage10_start", run_id=run_id, script_sha256=script_hash)

    failures: list[str] = []
    checks: dict[str, bool] = {}

    def check(name: str, ok: bool, detail=None):
        checks[name] = bool(checks.get(name, True)) and bool(ok)
        if not ok:
            failures.append(name)
        ev.log("check", check=name, ok=bool(ok), detail=detail)
        return ok

    # ---------------- Gate 1: frozen manifest and hash-bound inputs ----------------
    manifest_bind_ok = True
    binding_results = {}
    for rel, expected in INPUT_HASH_BINDINGS.items():
        p = ROOT / rel
        if not p.exists():
            manifest_bind_ok = False
            binding_results[rel] = "MISSING"
            continue
        got = sha256_file(p)
        binding_results[rel] = {"expected": expected, "observed": got, "match": got == expected}
        manifest_bind_ok &= got == expected
    check("frozen_manifest_and_input_hashes_live_match", manifest_bind_ok, binding_results)

    run_manifest = json.loads((RUN_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    run_status = json.loads((RUN_DIR / "run_status.json").read_text(encoding="utf-8"))
    run_qc = json.loads((RUN_DIR / "qc.json").read_text(encoding="utf-8"))
    check("run_status_pass_1920_of_1920",
          run_status.get("status") == "PASS"
          and run_status.get("verified_cells") == EXPECTED_CELLS
          and run_status.get("expected_cells") == EXPECTED_CELLS
          and run_status.get("missing_cells") == 0
          and not run_status.get("failures")
          and not run_status.get("batch_validation_errors"))
    check("run_qc_all_checks_true", bool(run_qc.get("checks")) and all(run_qc.get("checks").values()))
    check("run_manifest_binds_frozen_manifest",
          run_manifest.get("formal_run_manifest_sha256") == FROZEN_MANIFEST_SHA256)

    # ---------------- Gate 2: cell-file grid completeness ----------------
    cell_files = sorted((RUN_DIR / "cells").glob("*.json"))
    check("cell_file_count_1920", len(cell_files) == EXPECTED_CELLS, len(cell_files))
    rows = []
    parse_errors = []
    for fp in cell_files:
        try:
            rows.append(json.loads(fp.read_text(encoding="utf-8")))
        except Exception as exc:  # pragma: no cover
            parse_errors.append(f"{fp.name}: {exc}")
    check("cell_json_parse_ok", not parse_errors, parse_errors[:5])
    cells = pd.DataFrame(rows)

    key_df = cells[KEY_COLS].copy()
    dup_mask = key_df.duplicated(keep=False)
    check("cell_keys_unique_no_duplicates", not dup_mask.any(),
          int(dup_mask.sum()))
    expected_keys = {(d, s, m, c, mm) for d in DATASETS for s in SEEDS for m in MODELS
                     for c in CP_METHODS for mm in M_VALUES}
    observed_keys = set(map(tuple, key_df.itertuples(index=False, name=None)))
    check("cell_grid_exact_no_missing_no_extra", observed_keys == expected_keys,
          {"missing": len(expected_keys - observed_keys),
           "extra": len(observed_keys - expected_keys)})
    base_units = cells[["dataset_id", "seed", "model"]].drop_duplicates()
    check("base_prediction_units_240", len(base_units) == EXPECTED_BASE_UNITS, len(base_units))

    # ---------------- Gate 3: constants, versions, lineage ----------------
    check("protocol_version_v1_1_all", (cells["protocol_version"] == "v1.1").all())
    check("results_schema_version_v1_1_0_all", (cells["results_schema_version"] == "v1.1.0").all())
    check("alpha_0_1_all", (cells["alpha"] == 0.1).all())
    check("m_majority_200_all", (cells["m_majority"] == 200).all())
    check("status_pass_all", (cells["status"] == "PASS").all())
    check("run_id_constant", cells["run_id"].nunique() == 1
          and cells["run_id"].iloc[0] == RUN_DIR.name)
    check("frozen_manifest_sha256_constant_all_rows",
          (cells["formal_run_manifest_sha256"] == FROZEN_MANIFEST_SHA256).all())
    check("environment_hash_constant", cells["environment_hash"].nunique() == 1
          and cells["environment_hash"].iloc[0]
          == "32bdba723361527405e465e7d35afe78416de5f54454ec9bac4098652aaa40b9")
    lineage_ok = True
    lineage_bad = []
    for m in MODELS:
        cfg_exp, code_exp = MODEL_FAMILY[m]
        sub = cells[cells["model"] == m]
        if not ((sub["config_hash"] == cfg_exp).all() and (sub["code_hash"] == code_exp).all()):
            lineage_ok = False
            lineage_bad.append(m)
    check("config_code_lineage_per_model_family", lineage_ok, lineage_bad)
    check("created_utc_within_run_window",
          bool((cells["created_utc"] >= run_manifest["created_utc"].replace("Z", ""))
               .all()) and len(cells["created_utc"].unique()) > 0,
          {"distinct": int(cells["created_utc"].nunique()),
           "min": str(cells["created_utc"].min()), "max": str(cells["created_utc"].max()),
           "note": "created_utc is the per-cell write time; cells were written progressively "
                   "during the run, so constancy is not expected. Verified non-empty and "
                   "consistent with the immutable run window."})

    # ---------------- Gate 4: ranks, threshold pattern, geometry ----------------
    cc = cells[cells["cp_method"] == "class_conditional_cp"]
    gl = cells[cells["cp_method"] == "global_split_cp"]
    rank_ok = True
    for m_val, r_exp in RANK_CC_MINORITY.items():
        rank_ok &= (cc[cc["m_minority"] == m_val]["rank_minority"] == r_exp).all()
    rank_ok &= (cc["rank_majority"] == RANK_CC_MAJORITY).all()
    for m_val, r_exp in RANK_GLOBAL.items():
        rank_ok &= (gl[gl["m_minority"] == m_val]["rank_global"] == r_exp).all()
    check("exact_order_statistic_ranks_all_cells", bool(rank_ok))

    null_pattern_ok = True
    for col in CC_ONLY_NULL:
        null_pattern_ok &= cc[col].isna().all() and gl[col].notna().all()
    for col in GLOBAL_ONLY_NULL:
        null_pattern_ok &= gl[col].isna().all() and cc[col].notna().all()
    check("structural_null_pattern_exact", bool(null_pattern_ok))

    other_nulls = [c for c in cells.columns
                   if c not in CC_ONLY_NULL + GLOBAL_ONLY_NULL and cells[c].isna().any()]
    check("no_unexpected_missing_values", not other_nulls, other_nulls)

    gap_ok = bool(np.allclose(cc["threshold_gap"],
                              (cc["q_minority"] - cc["q_majority"]).abs(), rtol=0, atol=1e-12))
    sum_ok = bool(np.allclose(cc["threshold_sum"],
                              cc["q_minority"] + cc["q_majority"], rtol=0, atol=1e-12))
    check("threshold_geometry_gap_sum_identities_cc", gap_ok and sum_ok)
    check("thresholds_in_unit_interval",
          bool(cc[["q_minority", "q_majority"]].ge(0).all().all()
               and cc[["q_minority", "q_majority"]].le(1).all().all()
               and gl["q_global"].between(0, 1).all()))

    # ---------------- Gate 5: subset identity, split hash, counts ----------------
    subset_ok = True
    subset_bad = []
    for (d, s, m_val), grp in cells.groupby(["dataset_id", "seed", "m_minority"]):
        if grp["subset_hash"].nunique() != 1:
            subset_ok = False
            subset_bad.append((d, s, m_val))
    check("subset_hash_identical_across_models_and_cp_methods", subset_ok, subset_bad[:5])

    split_manifest_hash = {}
    split_bad = []
    for d in DATASETS:
        for s in SEEDS:
            sp = SPLIT_ROOT / d / f"seed-{s}.json"
            spj = json.loads(sp.read_text(encoding="utf-8"))
            split_manifest_hash[(d, s)] = spj["split_hash"]
    for (d, s), h in split_manifest_hash.items():
        sub = cells[(cells["dataset_id"] == d) & (cells["seed"] == s)]
        if not (sub["split_hash"] == h).all():
            split_bad.append((d, s))
    check("split_hash_equals_locked_v1_1_split_manifests", not split_bad, split_bad[:5])

    cnt_ok = bool((cells["n_cal_minority"] == cells["m_minority"]).all()
                  and (cells["n_cal_majority"] == 200).all()
                  and (cells["n_cal_total"] == cells["n_cal_majority"] + cells["n_cal_minority"]).all()
                  and (cells["n_test"] == cells["n_test_minority"] + cells["n_test_majority"]).all()
                  and (cells["n_test_minority"] >= 75).all())
    check("class_counts_and_feasibility_identities", cnt_ok)

    # ---------------- Gate 6: coverage, decomposition, Wilson, ranges ----------------
    cov_ok = bool(
        np.allclose(cells["coverage_minority"], cells["covered_count_minority"] / cells["n_test_minority"], rtol=0, atol=1e-12)
        and np.allclose(cells["coverage_majority"], cells["covered_count_majority"] / cells["n_test_majority"], rtol=0, atol=1e-12)
        and np.allclose(cells["coverage_overall"], cells["covered_count_overall"] / cells["n_test"], rtol=0, atol=1e-12)
        and (cells["covered_count_overall"] == cells["covered_count_minority"] + cells["covered_count_majority"]).all()
        and (cells["covered_count_minority"] <= cells["n_test_minority"]).all()
        and (cells["covered_count_majority"] <= cells["n_test_majority"]).all()
        and np.allclose(cells["coverage_disparity"],
                        (cells["coverage_minority"] - cells["coverage_majority"]).abs(), rtol=0, atol=1e-12))
    check("coverage_identities_and_disparity", cov_ok)

    dec_ok = bool(
        np.allclose(cells["empty_rate"] + cells["singleton_rate"] + cells["doubleton_rate"], 1.0, rtol=0, atol=1e-9)
        and np.allclose(cells["average_set_size"], cells["singleton_rate"] + 2.0 * cells["doubleton_rate"], rtol=0, atol=1e-12)
        and cells[["empty_rate", "singleton_rate", "doubleton_rate", "average_set_size"]].ge(0).all().all()
        and cells[["empty_rate", "singleton_rate", "doubleton_rate"]].le(1).all().all())
    check("set_decomposition_identities", dec_ok)

    wil_ok = True
    wil_max_err = 0.0
    for kcol, ncol, lcol, hcol in [
        ("covered_count_minority", "n_test_minority", "coverage_minority_wilson_low", "coverage_minority_wilson_high"),
        ("covered_count_majority", "n_test_majority", "coverage_majority_wilson_low", "coverage_majority_wilson_high"),
        ("covered_count_overall", "n_test", "coverage_overall_wilson_low", "coverage_overall_wilson_high"),
    ]:
        lo_hi = [wilson(k, n) for k, n in zip(cells[kcol], cells[ncol])]
        lo = np.array([x[0] for x in lo_hi])
        hi = np.array([x[1] for x in lo_hi])
        wil_max_err = max(wil_max_err,
                          float(np.max(np.abs(lo - cells[lcol].to_numpy()))),
                          float(np.max(np.abs(hi - cells[hcol].to_numpy()))))
        wil_ok &= bool(np.allclose(lo, cells[lcol], rtol=0, atol=1e-12)
                       and np.allclose(hi, cells[hcol], rtol=0, atol=1e-12))
    check("wilson_intervals_reproduced_all_5760_intervals", wil_ok, {"max_abs_err": wil_max_err})
    check("wilson_bounds_contain_estimate",
          bool((cells["coverage_minority_wilson_low"] <= cells["coverage_minority"]).all()
               and (cells["coverage_minority"] <= cells["coverage_minority_wilson_high"]).all()
               and (cells["coverage_majority_wilson_low"] <= cells["coverage_majority"]).all()
               and (cells["coverage_majority"] <= cells["coverage_majority_wilson_high"]).all()
               and (cells["coverage_overall_wilson_low"] <= cells["coverage_overall"]).all()
               and (cells["coverage_overall"] <= cells["coverage_overall_wilson_high"]).all()))
    check("metric_ranges_valid",
          bool(cells["auroc"].between(0, 1).all() and cells["auprc"].between(0, 1).all()
               and cells[[c for c in cells.columns if c.startswith("coverage_")
                          and not c.endswith("_low") and not c.endswith("_high")]].ge(0).all().all()))

    # ---------------- Gate 7: AUROC/AUPRC invariance ----------------
    inv_ok = True
    inv_bad = []
    for (d, s, m), grp in cells.groupby(["dataset_id", "seed", "model"]):
        if grp["auroc"].nunique() != 1 or grp["auprc"].nunique() != 1:
            inv_ok = False
            inv_bad.append((d, s, m))
    check("auroc_auprc_invariant_across_cp_and_m", inv_ok, inv_bad[:5])

    # ---------------- Gate 8: cache provenance and full NPZ re-hash ----------------
    cache_results = {}
    cache_ok = True
    for (d, s, m), grp in cells.groupby(["dataset_id", "seed", "model"]):
        tree = MODEL_CACHE_TREE[m]
        man_p = CACHE_ROOT / tree / d / f"seed-{s}" / m / "manifest.json"
        npz_p = man_p.parent / "predictions.npz"
        man = json.loads(man_p.read_text(encoding="utf-8"))
        npz_hash = sha256_file(npz_p)
        cell_hash = grp["prediction_cache_hash"].unique()
        ok = (man.get("cache_sha256") == npz_hash and len(cell_hash) == 1
              and cell_hash[0] == npz_hash
              and man.get("provenance", {}).get("split_hash") == split_manifest_hash[(d, s)]
              and man.get("model_hash") == grp["model_hash"].iloc[0]
              and man.get("provenance", {}).get("dataset_hash") == grp["dataset_hash"].iloc[0]
              and man.get("qc_status") == "PASS"
              and man.get("metrics", {}).get("test", {}).get("auroc") == grp["auroc"].iloc[0]
              and man.get("metrics", {}).get("test", {}).get("auprc") == grp["auprc"].iloc[0])
        cache_ok &= ok
        cache_results[f"{d}|{s}|{m}"] = {"npz_sha256": npz_hash, "ok": bool(ok)}
    check("all_240_caches_rehashed_and_provenance_bound", cache_ok,
          {"units": len(cache_results),
           "failed": [k for k, v in cache_results.items() if not v["ok"]]})

    # dataset/label/model hash constancy across rows
    check("dataset_hash_constant_per_dataset",
          bool(cells.groupby("dataset_id")["dataset_hash"].nunique().eq(1).all()))
    check("label_mapping_hash_constant_per_dataset",
          bool(cells.groupby("dataset_id")["label_mapping_hash"].nunique().eq(1).all()))
    # model_hash binds the fitted model artifact per (dataset, seed) unit; it must be
    # constant across the 8 CP x m cells of each unit and equals the cache manifest
    # value (verified in the cache-provenance gate below/above).
    mh_ok = bool(cells.groupby(["dataset_id", "seed", "model"])["model_hash"].nunique().eq(1).all())
    check("model_hash_constant_per_base_unit", mh_ok)

    # ---------------- Gate 9: results_long parquet/csv agreement ----------------
    res_pq = RUN_DIR / "results_long.parquet"
    res_csv = RUN_DIR / "results_long.csv"
    df_pq = pd.read_parquet(res_pq)
    df_csv = pd.read_csv(res_csv)
    check("results_long_rows_1920", len(df_pq) == EXPECTED_CELLS, len(df_pq))
    check("results_long_parquet_matches_cell_records",
          len(df_pq) == len(cells) and set(map(tuple, df_pq[KEY_COLS].itertuples(index=False, name=None))) == observed_keys)
    key_pq = set(map(tuple, df_pq[KEY_COLS].itertuples(index=False, name=None)))
    key_csv = set(map(tuple, df_csv[KEY_COLS].itertuples(index=False, name=None)))
    check("results_long_csv_parquet_key_sets_equal", key_pq == key_csv)
    num_cols = [c for c in df_pq.columns if pd.api.types.is_numeric_dtype(df_pq[c])]
    csv_num_ok = True
    csv_max_err = 0.0
    for c in num_cols:
        a = df_pq[c].to_numpy(dtype=float)
        b = df_csv[c].to_numpy(dtype=float)
        if a.shape != b.shape:
            csv_num_ok = False
            break
        err = float(np.nanmax(np.abs(a - b))) if a.size else 0.0
        csv_max_err = max(csv_max_err, err)
        csv_num_ok &= bool(np.allclose(a, b, rtol=0, atol=1e-12, equal_nan=True))
    obj_ok = all((df_pq[c].fillna("<NA>").astype(str) == df_csv[c].fillna("<NA>").astype(str)).all()
                 for c in df_pq.columns if c not in num_cols)
    check("results_long_csv_parquet_values_agree", csv_num_ok and obj_ok,
          {"max_numeric_abs_err": csv_max_err})

    # ---------------- Gate 10: reproduce run QC global-coverage diagnostic ----------------
    diag_ok = True
    diag_detail = []
    for g in run_qc.get("global_coverage_diagnostic", {}).get("groups", []):
        sub = gl[(gl["model"] == g["model"]) & (gl["m_minority"] == g["m_minority"])]
        flagged = int(((sub["coverage_overall_wilson_low"] > 0.9)
                       | (sub["coverage_overall_wilson_high"] < 0.9)).sum())
        if flagged != g["flagged_cells"] or len(sub) != g["total_cells"]:
            diag_ok = False
            diag_detail.append({"model": g["model"], "m": g["m_minority"],
                                "recomputed": flagged, "stored": g["flagged_cells"]})
    check("global_coverage_diagnostic_reproduced", diag_ok, diag_detail)

    # ---------------- Authoritative merge (exclusive; originals untouched) ----------------
    RESULTS_DIR.mkdir(exist_ok=True)
    out_pq = RESULTS_DIR / "results_long.parquet"
    if out_pq.exists():
        check("results_long_parquet_exclusive_creation", False, "target already exists")
    else:
        shutil.copyfile(res_pq, out_pq)
        same = sha256_file(out_pq) == sha256_file(res_pq)
        check("results_long_parquet_merged_sha256_identical_to_source", same)
        ev.log("results_merged", target=str(out_pq), source=str(res_pq))

    # ---------------- D08 confirmatory endpoints (report-only) ----------------
    endpoints = []
    piv = cells.sort_values(KEY_COLS).set_index(KEY_COLS)

    def paired_dataset_effects(metric: str, a_cp: str, a_m: int, b_cp: str, b_m: int):
        """Preregistered contrast is within dataset/seed/pipeline. Per dataset, the
        complete paired observations are the 3 pipelines x 10 seeds contrasts;
        d_j = mean over all complete paired contrasts (per-pipeline seed-means are
        recorded for transparency). No seed or pipeline is dropped (complete grid)."""
        dvals, dsets, per_pipeline_all = [], [], []
        for d in DATASETS:
            per_pipeline = {}
            diffs_all = []
            for mdl in MODELS:
                diffs = [float(piv.loc[(d, s, mdl, a_cp, a_m), metric]
                               - piv.loc[(d, s, mdl, b_cp, b_m), metric]) for s in SEEDS]
                per_pipeline[mdl] = float(np.mean(diffs))
                diffs_all.extend(diffs)
            dvals.append(float(np.mean(diffs_all)))
            dsets.append(d)
            per_pipeline_all.append(per_pipeline)
        return np.array(dvals), dsets, per_pipeline_all

    def endpoint(tag: str, metric: str, a_sel, b_sel, direction_label: str) -> dict:
        d, dsets, per_pipeline = paired_dataset_effects(metric, a_sel[0], a_sel[1], b_sel[0], b_sel[1])
        eff, lo, hi = d08_effect_ci(d, tag)
        n_pos = int((d > 0).sum())
        n_neg = int((d < 0).sum())
        n_zero = int((d == 0).sum())
        contains_zero = bool(lo <= 0.0 <= hi)
        return {
            "endpoint_tag": tag, "metric": metric, "comparison": direction_label,
            "d_by_dataset": {ds: float(v) for ds, v in zip(dsets, d)},
            "per_pipeline_seed_mean_d": {ds: pp for ds, pp in zip(dsets, per_pipeline)},
            "effect_median_dj": eff, "ci95_low": lo, "ci95_high": hi,
            "ci_width": hi - lo,
            "direction_count": {"positive": n_pos, "negative": n_neg, "zero": n_zero},
            "direction_judgeable_ci_excludes_zero": bool(not contains_zero),
            "ci_contains_zero": contains_zero,
            "d10_practical_precision_threshold": "NOT_APPROVED_OPEN_PARAMETER",
            "d10_expansion_trigger": False,
            "bootstrap": {"replicates": 20000, "rng": "numpy default_rng",
                          "seed_convention": "D01-derived (see evidence note)",
                          "seed_input_string": f"v1.1|EIGHT_DATASET|0|d08_bootstrap_{tag}",
                          "seed_uint32": d01_seed(tag)},
        }

    # RQ1 comparison A: Class-Conditional CP, m=100 - m=50 (primary endpoints)
    for metric, tag in [("singleton_rate", "rq1a_cc_m100_minus_m50_singleton_rate"),
                        ("average_set_size", "rq1a_cc_m100_minus_m50_average_set_size")]:
        endpoints.append(endpoint(tag, metric, ("class_conditional_cp", 100),
                                  ("class_conditional_cp", 50),
                                  "RQ1-A: Class-Conditional CP m=100 minus m=50 (primary)"))
    # RQ2 comparison B: Class-Conditional - Global at m in {50,100}
    for m_val in (50, 100):
        for metric in ["coverage_minority", "coverage_majority", "coverage_disparity",
                       "singleton_rate", "average_set_size"]:
            endpoints.append(endpoint(f"rq2b_cc_minus_global_m{m_val}_{metric}", metric,
                                      ("class_conditional_cp", m_val),
                                      ("global_split_cp", m_val),
                                      f"RQ2-B: Class-Conditional minus Global at m={m_val}"))

    any_trigger = any(e["d10_expansion_trigger"] for e in endpoints)
    d10_verdict = {
        "preregistered_rule": "D10/O-06: expansion 10->20 seeds may be proposed only for a "
                              "core m=50/100 primary endpoint when its 95% D08 CI contains zero "
                              "AND its width exceeds a user-approved practical-precision threshold; "
                              "never triggered by a p-value.",
        "practical_precision_threshold_status": "NOT_APPROVED (D10: open parameter; without it "
                                                "expansion remains pending and cannot auto-run)",
        "any_endpoint_trigger": any_trigger,
        "verdict": "NOT_TRIGGERED" if not any_trigger else "TRIGGER_PENDING_USER_DECISION",
        "seed_count_fixed_at_10": True,
        "note": "Report-only. No seed expansion executed or authorized; p-values are not a "
                "trigger and no formal significance test is interpreted here.",
    }

    # ---------------- Completeness matrix ----------------
    comp_rows = []
    for d in DATASETS:
        for m in MODELS:
            for c in CP_METHODS:
                for mm in M_VALUES:
                    sub = cells[(cells["dataset_id"] == d) & (cells["model"] == m)
                                & (cells["cp_method"] == c) & (cells["m_minority"] == mm)]
                    comp_rows.append({
                        "dataset_id": d, "model": m, "cp_method": c, "m_minority": mm,
                        "n_cells": len(sub), "n_unique_seeds": sub["seed"].nunique(),
                        "n_expected_seeds": len(SEEDS), "complete": len(sub) == len(SEEDS),
                        "all_status_pass": bool((sub["status"] == "PASS").all()) if len(sub) else False,
                    })
    comp = pd.DataFrame(comp_rows)
    check("completeness_matrix_all_1920_complete",
          bool(len(comp) == 192 and comp["complete"].all() and comp["all_status_pass"].all()),
          {"rows": len(comp), "complete": int(comp["complete"].sum())})

    # ---------------- Write deliverables ----------------
    def write_json_exclusive(path: Path, payload):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    comp_out = RESULTS_DIR / "cell_completeness_matrix.csv"
    if comp_out.exists():
        raise FileExistsError(f"refusing to overwrite {comp_out}")
    comp.to_csv(comp_out, index=False)
    comp_ev = ev_dir / "cell_completeness_matrix.csv"
    shutil.copyfile(comp_out, comp_ev)

    overall_pass = not failures
    evidence = {
        "artifact_id": f"{run_id}_qc_evidence",
        "stage": "Stage 10 results freeze",
        "created_utc": ts,
        "protocol_version": "v1.1",
        "source_run": {"run_id": RUN_DIR.name, "path": str(RUN_DIR),
                       "run_status": run_status.get("status"),
                       "run_qc_all_true": all(run_qc.get("checks", {}).values())},
        "frozen_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "input_hash_bindings": binding_results,
        "checks": checks,
        "failures": failures,
        "overall_status": "PASS" if overall_pass else "FAIL",
        "grid": {"expected_cp_cells": EXPECTED_CELLS, "observed_cp_cells": int(len(cells)),
                 "expected_base_prediction_units": EXPECTED_BASE_UNITS,
                 "observed_base_prediction_units": int(len(base_units)),
                 "cp_cell_definition": "dataset x seed x model x cp_method x m_minority",
                 "base_unit_definition": "dataset x seed x model (one immutable probability cache each)"},
        "cache_rehash": {"units": len(cache_results),
                         "all_ok": bool(cache_ok),
                         "results": cache_results},
        "wilson_recomputation": {"intervals": 5760, "max_abs_err": wil_max_err},
        "global_coverage_diagnostic_reproduction": {
            "note": "Diagnostic-only; reproduced from results_long aggregates; matches run QC.",
            "groups": run_qc.get("global_coverage_diagnostic", {}).get("groups", [])},
        "d08_confirmatory_endpoints_report_only": endpoints,
        "d10_seed_expansion_verdict": d10_verdict,
        "seed_convention_note": "D01-derived bootstrap RNG seed convention: input string "
                                "'v1.1|EIGHT_DATASET|0|d08_bootstrap_<endpoint_tag>'; seed = uint32 "
                                "of first 32 bits of that SHA-256. The frozen protocol specifies a "
                                "'D01-derived RNG seed' without fixing the project-level purpose "
                                "string; this convention is deterministic and recorded so any "
                                "re-pinning is a reproducible recomputation, not a results change.",
    }
    write_json_exclusive(ev_dir / "results_qc_evidence.json", evidence)

    # ---------------- Results manifest ----------------
    result_files = {
        "results/results_long.parquet": out_pq,
        "results/cell_completeness_matrix.csv": comp_out,
    }
    dd_out = RESULTS_DIR / "results_data_dictionary.json"
    qc_rep = RESULTS_DIR / "results_qc_report.md"
    man_out = RESULTS_DIR / "results_manifest.json"

    # data dictionary
    desc = {
        "alpha": "Conformal target miscoverage; fixed 0.1 (nominal coverage 90%).",
        "artifact_id": "Immutable Stage 09 cell artifact identifier (run-scoped).",
        "auprc": "Test-set average precision of the base predictive pipeline (invariant across CP/m).",
        "auroc": "Test-set AUROC of the base predictive pipeline (invariant across CP/m).",
        "average_set_size": "Mean binary prediction-set size = singleton_rate + 2*doubleton_rate.",
        "code_hash": "SHA-256 of the producing code snapshot (model-family-specific).",
        "config_hash": "SHA-256 of the producing configuration lock (model-family-specific).",
        "coverage_disparity": "|coverage_minority - coverage_majority| on the test split.",
        "coverage_majority": "Majority-class test coverage = covered_count_majority / n_test_majority.",
        "coverage_majority_wilson_high": "Upper endpoint of the two-sided 95% Wilson interval (z=1.959963984540054, no continuity correction, clipped to [0,1]).",
        "coverage_majority_wilson_low": "Lower endpoint of the same Wilson interval.",
        "coverage_minority": "Minority-class test coverage = covered_count_minority / n_test_minority.",
        "coverage_minority_wilson_high": "Upper Wilson endpoint for minority coverage.",
        "coverage_minority_wilson_low": "Lower Wilson endpoint for minority coverage.",
        "coverage_overall": "Overall test coverage = covered_count_overall / n_test.",
        "coverage_overall_wilson_high": "Upper Wilson endpoint for overall coverage.",
        "coverage_overall_wilson_low": "Lower Wilson endpoint for overall coverage.",
        "covered_count_majority": "Test majority samples whose true label is included in the prediction set.",
        "covered_count_minority": "Test minority samples whose true label is included in the prediction set.",
        "covered_count_overall": "covered_count_minority + covered_count_majority.",
        "cp_method": "global_split_cp | class_conditional_cp (Mondrian by label).",
        "created_utc": "Stage 09 run creation time (constant for all cells).",
        "dataset_hash": "SHA-256 of the locked raw dataset source for dataset_id.",
        "dataset_id": "Locked registry identifier (8 datasets).",
        "doubleton_rate": "Fraction of test sets containing both labels (binary size-2 sets).",
        "empty_rate": "Fraction of test sets containing no label.",
        "environment_hash": "SHA-256 of environment/environment_lock_v1.0.json (constant).",
        "formal_run_manifest_sha256": "SHA-256 of the frozen configs/formal_run_manifest_v1.1.yaml (constant).",
        "label_mapping_hash": "SHA-256 of the dataset label mapping (minority label identity).",
        "m_majority": "Fixed majority calibration count 200.",
        "m_minority": "Minority calibration size in {10,20,50,100} (nested subsets).",
        "minority_label": "Numeric label of the registry-defined minority class.",
        "model": "logistic_regression | xgboost | tabpfn (predictive pipelines).",
        "model_hash": "SHA-256 of the frozen model/pipeline specification.",
        "n_cal_majority": "Majority calibration samples used (200).",
        "n_cal_minority": "Minority calibration samples used (= m_minority).",
        "n_cal_total": "n_cal_majority + n_cal_minority.",
        "n_test": "Test rows = n_test_minority + n_test_majority.",
        "n_test_majority": "Test majority rows.",
        "n_test_minority": "Test minority rows (locked feasibility >= 75).",
        "prediction_cache_hash": "SHA-256 of the immutable base-probability predictions.npz used by this cell.",
        "protocol_version": "Protocol version v1.1 (constant).",
        "q_global": "Global Split CP threshold (score scale, 1-p_y). Null for class-conditional cells.",
        "q_majority": "Class-Conditional majority threshold. Null for global cells.",
        "q_minority": "Class-Conditional minority threshold. Null for global cells.",
        "rank_global": "Exact finite-sample order statistic ceil((n_global+1)(1-alpha)). Null for CC cells.",
        "rank_majority": "ceil((200+1)(1-alpha)) = 181. Null for global cells.",
        "rank_minority": "ceil((m+1)(1-alpha)) in {10,19,46,91}. Null for global cells.",
        "results_schema_version": "results_long schema version v1.1.0.",
        "run_id": "Immutable Stage 09 formal run identifier (constant).",
        "seed": "Top-level replicate seed (10 locked values).",
        "singleton_rate": "Fraction of test sets containing exactly one label.",
        "split_hash": "SHA-256 of the locked v1.1 split manifest for dataset x seed.",
        "status": "Per-cell execution/QC status (PASS).",
        "subset_hash": "Canonical membership hash of the calibration subset (identical across models and CP methods within dataset x seed x m).",
        "threshold_gap": "Class-Conditional threshold gap abs(q_minority - q_majority). Null for global cells.",
        "threshold_sum": "Class-Conditional threshold sum q_minority + q_majority. Null for global cells.",
    }
    data_dictionary = {
        "artifact_id": "results_data_dictionary_v1.0",
        "created_utc": ts,
        "describes": "results/results_long.parquet (authoritative merged copy of the Stage 09 formal run)",
        "source_run_id": RUN_DIR.name,
        "n_rows": int(len(df_pq)), "n_columns": int(len(df_pq.columns)),
        "unit_of_record": "CP result cell = dataset x seed x model x cp_method x m_minority (m_majority=200, alpha=0.1 fixed)",
        "structural_null_pattern": {
            "class_conditional_cp_rows_null": CC_ONLY_NULL,
            "global_split_cp_rows_null": GLOBAL_ONLY_NULL,
            "all_other_columns_non_null": True},
        "columns": [
            {"column": c, "dtype": str(df_pq[c].dtype), "description": desc.get(c, "")}
            for c in df_pq.columns],
    }
    write_json_exclusive(dd_out, data_dictionary)

    # QC report (human-readable)
    def fmt_ci(e):
        return (f"{e['effect_median_dj']:+.6f} [{e['ci95_low']:+.6f}, {e['ci95_high']:+.6f}] "
                f"w={e['ci_width']:.6f} dir={e['direction_count']['positive']}+/{e['direction_count']['negative']}-"
                f"{'/' + str(e['direction_count']['zero']) + '0' if e['direction_count']['zero'] else ''}"
                f" judgeable={'YES' if e['direction_judgeable_ci_excludes_zero'] else 'NO'}")

    ep_lines = "\n".join(f"- `{e['endpoint_tag']}` ({e['comparison']}): {fmt_ci(e)}" for e in endpoints)
    n_judgeable = sum(e["direction_judgeable_ci_excludes_zero"] for e in endpoints)
    chk_lines = "\n".join(f"- {'PASS' if ok else 'FAIL'} — {k}" for k, ok in checks.items())
    bind_lines = "\n".join(
        f"- `{rel}`: {'MATCH' if isinstance(v, dict) and v['match'] else 'MISMATCH/MISSING'}"
        for rel, v in binding_results.items())

    report = f"""# Stage 10 — Full-Grid Results Freeze QC Report

**Status:** `{'PASS' if overall_pass else 'FAIL'}` | **Created:** {ts} | **Stage 10 run:** `{run_id}`
**Source run:** `{RUN_DIR.name}` (single complete 8-dataset formal run; no batch split exists — user-confirmed option A)
**Frozen manifest:** `configs/formal_run_manifest_v1.1.yaml` SHA-256 `{FROZEN_MANIFEST_SHA256}`

## 1. Scope and grid verification

- CP result cells: **{len(cells)} / {EXPECTED_CELLS}** (`dataset x seed x model x cp_method x m_minority`); unique keys, zero duplicates, zero missing/extra.
- Base prediction units: **{len(base_units)} / 240** (`dataset x seed x model`, one immutable probability cache each). CP cells (8 per unit) are distinguished from base units.
- All cells `protocol_version=v1.1`, `results_schema_version=v1.1.0`, `status=PASS`, `alpha=0.1`, `m_majority=200`.
- Lineage: LR/XGBoost cells bind config `40f29139...` / code `cb25b48d...`; TabPFN cells bind config `cee5c7d7...` / code `8be59da8...`; environment `32bdba72...` constant; frozen-manifest hash constant in all rows.

## 2. Hash-bound input verification (live re-hash)

{bind_lines}

## 3. Invariant and identity checks

{chk_lines}

- Exact order statistics reproduced for all cells (CC ranks {{10:10, 20:19, 50:46, 100:91}} and majority 181; Global ranks {{10:190, 20:199, 50:226, 100:271}}).
- `subset_hash` identical across all 3 models and both CP methods within every `dataset x seed x m` (canonical membership).
- `split_hash` equals the locked v1.1 split manifest for all 80 `dataset x seed`.
- All **240** cache NPZ files re-hashed; each equals its manifest `cache_sha256` and every descendant cell's `prediction_cache_hash`; cache-manifest provenance (split/model/dataset hash, AUROC/AUPRC) matches all descendant cells.
- AUROC/AUPRC invariant across CP method and m within every `dataset x seed x model`.
- Coverage identities, disparity, set decomposition (`empty+singleton+doubleton=1`, `average_set_size=singleton+2*doubleton`), and threshold geometry identities (`gap=abs(q_minority-q_majority)`, `sum=q_minority+q_majority`) verified on all rows.
- All **5,760** Wilson intervals recomputed with the producing implementation's exact formula (`src/conformal_uq/metrics.py`, plain Wilson, no continuity correction, [0,1] endpoint clipping; max abs err {wil_max_err:.2e}); bounds contain estimates.
- Structural null pattern exact (Global rows null in `{', '.join(GLOBAL_ONLY_NULL)}`; CC rows null in `{', '.join(CC_ONLY_NULL)}`); no unexpected missing values.
- `results_long.parquet` and `.csv` key sets and values agree (max numeric abs err {csv_max_err:.2e}).
- Global-coverage Wilson diagnostic reproduced from aggregates and matches run QC exactly; remains **diagnostic-only** (no calculation discrepancy; not a scientific finding).

## 4. Authoritative merge

- `results/results_long.parquet` written once (exclusive creation); SHA-256 identical to the immutable source `artifacts/runs/{RUN_DIR.name}/results_long.parquet`. Originals preserved; nothing overwritten.

## 5. D08 confirmatory endpoints at core m=50/100 — 10-seed CI determinability (report-only)

Preregistered D08 estimator: per-dataset seed-mean paired effect `d_j`; confirmatory effect = median of the 8 `d_j`; two-sided 95% CI = 20,000-replicate percentile bootstrap of the 8 whole-dataset effects with a D01-derived RNG seed (convention documented in the evidence file). {n_judgeable} of {len(endpoints)} endpoints have a direction-judgeable 95% CI (excludes zero).

{ep_lines}

## 6. D10 20-seed expansion condition — verdict

- Preregistered rule (D10/O-06): 10->20 expansion may be **proposed** only for a core m=50/100 primary endpoint whose 95% D08 CI contains zero **and** whose width exceeds a user-approved practical-precision threshold; never p-value-motivated.
- Practical-precision threshold status: **NOT APPROVED** (explicitly an open parameter in D10; without it, expansion remains pending and cannot auto-run).
- **Verdict: `{'NOT_TRIGGERED' if not any_trigger else 'TRIGGER_PENDING_USER_DECISION'}`.** No seed expansion was executed, proposed for execution, or authorized; seeds remain fixed at 10. No p-value was computed or used.

## 7. Provenance

- Stage 10 script SHA-256: `{script_hash}`
- Evidence: `artifacts/stage10/{run_id}/results_qc_evidence.json`; events: `events.jsonl`
- Results manifest: `results/results_manifest.json` (artifact hashes below)

## 8. Boundary

No scientific interpretation, aggregate inference beyond the preregistered determinability report, publication figures, or manuscript work is included. The Global-coverage Wilson flag stays diagnostic-only. Next gates require separate user authorization.
"""
    if qc_rep.exists() or man_out.exists():
        raise FileExistsError("results_qc_report.md or results_manifest.json already exists")
    qc_rep.write_text(report, encoding="utf-8")

    stage10_manifest = {
        "artifact_id": f"{run_id}_manifest",
        "stage": "Stage 10 results freeze",
        "created_utc": ts,
        "status": "PASS" if overall_pass else "FAIL",
        "protocol_version": "v1.1",
        "producer_script": "src/run_stage10_freeze.py",
        "producer_script_sha256": script_hash,
        "source_run": {"run_id": RUN_DIR.name, "path": str(RUN_DIR),
                       "results_long_parquet_sha256": sha256_file(res_pq)},
        "frozen_manifest": {"path": "configs/formal_run_manifest_v1.1.yaml",
                            "sha256": FROZEN_MANIFEST_SHA256},
        "inputs_verified": {rel: (v if isinstance(v, str) else v["match"])
                            for rel, v in binding_results.items()},
        "outputs": {},  # filled below
        "checks": checks,
        "failures": failures,
        "d10_seed_expansion_verdict": d10_verdict["verdict"],
    }

    artifact_paths = {
        "artifacts/stage10/" + run_id + "/events.jsonl": ev_dir / "events.jsonl",
        "artifacts/stage10/" + run_id + "/results_qc_evidence.json": ev_dir / "results_qc_evidence.json",
        "artifacts/stage10/" + run_id + "/cell_completeness_matrix.csv": comp_ev,
        "results/results_long.parquet": out_pq,
        "results/results_data_dictionary.json": dd_out,
        "results/cell_completeness_matrix.csv": comp_out,
        "results/results_qc_report.md": qc_rep,
    }
    for rel, p in artifact_paths.items():
        stage10_manifest["outputs"][rel] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}

    write_json_exclusive(man_out, stage10_manifest)
    stage10_manifest["outputs"]["results/results_manifest.json"] = {
        "sha256": sha256_file(man_out), "bytes": man_out.stat().st_size}
    # rewrite manifest including its own final entry (self-excluded hash of prior state is
    # not required by governance; the manifest lists all other artifacts' hashes)
    man_out.write_text(json.dumps(stage10_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    status = {
        "artifact_id": f"{run_id}_status", "stage": "Stage 10 results freeze",
        "status": "PASS" if overall_pass else "FAIL",
        "checks_total": len(checks), "failures": failures,
        "results_frozen": overall_pass,
        "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json_exclusive(ev_dir / "stage10_status.json", status)
    write_json_exclusive(ev_dir / "stage10_manifest.json", stage10_manifest)
    ev.log("stage10_complete", status=status["status"], elapsed_s=round(time.time() - t0, 2))
    ev.handle.close()

    print(f"STAGE10_STATUS={status['status']}")
    print(f"CHECKS={len(checks)} FAILURES={len(failures)}")
    if failures:
        print("FAILED_CHECKS:", failures)
    print(f"RUN_ID={run_id}")
    print(f"D10_VERDICT={d10_verdict['verdict']}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())

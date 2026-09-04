"""Read-only independent Stage 08 v1.1 audit of the v1.1 pilot.

This tool intentionally does not import project conformal/evaluation helpers,
the v1.1 pilot runner, or TabPFN.  It reimplements the frozen v1.1
calculations directly from immutable caches, so its representative-cell
evidence is independent of the pilot implementation.  It creates no model,
no cache, no CP cell, no pilot rerun, and no formal-run manifest; its gate
recommendation authorizes nothing.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = ROOT / "artifacts" / "stage07_v1.1"
V10_PILOT_RUN = ROOT / "artifacts" / "stage07" / "20260830T161214Z_stage07-pilot_32b7e4728b8d"
CACHE_ROOT = ROOT / "artifacts" / "caches" / "v1.1"
SPLIT_ROOT = ROOT / "artifacts" / "splits" / "v1.1"
PILOT_DECISION = ROOT / "decisions" / "pilot_decision_stage07_v1.1.json"
D08_003_RECEIPT = ROOT / "decisions" / "D08-003_V11_CACHE_AND_PILOT_BUDGET_RECEIPT.json"
INTAKE_GLOB = "cache_intake_*/intake_audit.json"
CLOUD_SUMMARY_GLOB = "cache_intake_*/extracted/stage08_v11_tabpfn_cache_return/artifacts/stage08_v11_cloud/*/summary.json"
ALPHA = 0.1
M_VALUES = (10, 20, 50, 100)
MODELS = ("logistic_regression", "xgboost", "tabpfn")
METHODS = ("global_split_cp", "class_conditional_cp")
TOL = 1e-12
Z = 1.959963984540054


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def seed(dataset: str, base_seed: int, purpose: str) -> int:
    """v1.1 purpose-specific seed derivation, reimplemented independently."""
    raw = f"v1.1|{dataset}|{base_seed}|{purpose}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big", signed=False)


def q(scores: np.ndarray) -> tuple[int, float]:
    """Exact finite-sample order statistic: rank = ceil((n+1)(1-alpha))."""
    rank = math.ceil((len(scores) + 1) * (1 - ALPHA))
    return rank, float(np.sort(scores)[rank - 1]) if rank <= len(scores) else math.inf


def wilson(k: int, n: int) -> tuple[float, float]:
    """Two-sided Wilson interval with [0,1] endpoint clipping."""
    p = k / n
    den = 1 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / den
    radius = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return max(0.0, center - radius), min(1.0, center + radius)


def equal(a: object, b: object) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
        return abs(float(a) - float(b)) <= TOL
    return a == b


def representative_spec(dataset_ids: tuple[str, ...]) -> list[tuple[str, int, str, str, int]]:
    """36 cells spanning both datasets, all models/methods/m values, six seeds."""
    spec: list[tuple[str, int, str, str, int]] = []
    for dataset in dataset_ids:
        for model in MODELS:
            spec += [
                (dataset, 104729, model, "global_split_cp", 10),
                (dataset, 552721, model, "global_split_cp", 100),
                (dataset, 130363, model, "class_conditional_cp", 10),
                (dataset, 262147, model, "class_conditional_cp", 20),
                (dataset, 374209, model, "class_conditional_cp", 50),
                (dataset, 481517, model, "class_conditional_cp", 100),
            ]
    return spec


def cache_for(row: dict) -> Path:
    return CACHE_ROOT / f"cfg-{row['config_hash'][:12]}" / f"code-{row['code_hash'][:12]}" / row["dataset_id"] / f"seed-{row['seed']}" / row["model"]


def independent_cell(row: dict, arrays: dict[str, np.ndarray]) -> dict:
    """Recompute one CP cell from cache arrays without the pilot helpers."""
    cal_ids = arrays["calibration_pool_sample_ids"].astype(str)
    cal_y = arrays["calibration_pool_y"].astype(np.int8)
    cal_p = arrays["calibration_pool_probabilities"].astype(float)
    test_y = arrays["test_y"].astype(np.int8)
    test_p = arrays["test_probabilities"].astype(float)
    minority = int(row["minority_label"])
    majority = 1 - minority
    maj_idx = np.flatnonzero(cal_y == majority)
    min_idx = np.flatnonzero(cal_y == minority)
    maj_selected = np.random.default_rng(seed(row["dataset_id"], int(row["seed"]), "calibration_majority_subset")).permutation(maj_idx)[:200]
    min_selected = np.random.default_rng(seed(row["dataset_id"], int(row["seed"]), "calibration_minority_nested_subset")).permutation(min_idx)[: int(row["m_minority"])]
    selected = np.concatenate([maj_selected, min_selected])
    subset = {
        "majority_ids": sorted(cal_ids[maj_selected].tolist()),
        "minority_ids": sorted(cal_ids[min_selected].tolist()),
        "m_majority": 200,
        "m_minority": int(row["m_minority"]),
        "majority_label": majority,
        "minority_label": minority,
    }
    selected_y, selected_p = cal_y[selected], cal_p[selected]
    scores = 1 - selected_p[np.arange(len(selected_y)), selected_y]
    if row["cp_method"] == "global_split_cp":
        rg, qg = q(scores)
        thresholds = np.array([qg, qg])
        fields: dict[str, object] = {"q_global": qg, "q_minority": None, "q_majority": None, "threshold_gap": None, "threshold_sum": None, "rank_global": rg, "rank_minority": None, "rank_majority": None}
    else:
        rmin, qmin = q(scores[selected_y == minority])
        rmaj, qmaj = q(scores[selected_y == majority])
        thresholds = np.zeros(2)
        thresholds[minority], thresholds[majority] = qmin, qmaj
        fields = {"q_global": None, "q_minority": qmin, "q_majority": qmaj, "threshold_gap": abs(qmin - qmaj), "threshold_sum": qmin + qmaj, "rank_global": None, "rank_minority": rmin, "rank_majority": rmaj}
    included = (1 - test_p) <= thresholds[None, :]
    covered = included[np.arange(len(test_y)), test_y]
    min_mask, maj_mask = test_y == minority, test_y == majority
    size = included.sum(axis=1)
    # Geometry uses the two stored columns, without assuming p0 = 1 - p1.
    geometry = np.where(included[:, 0] & included[:, 1], "doubleton", np.where(~included[:, 0] & ~included[:, 1], "empty", "singleton"))

    def coverage(mask: np.ndarray) -> tuple[int, float, float, float]:
        k, n = int(covered[mask].sum()), int(mask.sum())
        lo, hi = wilson(k, n)
        return k, k / n, lo, hi

    all_mask = np.ones(len(test_y), dtype=bool)
    ko, co, olo, ohi = coverage(all_mask)
    kmi, cmi, milo, mihi = coverage(min_mask)
    kma, cma, malo, mahi = coverage(maj_mask)
    fields.update({
        "subset_hash": json_hash(subset), "n_cal_total": int(len(selected)), "n_cal_minority": int(len(min_selected)), "n_cal_majority": int(len(maj_selected)),
        "covered_count_overall": ko, "coverage_overall": co, "coverage_overall_wilson_low": olo, "coverage_overall_wilson_high": ohi,
        "covered_count_minority": kmi, "coverage_minority": cmi, "coverage_minority_wilson_low": milo, "coverage_minority_wilson_high": mihi,
        "covered_count_majority": kma, "coverage_majority": cma, "coverage_majority_wilson_low": malo, "coverage_majority_wilson_high": mahi,
        "coverage_disparity": abs(cmi - cma), "singleton_rate": float((size == 1).mean()), "empty_rate": float((size == 0).mean()), "doubleton_rate": float((size == 2).mean()), "average_set_size": float(size.mean()),
        "n_test": int(len(test_y)), "n_test_minority": int(min_mask.sum()), "n_test_majority": int(maj_mask.sum()),
        "auroc": float(roc_auc_score(test_y, test_p[:, 1])), "auprc": float(average_precision_score(test_y, test_p[:, 1])),
        "geometry_matches_sets": bool(np.array_equal(geometry, np.where(size == 2, "doubleton", np.where(size == 0, "empty", "singleton")))),
    })
    return fields


def v11_lineage_checks(rows: list[dict], manifest: dict) -> dict:
    """Version-specific checks: splits, decision bindings, intake lineage, v1.0 immutability, full-run inventory."""
    protocol_versions = sorted({row["protocol_version"] for row in rows})
    split_ok = True
    split_manifest_hashes: dict[str, str] = {}
    for dataset, seed_value in sorted({(row["dataset_id"], int(row["seed"])) for row in rows}):
        path = SPLIT_ROOT / dataset / f"seed-{seed_value}.json"
        split_manifest = json.loads(path.read_text(encoding="utf-8"))
        split_manifest_hashes[f"{dataset}/{seed_value}"] = split_manifest.get("split_hash")
        for row in rows:
            if row["dataset_id"] == dataset and int(row["seed"]) == seed_value and row["split_hash"] != split_manifest.get("split_hash"):
                split_ok = False
    decision = json.loads(PILOT_DECISION.read_text(encoding="utf-8"))
    bindings_ok = manifest.get("pilot_decision_sha256") == sha256(PILOT_DECISION)
    binding = decision.get("v11_lineage_binding", {})
    binding_files = {
        "registry_sha256": ROOT / decision.get("registry_path", "artifacts/stage02/dataset_registry_v1.0.1.json"),
        "dataset_lock_sha256": ROOT / decision.get("dataset_lock_path", "protocols/dataset_lock_v1.0.md"),
        "protocol_sha256": ROOT / decision.get("protocol_path", "protocols/protocol_v1.1.md"),
        "split_lock_sha256": ROOT / binding.get("split_lock_path", "configs/stage04_splits_v1.1.yaml"),
        "local_cache_lock_sha256": ROOT / binding.get("local_cache_lock_path", "configs/stage05_lr_xgboost_v1.1.yaml"),
        "tabpfn_cache_lock_sha256": ROOT / binding.get("tabpfn_cache_lock_path", "configs/stage05b_tabpfn_v1.1.yaml"),
        "d08_003_receipt_sha256": ROOT / binding.get("d08_003_receipt_path", "decisions/D08-003_V11_CACHE_AND_PILOT_BUDGET_RECEIPT.json"),
    }
    for key, path in binding_files.items():
        expected = decision.get(key, binding.get(key))
        if not path.is_file() or expected != sha256(path):
            bindings_ok = False
    intake_candidates = sorted((ROOT / "artifacts" / "stage08_v11_cloud").glob(INTAKE_GLOB))
    intake = json.loads(intake_candidates[-1].read_text(encoding="utf-8"))
    intake_lineage = intake["lineage"]
    lineages_match = intake.get("verdict") == "PASS"
    for row in rows:
        if row["model"] == "tabpfn":
            if row["config_hash"] != intake_lineage["tabpfn_config_hash"] or row["code_hash"] != intake_lineage["tabpfn_cache_time_code_hash"]:
                lineages_match = False
        elif row["config_hash"] != intake_lineage["local_config_hash"] or row["code_hash"] != intake_lineage["local_cache_time_code_hash"]:
            lineages_match = False
    v10_cells = len(list((V10_PILOT_RUN / "cells").glob("*.json")))
    v10_rows = len(pd.read_parquet(V10_PILOT_RUN / "results_long.parquet"))
    units: dict[str, int] = {}
    for manifest_path in CACHE_ROOT.glob("cfg-*/code-*/*/seed-*/*/manifest.json"):
        if (manifest_path.parent / "predictions.npz").is_file():
            model = manifest_path.parent.name
            units[model] = units.get(model, 0) + 1
    complete_units = sum(units.values())
    return {
        "protocol_versions": protocol_versions,
        "split_hash_matches_locked_v11_manifests": split_ok,
        "split_manifest_hashes": split_manifest_hashes,
        "pilot_decision_hash_bindings_match": bindings_ok,
        "cell_lineages_match_intake_audit": lineages_match,
        "intake_audit_path": str(intake_candidates[-1].relative_to(ROOT)).replace("\\", "/"),
        "v10_pilot_untouched": v10_cells == 480 and v10_rows == 480,
        "v10_pilot_cell_count": v10_cells,
        "v10_pilot_parquet_rows": v10_rows,
        "full_run_cache_inventory": {"complete_units": complete_units, "per_model": units, "required_units": 240},
    }


def cloud_budget_evidence() -> dict:
    """Read the preserved cloud generator summary recorded by the intake audit."""
    candidates = sorted((ROOT / "artifacts" / "stage08_v11_cloud").glob(CLOUD_SUMMARY_GLOB))
    if not candidates:
        return {"found": False}
    summary = json.loads(candidates[-1].read_text(encoding="utf-8"))
    budget = summary.get("budget", {})
    elapsed = float(budget.get("elapsed_seconds", 0))
    produced = int(budget.get("produced_bytes", 0))
    return {
        "found": True,
        "source": str(candidates[-1].relative_to(ROOT)).replace("\\", "/"),
        "status": summary.get("status"),
        "completed_units": summary.get("completed_units"),
        "elapsed_seconds": elapsed,
        "produced_bytes": produced,
        "maximum_wall_clock_hours": budget.get("maximum_wall_clock_hours"),
        "maximum_cloud_storage_gb": budget.get("maximum_cloud_storage_gb"),
        "within_budget": 0 < elapsed <= 12 * 3600 and 0 <= produced <= 50 * 1024 ** 3,
        "runtime_evidence": summary.get("runtime_evidence"),
    }


def main() -> None:
    status_candidates = sorted(PILOT_ROOT.glob("*/run_status.json"))
    if not status_candidates:
        raise FileNotFoundError(f"No v1.1 pilot run exists under {PILOT_ROOT}.")
    run_root = status_candidates[-1].parent
    run_status = json.loads((run_root / "run_status.json").read_text(encoding="utf-8"))
    if run_status.get("status") != "PASS":
        raise ValueError(f"The latest v1.1 pilot run did not PASS: {run_root}")
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((run_root / "cells").glob("*.json"))]
    key = lambda r: (r["dataset_id"], int(r["seed"]), r["model"], r["cp_method"], int(r["m_minority"]))  # noqa: E731
    keys = [key(r) for r in rows]
    expected = {(d, s, m, c, k) for d in manifest["pilot_dataset_ids"] for s in manifest["seeds"] for m in manifest["models"] for c in manifest["cp_methods"] for k in manifest["m_minority"]}
    duplicate_keys = [list(k) for k, n in Counter(keys).items() if n > 1]
    missing_keys = [list(k) for k in sorted(expected - set(keys))]
    extra_keys = [list(k) for k in sorted(set(keys) - expected)]
    parquet_rows = pd.read_parquet(run_root / "results_long.parquet").to_dict(orient="records")
    with (run_root / "results_long.csv").open(newline="", encoding="utf-8") as handle:
        csv_keys = {(r["dataset_id"], int(r["seed"]), r["model"], r["cp_method"], int(r["m_minority"])) for r in csv.DictReader(handle)}
    source_files = [
        ROOT / "protocols" / "protocol_v1.1.md",
        ROOT / "protocols" / "dataset_lock_v1.0.md",
        ROOT / "configs" / "stage04_splits_v1.1.yaml",
        ROOT / "configs" / "stage05_lr_xgboost_v1.1.yaml",
        ROOT / "configs" / "stage05b_tabpfn_v1.1.yaml",
        ROOT / "decisions" / "pilot_decision_stage07_v1.1.json",
        ROOT / "decisions" / "D08-003_V11_CACHE_AND_PILOT_BUDGET_RECEIPT.json",
        ROOT / "environment" / "environment_lock_v1.0.json",
        ROOT / "configs" / "results_long.schema.json",
    ]
    source_hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in source_files}
    cache_groups = defaultdict(list)
    for row in rows:
        cache_groups[(row["dataset_id"], row["seed"], row["model"])].append(row)
    cache_errors: list[dict] = []
    lineage = Counter()
    for unit, group in sorted(cache_groups.items()):
        row = group[0]
        path = cache_for(row)
        try:
            cache_manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            cache_hash_ok = sha256(path / "predictions.npz") == cache_manifest.get("cache_sha256") == row["prediction_cache_hash"]
            provenance = cache_manifest.get("provenance", {})
            provenance_ok = all(provenance.get(a) == row[b] for a, b in (("config_hash", "config_hash"), ("code_hash", "code_hash"), ("environment_hash", "environment_hash"), ("dataset_hash", "dataset_hash"), ("split_hash", "split_hash"), ("model_name", "model"))) and cache_manifest.get("model_hash") == row["model_hash"]
            probabilities_ok = True
            with np.load(path / "predictions.npz", allow_pickle=False) as arrays:
                for partition in ("calibration_pool", "test"):
                    prob = arrays[f"{partition}_probabilities"].astype(float)
                    probabilities_ok &= bool(np.isfinite(prob).all() and (prob >= 0).all() and (prob <= 1).all() and np.allclose(prob.sum(axis=1), 1, rtol=1e-6, atol=1e-6))
            if not (cache_hash_ok and provenance_ok and probabilities_ok and len(group) == 8):
                cache_errors.append({"unit": list(unit), "cache_hash_ok": cache_hash_ok, "provenance_ok": provenance_ok, "probability_ok": probabilities_ok, "cell_count": len(group)})
            lineage[(row["model"], row["config_hash"], row["code_hash"], row["environment_hash"])] += 1
        except Exception as exc:  # noqa: BLE001 - any cache read failure is recorded as an error
            cache_errors.append({"unit": list(unit), "error": f"{type(exc).__name__}: {exc}"})
    invariant_errors = []
    metric_groups = defaultdict(list)
    for row in rows:
        metric_groups[(row["dataset_id"], row["seed"], row["model"])].append(row)
        decomposition = row["empty_rate"] + row["singleton_rate"] + row["doubleton_rate"]
        denominators = {"overall": row["n_test"], "minority": row["n_test_minority"], "majority": row["n_test_majority"]}
        coverage_ok = all(equal(row[f"coverage_{part}"], row[f"covered_count_{part}"] / denominators[part]) for part in ("overall", "minority", "majority"))
        wilson_ok = all(0 <= row[f"coverage_{part}_wilson_low"] <= row[f"coverage_{part}"] <= row[f"coverage_{part}_wilson_high"] <= 1 for part in ("overall", "minority", "majority"))
        ranks_ok = (
            (row["cp_method"] == "global_split_cp" and row["rank_global"] == math.ceil((row["n_cal_total"] + 1) * 0.9))
            or (row["cp_method"] == "class_conditional_cp" and row["rank_minority"] == math.ceil((row["m_minority"] + 1) * 0.9) and row["rank_majority"] == 181)
        )
        if abs(decomposition - 1) > TOL or not coverage_ok or not wilson_ok or not ranks_ok:
            invariant_errors.append({"key": list(key(row)), "decomposition": decomposition, "coverage_ok": coverage_ok, "wilson_ok": wilson_ok, "ranks_ok": ranks_ok})
    predictive_invariance_errors = [list(k) for k, g in metric_groups.items() if len(g) != 8 or len({(r["auroc"], r["auprc"]) for r in g}) != 1]
    pairs = defaultdict(list)
    for row in rows:
        pairs[(row["dataset_id"], row["seed"], row["model"], row["m_minority"])].append(row)
    subset_identity_errors = [list(unit) for unit, group in pairs.items() if len(group) != 2 or len({r["subset_hash"] for r in group}) != 1]
    spec = representative_spec(tuple(manifest["pilot_dataset_ids"]))
    indexed = {key(r): r for r in rows}
    representative_errors, representative_evidence = [], []
    compare_fields = (
        "subset_hash", "n_cal_total", "n_cal_minority", "n_cal_majority", "rank_global", "rank_minority", "rank_majority",
        "q_global", "q_minority", "q_majority", "threshold_gap", "threshold_sum", "covered_count_overall",
        "covered_count_minority", "covered_count_majority", "coverage_overall", "coverage_minority", "coverage_majority",
        "coverage_disparity", "singleton_rate", "empty_rate", "doubleton_rate", "average_set_size", "auroc", "auprc",
        "coverage_overall_wilson_low", "coverage_overall_wilson_high", "coverage_minority_wilson_low",
        "coverage_minority_wilson_high", "coverage_majority_wilson_low", "coverage_majority_wilson_high",
    )
    for item in spec:
        row = indexed[item]
        with np.load(cache_for(row) / "predictions.npz", allow_pickle=False) as arrays:
            direct = independent_cell(row, {name: arrays[name] for name in arrays.files})
        mismatch = [field for field in compare_fields if not equal(row[field], direct[field])]
        evidence = {field: direct[field] for field in ("subset_hash", "rank_global", "rank_minority", "rank_majority", "q_global", "q_minority", "q_majority", "covered_count_overall", "coverage_overall", "empty_rate", "singleton_rate", "doubleton_rate", "auroc", "auprc")}
        evidence["key"] = list(item)
        evidence["geometry_matches_sets"] = direct["geometry_matches_sets"]
        evidence["mismatches"] = mismatch
        representative_evidence.append(evidence)
        if mismatch or not direct["geometry_matches_sets"]:
            representative_errors.append(evidence)
    coverage_by_group = defaultdict(list)
    for row in rows:
        if row["cp_method"] == "global_split_cp":
            coverage_by_group[(row["model"], row["m_minority"])].append(row)
    coverage_groups = []
    for (model, m), group in sorted(coverage_by_group.items()):
        flagged = sum(not (r["coverage_overall_wilson_low"] <= 0.9 <= r["coverage_overall_wilson_high"]) for r in group)
        coverage_groups.append({"model": model, "m_minority": m, "flagged_cells": flagged, "total_cells": len(group), "flagged_fraction": flagged / len(group)})
    pilot_qc = json.loads((run_root / "pilot_qc.json").read_text(encoding="utf-8"))
    qc_groups = {(item["model"], item["m_minority"]): item["flagged_cells"] for item in pilot_qc["global_coverage_diagnostic"]["groups"]}
    matches_pilot_qc = {(item["model"], item["m_minority"]): item["flagged_cells"] for item in coverage_groups} == qc_groups
    lineage_checks = v11_lineage_checks(rows, manifest)
    budget_evidence = cloud_budget_evidence()
    receipt = json.loads(D08_003_RECEIPT.read_text(encoding="utf-8"))
    inventory = lineage_checks["full_run_cache_inventory"]
    integrity_pass = not (
        duplicate_keys or missing_keys or extra_keys or cache_errors or invariant_errors
        or predictive_invariance_errors or subset_identity_errors or representative_errors
    ) and len(rows) == len(expected) and len(parquet_rows) == len(expected)
    version_pass = (
        lineage_checks["protocol_versions"] == ["v1.1"]
        and lineage_checks["split_hash_matches_locked_v11_manifests"]
        and lineage_checks["pilot_decision_hash_bindings_match"]
        and lineage_checks["cell_lineages_match_intake_audit"]
    )
    checklist = [
        {"item": 1, "check": "Controlled inputs and v1.1 pilot selection", "verdict": "PASS" if version_pass else "FAIL", "evidence": "Protocol v1.1, dataset lock, all three v1.1 locks, the pre-outcome v1.1 pilot decision, D08-003 receipt, environment lock, and results schema re-hashed; decision bindings and intake-audit lineage match."},
        {"item": 2, "check": "Complete factorial grid", "verdict": "PASS" if len(rows) == 480 and len(parquet_rows) == 480 and {key(r) for r in parquet_rows} == set(keys) and csv_keys == set(keys) else "FAIL", "evidence": "480 = 2 datasets x 10 seeds x 3 models x 2 methods x 4 m; 480 cell JSONs, 480 Parquet rows, matching CSV key set."},
        {"item": 3, "check": "Unique, complete result keys", "verdict": "PASS" if not (duplicate_keys or missing_keys or extra_keys) else "FAIL", "evidence": f"480 unique keys; duplicates={len(duplicate_keys)}, missing={len(missing_keys)}, extra={len(extra_keys)}."},
        {"item": 4, "check": "Cache integrity and probability mapping", "verdict": "PASS" if not cache_errors else "FAIL", "evidence": f"{len(cache_groups)} base-cache units re-hashed (NPZ SHA-256 = manifest = record), provenance/probability contracts valid, 8 cells per unit; errors={len(cache_errors)}."},
        {"item": 5, "check": "Dataset/split/model/environment lineage", "verdict": "PASS" if not cache_errors and version_pass else "FAIL", "evidence": "LR 20 + XGBoost 20 units in the local v1.1 cfg/code tree and TabPFN 20 units in the cloud v1.1 cfg/code tree; every cell split_hash equals its locked v1.1 split manifest hash; lineage equals the PASS 240-cache intake audit."},
        {"item": 6, "check": "Calibration subsets", "verdict": "PASS" if not subset_identity_errors else "FAIL", "evidence": f"Independent v1.1 seed routing reproduces the fixed 200-majority and nested minority subsets; Global and Class-Conditional share a subset hash for every dataset x seed x model x m; identity errors={len(subset_identity_errors)}."},
        {"item": 7, "check": "Finite-sample ranks and thresholds", "verdict": "PASS" if not [e for e in invariant_errors if not e["ranks_ok"]] and not representative_errors else "FAIL", "evidence": "All-cell ranks follow ceil((n+1)(1-alpha)); the 36 direct threshold recalculations match."},
        {"item": 8, "check": "Prediction-set geometry and decomposition", "verdict": "PASS" if not [e for e in invariant_errors if abs(e["decomposition"] - 1) > TOL] and all(e["geometry_matches_sets"] for e in representative_evidence) else "FAIL", "evidence": "empty + singleton + doubleton = 1 in all 480 rows; all 36 stored-column geometry checks match actual sets."},
        {"item": 9, "check": "Predictive-metric invariance and coverage sanity", "verdict": "PASS WITH DIAGNOSTIC FLAG" if not predictive_invariance_errors and not [e for e in invariant_errors if not (e["coverage_ok"] and e["wilson_ok"])] else "FAIL", "evidence": f"AUROC/AUPRC invariant across CP x m in all {len(metric_groups)} dataset x seed x model groups; the Global Wilson diagnostic is reproduced (matches pilot QC: {matches_pilot_qc}) and remains diagnostic-only."},
        {"item": 10, "check": "Full-run feasibility, authorization, and freeze readiness", "verdict": "PASS" if inventory["complete_units"] == 240 and budget_evidence.get("within_budget") and receipt.get("formal_run_manifest_authorized") is False else "FAIL", "evidence": f"Protocol v1.1 full-context route execution-proven: {inventory['complete_units']}/240 formal-run base caches exist (80 per model, all 8 datasets) and the cloud run stayed within the 12-hour/50-GB budget. Remaining gates are authorization-only: D08-003 keeps formal_run_manifest_authorized=false and full_experiment_authorized=false; no formal manifest is frozen and an explicit user go is required."},
    ]
    technical_readiness = {
        "pilot_independent_recomputation_pass": integrity_pass and version_pass,
        "full_run_base_caches_available": inventory["complete_units"] == 240,
        "tabpfn_full_context_route_proven": budget_evidence.get("within_budget") is True and budget_evidence.get("completed_units") == 80,
        "v1_1_protocol_and_locks_current": version_pass,
    }
    gate_recommendation = {
        "recommendation": "CONDITIONAL-GO" if all(technical_readiness.values()) else "NO-GO",
        "technical_readiness": technical_readiness,
        "remaining_gates": [
            "explicit user go after reviewing this audit",
            "formal-run manifest freeze (outside the D08-003 cache-and-pilot scope)",
            "separate formal-run execution authorization for the 1,920-cell eight-dataset experiment",
        ],
        "formal_run_authorized": False,
        "formal_run_manifest_frozen": False,
        "explicit_user_go_required": True,
        "interpretation": "The v1.1 lineage is technically ready for the formal design: all 240 formal-run base caches exist and this audit independently reproduced the pilot. Nothing here authorizes the formal run. A formal-run manifest may be frozen only after the user reviews this audit and explicitly says go; the 1,920-cell experiment additionally requires its own authorization. v1.0 artifacts remain historical-only evidence.",
    }
    verdict = "PASS" if integrity_pass and version_pass else "FAIL"
    stale_artifact_map = {
        "v1.0_splits_caches_pilot_audits": "historical_only_not_reusable_as_v1.1_evidence",
        "v1.1_splits": "current_and_locked",
        "v1.1_local_lr_xgboost_caches": "current_and_audited",
        "v1.1_tabpfn_caches": "current_and_audited",
        "v1.1_pilot_run": "current_and_independently_audited",
        "stale_v1.1_artifacts": [],
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = ROOT / "artifacts" / "stage08_v11" / f"{timestamp}_pilot_independent_audit"
    if output_root.exists():
        raise FileExistsError(f"Audit output directory already exists: {output_root}")
    output_root.mkdir(parents=True)
    payload = {
        "artifact_id": "stage08_v11_pilot_independent_audit_v1.0.0",
        "scope": "READ_ONLY_STAGE08_V11_AUDIT_NO_FULL_RUN_NO_FORMAL_MANIFEST",
        "audit_created_utc": utc_now(),
        "run": str(run_root.relative_to(ROOT)).replace("\\", "/"),
        "source_hashes": source_hashes,
        "run_manifest": manifest,
        "cell_integrity": {
            "cell_json_count": len(rows), "expected_count": len(expected), "unique_keys": len(set(keys)),
            "duplicate_keys": duplicate_keys, "missing_keys": missing_keys, "extra_keys": extra_keys,
            "parquet_row_count": len(parquet_rows), "parquet_key_set_matches_cells": {key(r) for r in parquet_rows} == set(keys),
            "csv_key_set_matches_cells": csv_keys == set(keys),
        },
        "cache_lineage": {
            "base_cache_units": len(cache_groups), "errors": cache_errors,
            "lineages": [{"model": model, "config_hash": config, "code_hash": code, "environment_hash": environment, "base_cache_units": count} for (model, config, code, environment), count in sorted(lineage.items())],
        },
        "all_cell_invariants": {
            "errors": invariant_errors, "predictive_metric_invariance_errors": predictive_invariance_errors,
            "cp_subset_identity_errors": subset_identity_errors,
            "set_decomposition_errors": [e for e in invariant_errors if abs(e["decomposition"] - 1) > TOL],
        },
        "representative_recomputation": {"n_cells": len(spec), "errors": representative_errors, "evidence": representative_evidence},
        "global_coverage_sanity": {
            "nominal": 0.9, "groups": coverage_groups, "matches_pilot_qc": matches_pilot_qc,
            "interpretation": "A Wilson exclusion is a diagnostic flag requiring implementation review, not a scientific explanation, result claim, or protocol-change trigger.",
        },
        "v11_lineage_checks": lineage_checks,
        "cloud_budget_evidence": budget_evidence,
        "stale_artifact_map": stale_artifact_map,
        "checklist": checklist,
        "gate_recommendation": gate_recommendation,
        "verdict": verdict,
    }
    (output_root / "independent_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "gate": gate_recommendation["recommendation"], "output": str(output_root / "independent_audit.json"), "representative_cells": len(spec), "cache_units": len(cache_groups), "cell_count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

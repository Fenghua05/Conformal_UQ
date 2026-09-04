"""Read-only independent Stage 08 audit of the Stage 07 pilot.

This tool intentionally does not import project conformal/evaluation helpers.
It reimplements the frozen calculations directly from immutable caches, so its
representative-cell evidence is independent of the Stage 07 implementation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts" / "stage07" / "20260830T161214Z_stage07-pilot_32b7e4728b8d"
OUT = ROOT / "artifacts" / "stage08" / "20260831T000000Z_stage08_pilot_independent_audit"
ALPHA = 0.1
M_VALUES = (10, 20, 50, 100)
MODELS = ("logistic_regression", "xgboost", "tabpfn")
METHODS = ("global_split_cp", "class_conditional_cp")
TOL = 1e-12
Z = 1.959963984540054


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def seed(dataset: str, base_seed: int, purpose: str) -> int:
    raw = f"v1.0|{dataset}|{base_seed}|{purpose}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big", signed=False)


def q(scores: np.ndarray) -> tuple[int, float]:
    rank = math.ceil((len(scores) + 1) * (1 - ALPHA))
    return rank, float(np.sort(scores)[rank - 1]) if rank <= len(scores) else math.inf


def wilson(k: int, n: int) -> tuple[float, float]:
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


def cache_for(row: dict) -> Path:
    return ROOT / "artifacts" / "caches" / f"cfg-{row['config_hash'][:12]}" / f"code-{row['code_hash'][:12]}" / row["dataset_id"] / f"seed-{row['seed']}" / row["model"]


def independent_cell(row: dict, arrays: dict[str, np.ndarray]) -> dict:
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
    # Geometry is calculated from the two stored columns, without p0=1-p1.
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


def main() -> None:
    manifest = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((RUN / "cells").glob("*.json"))]
    expected = {(d, s, m, c, k) for d in manifest["pilot_dataset_ids"] for s in manifest["seeds"] for m in manifest["models"] for c in manifest["cp_methods"] for k in manifest["m_minority"]}
    key = lambda r: (r["dataset_id"], int(r["seed"]), r["model"], r["cp_method"], int(r["m_minority"]))
    keys = [key(r) for r in rows]
    duplicate_keys = [list(k) for k, n in Counter(keys).items() if n > 1]
    missing_keys = [list(k) for k in sorted(expected - set(keys))]
    extra_keys = [list(k) for k in sorted(set(keys) - expected)]
    parquet_rows = pd.read_parquet(RUN / "results_long.parquet").to_dict(orient="records")
    with (RUN / "results_long.csv").open(newline="", encoding="utf-8") as f:
        csv_keys = {(r["dataset_id"], int(r["seed"]), r["model"], r["cp_method"], int(r["m_minority"])) for r in csv.DictReader(f)}
    source_files = [ROOT / "protocols" / "protocol_v1.0.md", ROOT / "protocols" / "dataset_lock_v1.0.md", ROOT / "configs" / "stage05b_tabpfn_v1.0.yaml", ROOT / "decisions" / "pilot_decision_stage07_v1.0.json", ROOT / "environment" / "environment_lock_v1.0.json", ROOT / "configs" / "results_long.schema.json"]
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in source_files}
    cache_groups = defaultdict(list)
    for row in rows:
        cache_groups[(row["dataset_id"], row["seed"], row["model"])].append(row)
    cache_errors: list[dict] = []
    lineage = Counter()
    for unit, group in sorted(cache_groups.items()):
        row = group[0]
        path = cache_for(row)
        try:
            m = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            cache_hash_ok = sha256(path / "predictions.npz") == m.get("cache_sha256") == row["prediction_cache_hash"]
            p = m.get("provenance", {})
            prov_ok = all(p.get(a) == row[b] for a, b in (("config_hash", "config_hash"), ("code_hash", "code_hash"), ("environment_hash", "environment_hash"), ("dataset_hash", "dataset_hash"), ("split_hash", "split_hash"), ("model_name", "model"))) and m.get("model_hash") == row["model_hash"]
            probs_ok = True
            with np.load(path / "predictions.npz", allow_pickle=False) as z:
                for partition in ("calibration_pool", "test"):
                    prob = z[f"{partition}_probabilities"].astype(float)
                    probs_ok &= bool(np.isfinite(prob).all() and (prob >= 0).all() and (prob <= 1).all() and np.allclose(prob.sum(axis=1), 1, rtol=1e-6, atol=1e-6))
            if not (cache_hash_ok and prov_ok and probs_ok and len(group) == 8):
                cache_errors.append({"unit": unit, "cache_hash_ok": cache_hash_ok, "provenance_ok": prov_ok, "probability_ok": probs_ok, "cell_count": len(group)})
            lineage[(row["model"], row["config_hash"], row["code_hash"], row["environment_hash"])] += 1
        except Exception as exc:
            cache_errors.append({"unit": unit, "error": f"{type(exc).__name__}: {exc}"})
    inv_errors = []
    metric_groups = defaultdict(list)
    for row in rows:
        metric_groups[(row["dataset_id"], row["seed"], row["model"])].append(row)
        parts = row["empty_rate"] + row["singleton_rate"] + row["doubleton_rate"]
        denominators = {"overall": row["n_test"], "minority": row["n_test_minority"], "majority": row["n_test_majority"]}
        coverage_ok = all(equal(row[f"coverage_{part}"], row[f"covered_count_{part}"] / denominators[part]) for part in ("overall", "minority", "majority"))
        wilson_ok = all(0 <= row[f"coverage_{part}_wilson_low"] <= row[f"coverage_{part}"] <= row[f"coverage_{part}_wilson_high"] <= 1 for part in ("overall", "minority", "majority"))
        ranks_ok = ((row["cp_method"] == "global_split_cp" and row["rank_global"] == math.ceil((row["n_cal_total"] + 1) * .9)) or (row["cp_method"] == "class_conditional_cp" and row["rank_minority"] == math.ceil((row["m_minority"] + 1) * .9) and row["rank_majority"] == 181))
        if abs(parts - 1) > TOL or not coverage_ok or not wilson_ok or not ranks_ok:
            inv_errors.append({"key": list(key(row)), "decomposition": parts, "coverage_ok": coverage_ok, "wilson_ok": wilson_ok, "ranks_ok": ranks_ok})
    predictive_invariance_errors = [list(k) for k, g in metric_groups.items() if len(g) != 8 or len({(r["auroc"], r["auprc"]) for r in g}) != 1]
    subset_identity_errors = []
    for (dataset, base_seed, model, m), g in defaultdict(list, {k: [] for k in []}).items():
        pass
    pairs = defaultdict(list)
    for row in rows:
        pairs[(row["dataset_id"], row["seed"], row["model"], row["m_minority"])].append(row)
    for unit, group in pairs.items():
        if len(group) != 2 or len({r["subset_hash"] for r in group}) != 1:
            subset_identity_errors.append(list(unit))
    # Representative cells span each dataset/model/CP method/m value, on four different seeds.
    representative_spec = []
    for dataset in manifest["pilot_dataset_ids"]:
        for model in MODELS:
            representative_spec += [(dataset, 104729, model, "global_split_cp", 10), (dataset, 552721, model, "global_split_cp", 100), (dataset, 130363, model, "class_conditional_cp", 10), (dataset, 262147, model, "class_conditional_cp", 20), (dataset, 374209, model, "class_conditional_cp", 50), (dataset, 481517, model, "class_conditional_cp", 100)]
    indexed = {key(r): r for r in rows}
    rep_errors, rep_evidence = [], []
    compare_fields = ("subset_hash", "n_cal_total", "n_cal_minority", "n_cal_majority", "rank_global", "rank_minority", "rank_majority", "q_global", "q_minority", "q_majority", "threshold_gap", "threshold_sum", "covered_count_overall", "covered_count_minority", "covered_count_majority", "coverage_overall", "coverage_minority", "coverage_majority", "coverage_disparity", "singleton_rate", "empty_rate", "doubleton_rate", "average_set_size", "auroc", "auprc", "coverage_overall_wilson_low", "coverage_overall_wilson_high", "coverage_minority_wilson_low", "coverage_minority_wilson_high", "coverage_majority_wilson_low", "coverage_majority_wilson_high")
    for item in representative_spec:
        row = indexed[item]
        with np.load(cache_for(row) / "predictions.npz", allow_pickle=False) as z:
            direct = independent_cell(row, {k: z[k] for k in z.files})
        mismatch = [field for field in compare_fields if not equal(row[field], direct[field])]
        evidence = {field: direct[field] for field in ("subset_hash", "rank_global", "rank_minority", "rank_majority", "q_global", "q_minority", "q_majority", "covered_count_overall", "coverage_overall", "empty_rate", "singleton_rate", "doubleton_rate", "auroc", "auprc")}
        evidence["key"] = list(item); evidence["geometry_matches_sets"] = direct["geometry_matches_sets"]; evidence["mismatches"] = mismatch
        rep_evidence.append(evidence)
        if mismatch or not direct["geometry_matches_sets"]:
            rep_errors.append(evidence)
    global_rows = [r for r in rows if r["cp_method"] == "global_split_cp"]
    coverage_groups = []
    coverage_by_group = defaultdict(list)
    for row in global_rows:
        coverage_by_group[(row["model"], row["m_minority"])].append(row)
    for (model, m), g in sorted(coverage_by_group.items()):
        flagged = sum(not (r["coverage_overall_wilson_low"] <= .9 <= r["coverage_overall_wilson_high"]) for r in g)
        coverage_groups.append({"model": model, "m_minority": m, "flagged_cells": flagged, "total_cells": len(g), "flagged_fraction": flagged / len(g)})
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_id": "stage08_pilot_independent_audit_v1.0.0", "scope": "READ_ONLY_STAGE08_AUDIT_NO_FULL_RUN_NO_FORMAL_MANIFEST", "run": str(RUN.relative_to(ROOT)),
        "source_hashes": source_hashes, "run_manifest": manifest,
        "cell_integrity": {"cell_json_count": len(rows), "expected_count": len(expected), "unique_keys": len(set(keys)), "duplicate_keys": duplicate_keys, "missing_keys": missing_keys, "extra_keys": extra_keys, "parquet_row_count": len(parquet_rows), "parquet_key_set_matches_cells": {key(r) for r in parquet_rows} == set(keys), "csv_key_set_matches_cells": csv_keys == set(keys)},
        "cache_lineage": {"base_cache_units": len(cache_groups), "errors": cache_errors, "lineages": [{"model": a, "config_hash": b, "code_hash": c, "environment_hash": d, "base_cache_units": n} for (a, b, c, d), n in sorted(lineage.items())]},
        "all_cell_invariants": {"errors": inv_errors, "predictive_metric_invariance_errors": predictive_invariance_errors, "cp_subset_identity_errors": subset_identity_errors, "set_decomposition_errors": [e for e in inv_errors if abs(e["decomposition"] - 1) > TOL]},
        "representative_recomputation": {"n_cells": len(representative_spec), "errors": rep_errors, "evidence": rep_evidence},
        "global_coverage_sanity": {"nominal": .9, "groups": coverage_groups, "interpretation": "A Wilson exclusion is a diagnostic flag, not a scientific explanation."},
        "verdict": "PASS" if not (duplicate_keys or missing_keys or extra_keys or cache_errors or inv_errors or predictive_invariance_errors or subset_identity_errors or rep_errors) and len(rows) == len(expected) and len(parquet_rows) == len(expected) else "FAIL",
    }
    (OUT / "independent_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"], "output": str(OUT / "independent_audit.json"), "representative_cells": len(representative_spec), "cache_units": len(cache_groups), "cell_count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

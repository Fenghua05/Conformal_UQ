"""Read-only Stage 06 validation on one existing Stage 05 probability cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conformal_uq.conformal import CP_METHODS, M_MINORITY, evaluate_split_cp, select_nested_calibration_subsets
from conformal_uq.data import load_dataset_registry, load_locked_dataset, registry_record
from conformal_uq.metrics import binary_predictive_metrics, global_coverage_sanity
from conformal_uq.prediction_cache import read_valid_cache
from conformal_uq.provenance import sha256_path
from conformal_uq.split import make_stratified_split


def _label_mapping_hash(mapping: dict[str, int]) -> str:
    value = json.dumps(mapping, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _minority_label(root: Path, dataset_id: str) -> int:
    record = registry_record(load_dataset_registry(root), dataset_id)
    original = str(record["minority_original_label"])
    return int(record["label_mapping_to_protocol_binary"][original])


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate one existing probability cache without training or writing research results.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dataset", default="openml_3_kr_vs_kp")
    parser.add_argument("--seed", type=int, default=104729)
    parser.add_argument("--model", default="logistic_regression", choices=("logistic_regression", "xgboost"))
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    cache_dir = args.cache_dir.resolve()
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    provenance = manifest["provenance"]
    if provenance["model_name"] != args.model or int(provenance["base_seed"]) != args.seed:
        raise ValueError("Selected cache path does not match the requested model/seed.")
    table = load_locked_dataset(root, args.dataset)
    split = make_stratified_split(table, args.seed, protocol_version="v1.0")
    split_manifest = json.loads((root / "artifacts" / "splits" / "v1.0" / args.dataset / f"seed-{args.seed}.json").read_text(encoding="utf-8"))
    if split_manifest["split_hash"] != split.split_hash or split_manifest["split_ids"] != split.ids.as_dict():
        raise ValueError("Regenerated Stage 04 split differs from its locked manifest.")
    expected_ids = {"calibration_pool": split.ids.calibration_pool, "test": split.ids.test}
    expected_y = {name: table.subset_labels(ids).to_numpy(dtype="int8", copy=True) for name, ids in expected_ids.items()}
    cached = read_valid_cache(cache_dir, provenance, expected_ids, expected_y)
    minority_label = _minority_label(root, args.dataset)
    subsets = select_nested_calibration_subsets(
        cached["ids"]["calibration_pool"], cached["labels"]["calibration_pool"],
        protocol_version="v1.0", dataset_id=args.dataset, base_seed=args.seed, minority_label=minority_label,
    )
    base_metrics = binary_predictive_metrics(cached["labels"]["test"], cached["probabilities"]["test"])
    cached_base = cached["manifest"]["metrics"]["test"]
    if any(abs(float(base_metrics[key]) - float(cached_base[key])) > 1e-15 for key in ("auroc", "auprc")):
        raise AssertionError("Read-only cache predictive metrics disagree with its Stage 05 manifest.")
    cells = []
    for m_minority in M_MINORITY:
        for method in CP_METHODS:
            cell = evaluate_split_cp(
                cached["ids"]["calibration_pool"], cached["labels"]["calibration_pool"], cached["probabilities"]["calibration_pool"],
                cached["labels"]["test"], cached["probabilities"]["test"], subsets,
                m_minority=m_minority, cp_method=method,
            )
            cell["auroc"] = base_metrics["auroc"]
            cell["auprc"] = base_metrics["auprc"]
            cells.append(cell)
    if len({(cell["m_minority"], cell["subset_hash"]) for cell in cells if cell["cp_method"] == "global_split_cp"}) != 4:
        raise AssertionError("Global subset hashes are not distinct per minority size.")
    for m_minority in M_MINORITY:
        pair = [cell for cell in cells if cell["m_minority"] == m_minority]
        if pair[0]["subset_hash"] != pair[1]["subset_hash"]:
            raise AssertionError("Global and Class-Conditional CP do not share the same subset identity.")
        if any(abs(float(cell["auroc"]) - base_metrics["auroc"]) > 0 or abs(float(cell["auprc"]) - base_metrics["auprc"]) > 0 for cell in pair):
            raise AssertionError("AUROC/AUPRC changed across the CP×m grid.")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = root / "artifacts" / "stage06"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{timestamp}_limited_cache_validation.json"
    global_sanity = [global_coverage_sanity(int(cell["covered_count_overall"]), int(cell["n_test"])) for cell in cells if cell["cp_method"] == "global_split_cp"]
    payload = {
        "artifact_id": output.stem, "stage": "Stage 06", "status": "PASS", "scope": "limited_cache_validation_not_pilot_not_results_long",
        "created_utc": timestamp, "dataset_id": args.dataset, "seed": args.seed, "model": args.model,
        "cache_dir": str(cache_dir), "prediction_cache_hash": cached["manifest"]["cache_sha256"],
        "cache_provenance": provenance, "minority_label": minority_label, "label_mapping_hash": _label_mapping_hash(table.label_mapping),
        "split_hash": split.split_hash, "schema_sha256": sha256_path(root / "configs" / "results_long.schema.json"),
        "base_metrics": base_metrics, "cp_cells": cells, "global_coverage_sanity": global_sanity,
        "qc": {"cache_read_only_validated": True, "fixed_200_majority": True, "nested_10_20_50_100": True, "subset_identity": True, "exact_rank": True, "geometry_consistency": True, "set_decomposition": True, "auroc_auprc_invariant": True, "coverage_sanity_recorded": True},
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "validation_artifact": str(output), "cp_cells": len(cells), "trained_models": 0, "formal_results_written": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

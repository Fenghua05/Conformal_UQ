"""Read-only independent intake audit for the Stage 08 v1.1 cloud preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

import yaml

from conformal_uq.data import load_locked_dataset
from conformal_uq.preprocessing import TrainOnlyPreprocessor
from conformal_uq.provenance import sha256_path
from conformal_uq.split import make_stratified_split


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "stage08_tabpfn_full_context_preflight_v1.1.yaml"
RECEIPT = ROOT / "decisions" / "D08-002_CLOUD_PREFLIGHT_BUDGET_RECEIPT.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return sha256_path(path)


def dense_shape(value: Any) -> list[int]:
    return [int(item) for item in value.shape]


def expected_shapes(config: dict[str, Any], dataset_id: str, seed: int) -> dict[str, list[int]]:
    registry_path = ROOT / config["registry_path"]
    table = load_locked_dataset(ROOT, dataset_id, registry_path=registry_path)
    split = make_stratified_split(table, seed, protocol_version="v1.1")
    processor = TrainOnlyPreprocessor("xgboost").fit(table, split)
    return {
        "dataset_hash": table.raw_sha256,
        "split_hash": split.split_hash,
        "matrix_shapes": {
            "train": dense_shape(processor.transform(table, split.ids.train, partition="train")),
            "calibration_pool": dense_shape(processor.transform(table, split.ids.calibration_pool, partition="calibration_pool")),
            "test": dense_shape(processor.transform(table, split.ids.test, partition="test")),
        },
    }


def probability_summary_valid(summary: dict[str, Any]) -> bool:
    shape = summary.get("shape")
    return (
        isinstance(shape, list)
        and len(shape) == 2
        and shape[0] > 0
        and shape[1] == 2
        and 0 <= float(summary.get("min")) <= float(summary.get("max")) <= 1
        and 0 <= float(summary.get("max_row_sum_error")) <= 1e-6
        and isinstance(summary.get("sha256"), str)
        and len(summary["sha256"]) == 64
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intake-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    intake = args.intake_dir.resolve()
    archive = intake / "stage08_v11_preflight_return_r4.tar.gz"
    output_root = intake / "preflight_20260831_r4"
    manifest_path, events_path = output_root / "preflight_manifest.json", output_root / "events.jsonl"
    manifest, config, receipt = read_json(manifest_path), yaml.safe_load(CONFIG.read_text(encoding="utf-8")), read_json(RECEIPT)
    expected_members = {"preflight_20260831_r4", "preflight_20260831_r4/events.jsonl", "preflight_20260831_r4/preflight_manifest.json"}
    with tarfile.open(archive, "r:gz") as handle:
        members = {member.name for member in handle.getmembers()}
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_units = config["units"]
    units = manifest.get("units", [])
    checks: dict[str, Any] = {
        "archive_sha256": sha256(archive),
        "archive_members_exact": members == expected_members,
        "manifest_status_pass": manifest.get("status") == "PASS",
        "scope_exact": manifest.get("scope") == "V1.1_FULL_CONTEXT_COMPATIBILITY_PREFLIGHT_ONLY",
        "config_hash_exact": manifest.get("config_sha256") == sha256(CONFIG),
        "budget_receipt_exact": manifest.get("budget") == receipt,
        "runtime_exact": manifest.get("runtime") == config["runtime"],
        "safety_limits_exact": manifest.get("safety_limits") == config["safety_limits"],
        "events_exact": [event.get("event") for event in events] == ["preflight_started", "preflight_complete"] and all(event.get("level") == "INFO" for event in events),
        "expected_units_exact": [{"dataset_id": item.get("dataset_id"), "seed": item.get("seed")} for item in units] == expected_units,
        "unit_checks": [],
    }
    for unit, expected in zip(units, expected_units, strict=True):
        local = expected_shapes(config, expected["dataset_id"], int(expected["seed"]))
        shapes = unit.get("matrix_shapes", {})
        limits = config["safety_limits"]
        checks["unit_checks"].append({
            "dataset_id": expected["dataset_id"],
            "raw_hash_exact": unit.get("dataset_hash") == local["dataset_hash"],
            "split_hash_exact": unit.get("split_hash") == local["split_hash"],
            "matrix_shapes_exact": shapes == local["matrix_shapes"],
            "within_row_limit": int(shapes.get("train", [limits["max_train_rows"] + 1])[0]) <= limits["max_train_rows"],
            "within_feature_limit": all(int(shape[1]) <= limits["max_transformed_features"] for shape in shapes.values()),
            "classes_exact": unit.get("estimator_classes") == [0, 1],
            "calibration_probability_summary_valid": probability_summary_valid(unit.get("calibration_probabilities", {})),
            "test_probability_summary_valid": probability_summary_valid(unit.get("test_probabilities", {})),
            "predictive_metric_counts_valid": unit.get("test_predictive_contract_metrics", {}).get("n_rows") == shapes.get("test", [None])[0],
        })
    first_higgs = units[0] if units else {}
    repeat = manifest.get("repeat", {})
    checks["repeat_exact"] = (
        repeat.get("unit") == config["repeat_unit"]
        and float(repeat.get("max_abs_probability_difference", float("inf"))) <= float(config["repeat_unit"]["max_abs_probability_difference"])
        and repeat.get("repeat_summary", {}).get("sha256") == first_higgs.get("test_probabilities", {}).get("sha256")
        and repeat.get("repeat_runtime_evidence", {}).get("test_probabilities", {}).get("sha256") == first_higgs.get("test_probabilities", {}).get("sha256")
    )
    checks["elapsed_within_budget"] = 0 < float(manifest.get("elapsed_seconds", 0)) <= float(receipt["maximum_wall_clock_hours"]) * 3600
    unit_flags = [value for item in checks["unit_checks"] for key, value in item.items() if key != "dataset_id"]
    scalar_flags = [value for key, value in checks.items() if key not in {"archive_sha256", "unit_checks"}]
    verdict = "PASS" if all(scalar_flags) and all(unit_flags) else "FAIL"
    result = {"artifact_id": "stage08_v11_preflight_intake_audit", "intake_archive": str(archive), "verdict": verdict, "checks": checks, "limitations": ["The designed preflight return contains probability summaries and hashes, not probability arrays; this audit validates the summary contracts and deterministic repeat hash, not individual probabilities.", "The returned archive contains no terminal log despite the requested return contract; events.jsonl provides the available execution timeline."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "output": str(args.output)}, ensure_ascii=False))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

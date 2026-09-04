"""User-operated, cache-free v1.1 TabPFN full-context compatibility preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from conformal_uq.data import load_dataset_registry, load_locked_dataset, registry_record
from conformal_uq.identity import derive_seed
from conformal_uq.metrics import binary_predictive_metrics
from conformal_uq.preprocessing import TrainOnlyPreprocessor
from conformal_uq.provenance import sha256_path
from conformal_uq.split import make_stratified_split


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def config_sha256(path: Path) -> str:
    return sha256_path(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _event(path: Path, level: str, event: str, message: str, **scope: Any) -> None:
    payload = {"timestamp_utc": utc_now(), "stage": "Stage 08", "level": level, "event": event, "message": message, **scope}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def load_and_validate_config(root: Path, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Preflight configuration is absent: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("artifact_status") != "APPROVED_PACKAGE_PREPARATION_ONLY" or config.get("protocol_version") != "v1.1":
        raise ValueError("Configuration is not the approved v1.1 preflight contract.")
    runtime, limits = config.get("runtime"), config.get("safety_limits")
    if not isinstance(runtime, dict) or runtime.get("device") != "cuda" or runtime.get("tabpfn_version") != "8.5.0" or runtime.get("ignore_pretraining_limits") is not False:
        raise ValueError("Preflight requires the approved CUDA TabPFN 8.5.0 runtime with ignored limits disabled.")
    if not isinstance(limits, dict) or limits != {"max_train_rows": 100000, "max_transformed_features": 2000}:
        raise ValueError("Preflight safety limits differ from approved v1.1 values.")
    expected_units = [{"dataset_id": "openml_23512_higgs", "seed": 104729}, {"dataset_id": "openml_23517_numerai28_6", "seed": 104729}, {"dataset_id": "openml_1590_adult", "seed": 104729}]
    if config.get("units") != expected_units:
        raise ValueError("Preflight units differ from the approved three-unit scope.")
    expected_hashes = {"protocol_sha256": root / "protocols" / "protocol_v1.1.md", "approval_sha256": root / "decisions" / "D08-001_APPROVAL_RECEIPT.md", "dataset_lock_sha256": root / "protocols" / "dataset_lock_v1.0.md", "registry_sha256": root / str(config.get("registry_path", "")), "environment_sha256": root / "environment" / "environment_lock_v1.0.json"}
    for key, target in expected_hashes.items():
        if not target.is_file() or config.get("input_hashes", {}).get(key) != sha256_path(target):
            raise ValueError(f"Controlled input hash mismatch: {key}.")
    return config


def load_budget_receipt(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Approved numeric cloud budget receipt is absent: {path}")
    receipt = _read_json(path)
    required = {"status": "APPROVED_FOR_STAGE08_CLOUD_PREFLIGHT_EXECUTION", "config_sha256": config_sha256(Path(config["_config_path"]))}
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError("Budget receipt is not approved for this exact preflight configuration.")
    for field in ("maximum_wall_clock_hours", "maximum_cloud_storage_gb"):
        if not isinstance(receipt.get(field), (int, float)) or receipt[field] <= 0:
            raise ValueError(f"Budget receipt requires a positive numeric {field}.")
    return receipt


def validate_matrix_shape(shape: tuple[int, int], limits: dict[str, int]) -> tuple[int, int]:
    rows, features = int(shape[0]), int(shape[1])
    if rows <= 0 or features <= 0:
        raise ValueError("Transformed matrix has an invalid empty shape.")
    if rows > limits["max_train_rows"]:
        raise ValueError(f"TabPFN train rows {rows} exceed approved safety limit {limits['max_train_rows']}.")
    if features > limits["max_transformed_features"]:
        raise ValueError(f"TabPFN transformed features {features} exceed approved safety limit {limits['max_transformed_features']}.")
    return rows, features


def assert_budget_not_exhausted(started: float, receipt: dict[str, Any]) -> float:
    elapsed = time.perf_counter() - started
    maximum = float(receipt["maximum_wall_clock_hours"]) * 3600.0
    if elapsed > maximum:
        raise TimeoutError(f"Approved wall-clock budget exhausted ({elapsed:.1f}s > {maximum:.1f}s).")
    return elapsed


def _to_dense(value: Any, limits: dict[str, int], *, is_train: bool) -> np.ndarray:
    result = value.toarray() if hasattr(value, "toarray") else np.asarray(value)
    if result.ndim != 2:
        raise ValueError("Preprocessing did not produce a two-dimensional matrix.")
    rows, features = int(result.shape[0]), int(result.shape[1])
    if is_train:
        validate_matrix_shape((rows, features), limits)
    elif features > limits["max_transformed_features"]:
        raise ValueError("Calibration/test transformed features exceed the approved safety limit.")
    dense = np.asarray(result, dtype=np.float64)
    if not np.isfinite(dense).all():
        raise ValueError("Preprocessing produced non-finite features.")
    return dense


def _aligned_probabilities(classes: Any, probabilities: Any) -> np.ndarray:
    labels = tuple(int(item) for item in classes)
    value = np.asarray(probabilities, dtype=np.float64)
    if value.ndim != 2 or set(labels) != {0, 1} or value.shape[1] != 2:
        raise ValueError("TabPFN does not expose binary protocol classes.")
    aligned = np.column_stack((value[:, labels.index(0)], value[:, labels.index(1)]))
    if not np.isfinite(aligned).all() or (aligned < 0).any() or (aligned > 1).any() or not np.allclose(aligned.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("TabPFN probabilities violate the [0,1] row-sum contract.")
    return aligned


def _probability_summary(value: np.ndarray) -> dict[str, Any]:
    return {"shape": [int(x) for x in value.shape], "min": float(value.min()), "max": float(value.max()), "max_row_sum_error": float(np.abs(value.sum(axis=1) - 1.0).max()), "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}


def _run_unit(root: Path, config: dict[str, Any], unit: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    runtime, limits, dataset_id, base_seed = config["runtime"], config["safety_limits"], unit["dataset_id"], int(unit["seed"])
    registry_path = root / config["registry_path"]
    registry = load_dataset_registry(root, registry_path)
    record = registry_record(registry, dataset_id)
    table = load_locked_dataset(root, dataset_id, registry_path=registry_path)
    split = make_stratified_split(table, base_seed, protocol_version="v1.1")
    processor = TrainOnlyPreprocessor("xgboost").fit(table, split)
    train_x = _to_dense(processor.transform(table, split.ids.train, partition="train"), limits, is_train=True)
    calibration_x = _to_dense(processor.transform(table, split.ids.calibration_pool, partition="calibration_pool"), limits, is_train=False)
    test_x = _to_dense(processor.transform(table, split.ids.test, partition="test"), limits, is_train=False)
    train_y = table.subset_labels(split.ids.train).to_numpy(dtype=np.int8, copy=True)
    calibration_y = table.subset_labels(split.ids.calibration_pool).to_numpy(dtype=np.int8, copy=True)
    test_y = table.subset_labels(split.ids.test).to_numpy(dtype=np.int8, copy=True)
    if set(train_y.tolist()) != {0, 1}:
        raise ValueError("Fixed v1.1 train partition lacks a protocol class.")
    try:
        import torch
        import tabpfn
        from tabpfn import TabPFNClassifier
    except ImportError as exc:
        raise RuntimeError("TabPFN and torch must be installed only in the approved cloud runtime.") from exc
    if not torch.cuda.is_available() or str(getattr(tabpfn, "__version__", "")) != str(runtime["tabpfn_version"]):
        raise ValueError("Cloud CUDA availability or TabPFN version differs from the approved runtime.")
    checkpoint = Path(str(runtime["checkpoint_path"]))
    if not checkpoint.is_file() or sha256_path(checkpoint) != runtime["checkpoint_sha256"]:
        raise ValueError("Cloud checkpoint is absent or does not match the approved SHA-256.")
    derived_seed = derive_seed("v1.1", dataset_id, base_seed, "tabpfn")[1]
    torch.cuda.reset_peak_memory_stats()
    fit_start = time.perf_counter()
    classifier = TabPFNClassifier(device="cuda", model_path=str(checkpoint), random_state=derived_seed, **dict(runtime["constructor_kwargs"]))
    classifier.fit(train_x, train_y)
    fit_seconds = time.perf_counter() - fit_start
    cal_start = time.perf_counter(); calibration_p = _aligned_probabilities(classifier.classes_, classifier.predict_proba(calibration_x)); calibration_seconds = time.perf_counter() - cal_start
    test_start = time.perf_counter(); test_p = _aligned_probabilities(classifier.classes_, classifier.predict_proba(test_x)); test_seconds = time.perf_counter() - test_start
    evidence = {"dataset_id": dataset_id, "seed": base_seed, "dataset_hash": table.raw_sha256, "split_hash": split.split_hash, "label_mapping": table.label_mapping, "label_mapping_hash": hashlib.sha256(json.dumps(table.label_mapping, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "registry_rank": record["frozen_selection_rank"], "derived_model_seed": derived_seed, "matrix_shapes": {"train": list(train_x.shape), "calibration_pool": list(calibration_x.shape), "test": list(test_x.shape)}, "dense_bytes": {"train": int(train_x.nbytes), "calibration_pool": int(calibration_x.nbytes), "test": int(test_x.nbytes)}, "timing_seconds": {"fit": fit_seconds, "calibration_predict": calibration_seconds, "test_predict": test_seconds}, "gpu": {"device": torch.cuda.get_device_name(), "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())}, "estimator_classes": [int(x) for x in classifier.classes_], "calibration_probabilities": _probability_summary(calibration_p), "test_probabilities": _probability_summary(test_p), "test_predictive_contract_metrics": binary_predictive_metrics(test_y, test_p), "preprocessor_report": processor.report()}
    return evidence, test_p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--budget-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Preflight output directory already exists and cannot be overwritten.")
    config = load_and_validate_config(ROOT, args.config)
    config["_config_path"] = str(args.config.resolve())
    receipt = load_budget_receipt(args.budget_receipt, config)
    args.output_dir.mkdir(parents=True)
    events = args.output_dir / "events.jsonl"
    _event(events, "INFO", "preflight_started", "Approved cloud preflight started; no cache or CP output is allowed.", config_sha256=config_sha256(args.config), budget_receipt=str(args.budget_receipt))
    try:
        preflight_started = time.perf_counter()
        evidence: list[dict[str, Any]] = []
        repeat_reference: np.ndarray | None = None
        for unit in config["units"]:
            assert_budget_not_exhausted(preflight_started, receipt)
            item, test_probabilities = _run_unit(ROOT, config, unit)
            evidence.append(item)
            if unit["dataset_id"] == config["repeat_unit"]["dataset_id"]:
                repeat_reference = test_probabilities
        assert_budget_not_exhausted(preflight_started, receipt)
        repeat_item, repeat_probabilities = _run_unit(ROOT, config, config["repeat_unit"])
        elapsed_seconds = assert_budget_not_exhausted(preflight_started, receipt)
        if repeat_reference is None or repeat_reference.shape != repeat_probabilities.shape:
            raise ValueError("Repeat unit does not have an aligned first prediction.")
        max_difference = float(np.max(np.abs(repeat_reference - repeat_probabilities)))
        if max_difference > float(config["repeat_unit"]["max_abs_probability_difference"]):
            raise ValueError(f"Repeat probability difference {max_difference} exceeds approved tolerance.")
        manifest = {"artifact_id": f"{args.output_dir.name}_manifest", "stage": "Stage 08", "status": "PASS", "scope": "V1.1_FULL_CONTEXT_COMPATIBILITY_PREFLIGHT_ONLY", "created_utc": utc_now(), "config_sha256": config_sha256(args.config), "budget_receipt": str(args.budget_receipt), "budget": receipt, "elapsed_seconds": elapsed_seconds, "runtime": config["runtime"], "safety_limits": config["safety_limits"], "units": evidence, "repeat": {"unit": config["repeat_unit"], "max_abs_probability_difference": max_difference, "repeat_summary": _probability_summary(repeat_probabilities), "repeat_runtime_evidence": repeat_item}, "prohibited_artifacts": config["output_contract"]["prohibited_artifacts"]}
        _write_json(args.output_dir / "preflight_manifest.json", manifest)
        _event(events, "INFO", "preflight_complete", "All approved full-context compatibility checks passed.", config_sha256=config_sha256(args.config))
        return 0
    except Exception as exc:
        failure = {"stage": "Stage 08", "status": "FAIL", "created_utc": utc_now(), "config_sha256": config_sha256(args.config), "exception_type": type(exc).__name__, "exception": str(exc), "action": "Stop. Preserve this package and obtain a user decision; do not enable overrides, sampling, truncation, or reruns."}
        _write_json(args.output_dir / "failure_records" / "preflight_failure.json", failure)
        _event(events, "ERROR", "preflight_failed", str(exc), config_sha256=config_sha256(args.config))
        raise


if __name__ == "__main__":
    raise SystemExit(main())

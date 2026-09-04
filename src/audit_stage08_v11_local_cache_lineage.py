"""Independent, read-only verifier for Stage 08 Task 4 local v1.1 caches.

This module intentionally does not import the Stage 05 runner or prediction
cache helpers.  It recomputes cache paths, hashes, split identities, labels,
probability checks, and predictive metrics from controlled inputs and stored
arrays.  It never fits a model or imports TabPFN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from conformal_uq.config import FROZEN_DATASETS, FROZEN_SEEDS
from conformal_uq.data import load_locked_dataset
from conformal_uq.split import make_stratified_split


ROOT = Path(__file__).resolve().parents[1]
LOCAL_LOCK = Path("configs/stage05_lr_xgboost_v1.1.yaml")
SPLIT_LOCK = Path("configs/stage04_splits_v1.1.yaml")
FINAL_LOCK = Path("configs/stage05b_tabpfn_v1.1.yaml")
RECEIPT = Path("decisions/D08-003_V11_CACHE_AND_PILOT_BUDGET_RECEIPT.json")
MODELS = ("logistic_regression", "xgboost")
PARTITIONS = ("calibration_pool", "test")


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "src").rglob("*.py")):
        digest.update(str(path.relative_to(root)).replace("\\", "/").encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _config_hash(root: Path) -> str:
    split_bytes = (root / SPLIT_LOCK).read_bytes()
    local_bytes = (root / LOCAL_LOCK).read_bytes()
    interim = hashlib.sha256(split_bytes + local_bytes).hexdigest()
    final_hash = _sha256_path(root / FINAL_LOCK)
    return hashlib.sha256(bytes.fromhex(interim) + bytes.fromhex(final_hash)).hexdigest()


def _cache_dir(root: Path, config_hash: str, code_hash: str, dataset_id: str, seed: int, model: str) -> Path:
    return root / "artifacts" / "caches" / "v1.1" / f"cfg-{config_hash[:12]}" / f"code-{code_hash[:12]}" / dataset_id / f"seed-{seed}" / model


def cache_key_from_directory(directory: Path) -> tuple[str, int, str]:
    """Extract a cache key from the fixed cfg/code/dataset/seed/model layout."""
    model = directory.name
    seed_component = directory.parents[0].name
    dataset_id = directory.parents[1].name
    if not seed_component.startswith("seed-") or model not in MODELS:
        raise ValueError(f"invalid v1.1 cache directory layout: {directory}")
    return dataset_id, int(seed_component.removeprefix("seed-")), model


def cache_time_source_hash(root: Path, config_hash: str) -> str:
    """Return the one immutable source hash recorded by this cache set itself."""
    cache_root = root / "artifacts" / "caches" / "v1.1" / f"cfg-{config_hash[:12]}"
    hashes: set[str] = set()
    for manifest_path in cache_root.glob("code-*/*/seed-*/*/manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        provenance = manifest.get("provenance", {})
        code_hash = provenance.get("code_hash")
        if provenance.get("config_hash") == config_hash and isinstance(code_hash, str) and len(code_hash) == 64:
            if manifest_path.parents[3].name != f"code-{code_hash[:12]}":
                raise ValueError(f"cache path/source-hash prefix mismatch: {manifest_path}")
            hashes.add(code_hash)
    if len(hashes) != 1:
        raise ValueError(f"expected exactly one cache-time source hash for config {config_hash}, found {sorted(hashes)}")
    return hashes.pop()


def validate_exact_v11_cache_relative_entries(
    entries: set[PurePosixPath], config_hash: str, code_hash: str, expected_keys: set[tuple[str, int, str]],
) -> list[str]:
    """Require one exact cfg/code tree and only complete expected cache units."""
    cfg, code = f"cfg-{config_hash[:12]}", f"code-{code_hash[:12]}"
    expected: set[PurePosixPath] = {PurePosixPath(cfg), PurePosixPath(cfg, code)}
    expected_unit_files: set[PurePosixPath] = set()
    for dataset_id, seed, model in expected_keys:
        unit = PurePosixPath(cfg, code, dataset_id, f"seed-{seed}", model)
        expected.update({unit.parents[2], unit.parents[1], unit.parents[0], unit})
        expected_unit_files.update({unit / "manifest.json", unit / "predictions.npz"})
    expected.update(expected_unit_files)
    errors: list[str] = []
    for entry in sorted(entries):
        if entry not in expected:
            if len(entry.parts) >= 2 and (entry.parts[0] != cfg or entry.parts[1] != code):
                errors.append(f"foreign cfg/code tree entry under v1.1 cache root: {entry}")
            elif len(entry.parts) >= 5:
                errors.append(f"incomplete or unexpected cache directory/file under expected tree: {entry}")
            else:
                errors.append(f"unexpected v1.1 cache path: {entry}")
    complete_units = 0
    for dataset_id, seed, model in sorted(expected_keys):
        unit = PurePosixPath(cfg, code, dataset_id, f"seed-{seed}", model)
        manifest, prediction = unit / "manifest.json", unit / "predictions.npz"
        if manifest in entries and prediction in entries:
            complete_units += 1
        else:
            errors.append(f"incomplete expected v1.1 cache directory: {unit}")
    if complete_units != len(expected_keys):
        errors.append(f"complete cache directory count mismatch: observed={complete_units} expected={len(expected_keys)}")
    return errors


def validate_exact_v11_cache_tree(
    cache_root: Path, config_hash: str, code_hash: str, expected_keys: set[tuple[str, int, str]],
) -> list[str]:
    entries = {PurePosixPath(path.relative_to(cache_root).as_posix()) for path in cache_root.rglob("*")}
    return validate_exact_v11_cache_relative_entries(entries, config_hash, code_hash, expected_keys)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _metric_payload(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    return {
        "auroc": float(roc_auc_score(labels, probabilities[:, 1])),
        "auprc": float(average_precision_score(labels, probabilities[:, 1])),
        "n_rows": int(len(labels)),
        "positive_count": int(labels.sum()),
    }


def _v10_file_inventory(root: Path) -> list[dict[str, str]]:
    cache_root = root / "artifacts" / "caches"
    if not cache_root.exists():
        return []
    records = []
    for path in sorted(cache_root.rglob("*")):
        if path.is_file() and "v1.1" not in path.relative_to(cache_root).parts:
            records.append({"path": str(path.relative_to(root)).replace("\\", "/"), "sha256": _sha256_path(path)})
    return records


def snapshot_v10_lineage(root: Path, output_path: Path) -> dict[str, Any]:
    """Record a hash inventory of v1.0 cache history without modifying it."""
    if output_path.exists():
        raise FileExistsError(f"Immutable v1.0 snapshot already exists: {output_path}")
    inventory = _v10_file_inventory(root)
    payload = {
        "artifact_type": "stage08_v11_v10_cache_lineage_snapshot",
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Read-only hash inventory of v1.0 cache history before v1.1 local cache execution.",
        "files": inventory,
        "file_count": len(inventory),
        "inventory_sha256": _json_hash(inventory),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return payload


def audit(root: Path, v10_snapshot_path: Path) -> dict[str, Any]:
    """Independently validate exact Stage 08 v1.1 local-cache lineage."""
    receipt = json.loads((root / RECEIPT).read_text(encoding="utf-8"))
    local_lock = yaml.safe_load((root / LOCAL_LOCK).read_text(encoding="utf-8"))
    split_lock = yaml.safe_load((root / SPLIT_LOCK).read_text(encoding="utf-8"))
    final_lock = yaml.safe_load((root / FINAL_LOCK).read_text(encoding="utf-8"))
    expected_final_hash = _sha256_path(root / FINAL_LOCK)
    errors: list[str] = []
    if receipt.get("status") != "APPROVED_FOR_STAGE08_V11_CACHE_AND_PILOT_ONLY" or receipt.get("protocol_version") != "v1.1":
        errors.append("D08-003 receipt status/protocol is invalid")
    if (receipt.get("maximum_wall_clock_hours"), receipt.get("maximum_cloud_storage_gb")) != (12, 50):
        errors.append("D08-003 numeric budget differs from approved 12h/50GB")
    if (receipt.get("authorized_local_lr_xgboost_units"), receipt.get("authorized_tabpfn_units"), receipt.get("authorized_pilot_cells")) != (160, 80, 480):
        errors.append("D08-003 unit limits differ from authorized scope")
    if receipt.get("cache_lock_sha256") != expected_final_hash or final_lock.get("protocol_version") != "v1.1":
        errors.append("D08-003 final TabPFN lock binding is invalid")
    if local_lock.get("protocol_version") != "v1.1" or split_lock.get("protocol_version") != "v1.1":
        errors.append("local/split lock is not v1.1")
    if local_lock.get("registry", {}).get("locked_primary_ids") != FROZEN_DATASETS or local_lock.get("seeds") != FROZEN_SEEDS or tuple(local_lock.get("authorized_local_models", ())) != MODELS:
        errors.append("local cache lock scope differs from 8x10x2")
    if local_lock.get("paths", {}).get("split_root") != "artifacts/splits/v1.1" or local_lock.get("paths", {}).get("cache_root") != "artifacts/caches/v1.1":
        errors.append("local cache paths are not isolated v1.1 roots")
    if any(local_lock.get("execution_gate", {}).get(key) is not False for key in ("split_regeneration_authorized", "local_cache_generation_authorized", "tabpfn_cache_generation_authorized", "pilot_authorized", "formal_run_manifest_authorized", "full_experiment_authorized")):
        errors.append("local lock authorization gates were mutated")

    config_hash, audit_code_hash = _config_hash(root), _code_hash(root)
    try:
        cache_code_hash = cache_time_source_hash(root, config_hash)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"unable to establish one cache-time source hash: {exc}")
        cache_code_hash = "0" * 64
    environment_hash = _sha256_path(root / "environment" / "environment_lock_v1.0.json")
    expected_provenance_common = {
        "config_hash": config_hash,
        "code_hash": cache_code_hash,
        "environment_hash": environment_hash,
        "protocol_version": "v1.1",
        "local_cache_lock_sha256": _sha256_path(root / LOCAL_LOCK),
        "split_lock_sha256": _sha256_path(root / SPLIT_LOCK),
        "d08_003_cache_lock_sha256": expected_final_hash,
        "class_labels": [0, 1],
    }
    expected_keys = {(dataset_id, seed, model) for dataset_id in FROZEN_DATASETS for seed in FROZEN_SEEDS for model in MODELS}
    cache_v11_root = root / "artifacts" / "caches" / "v1.1"
    if not cache_v11_root.is_dir():
        errors.append(f"v1.1 cache root is absent: {cache_v11_root}")
    else:
        errors.extend(validate_exact_v11_cache_tree(cache_v11_root, config_hash, cache_code_hash, expected_keys))

    checked_units, per_model = [], {model: 0 for model in MODELS}
    tables: dict[str, Any] = {}
    for dataset_id, seed, model in sorted(expected_keys):
        unit_errors: list[str] = []
        directory = _cache_dir(root, config_hash, cache_code_hash, dataset_id, seed, model)
        manifest_path, prediction_path = directory / "manifest.json", directory / "predictions.npz"
        if not manifest_path.is_file() or not prediction_path.is_file():
            unit_errors.append("missing complete cache")
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            table = tables.setdefault(dataset_id, load_locked_dataset(root, dataset_id))
            split = make_stratified_split(table, seed, protocol_version="v1.1")
            split_manifest = json.loads((root / "artifacts" / "splits" / "v1.1" / dataset_id / f"seed-{seed}.json").read_text(encoding="utf-8"))
            if split_manifest.get("split_hash") != split.split_hash or split_manifest.get("split_ids") != split.ids.as_dict():
                unit_errors.append("v1.1 split recomputation/manifest mismatch")
            provenance = dict(expected_provenance_common)
            provenance.update({"dataset_hash": table.raw_sha256, "split_hash": split.split_hash, "model_name": model, "base_seed": seed, "label_mapping": table.label_mapping})
            if manifest.get("artifact_type") != "stage05_base_prediction_cache" or manifest.get("format_version") != "v1.1.0" or manifest.get("qc_status") != "PASS":
                unit_errors.append("cache manifest identity/status/version mismatch")
            if manifest.get("provenance") != provenance:
                unit_errors.append("cache provenance mismatch")
            if manifest.get("cache_sha256") != _sha256_path(prediction_path):
                unit_errors.append("prediction cache SHA-256 mismatch")
            if len(str(manifest.get("model_hash", ""))) != 64:
                unit_errors.append("fitted model hash is missing")
            with np.load(prediction_path, allow_pickle=False) as arrays:
                for partition, split_ids in (("calibration_pool", split.ids.calibration_pool), ("test", split.ids.test)):
                    ids = tuple(str(value) for value in arrays[f"{partition}_sample_ids"].tolist())
                    labels = np.asarray(arrays[f"{partition}_y"], dtype=np.int8)
                    probabilities = np.asarray(arrays[f"{partition}_probabilities"], dtype=np.float64)
                    expected_labels = table.subset_labels(split_ids).to_numpy(dtype=np.int8, copy=True)
                    if ids != tuple(str(value) for value in split_ids) or not np.array_equal(labels, expected_labels):
                        unit_errors.append(f"{partition} IDs or labels are not aligned to v1.1 split")
                    if probabilities.shape != (len(ids), 2) or not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6):
                        unit_errors.append(f"{partition} probabilities are not finite aligned [0,1] rows")
                    details = {"n_rows": len(ids), "sample_ids_hash": _json_hash(list(ids)), "y_hash": _array_hash(labels), "probabilities_hash": _array_hash(probabilities), "class_counts": {"0": int((labels == 0).sum()), "1": int((labels == 1).sum())}}
                    if manifest.get("partitions", {}).get(partition) != details:
                        unit_errors.append(f"{partition} partition-hash detail mismatch")
                    try:
                        if manifest.get("metrics", {}).get(partition) != _metric_payload(labels, probabilities):
                            unit_errors.append(f"{partition} AUROC/AUPRC invariance mismatch")
                    except ValueError as exc:
                        unit_errors.append(f"{partition} predictive metric unavailable: {exc}")
        per_model[model] += 1 if not unit_errors else 0
        checked_units.append({"dataset_id": dataset_id, "seed": seed, "model": model, "status": "PASS" if not unit_errors else "FAIL", "errors": unit_errors})
        errors.extend(f"{dataset_id}/{seed}/{model}: {error}" for error in unit_errors)

    snapshot = json.loads(v10_snapshot_path.read_text(encoding="utf-8"))
    v10_inventory = _v10_file_inventory(root)
    v10_unchanged = snapshot.get("files") == v10_inventory and snapshot.get("inventory_sha256") == _json_hash(v10_inventory)
    if not v10_unchanged:
        errors.append("v1.0 cache file inventory/hash snapshot changed")
    prohibited = []
    for root_path in (root / "artifacts" / "caches" / "v1.1", root / "artifacts" / "runs"):
        if root_path.exists():
            prohibited.extend(str(path.relative_to(root)) for path in root_path.rglob("results_long.parquet"))
            prohibited.extend(str(path.relative_to(root)) for path in root_path.rglob("results_long.csv"))
            prohibited.extend(str(path.relative_to(root)) for path in root_path.rglob("formal_run_manifest.json"))
    if (root / "artifacts" / "stage07_v1.1").exists():
        prohibited.append("artifacts/stage07_v1.1")
    if prohibited:
        errors.append(f"prohibited v1.1 CP/pilot/formal output detected: {prohibited}")
    return {
        "artifact_type": "stage08_v11_local_cache_independent_lineage_audit",
        "status": "PASS" if not errors else "FAIL",
        "protocol_version": "v1.1",
        "authorized_local_cache_units": 160,
        "checked_units": len(checked_units),
        "passed_units": sum(unit["status"] == "PASS" for unit in checked_units),
        "per_model_passed_units": per_model,
        "controlled_hashes": {"local_cache_lock_sha256": expected_provenance_common["local_cache_lock_sha256"], "split_lock_sha256": expected_provenance_common["split_lock_sha256"], "d08_003_cache_lock_sha256": expected_final_hash, "config_hash": config_hash, "cache_time_source_code_hash": cache_code_hash, "audit_source_code_hash": audit_code_hash, "environment_hash": environment_hash},
        "v10_snapshot_path": str(v10_snapshot_path.relative_to(root)),
        "v10_cache_lineage_unchanged": v10_unchanged,
        "prohibited_outputs": prohibited,
        "errors": errors,
        "units": checked_units,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--snapshot-v10", action="store_true")
    parser.add_argument("--v10-snapshot", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.snapshot_v10:
        output = root / "artifacts" / "stage08_v11" / "local_cache_pre_generation" / "v10_cache_lineage_snapshot.json"
        payload = snapshot_v10_lineage(root, output)
        print(json.dumps({"status": payload["status"], "snapshot": str(output), "file_count": payload["file_count"]}))
        return 0
    if args.v10_snapshot is None:
        parser.error("--v10-snapshot is required for a local-cache audit")
    snapshot_path = args.v10_snapshot if args.v10_snapshot.is_absolute() else root / args.v10_snapshot
    payload = audit(root, snapshot_path)
    evidence_root = root / "artifacts" / "stage08_v11" / f"local_cache_lineage_audit_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    evidence_root.mkdir(parents=True, exist_ok=False)
    (evidence_root / "independent_audit.json").write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "audit": str(evidence_root / "independent_audit.json"), "passed_units": payload["passed_units"], "errors": len(payload["errors"])}))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

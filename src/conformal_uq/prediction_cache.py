"""Immutable, QC-gated base-prediction cache format for Stage 05."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .provenance import sha256_path

PARTITIONS = ("calibration_pool", "test")


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _validate_arrays(ids: Mapping[str, Sequence[str]], labels: Mapping[str, np.ndarray], probabilities: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    if set(ids) != set(PARTITIONS) or set(labels) != set(PARTITIONS) or set(probabilities) != set(PARTITIONS):
        raise ValueError("Prediction cache requires exactly calibration_pool and test partitions.")
    all_ids: list[str] = []
    details: dict[str, dict[str, Any]] = {}
    for partition in PARTITIONS:
        ordered_ids = tuple(str(value) for value in ids[partition])
        y = np.asarray(labels[partition], dtype=np.int8)
        proba = np.asarray(probabilities[partition], dtype=np.float64)
        if not ordered_ids or len(set(ordered_ids)) != len(ordered_ids):
            raise ValueError(f"{partition}: sample IDs must be non-empty and unique.")
        if y.ndim != 1 or proba.shape != (len(ordered_ids), 2):
            raise ValueError(f"{partition}: y and probabilities have incompatible dimensions.")
        if set(y.tolist()).difference({0, 1}):
            raise ValueError(f"{partition}: labels are not protocol binary labels.")
        if not np.isfinite(proba).all() or (proba < 0).any() or (proba > 1).any() or not np.allclose(proba.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6):
            raise ValueError(f"{partition}: invalid probability range or row sum.")
        all_ids.extend(ordered_ids)
        details[partition] = {"n_rows": len(ordered_ids), "sample_ids_hash": _json_hash(list(ordered_ids)), "y_hash": _array_hash(y), "probabilities_hash": _array_hash(proba), "class_counts": {"0": int((y == 0).sum()), "1": int((y == 1).sum())}}
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Calibration-pool and test sample IDs overlap.")
    return details


def _required_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    required = {"config_hash", "code_hash", "environment_hash", "dataset_hash", "split_hash", "model_name", "base_seed", "label_mapping", "class_labels"}
    missing = required.difference(provenance)
    if missing:
        raise ValueError(f"Cache provenance missing {sorted(missing)}.")
    result = {key: provenance[key] for key in sorted(required)}
    # Stage 08 v1.1 caches carry their explicit versioned lock lineage.  These
    # fields are optional here so immutable v1.0 history remains readable.
    for key in ("protocol_version", "local_cache_lock_sha256", "split_lock_sha256", "d08_003_cache_lock_sha256"):
        if key in provenance:
            result[key] = provenance[key]
    if result["class_labels"] != [0, 1] or sorted(int(value) for value in result["label_mapping"].values()) != [0, 1]:
        raise ValueError("Cache provenance does not preserve the locked binary label mapping.")
    if result.get("protocol_version") == "v1.1":
        required_v11 = {"local_cache_lock_sha256", "split_lock_sha256", "d08_003_cache_lock_sha256"}
        missing_v11 = required_v11.difference(result)
        if missing_v11 or any(len(str(result[key])) != 64 for key in required_v11):
            raise ValueError("Stage 08 v1.1 cache provenance lacks a complete explicit lock lineage.")
    return result


def write_prediction_cache(cache_dir: Path, provenance: Mapping[str, Any], ids: Mapping[str, Sequence[str]], labels: Mapping[str, np.ndarray], probabilities: Mapping[str, np.ndarray], model_hash: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Write data then manifest once; existing material is never overwritten."""
    cached_provenance = _required_provenance(provenance)
    details = _validate_arrays(ids, labels, probabilities)
    if len(model_hash) != 64:
        raise ValueError("A fitted-model SHA-256 hash is required.")
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path, manifest_path = cache_dir / "predictions.npz", cache_dir / "manifest.json"
    if data_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Cache exists or is partial and cannot be overwritten: {cache_dir}")
    temporary = cache_dir / f".predictions.{uuid.uuid4().hex}.npz"
    try:
        np.savez_compressed(
            temporary,
            calibration_pool_sample_ids=np.asarray(ids["calibration_pool"], dtype=str), calibration_pool_y=np.asarray(labels["calibration_pool"], dtype=np.int8), calibration_pool_probabilities=np.asarray(probabilities["calibration_pool"], dtype=np.float64),
            test_sample_ids=np.asarray(ids["test"], dtype=str), test_y=np.asarray(labels["test"], dtype=np.int8), test_probabilities=np.asarray(probabilities["test"], dtype=np.float64),
        )
        os.replace(temporary, data_path)
        format_version = "v1.1.0" if cached_provenance.get("protocol_version") == "v1.1" else "v1.0.0"
        manifest = {"artifact_type": "stage05_base_prediction_cache", "format_version": format_version, "qc_status": "PASS", "provenance": cached_provenance, "model_hash": model_hash, "cache_file": data_path.name, "cache_sha256": sha256_path(data_path), "partitions": details, "metrics": dict(metrics), "checks": {"class_probability_columns_protocol_order_0_1": True, "probability_range_finite_and_row_sums": True, "sample_ids_unique_and_disjoint": True, "stored_order_matches_locked_split": True, "stored_y_matches_locked_mapping": True}}
        encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        manifest_path.write_text(encoded, encoding="utf-8")
        return manifest
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def read_valid_cache(cache_dir: Path, provenance: Mapping[str, Any], expected_ids: Mapping[str, Sequence[str]], expected_labels: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Load only a complete cache proven to match this exact locked prediction unit."""
    data_path, manifest_path = cache_dir / "predictions.npz", cache_dir / "manifest.json"
    if not data_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Complete cache is absent: {cache_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_provenance = _required_provenance(provenance)
    if manifest.get("artifact_type") != "stage05_base_prediction_cache" or manifest.get("qc_status") != "PASS" or manifest.get("provenance") != required_provenance:
        raise ValueError("Cache manifest does not match this controlled prediction unit.")
    if manifest.get("cache_sha256") != sha256_path(data_path):
        raise ValueError("Prediction cache file SHA-256 mismatch.")
    with np.load(data_path, allow_pickle=False) as arrays:
        ids = {partition: tuple(str(value) for value in arrays[f"{partition}_sample_ids"].tolist()) for partition in PARTITIONS}
        labels = {partition: np.asarray(arrays[f"{partition}_y"], dtype=np.int8) for partition in PARTITIONS}
        probabilities = {partition: np.asarray(arrays[f"{partition}_probabilities"], dtype=np.float64) for partition in PARTITIONS}
    details = _validate_arrays(ids, labels, probabilities)
    for partition in PARTITIONS:
        if tuple(str(value) for value in expected_ids[partition]) != ids[partition]:
            raise ValueError(f"{partition}: cached sample-ID order differs from the locked split.")
        if not np.array_equal(np.asarray(expected_labels[partition], dtype=np.int8), labels[partition]):
            raise ValueError(f"{partition}: cached labels differ from the locked label mapping.")
        if manifest.get("partitions", {}).get(partition) != details[partition]:
            raise ValueError(f"{partition}: cache partition hash/detail mismatch.")
    return {"manifest": manifest, "model_hash": manifest["model_hash"], "probabilities": probabilities, "labels": labels, "ids": ids}

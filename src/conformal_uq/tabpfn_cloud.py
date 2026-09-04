"""Cloud-only TabPFN adapter for immutable Stage 05B probability caches.

This module imports TabPFN only inside the execution function so all local
contract tests remain runnable without the cloud-only dependency.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .config import FROZEN_SEEDS
from .data import BinaryTable
from .preprocessing import TrainOnlyPreprocessor
from .split import StratifiedSplit


class CloudLockError(ValueError):
    """Raised when a cloud execution input is not explicitly locked."""


@dataclass(frozen=True)
class TabPFNPrediction:
    model_name: str
    class_labels: tuple[int, int]
    calibration_probabilities: np.ndarray
    test_probabilities: np.ndarray
    calibration_y: np.ndarray
    test_y: np.ndarray
    preprocessor_report: dict[str, Any]
    model_hash: str
    estimator_version: str


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise CloudLockError(f"Required controlled input is absent: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.suffix in {".yaml", ".yml"} else json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CloudLockError(f"Controlled input must be a mapping: {path}")
    return payload


def load_approved_stage05b_lock(lock_path: Path, pilot_decision_path: Path) -> dict[str, Any]:
    """Load only a user-approved cloud lock matching a pre-outcome pilot pair."""
    lock, decision = _read_mapping(lock_path), _read_mapping(pilot_decision_path)
    if lock.get("artifact_status") != "APPROVED_FOR_STAGE05B":
        raise CloudLockError("Stage 05B lock is not approved for cloud cache generation.")
    if decision.get("status") != "APPROVED_PRE_OUTCOME_PILOT_DECISION":
        raise CloudLockError("Pilot decision is absent or not approved before outcomes.")
    if lock.get("protocol_version") != "v1.0" or decision.get("protocol_version") != "v1.0":
        raise CloudLockError("Stage 05B lock and pilot decision must use protocol v1.0.")
    datasets = lock.get("pilot_dataset_ids")
    decision_datasets = decision.get("pilot_dataset_ids")
    if not isinstance(datasets, list) or len(datasets) != 2 or len(set(datasets)) != 2:
        raise CloudLockError("Stage 05B lock requires exactly two unique pilot dataset IDs.")
    if datasets != decision_datasets:
        raise CloudLockError("Stage 05B lock pilot datasets differ from the approved pilot decision.")
    if lock.get("seeds") != FROZEN_SEEDS:
        raise CloudLockError("Stage 05B lock seeds differ from the frozen ten-seed protocol.")
    runtime = lock.get("runtime")
    if not isinstance(runtime, dict):
        raise CloudLockError("Stage 05B lock has no runtime mapping.")
    required_runtime = {"device", "tabpfn_version", "checkpoint_path", "checkpoint_sha256", "context_limit", "constructor_kwargs", "preprocessing_contract", "ignore_pretraining_limits"}
    missing = required_runtime.difference(runtime)
    if missing:
        raise CloudLockError(f"Stage 05B runtime lock missing {sorted(missing)}.")
    if runtime["device"] != "cuda":
        raise CloudLockError("Stage 05B requires CUDA; CPU and auto-device fallback are prohibited.")
    if runtime["ignore_pretraining_limits"] is not False:
        raise CloudLockError("ignore_pretraining_limits must remain false.")
    if not isinstance(runtime["checkpoint_sha256"], str) or len(runtime["checkpoint_sha256"]) != 64:
        raise CloudLockError("Stage 05B checkpoint SHA-256 must be a 64-character hex digest.")
    if not isinstance(runtime["constructor_kwargs"], dict) or "random_state" in runtime["constructor_kwargs"]:
        raise CloudLockError("constructor_kwargs must be a mapping and must not override the derived random_state.")
    if runtime["preprocessing_contract"] != "stage04_train_only_unscaled_onehot_dense":
        raise CloudLockError("Stage 05B preprocessing contract is not the approved train-only dense unscaled one-hot path.")
    return lock


def expected_tabpfn_units(pilot_dataset_ids: Sequence[str], seeds: Sequence[int]) -> tuple[tuple[str, int, str], ...]:
    if len(pilot_dataset_ids) != 2 or len(set(pilot_dataset_ids)) != 2 or list(seeds) != FROZEN_SEEDS:
        raise CloudLockError("Expected units require exactly the approved two datasets and ten frozen seeds.")
    return tuple((dataset_id, int(seed), "tabpfn") for dataset_id in pilot_dataset_ids for seed in seeds)


def align_protocol_probabilities(classes: Sequence[int], probabilities: Any) -> np.ndarray:
    """Return TabPFN probabilities ordered by protocol labels 0 then 1."""
    values = np.asarray(probabilities, dtype=np.float64)
    labels = tuple(int(value) for value in classes)
    if values.ndim != 2 or values.shape[1] != len(labels) or set(labels) != {0, 1}:
        raise ValueError("TabPFN classes/probabilities must encode protocol labels [0, 1].")
    aligned = np.column_stack((values[:, labels.index(0)], values[:, labels.index(1)]))
    if not np.isfinite(aligned).all() or (aligned < 0).any() or (aligned > 1).any() or not np.allclose(aligned.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("TabPFN emitted invalid protocol-aligned probabilities.")
    return aligned


def _dense_matrix(value: Any, *, context_limit: int) -> np.ndarray:
    result = value.toarray() if hasattr(value, "toarray") else np.asarray(value)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError("TabPFN preprocessing produced an invalid feature matrix.")
    if result.shape[0] > context_limit:
        raise ValueError(f"TabPFN train rows {result.shape[0]} exceed locked context limit {context_limit}.")
    if not np.isfinite(result).all():
        raise ValueError("TabPFN preprocessing produced non-finite features.")
    return np.asarray(result, dtype=np.float64)


def _model_hash(*, derived_seed: int, runtime: Mapping[str, Any], report: Mapping[str, Any], calibration_probabilities: np.ndarray, test_probabilities: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps({"model": "tabpfn", "derived_seed": derived_seed, "runtime": dict(runtime), "preprocessor": dict(report)}, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(np.ascontiguousarray(calibration_probabilities).tobytes())
    digest.update(np.ascontiguousarray(test_probabilities).tobytes())
    return digest.hexdigest()


def fit_predict_tabpfn_locked(table: BinaryTable, split: StratifiedSplit, runtime: Mapping[str, Any], derived_seed: int) -> TabPFNPrediction:
    """Fit TabPFN only on locked train IDs and predict the held-out fixed orders."""
    if runtime.get("device") != "cuda" or runtime.get("ignore_pretraining_limits") is not False:
        raise CloudLockError("TabPFN runtime lock forbids CPU/auto device and ignored pretraining limits.")
    if not isinstance(derived_seed, int) or not 0 <= derived_seed <= 0xFFFFFFFF:
        raise ValueError("TabPFN requires an unsigned 32-bit derived seed.")
    try:
        from tabpfn import TabPFNClassifier
        import tabpfn
    except ImportError as exc:  # pragma: no cover - expected on the local machine
        raise RuntimeError("TabPFN is required only on the approved cloud runtime.") from exc
    if str(getattr(tabpfn, "__version__", "")) != str(runtime["tabpfn_version"]):
        raise CloudLockError("Installed TabPFN version differs from the approved Stage 05B lock.")
    processor = TrainOnlyPreprocessor("xgboost").fit(table, split)
    train_x = _dense_matrix(processor.transform(table, split.ids.train, partition="train"), context_limit=int(runtime["context_limit"]))
    calibration_x = _dense_matrix(processor.transform(table, split.ids.calibration_pool, partition="calibration_pool"), context_limit=max(int(runtime["context_limit"]), len(split.ids.calibration_pool)))
    test_x = _dense_matrix(processor.transform(table, split.ids.test, partition="test"), context_limit=max(int(runtime["context_limit"]), len(split.ids.test)))
    train_y = table.subset_labels(split.ids.train).to_numpy(dtype=np.int8, copy=True)
    calibration_y = table.subset_labels(split.ids.calibration_pool).to_numpy(dtype=np.int8, copy=True)
    test_y = table.subset_labels(split.ids.test).to_numpy(dtype=np.int8, copy=True)
    if set(train_y.tolist()) != {0, 1}:
        raise ValueError("Locked train split lacks a protocol class.")
    classifier = TabPFNClassifier(device="cuda", model_path=str(runtime["checkpoint_path"]), random_state=derived_seed, **dict(runtime["constructor_kwargs"]))
    classifier.fit(train_x, train_y)
    calibration_probabilities = align_protocol_probabilities(classifier.classes_, classifier.predict_proba(calibration_x))
    test_probabilities = align_protocol_probabilities(classifier.classes_, classifier.predict_proba(test_x))
    report = processor.report()
    return TabPFNPrediction(
        model_name="tabpfn", class_labels=(0, 1), calibration_probabilities=calibration_probabilities,
        test_probabilities=test_probabilities, calibration_y=calibration_y, test_y=test_y,
        preprocessor_report=report,
        model_hash=_model_hash(derived_seed=derived_seed, runtime=runtime, report=report, calibration_probabilities=calibration_probabilities, test_probabilities=test_probabilities),
        estimator_version=f"tabpfn.{getattr(tabpfn, '__version__', 'unknown')}",
    )

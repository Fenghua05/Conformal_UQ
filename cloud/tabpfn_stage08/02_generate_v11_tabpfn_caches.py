"""User-operated, cache-only v1.1 TabPFN base-probability cache generator.

Runs ONLY on the locked AutoDL Ubuntu 22.04 / RTX 4090 24 GB / CUDA /
TabPFN 8.5.0 runtime and generates exactly 80 immutable probability caches
(8 locked datasets x 10 frozen seeds).  TabPFN is fitted on each COMPLETE
fixed v1.1 train partition with no truncation, sampling, or subsampling and
with ``ignore_pretraining_limits`` left false.  Per-cache provenance records,
for the cloud TabPFN lineage, ``local_cache_lock_sha256`` as the SHA-256 of
the generating TabPFN cache lock (``configs/stage05b_tabpfn_v1.1.yaml``),
which is also the D08-003-bound final lock.

Outputs are limited to probability caches, events, immutable failure
records, and one summary: no conformal prediction, no pilot, no
``results_long``, no figures, and no formal run manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from conformal_uq.config import FROZEN_DATASETS, FROZEN_SEEDS
from conformal_uq.data import load_locked_dataset
from conformal_uq.identity import derive_seed, run_id
from conformal_uq.metrics import binary_predictive_metrics
from conformal_uq.paths import cache_path, create_immutable_run_dir
from conformal_uq.prediction_cache import read_valid_cache, write_prediction_cache
from conformal_uq.preprocessing import TrainOnlyPreprocessor
from conformal_uq.provenance import sha256_path
from conformal_uq.split import make_stratified_split
from conformal_uq.stage08_authorization import load_d08_003_authorization

CANONICAL_LOCK_PATH = Path("configs/stage05b_tabpfn_v1.1.yaml")
CANONICAL_SPLIT_LOCK_PATH = Path("configs/stage04_splits_v1.1.yaml")
CANONICAL_RECEIPT_PATH = Path("decisions/D08-003_V11_CACHE_AND_PILOT_BUDGET_RECEIPT.json")
CANONICAL_ENVIRONMENT_LOCK_PATH = Path("environment/environment_lock_v1.0.json")
AUTHORIZED_TABPFN_UNITS = 80


class V11TabPFNCacheLockError(ValueError):
    """Raised when the v1.1 TabPFN cache contract is violated."""


class V11TabPFNCacheRunLockError(RuntimeError):
    """Raised when another process already owns this exact cloud cache scope."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_bytes(*chunks: bytes) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def append_event(path: Path | None, *, run_identifier: str, level: str, event: str, config_hash: str, message: str, **scope: Any) -> None:
    """Append one structured event; a None path disables event recording."""
    if path is None:
        return
    payload = {
        "timestamp_utc": utc_now(), "run_id": run_identifier, "stage": "Stage 08 / Task 5",
        "level": level, "event": event, "config_hash": config_hash, "message": message,
    }
    payload.update({key: value for key, value in scope.items() if value is not None})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def load_and_validate_v11_cache_lock(root: Path, lock_path: Path) -> dict[str, Any]:
    """Accept only the canonical final v1.1 TabPFN cache lock with its full contract."""
    canonical = (root / CANONICAL_LOCK_PATH).resolve()
    actual = (lock_path if lock_path.is_absolute() else root / lock_path).resolve()
    if actual != canonical:
        raise V11TabPFNCacheLockError("Only the canonical final v1.1 TabPFN cache lock may drive the 80-unit cache run.")
    lock = yaml.safe_load(actual.read_text(encoding="utf-8"))
    if not isinstance(lock, dict) or lock.get("protocol_version") != "v1.1":
        raise V11TabPFNCacheLockError("The final TabPFN cache lock is not a v1.1 mapping.")
    if lock.get("artifact_status") != "FINAL_CACHE_LOCK_D08_003_RECEIPT_BOUND_IMPLEMENTATION_PENDING":
        raise V11TabPFNCacheLockError("The final TabPFN cache lock status differs from the D08-003-bound lock.")
    if lock.get("registry", {}).get("locked_primary_ids") != FROZEN_DATASETS or lock.get("seeds") != FROZEN_SEEDS:
        raise V11TabPFNCacheLockError("The final TabPFN cache lock registry or seeds differ from the frozen v1.1 protocol.")
    if lock.get("split_lock_path") != str(CANONICAL_SPLIT_LOCK_PATH).replace("\\", "/"):
        raise V11TabPFNCacheLockError("The final TabPFN cache lock must bind the canonical v1.1 split lock.")
    paths = lock.get("paths", {})
    if paths.get("split_root") != "artifacts/splits/v1.1" or paths.get("cache_root") != "artifacts/caches/v1.1":
        raise V11TabPFNCacheLockError("The final TabPFN cache lock must use the isolated v1.1 split/cache roots.")
    runtime = lock.get("runtime", {})
    if runtime.get("device") != "cuda" or runtime.get("tabpfn_version") != "8.5.0":
        raise V11TabPFNCacheLockError("The locked runtime must be the approved CUDA TabPFN 8.5.0 environment.")
    if runtime.get("ignore_pretraining_limits") is not False:
        raise V11TabPFNCacheLockError("ignore_pretraining_limits must remain false.")
    if runtime.get("no_truncation_or_subsampling") is not True or runtime.get("train_partition_contract") != "full_fixed_train_partition":
        raise V11TabPFNCacheLockError("The complete fixed train-partition contract must stay locked.")
    if runtime.get("preprocessing_contract") != "stage04_train_only_unscaled_onehot_dense":
        raise V11TabPFNCacheLockError("The train-only unscaled one-hot dense preprocessing contract must stay locked.")
    checkpoint_sha = runtime.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64 or not runtime.get("checkpoint_path"):
        raise V11TabPFNCacheLockError("The locked checkpoint path/SHA-256 is invalid.")
    if dict(runtime.get("constructor_kwargs", {}) or {}) != {}:
        raise V11TabPFNCacheLockError("Constructor overrides are forbidden; the derived random_state must stay in control.")
    if lock.get("safety_limits") != {"max_train_rows": 100000, "max_transformed_features": 2000}:
        raise V11TabPFNCacheLockError("The v1.1 full-context safety limits differ from the approved values.")
    scope = lock.get("authorization", {}).get("cache_only_scope", {})
    if scope.get("authorized_tabpfn_units") != AUTHORIZED_TABPFN_UNITS or scope.get("datasets") != 8 or scope.get("seeds_per_dataset") != 10:
        raise V11TabPFNCacheLockError("The cache-only scope must be exactly 8 datasets x 10 seeds = 80 TabPFN units.")
    if scope.get("probability_cache_only") is not True or any(scope.get(key) is not False for key in ("conformal_prediction_allowed", "pilot_output_allowed", "formal_output_allowed")):
        raise V11TabPFNCacheLockError("The cache-only scope must prohibit CP, pilot, and formal outputs.")
    output = lock.get("output_contract", {})
    if any(output.get(key) is not False for key in ("conformal_prediction_allowed", "pilot_allowed", "formal_outputs_allowed")):
        raise V11TabPFNCacheLockError("The output contract must prohibit CP, pilot, and formal outputs.")
    split_lock_path = root / CANONICAL_SPLIT_LOCK_PATH
    split_lock = yaml.safe_load(split_lock_path.read_text(encoding="utf-8"))
    if not isinstance(split_lock, dict) or split_lock.get("protocol_version") != "v1.1":
        raise V11TabPFNCacheLockError("The canonical v1.1 split lock is invalid.")
    if split_lock.get("registry", {}).get("locked_primary_ids") != FROZEN_DATASETS or split_lock.get("seeds") != FROZEN_SEEDS:
        raise V11TabPFNCacheLockError("The canonical v1.1 split lock registry or seeds differ from the frozen protocol.")
    if split_lock.get("paths", {}).get("split_root") != "artifacts/splits/v1.1":
        raise V11TabPFNCacheLockError("The canonical v1.1 split lock must use the isolated v1.1 split root.")
    if not (root / CANONICAL_ENVIRONMENT_LOCK_PATH).is_file():
        raise V11TabPFNCacheLockError("The environment lock is absent from the package.")
    return lock


def load_validated_d08_003_authorization(root: Path, receipt_path: Path) -> dict[str, Any]:
    """Validate the canonical D08-003 receipt before any TabPFN import or output."""
    canonical = (root / CANONICAL_RECEIPT_PATH).resolve()
    actual = (receipt_path if receipt_path.is_absolute() else root / receipt_path).resolve()
    if actual != canonical:
        raise V11TabPFNCacheLockError("Only the canonical D08-003 budget receipt may authorize the 80-unit cache run.")
    return load_d08_003_authorization(root)


def v11_tabpfn_config_hash(root: Path, lock: Mapping[str, Any]) -> str:
    """Deterministic v1.1 TabPFN cache config hash over the split and final locks."""
    split_lock_path = root / str(lock.get("split_lock_path", CANONICAL_SPLIT_LOCK_PATH))
    tabpfn_lock_path = root / CANONICAL_LOCK_PATH
    interim = sha256_bytes(split_lock_path.read_bytes(), tabpfn_lock_path.read_bytes())
    final = sha256_bytes(tabpfn_lock_path.read_bytes())
    return sha256_bytes(bytes.fromhex(interim), bytes.fromhex(final))


def v11_tabpfn_code_hash(root: Path) -> str:
    """Hash the exact src and stage08 cloud code that produces the caches.

    Entries are ordered by their POSIX relative path STRING so the digest is
    identical on Windows (bundle build) and Linux (cloud execution); sorting
    Path objects directly is platform-dependent (``data/`` vs ``data.py``).
    """
    entries = [
        (str(path.relative_to(root)).replace("\\", "/"), path)
        for directory in (root / "src", root / "cloud" / "tabpfn_stage08")
        for path in directory.rglob("*.py")
    ]
    digest = hashlib.sha256()
    for relative, path in sorted(entries, key=lambda item: item[0]):
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def expected_v11_tabpfn_units(lock: Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    """Exactly the 8 locked datasets x 10 frozen seeds, in registry/seed order."""
    units = tuple(
        (str(dataset_id), int(seed))
        for dataset_id in lock["registry"]["locked_primary_ids"]
        for seed in lock["seeds"]
    )
    if len(units) != AUTHORIZED_TABPFN_UNITS or len(set(units)) != len(units):
        raise V11TabPFNCacheLockError("The v1.1 TabPFN cache scope is exactly 8 datasets x 10 frozen seeds = 80 units.")
    return units


def assert_budget_not_exhausted(started: float, receipt: Mapping[str, Any]) -> float:
    elapsed = time.perf_counter() - started
    maximum = float(receipt["maximum_wall_clock_hours"]) * 3600.0
    if elapsed > maximum:
        raise TimeoutError(f"Approved D08-003 wall-clock budget exhausted ({elapsed:.1f}s > {maximum:.1f}s).")
    return elapsed


def assert_storage_budget(receipt: Mapping[str, Any], *roots: Path) -> int:
    """Bound the produced cache/run bytes by the approved cloud-storage budget."""
    total = 0
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
    maximum = float(receipt["maximum_cloud_storage_gb"]) * (1024 ** 3)
    if total > maximum:
        raise V11TabPFNCacheLockError(f"Approved D08-003 cloud-storage budget exceeded ({total} bytes > {maximum:.0f} bytes).")
    return total


def validate_matrix_shape(shape: tuple[int, int], limits: Mapping[str, int]) -> tuple[int, int]:
    rows, features = int(shape[0]), int(shape[1])
    if rows <= 0 or features <= 0:
        raise ValueError("Transformed matrix has an invalid empty shape.")
    if rows > limits["max_train_rows"]:
        raise ValueError(f"TabPFN train rows {rows} exceed approved safety limit {limits['max_train_rows']}.")
    if features > limits["max_transformed_features"]:
        raise ValueError(f"TabPFN transformed features {features} exceed approved safety limit {limits['max_transformed_features']}.")
    return rows, features


def _to_dense(value: Any, limits: Mapping[str, int], *, is_train: bool) -> np.ndarray:
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


def aligned_probabilities(classes: Any, probabilities: Any) -> np.ndarray:
    """Return TabPFN probabilities ordered by protocol labels 0 then 1."""
    labels = tuple(int(item) for item in classes)
    value = np.asarray(probabilities, dtype=np.float64)
    if value.ndim != 2 or set(labels) != {0, 1} or value.shape[1] != 2:
        raise ValueError("TabPFN does not expose binary protocol classes.")
    aligned = np.column_stack((value[:, labels.index(0)], value[:, labels.index(1)]))
    if not np.isfinite(aligned).all() or (aligned < 0).any() or (aligned > 1).any() or not np.allclose(aligned.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("TabPFN probabilities violate the [0,1] row-sum contract.")
    return aligned


def v11_tabpfn_cache_provenance(
    *, config_hash: str, code_hash: str, environment_hash: str, table: Any, split: Any, base_seed: int,
    split_lock_sha256: str, tabpfn_cache_lock_sha256: str,
) -> dict[str, Any]:
    """Full v1.1 provenance; the generating TabPFN lock is recorded twice by design.

    For the cloud TabPFN lineage ``local_cache_lock_sha256`` carries the
    SHA-256 of the generating TabPFN cache lock, which is also the
    D08-003-bound final lock, so both fields hold the same value.
    """
    return {
        "config_hash": config_hash, "code_hash": code_hash, "environment_hash": environment_hash,
        "dataset_hash": table.raw_sha256, "split_hash": split.split_hash,
        "model_name": "tabpfn", "base_seed": base_seed,
        "label_mapping": table.label_mapping, "class_labels": [0, 1],
        "protocol_version": "v1.1",
        "local_cache_lock_sha256": tabpfn_cache_lock_sha256,
        "split_lock_sha256": split_lock_sha256,
        "d08_003_cache_lock_sha256": tabpfn_cache_lock_sha256,
    }


def locked_v11_split(root: Path, lock: Mapping[str, Any], table: Any, base_seed: int) -> Any:
    """Regenerate and verify the v1.1 split against its packaged locked manifest."""
    if table.dataset_id not in lock["registry"]["locked_primary_ids"] or base_seed not in lock["seeds"]:
        raise V11TabPFNCacheLockError("Split requested outside the explicit v1.1 TabPFN cache lock scope.")
    split = make_stratified_split(table, base_seed, protocol_version=lock["protocol_version"])
    path = root / lock["paths"]["split_root"] / table.dataset_id / f"seed-{base_seed}.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("raw_sha256") != table.raw_sha256 or manifest.get("split_hash") != split.split_hash or manifest.get("split_ids") != split.ids.as_dict():
        raise V11TabPFNCacheLockError(f"{table.dataset_id}/{base_seed}: regenerated v1.1 split differs from its locked manifest.")
    return split


def v11_tabpfn_cache_run_lock_path(root: Path, config_hash: str, final_lock_sha256: str) -> Path:
    return root / "artifacts" / "stage08_v11" / "locks" / (
        f"v11_tabpfn_cache_cfg-{config_hash[:12]}_final-{final_lock_sha256[:12]}.lock"
    )


def v11_tabpfn_cache_run_identifier(config_hash: str) -> str:
    """Unique v1.1 cloud cache run identifier (uuid suffix prevents collisions)."""
    return f"{run_id('stage08-v11-tabpfn-cache', config_hash)}_{uuid.uuid4().hex[:12]}"


@contextmanager
def exclusive_v11_tabpfn_cache_run_lock(root: Path, config_hash: str, final_lock_sha256: str):
    """Own the exact cloud cache scope before any data/model/cache work.

    A pre-existing lock is never reclaimed automatically: an operator must
    inspect a possible crashed holder instead of allowing overlapping cache
    writers.  The owner releases only its own token in a finally block.
    """
    path = v11_tabpfn_cache_run_lock_path(root, config_hash, final_lock_sha256)
    token = uuid.uuid4().hex
    metadata = {
        "artifact_type": "stage08_v11_tabpfn_cache_exclusive_run_lock",
        "scope": "D08-003 authorized 80-unit v1.1 cloud TabPFN cache run",
        "config_hash": config_hash,
        "d08_003_cache_lock_sha256": final_lock_sha256,
        "owner_token": token,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(metadata, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        try:
            holder = path.read_text(encoding="utf-8").strip()
        except OSError:
            holder = "unreadable holder metadata"
        raise V11TabPFNCacheRunLockError(
            f"The exact v1.1 TabPFN cache scope is already held: {path}. "
            f"No data/model/cache work was started. Holder metadata: {holder}"
        ) from exc
    try:
        yield path
    finally:
        try:
            holder = json.loads(path.read_text(encoding="utf-8"))
            if holder.get("owner_token") == token:
                path.unlink()
        except FileNotFoundError:
            pass


def immutable_failure(root: Path, *, run_identifier: str, config_hash: str, scope: Mapping[str, Any], exception: BaseException, retry_count: int) -> Path:
    """Write one exclusive, never-overwritten failure record."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = root / "artifacts" / "failures" / (
        f"{timestamp}_stage08_v11_tabpfn_{scope['dataset_id']}_{scope['seed']}_{uuid.uuid4().hex[:10]}.json"
    )
    payload = {
        "failure_id": path.stem, "stage": "Stage 08 / Task 5", "run_id": run_identifier,
        "timestamp_utc": utc_now(), "classification": "bug_or_data_or_environment_pending_triage",
        "retry_count": retry_count, "scope": dict(scope), "config_hash": config_hash,
        "exception_type": type(exception).__name__, "exception": str(exception),
        "action": "No blind retry and no cache overwrite; inspect this record before any rerun.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return path


def retryable_exception(exc: BaseException) -> bool:
    return isinstance(exc, (TimeoutError, ConnectionError)) or type(exc).__name__ in {"OutOfMemoryError", "CUDAError"}


def verify_cloud_runtime(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Import torch/TabPFN only after receipt validation; verify the locked runtime."""
    try:
        import tabpfn
        import torch
    except ImportError as exc:
        raise V11TabPFNCacheLockError("TabPFN and torch must be installed only in the approved cloud runtime.") from exc
    if not torch.cuda.is_available():
        raise V11TabPFNCacheLockError("The locked runtime requires CUDA; no CUDA device is available.")
    if str(getattr(tabpfn, "__version__", "")) != str(runtime["tabpfn_version"]):
        raise V11TabPFNCacheLockError("Installed TabPFN version differs from the approved v1.1 lock.")
    checkpoint = Path(str(runtime["checkpoint_path"])).expanduser()
    if not checkpoint.is_file() or sha256_path(checkpoint) != runtime["checkpoint_sha256"]:
        raise V11TabPFNCacheLockError("Cloud checkpoint is absent or does not match the approved SHA-256.")
    return {
        "cuda_available": True,
        "gpu_name": torch.cuda.get_device_name(),
        "torch_version": str(getattr(torch, "__version__", "unknown")),
        "tabpfn_version": str(getattr(tabpfn, "__version__", "unknown")),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_path(checkpoint),
    }


def _tabpfn_model_hash(*, derived_seed: int, runtime: Mapping[str, Any], report: Mapping[str, Any], calibration_probabilities: np.ndarray, test_probabilities: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps({"model": "tabpfn", "derived_seed": derived_seed, "runtime": dict(runtime), "preprocessor": dict(report)}, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(np.ascontiguousarray(calibration_probabilities).tobytes())
    digest.update(np.ascontiguousarray(test_probabilities).tobytes())
    return digest.hexdigest()


def full_context_tabpfn_fit_predict(table: Any, split: Any, runtime: Mapping[str, Any], limits: Mapping[str, int], derived_seed: int) -> dict[str, Any]:
    """Fit TabPFN on the COMPLETE fixed train partition; predict the fixed orders.

    The train matrix must have exactly as many rows as the locked split train
    IDs: no truncation, sampling, or subsampling is performed or tolerated,
    and no ``ignore_pretraining_limits`` override is ever passed.
    """
    import torch
    from tabpfn import TabPFNClassifier

    processor = TrainOnlyPreprocessor("xgboost").fit(table, split)
    train_x = _to_dense(processor.transform(table, split.ids.train, partition="train"), limits, is_train=True)
    if int(train_x.shape[0]) != len(split.ids.train):
        raise V11TabPFNCacheLockError("The full fixed train-partition contract was violated: fitted rows differ from the locked split.")
    calibration_x = _to_dense(processor.transform(table, split.ids.calibration_pool, partition="calibration_pool"), limits, is_train=False)
    test_x = _to_dense(processor.transform(table, split.ids.test, partition="test"), limits, is_train=False)
    train_y = table.subset_labels(split.ids.train).to_numpy(dtype=np.int8, copy=True)
    calibration_y = table.subset_labels(split.ids.calibration_pool).to_numpy(dtype=np.int8, copy=True)
    test_y = table.subset_labels(split.ids.test).to_numpy(dtype=np.int8, copy=True)
    if set(train_y.tolist()) != {0, 1}:
        raise V11TabPFNCacheLockError("The fixed v1.1 train partition lacks a protocol class.")
    torch.cuda.reset_peak_memory_stats()
    fit_start = time.perf_counter()
    classifier = TabPFNClassifier(device="cuda", model_path=str(runtime["checkpoint_path"]), random_state=derived_seed)
    classifier.fit(train_x, train_y)
    fit_seconds = time.perf_counter() - fit_start
    calibration_start = time.perf_counter()
    calibration_p = aligned_probabilities(classifier.classes_, classifier.predict_proba(calibration_x))
    calibration_seconds = time.perf_counter() - calibration_start
    test_start = time.perf_counter()
    test_p = aligned_probabilities(classifier.classes_, classifier.predict_proba(test_x))
    test_seconds = time.perf_counter() - test_start
    evidence = {
        "matrix_shapes": {"train": list(train_x.shape), "calibration_pool": list(calibration_x.shape), "test": list(test_x.shape)},
        "train_rows_equal_locked_split_train_ids": True,
        "timing_seconds": {"fit": fit_seconds, "calibration_predict": calibration_seconds, "test_predict": test_seconds},
        "gpu": {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated()), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())},
        "estimator_classes": [int(value) for value in classifier.classes_],
        "preprocessor_report": processor.report(),
    }
    model_hash = _tabpfn_model_hash(derived_seed=derived_seed, runtime=runtime, report=processor.report(), calibration_probabilities=calibration_p, test_probabilities=test_p)
    del classifier
    torch.cuda.empty_cache()
    return {
        "calibration_y": calibration_y, "test_y": test_y,
        "calibration_probabilities": calibration_p, "test_probabilities": test_p,
        "model_hash": model_hash, "evidence": evidence,
    }


def reuse_or_generate_v11_tabpfn_cache(
    *, root: Path, lock: Mapping[str, Any], config_hash: str, code_hash: str, environment_hash: str,
    split_lock_sha256: str, tabpfn_cache_lock_sha256: str, table: Any, split: Any, base_seed: int,
    cache_root: Path, event_path: Path | None, run_identifier: str, retry_event: Callable[[], None] | None,
) -> dict[str, Any]:
    """Reuse one complete valid cache, fitting only when it is truly absent."""
    dataset_id = table.dataset_id
    canonical_seed_input, derived_seed = derive_seed(lock["protocol_version"], dataset_id, base_seed, "tabpfn")
    provenance = v11_tabpfn_cache_provenance(
        config_hash=config_hash, code_hash=code_hash, environment_hash=environment_hash,
        table=table, split=split, base_seed=base_seed,
        split_lock_sha256=split_lock_sha256, tabpfn_cache_lock_sha256=tabpfn_cache_lock_sha256,
    )
    destination = cache_path(cache_root, config_hash, code_hash, dataset_id, base_seed, "tabpfn")
    expected_ids = {"calibration_pool": split.ids.calibration_pool, "test": split.ids.test}
    expected_y = {partition: table.subset_labels(ids).to_numpy(dtype="int8", copy=True) for partition, ids in expected_ids.items()}
    try:
        cached = read_valid_cache(destination, provenance, expected_ids, expected_y)
        return {
            "action": "reused_validated_v11_tabpfn_cache", "model_hash": cached["model_hash"],
            "metrics": cached["manifest"]["metrics"], "evidence": None,
            "derived_seed": derived_seed, "canonical_seed_input": canonical_seed_input,
            "cache_path": str(destination.relative_to(root)),
        }
    except FileNotFoundError:
        output = None
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                output = full_context_tabpfn_fit_predict(table, split, lock["runtime"], {"max_train_rows": 100000, "max_transformed_features": 2000}, derived_seed)
                break
            except Exception as exc:  # noqa: BLE001 - retried only for transient cloud failures
                last_exc = exc
                if attempt == 0 and retryable_exception(exc):
                    if retry_event is not None:
                        retry_event()
                    continue
                raise
        if output is None:  # defensive; reached only if the retry loop is changed
            raise RuntimeError(f"TabPFN output is absent after cache-generation attempts: {last_exc}")
        metrics = {
            "calibration_pool": binary_predictive_metrics(output["calibration_y"], output["calibration_probabilities"]),
            "test": binary_predictive_metrics(output["test_y"], output["test_probabilities"]),
        }
        manifest = write_prediction_cache(
            destination, provenance, expected_ids,
            {"calibration_pool": output["calibration_y"], "test": output["test_y"]},
            {"calibration_pool": output["calibration_probabilities"], "test": output["test_probabilities"]},
            output["model_hash"], metrics,
        )
        return {
            "action": "trained_and_cached_v11_tabpfn", "model_hash": manifest["model_hash"],
            "metrics": metrics, "evidence": output["evidence"],
            "derived_seed": derived_seed, "canonical_seed_input": canonical_seed_input,
            "cache_path": str(destination.relative_to(root)),
        }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate exactly the 80 D08-003-authorized v1.1 TabPFN probability caches.")
    parser.add_argument("--lock", type=Path, default=CANONICAL_LOCK_PATH)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("The v1.1 TabPFN cache run directory already exists and cannot be overwritten.")
    lock = load_and_validate_v11_cache_lock(ROOT, args.lock)
    authorization = load_validated_d08_003_authorization(ROOT, args.receipt)
    receipt, final_lock_sha256 = authorization["receipt"], authorization["final_lock_sha256"]
    config_hash = v11_tabpfn_config_hash(ROOT, lock)
    code_hash = v11_tabpfn_code_hash(ROOT)
    environment_hash = sha256_path(ROOT / CANONICAL_ENVIRONMENT_LOCK_PATH)
    split_lock_sha256 = sha256_path(ROOT / CANONICAL_SPLIT_LOCK_PATH)
    tabpfn_cache_lock_sha256 = sha256_path(ROOT / CANONICAL_LOCK_PATH)
    cache_root = ROOT / lock["paths"]["cache_root"]
    units = expected_v11_tabpfn_units(lock)
    with exclusive_v11_tabpfn_cache_run_lock(ROOT, config_hash, final_lock_sha256):
        runtime_evidence = verify_cloud_runtime(lock["runtime"])
        output_dir.mkdir(parents=True)
        events = output_dir / "events.jsonl"
        identifier = v11_tabpfn_cache_run_identifier(config_hash)
        append_event(events, run_identifier=identifier, level="INFO", event="v11_tabpfn_cache_run_started", config_hash=config_hash, message="Starting exactly 80 D08-003-authorized v1.1 TabPFN cache units; CP, pilot, and formal outputs are prohibited.", expected_units=len(units))
        started = time.perf_counter()
        rows: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        tables: dict[str, Any] = {}
        for dataset_id, base_seed in units:
            assert_budget_not_exhausted(started, receipt)
            scope = {"dataset_id": dataset_id, "seed": base_seed, "model": "tabpfn"}
            try:
                if dataset_id not in tables:
                    tables[dataset_id] = load_locked_dataset(ROOT, dataset_id, registry_path=ROOT / lock["registry"]["path"])
                table = tables[dataset_id]
                split = locked_v11_split(ROOT, lock, table, base_seed)
                row = reuse_or_generate_v11_tabpfn_cache(
                    root=ROOT, lock=lock, config_hash=config_hash, code_hash=code_hash,
                    environment_hash=environment_hash, split_lock_sha256=split_lock_sha256,
                    tabpfn_cache_lock_sha256=tabpfn_cache_lock_sha256, table=table, split=split,
                    base_seed=base_seed, cache_root=cache_root, event_path=events, run_identifier=identifier,
                    retry_event=lambda: append_event(events, run_identifier=identifier, level="WARN", event="v11_tabpfn_prediction_unit_retry", config_hash=config_hash, message="Transient cloud failure; retrying once before recording a failure.", **scope, retry_count=1),
                )
                rows.append({
                    **scope, "action": row["action"], "cache_path": row["cache_path"],
                    "model_hash": row["model_hash"], "split_hash": split.split_hash,
                    "dataset_hash": table.raw_sha256, "derived_seed": row["derived_seed"],
                    "canonical_seed_input": row["canonical_seed_input"],
                    "test_auroc": row["metrics"]["test"]["auroc"], "test_auprc": row["metrics"]["test"]["auprc"],
                    "calibration_pool_auroc": row["metrics"]["calibration_pool"]["auroc"],
                    "calibration_pool_auprc": row["metrics"]["calibration_pool"]["auprc"],
                    "fit_evidence": row["evidence"],
                })
                append_event(events, run_identifier=identifier, level="INFO", event="v11_tabpfn_cache_unit_complete", config_hash=config_hash, message=row["action"], **scope, model_hash=row["model_hash"])
            except Exception as exc:  # noqa: BLE001 - every unit failure is preserved immutably
                failure_path = immutable_failure(ROOT, run_identifier=identifier, config_hash=config_hash, scope=scope, exception=exc, retry_count=1 if retryable_exception(exc) else 0)
                failures.append({**scope, "failure_record": str(failure_path.relative_to(ROOT)), "exception_type": type(exc).__name__, "exception": str(exc)})
                append_event(events, run_identifier=identifier, level="ERROR", event="v11_tabpfn_cache_unit_failed", config_hash=config_hash, message=str(exc), **scope, exception_id=failure_path.stem)
        elapsed_seconds = assert_budget_not_exhausted(started, receipt)
        produced_bytes = assert_storage_budget(receipt, cache_root, output_dir)
        status = "PASS" if not failures and len(rows) == AUTHORIZED_TABPFN_UNITS else "FAIL"
        summary = {
            "artifact_id": f"{identifier}_summary", "stage": "Stage 08 / Task 5", "run_id": identifier,
            "protocol_version": "v1.1", "status": status,
            "scope": "D08_003_V11_TABPFN_PROBABILITY_CACHES_ONLY",
            "config_hash": config_hash, "code_hash": code_hash, "environment_hash": environment_hash,
            "split_lock_sha256": split_lock_sha256,
            "tabpfn_cache_lock_sha256": tabpfn_cache_lock_sha256,
            "d08_003_cache_lock_sha256": final_lock_sha256,
            "budget": {
                "maximum_wall_clock_hours": receipt["maximum_wall_clock_hours"],
                "maximum_cloud_storage_gb": receipt["maximum_cloud_storage_gb"],
                "elapsed_seconds": elapsed_seconds, "produced_bytes": produced_bytes,
            },
            "runtime_evidence": runtime_evidence,
            "expected_units": AUTHORIZED_TABPFN_UNITS, "completed_units": len(rows),
            "reused_units": sum(row["action"].startswith("reused") for row in rows),
            "trained_units": sum(row["action"].startswith("trained") for row in rows),
            "units": rows, "failures": failures,
            "cp_evaluated": False, "pilot_outputs": False,
            "formal_run_manifest_created": False, "full_experiment_executed": False,
        }
        _write_json(output_dir / "summary.json", summary)
        append_event(events, run_identifier=identifier, level="INFO" if status == "PASS" else "ERROR", event="v11_tabpfn_cache_run_finished", config_hash=config_hash, message=status, completed=len(rows), failures=len(failures))
        print(json.dumps({"status": status, "run_root": str(output_dir), "completed": len(rows), "failures": len(failures), "cp_evaluated": False, "pilot_outputs": False, "formal_outputs": False}))
        return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

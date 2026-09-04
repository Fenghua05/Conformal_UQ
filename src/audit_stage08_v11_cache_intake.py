"""Read-only independent intake audit for the returned 80 v1.1 TabPFN caches.

This auditor never imports TabPFN, fits a model, or calls the cloud runners'
validation helpers.  It recomputes every controlled hash from the local v1.1
locks, regenerates each v1.1 split from the locked raw data, and independently
revalidates all 240 v1.1 cache units (80 returned cloud TabPFN caches plus the
160 existing local LR/XGBoost caches): provenance, file hashes, sample order,
labels, probability contracts, partition hashes, and AUROC/AUPRC metrics.

Only after every check passes does it install the 80 TabPFN caches into the
previously absent local v1.1 cache tree and verify the combined 240-unit
layout.  No CP, pilot, or formal output is created or authorized here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from conformal_uq.config import FROZEN_DATASETS, FROZEN_SEEDS
from conformal_uq.data import load_locked_dataset
from conformal_uq.split import make_stratified_split

ROOT = Path(__file__).resolve().parents[1]
SPLIT_LOCK = ROOT / "configs" / "stage04_splits_v1.1.yaml"
LOCAL_LOCK = ROOT / "configs" / "stage05_lr_xgboost_v1.1.yaml"
TABPFN_LOCK = ROOT / "configs" / "stage05b_tabpfn_v1.1.yaml"
ENV_LOCK = ROOT / "environment" / "environment_lock_v1.0.json"
D08_003_RECEIPT = ROOT / "decisions" / "D08-003_V11_CACHE_AND_PILOT_BUDGET_RECEIPT.json"
REGISTRY = ROOT / "artifacts" / "stage02" / "dataset_registry_v1.0.1.json"
CACHE_ROOT = ROOT / "artifacts" / "caches" / "v1.1"
SPLIT_ROOT = ROOT / "artifacts" / "splits" / "v1.1"
LOCAL_MODELS = ("logistic_regression", "xgboost")
PARTITIONS = ("calibration_pool", "test")
AUTHORIZED_TABPFN_UNITS = 80
AUTHORIZED_TOTAL_UNITS = 240

ENVIRONMENT_HASH = hashlib.sha256(ENV_LOCK.read_bytes()).hexdigest()
SPLIT_LOCK_SHA256 = hashlib.sha256(SPLIT_LOCK.read_bytes()).hexdigest()
LOCAL_LOCK_SHA256 = hashlib.sha256(LOCAL_LOCK.read_bytes()).hexdigest()
TABPFN_LOCK_SHA256 = hashlib.sha256(TABPFN_LOCK.read_bytes()).hexdigest()
LOCKED_RUNTIME = {
    "gpu_name": "NVIDIA GeForce RTX 4090",
    "tabpfn_version": "8.5.0",
    "checkpoint_sha256": "d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988",
}


def _load_upload_bundle_hashes() -> tuple[str, str, str]:
    """Bind the cache-time TabPFN code hash to the immutable upload receipt.

    The cloud caches were produced by the code shipped in the execution-
    authorized upload archive, not by the current local tree (this auditor
    itself lives under src/).  Cache-time and audit-time source hashes are
    therefore recorded separately; the current tree hash can never negate a
    historical cache.
    """
    candidates = sorted((ROOT / "artifacts" / "stage08_v11_transfer").glob("cache_upload_authorized_*/archive_receipt.json"))
    if not candidates:
        raise FileNotFoundError("No v1.1 TabPFN cache upload archive receipt exists under artifacts/stage08_v11_transfer.")
    upload_receipt = json.loads(candidates[-1].read_text(encoding="utf-8"))
    config_hash, code_hash = upload_receipt.get("bundle_config_sha256"), upload_receipt.get("bundle_code_sha256")
    if not isinstance(config_hash, str) or len(config_hash) != 64 or not isinstance(code_hash, str) or len(code_hash) != 64:
        raise ValueError("The upload archive receipt does not record valid bundle hashes.")
    if upload_receipt.get("cloud_execution_authorized") is not True:
        raise ValueError("The upload archive receipt is not execution-authorized.")
    return config_hash, code_hash, str(candidates[-1].relative_to(ROOT)).replace("\\", "/")


CACHE_TIME_TABPFN_CONFIG_HASH, CACHE_TIME_TABPFN_CODE_HASH, UPLOAD_RECEIPT_PATH = _load_upload_bundle_hashes()


def sha256_bytes(*chunks: bytes) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recompute_tabpfn_config_hash(root: Path) -> str:
    """Recompute the cloud TabPFN cache config hash from the v1.1 locks."""
    interim = sha256_bytes((root / "configs/stage04_splits_v1.1.yaml").read_bytes(), (root / "configs/stage05b_tabpfn_v1.1.yaml").read_bytes())
    final = sha256_bytes((root / "configs/stage05b_tabpfn_v1.1.yaml").read_bytes())
    return sha256_bytes(bytes.fromhex(interim), bytes.fromhex(final))


def recompute_tabpfn_code_hash(root: Path) -> str:
    """Recompute the cloud code hash (src + cloud/tabpfn_stage08), platform-independent."""
    entries = sorted(
        str(path.relative_to(root)).replace("\\", "/")
        for directory in (root / "src", root / "cloud" / "tabpfn_stage08")
        for path in directory.rglob("*.py")
    )
    digest = hashlib.sha256()
    for relative in entries:
        digest.update(relative.encode("utf-8"))
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def recompute_local_config_hash(root: Path) -> str:
    """Recompute the local LR/XGBoost cache config hash from the v1.1 locks."""
    interim = sha256_bytes((root / "configs/stage04_splits_v1.1.yaml").read_bytes(), (root / "configs/stage05_lr_xgboost_v1.1.yaml").read_bytes())
    final = sha256_bytes((root / "configs/stage05b_tabpfn_v1.1.yaml").read_bytes())
    return sha256_bytes(bytes.fromhex(interim), bytes.fromhex(final))


def expected_intake_unit_keys() -> set[tuple[str, int, str]]:
    """Exactly 240 unique (dataset, seed, model) keys: 80 per model."""
    return {
        (dataset, seed, model)
        for dataset in FROZEN_DATASETS
        for seed in FROZEN_SEEDS
        for model in (*LOCAL_MODELS, "tabpfn")
    }


def expected_tabpfn_cache_provenance(root: Path, table: Any, split: Any, base_seed: int, *, config_hash: str, code_hash: str) -> dict[str, Any]:
    return {
        "config_hash": config_hash, "code_hash": code_hash, "environment_hash": ENVIRONMENT_HASH,
        "dataset_hash": table.raw_sha256, "split_hash": split.split_hash,
        "model_name": "tabpfn", "base_seed": base_seed,
        "label_mapping": table.label_mapping, "class_labels": [0, 1],
        "protocol_version": "v1.1",
        "local_cache_lock_sha256": TABPFN_LOCK_SHA256,
        "split_lock_sha256": SPLIT_LOCK_SHA256,
        "d08_003_cache_lock_sha256": TABPFN_LOCK_SHA256,
    }


def expected_local_cache_provenance(root: Path, table: Any, split: Any, base_seed: int, model: str, *, config_hash: str, code_hash: str) -> dict[str, Any]:
    return {
        "config_hash": config_hash, "code_hash": code_hash, "environment_hash": ENVIRONMENT_HASH,
        "dataset_hash": table.raw_sha256, "split_hash": split.split_hash,
        "model_name": model, "base_seed": base_seed,
        "label_mapping": table.label_mapping, "class_labels": [0, 1],
        "protocol_version": "v1.1",
        "local_cache_lock_sha256": LOCAL_LOCK_SHA256,
        "split_lock_sha256": SPLIT_LOCK_SHA256,
        "d08_003_cache_lock_sha256": TABPFN_LOCK_SHA256,
    }


def validate_return_receipt(
    receipt: Mapping[str, Any], *, archive_sha256: str, archive_bytes: int,
    config_hash: str, code_hash: str, inventory_sha256: str, final_lock_sha256: str,
) -> list[str]:
    """Validate the returned archive receipt against independently observed values."""
    errors: list[str] = []
    if receipt.get("artifact_id") != "stage08_v11_tabpfn_cache_return_archive":
        errors.append("receipt artifact_id is not the v1.1 TabPFN cache return archive")
    if not str(receipt.get("archive", "")).endswith(".tar.gz"):
        errors.append("receipt archive name is not the expected .tar.gz return archive")
    if receipt.get("archive_sha256") != archive_sha256:
        errors.append("receipt archive_sha256 does not match the observed archive hash")
    if receipt.get("archive_bytes") != archive_bytes:
        errors.append("receipt archive_bytes does not match the observed archive size")
    if receipt.get("inventory_sha256") != inventory_sha256:
        errors.append("receipt inventory_sha256 does not match the returned inventory hash")
    if receipt.get("verified_units") != AUTHORIZED_TABPFN_UNITS:
        errors.append("receipt verified_units is not exactly 80")
    if receipt.get("config_hash") != config_hash:
        errors.append("receipt config_hash differs from the recomputed v1.1 TabPFN cache config hash")
    if receipt.get("code_hash") != code_hash:
        errors.append("receipt code_hash differs from the recomputed cloud code hash")
    if receipt.get("d08_003_cache_lock_sha256") != final_lock_sha256:
        errors.append("receipt final-lock hash differs from the current D08-003-bound lock")
    if not isinstance(receipt.get("source_run"), str) or not receipt["source_run"]:
        errors.append("receipt source_run is absent")
    for key in ("cp_evaluated", "pilot_outputs", "formal_run_manifest_created"):
        if receipt.get(key) is not False:
            errors.append(f"receipt must keep {key} false")
    return errors


def validate_generator_summary(
    summary: Mapping[str, Any], *, config_hash: str, code_hash: str, budget: Mapping[str, Any],
) -> list[str]:
    """Validate the packed generator summary for a complete authorized PASS run."""
    errors: list[str] = []
    if summary.get("status") != "PASS":
        errors.append("generator summary status is not PASS")
    if summary.get("protocol_version") != "v1.1" or summary.get("scope") != "D08_003_V11_TABPFN_PROBABILITY_CACHES_ONLY":
        errors.append("generator summary is not the v1.1 cache-only run record")
    if summary.get("expected_units") != AUTHORIZED_TABPFN_UNITS or summary.get("completed_units") != AUTHORIZED_TABPFN_UNITS:
        errors.append("generator summary must report exactly 80 completed units")
    if summary.get("config_hash") != config_hash or summary.get("code_hash") != code_hash:
        errors.append("generator summary lineage differs from the recomputed hashes")
    if summary.get("environment_hash") != ENVIRONMENT_HASH:
        errors.append("generator summary environment hash differs from the local environment lock")
    if summary.get("split_lock_sha256") != SPLIT_LOCK_SHA256:
        errors.append("generator summary split-lock hash differs from the local v1.1 split lock")
    if summary.get("tabpfn_cache_lock_sha256") != TABPFN_LOCK_SHA256 or summary.get("d08_003_cache_lock_sha256") != TABPFN_LOCK_SHA256:
        errors.append("generator summary TabPFN/final lock hash differs from the local lock")
    summary_budget = summary.get("budget", {})
    if summary_budget.get("maximum_wall_clock_hours") != budget["maximum_wall_clock_hours"] or summary_budget.get("maximum_cloud_storage_gb") != budget["maximum_cloud_storage_gb"]:
        errors.append("generator summary budget differs from the D08-003 receipt")
    elapsed, produced = float(summary_budget.get("elapsed_seconds", 0)), int(summary_budget.get("produced_bytes", 0))
    if not 0 < elapsed <= float(budget["maximum_wall_clock_hours"]) * 3600.0:
        errors.append("generator elapsed time is outside the approved wall-clock budget")
    if not 0 <= produced <= float(budget["maximum_cloud_storage_gb"]) * 1024 ** 3:
        errors.append("generator produced bytes are outside the approved storage budget")
    runtime = summary.get("runtime_evidence", {})
    if runtime.get("cuda_available") is not True:
        errors.append("generator runtime evidence does not confirm CUDA")
    if runtime.get("gpu_name") != LOCKED_RUNTIME["gpu_name"]:
        errors.append("generator runtime GPU differs from the locked RTX 4090")
    if str(runtime.get("tabpfn_version")) != LOCKED_RUNTIME["tabpfn_version"]:
        errors.append("generator runtime TabPFN version differs from the locked 8.5.0")
    if runtime.get("checkpoint_sha256") != LOCKED_RUNTIME["checkpoint_sha256"]:
        errors.append("generator runtime checkpoint hash differs from the locked checkpoint")
    for key in ("cp_evaluated", "pilot_outputs", "formal_run_manifest_created", "full_experiment_executed"):
        if summary.get(key) is not False:
            errors.append(f"generator summary must keep {key} false")
    if summary.get("failures"):
        errors.append("generator summary records unit failures; the PASS run must be failure-free")
    return errors


def scan_prohibited_return_members(names: Sequence[str], *, tabpfn_cfg_prefix: str | None = None, tabpfn_code_prefix: str | None = None) -> list[str]:
    """Flag CP/formal outputs, credentials, checkpoints, raw data, and foreign cache trees."""
    cfg = tabpfn_cfg_prefix or f"cfg-{CACHE_TIME_TABPFN_CONFIG_HASH[:12]}"
    code = tabpfn_code_prefix or f"code-{CACHE_TIME_TABPFN_CODE_HASH[:12]}"
    allowed_cache_prefix = f"artifacts/caches/v1.1/{cfg}/{code}/"
    markers = ("results_long", "formal_run_manifest", "figures/", "cred", "token", "secret", "password", ".ckpt", ".arff")
    flagged: list[str] = []
    for name in names:
        normalized = str(name).replace("\\", "/").lower()
        bad = any(marker in normalized for marker in markers) or normalized.endswith((".pt", ".pth"))
        if "artifacts/caches/v1.0/" in normalized or "artifacts/splits/v1.0/" in normalized:
            bad = True
        if "artifacts/caches/" in normalized and allowed_cache_prefix not in normalized:
            bad = True
        if bad:
            flagged.append(name)
    return flagged


def single_cache_time_source_hash(cache_root: Path, config_hash: str) -> str:
    """Return the one source hash recorded by an existing cache set itself."""
    hashes: set[str] = set()
    for manifest_path in (cache_root / f"cfg-{config_hash[:12]}").glob("code-*/*/seed-*/*/manifest.json"):
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


def validate_combined_cache_relative_entries(
    entries: set[PurePosixPath], local_config_hash: str, local_code_hash: str,
    tabpfn_config_hash: str, tabpfn_code_hash: str,
    expected_keys: set[tuple[str, int, str]],
) -> list[str]:
    """Require exactly the two authorized cfg/code trees and 240 complete units."""
    local_pair = (f"cfg-{local_config_hash[:12]}", f"code-{local_code_hash[:12]}")
    tabpfn_pair = (f"cfg-{tabpfn_config_hash[:12]}", f"code-{tabpfn_code_hash[:12]}")
    allowed_pairs = {local_pair, tabpfn_pair}
    expected: set[PurePosixPath] = {PurePosixPath(pair[0]) for pair in allowed_pairs}
    expected.update(PurePosixPath(*pair) for pair in allowed_pairs)
    for dataset, seed, model in expected_keys:
        pair = tabpfn_pair if model == "tabpfn" else local_pair
        unit = PurePosixPath(pair[0], pair[1], dataset, f"seed-{seed}", model)
        expected.update({unit.parents[2], unit.parents[1], unit.parents[0], unit})
        expected.update({unit / "manifest.json", unit / "predictions.npz"})
    errors: list[str] = []
    for entry in sorted(entries):
        if entry not in expected:
            if len(entry.parts) >= 2 and (entry.parts[0], entry.parts[1]) not in allowed_pairs:
                errors.append(f"foreign cfg/code tree entry under the v1.1 cache root: {entry}")
            elif len(entry.parts) >= 5:
                errors.append(f"incomplete or unexpected cache directory/file under an authorized tree: {entry}")
            else:
                errors.append(f"unexpected v1.1 cache path: {entry}")
    complete_units = 0
    for dataset, seed, model in sorted(expected_keys):
        pair = tabpfn_pair if model == "tabpfn" else local_pair
        unit = PurePosixPath(pair[0], pair[1], dataset, f"seed-{seed}", model)
        if (unit / "manifest.json") in entries and (unit / "predictions.npz") in entries:
            complete_units += 1
        else:
            errors.append(f"incomplete expected v1.1 cache directory: {unit}")
    if complete_units != len(expected_keys):
        errors.append(f"complete cache directory count mismatch: observed={complete_units} expected={len(expected_keys)}")
    return errors


def install_tabpfn_cache_units(extracted_cache_root: Path, cache_root: Path, units: Iterable[tuple[str, int]], config_hash: str, code_hash: str) -> list[str]:
    """Copy the verified TabPFN caches into the previously absent local tree."""
    installed: list[str] = []
    for dataset_id, base_seed in units:
        relative = Path(f"cfg-{config_hash[:12]}") / f"code-{code_hash[:12]}" / dataset_id / f"seed-{base_seed}" / "tabpfn"
        source = extracted_cache_root / relative
        if not source.is_dir():
            raise FileNotFoundError(f"verified TabPFN cache unit is absent from the return archive: {source}")
        for name in ("manifest.json", "predictions.npz"):
            if not (source / name).is_file():
                raise FileNotFoundError(f"verified TabPFN cache unit file is absent: {source / name}")
        target = cache_root / relative
        if target.exists():
            raise FileExistsError(f"local TabPFN cache unit already exists and must never be overwritten: {target}")
        target.mkdir(parents=True, exist_ok=False)
        for name in ("manifest.json", "predictions.npz"):
            shutil.copy2(source / name, target / name)
        installed.append(relative.as_posix())
    return installed


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


def validate_cache_unit(cache_dir: Path, expected_provenance: Mapping[str, Any], expected_ids: Mapping[str, Sequence[str]], expected_labels: Mapping[str, np.ndarray]) -> list[str]:
    """Independently revalidate one cache unit against its expected lineage."""
    errors: list[str] = []
    manifest_path, data_path = cache_dir / "manifest.json", cache_dir / "predictions.npz"
    if not manifest_path.is_file() or not data_path.is_file():
        return [f"complete cache unit is absent: {cache_dir}"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("provenance") != dict(expected_provenance):
        errors.append(f"provenance mismatch: {cache_dir}")
    if manifest.get("qc_status") != "PASS" or manifest.get("format_version") != "v1.1.0":
        errors.append(f"manifest qc/format is not PASS v1.1.0: {cache_dir}")
    if manifest.get("cache_sha256") != _sha256_path(data_path):
        errors.append(f"cache file SHA-256 mismatch: {cache_dir}")
    if not isinstance(manifest.get("model_hash"), str) or len(manifest["model_hash"]) != 64:
        errors.append(f"model_hash is not a SHA-256 digest: {cache_dir}")
    try:
        with np.load(data_path, allow_pickle=False) as arrays:
            ids = {partition: tuple(str(value) for value in arrays[f"{partition}_sample_ids"].tolist()) for partition in PARTITIONS}
            labels = {partition: np.asarray(arrays[f"{partition}_y"], dtype=np.int8) for partition in PARTITIONS}
            probabilities = {partition: np.asarray(arrays[f"{partition}_probabilities"], dtype=np.float64) for partition in PARTITIONS}
    except Exception as exc:  # noqa: BLE001 - any load failure is a cache defect
        return errors + [f"cache arrays cannot be loaded ({type(exc).__name__}): {cache_dir}"]
    all_ids: list[str] = []
    for partition in PARTITIONS:
        ordered_ids, y, proba = ids[partition], labels[partition], probabilities[partition]
        if len(set(ordered_ids)) != len(ordered_ids) or not ordered_ids:
            errors.append(f"{partition}: sample IDs are not unique and non-empty: {cache_dir}")
        if proba.shape != (len(ordered_ids), 2):
            errors.append(f"{partition}: probability shape is invalid: {cache_dir}")
        if set(y.tolist()).difference({0, 1}):
            errors.append(f"{partition}: labels are not protocol binary: {cache_dir}")
        if not np.isfinite(proba).all() or (proba < 0).any() or (proba > 1).any() or not np.allclose(proba.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6):
            errors.append(f"{partition}: probability contract violated: {cache_dir}")
        if list(ordered_ids) != [str(value) for value in expected_ids[partition]]:
            errors.append(f"{partition}: cached sample-ID order differs from the regenerated v1.1 split: {cache_dir}")
        if not np.array_equal(y, np.asarray(expected_labels[partition], dtype=np.int8)):
            errors.append(f"{partition}: cached labels differ from the locked label mapping: {cache_dir}")
        details = {
            "n_rows": len(ordered_ids),
            "sample_ids_hash": _json_hash(list(ordered_ids)),
            "y_hash": _array_hash(y),
            "probabilities_hash": _array_hash(proba),
            "class_counts": {"0": int((y == 0).sum()), "1": int((y == 1).sum())},
        }
        if manifest.get("partitions", {}).get(partition) != details:
            errors.append(f"{partition}: partition hash/detail mismatch: {cache_dir}")
        if set(y.tolist()) == {0, 1} and proba.shape == (len(ordered_ids), 2):
            if manifest.get("metrics", {}).get(partition) != _metric_payload(y, proba):
                errors.append(f"{partition}: recomputed AUROC/AUPRC differ from the manifest metrics: {cache_dir}")
        all_ids.extend(ordered_ids)
    if len(set(all_ids)) != len(all_ids):
        errors.append(f"calibration-pool and test sample IDs overlap: {cache_dir}")
    return errors


def _locked_split_and_table(tables: dict[str, Any], splits: dict[tuple[str, int], Any], dataset_id: str, base_seed: int) -> tuple[Any, Any]:
    if dataset_id not in tables:
        tables[dataset_id] = load_locked_dataset(ROOT, dataset_id, registry_path=REGISTRY)
    table = tables[dataset_id]
    if (dataset_id, base_seed) not in splits:
        split = make_stratified_split(table, base_seed, protocol_version="v1.1")
        manifest = json.loads((SPLIT_ROOT / dataset_id / f"seed-{base_seed}.json").read_text(encoding="utf-8"))
        if manifest.get("raw_sha256") != table.raw_sha256 or manifest.get("split_hash") != split.split_hash or manifest.get("split_ids") != split.ids.as_dict():
            raise ValueError(f"{dataset_id}/{base_seed}: regenerated v1.1 split differs from the locked split manifest.")
        splits[(dataset_id, base_seed)] = split
    return table, splits[(dataset_id, base_seed)]


def _expected_ids_and_labels(table: Any, split: Any) -> tuple[dict[str, Sequence[str]], dict[str, np.ndarray]]:
    ids = {"calibration_pool": split.ids.calibration_pool, "test": split.ids.test}
    labels = {partition: table.subset_labels(partition_ids).to_numpy(dtype=np.int8, copy=True) for partition, partition_ids in ids.items()}
    return ids, labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Independently audit the returned 80 TabPFN caches plus the 160 local caches (240 units).")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--intake-dir", type=Path, required=True)
    args = parser.parse_args()
    archive = args.archive.resolve()
    receipt_file = args.receipt.resolve()
    intake_dir = args.intake_dir.resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"Returned archive is absent: {archive}")
    if not receipt_file.is_file():
        raise FileNotFoundError(f"Returned archive receipt is absent: {receipt_file}")
    if intake_dir.exists():
        raise FileExistsError(f"Intake directory already exists and cannot be overwritten: {intake_dir}")
    d08_003 = json.loads(D08_003_RECEIPT.read_text(encoding="utf-8"))
    budget = {"maximum_wall_clock_hours": d08_003["maximum_wall_clock_hours"], "maximum_cloud_storage_gb": d08_003["maximum_cloud_storage_gb"]}
    tabpfn_config_hash = recompute_tabpfn_config_hash(ROOT)
    audit_time_code_hash = recompute_tabpfn_code_hash(ROOT)
    local_config_hash = recompute_local_config_hash(ROOT)
    errors: list[str] = []
    if CACHE_TIME_TABPFN_CONFIG_HASH != tabpfn_config_hash:
        errors.append("the upload bundle config hash differs from the recomputed v1.1 lock hashes; a lock changed after packaging")
    # The cache-time code hash is bound to the immutable upload receipt; the
    # current tree hash is recorded separately and never negates the caches.
    tabpfn_code_hash = CACHE_TIME_TABPFN_CODE_HASH
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    archive_bytes = archive.stat().st_size
    intake_dir.mkdir(parents=True)
    preserved_archive = intake_dir / "stage08_v11_tabpfn_cache_return.tar.gz"
    preserved_receipt = intake_dir / "returned_archive_receipt.json"
    shutil.copy2(archive, preserved_archive)
    shutil.copy2(receipt_file, preserved_receipt)
    archive_sha256 = _sha256_path(preserved_archive)
    extraction_root = intake_dir / "extracted"
    extraction_root.mkdir()
    with tarfile.open(preserved_archive, "r:gz") as handle:
        handle.extractall(extraction_root, filter="data")
    stage_root = extraction_root / "stage08_v11_tabpfn_cache_return"
    if not stage_root.is_dir():
        raise ValueError(f"Return archive does not contain the expected stage root: {stage_root}")
    inventory_path = stage_root / "return_inventory.json"
    inventory_sha256 = _sha256_path(inventory_path)
    errors.extend(validate_return_receipt(
        receipt, archive_sha256=archive_sha256, archive_bytes=archive_bytes,
        config_hash=tabpfn_config_hash, code_hash=tabpfn_code_hash,
        inventory_sha256=inventory_sha256, final_lock_sha256=TABPFN_LOCK_SHA256,
    ))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    observed_files = {str(path.relative_to(stage_root)).replace("\\", "/") for path in stage_root.rglob("*") if path.is_file()}
    inventory_files = {str(record["path"]).replace("\\", "/") for record in inventory.get("files", [])}
    if observed_files != inventory_files | {"return_inventory.json"}:
        errors.append("returned file set differs from the return inventory")
    for record in inventory.get("files", []):
        member = stage_root / str(record["path"]).replace("\\", "/")
        if not member.is_file() or member.stat().st_size != record["bytes"] or _sha256_path(member) != record["sha256"]:
            errors.append(f"return inventory hash/size mismatch: {record['path']}")
    member_names = sorted(observed_files)
    prohibited = scan_prohibited_return_members(member_names)
    errors.extend(f"prohibited return member: {name}" for name in prohibited)
    summary_path = stage_root / str(receipt.get("source_run", "")) / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Packed generator summary is absent: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    errors.extend(validate_generator_summary(summary, config_hash=tabpfn_config_hash, code_hash=tabpfn_code_hash, budget=budget))
    events_path = stage_root / str(receipt["source_run"]) / "events.jsonl"
    if events_path.is_file():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                json.loads(line)
    else:
        errors.append("packed generator events.jsonl is absent")
    expected_keys = expected_intake_unit_keys()
    tabpfn_units = sorted({(dataset, seed) for dataset, seed, model in expected_keys if model == "tabpfn"})
    tables: dict[str, Any] = {}
    splits: dict[tuple[str, int], Any] = {}
    tabpfn_cache_root = stage_root / "artifacts" / "caches" / "v1.1"
    summary_rows = {(row.get("dataset_id"), int(row.get("seed", -1))): row for row in summary.get("units", [])}
    if set(summary_rows) != set(tabpfn_units):
        errors.append("generator summary unit keys do not cover exactly the 80 expected TabPFN units")
    observed_keys: set[tuple[str, int, str]] = set()
    unit_results: list[dict[str, Any]] = []
    for dataset_id, base_seed in tabpfn_units:
        table, split = _locked_split_and_table(tables, splits, dataset_id, base_seed)
        ids, labels = _expected_ids_and_labels(table, split)
        provenance = expected_tabpfn_cache_provenance(ROOT, table, split, base_seed, config_hash=tabpfn_config_hash, code_hash=tabpfn_code_hash)
        cache_dir = tabpfn_cache_root / f"cfg-{tabpfn_config_hash[:12]}" / f"code-{tabpfn_code_hash[:12]}" / dataset_id / f"seed-{base_seed}" / "tabpfn"
        unit_errors = validate_cache_unit(cache_dir, provenance, ids, labels)
        observed_keys.add((dataset_id, base_seed, "tabpfn"))
        row = summary_rows.get((dataset_id, base_seed), {})
        if row.get("split_hash") != split.split_hash or row.get("dataset_hash") != table.raw_sha256:
            unit_errors.append(f"summary unit lineage differs from the regenerated v1.1 split/raw data: {dataset_id}/{base_seed}")
        if row.get("model") != "tabpfn" or row.get("action") not in {"trained_and_cached_v11_tabpfn", "reused_validated_v11_tabpfn_cache"}:
            unit_errors.append(f"summary unit action is not an authorized v1.1 TabPFN cache action: {dataset_id}/{base_seed}")
        if row.get("test_auroc") is not None and cache_dir.joinpath("manifest.json").is_file():
            manifest_metrics = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8")).get("metrics", {})
            if row.get("test_auroc") != manifest_metrics.get("test", {}).get("auroc") or row.get("test_auprc") != manifest_metrics.get("test", {}).get("auprc"):
                unit_errors.append(f"summary unit metrics differ from the cache manifest metrics: {dataset_id}/{base_seed}")
        unit_results.append({"dataset_id": dataset_id, "seed": base_seed, "model": "tabpfn", "errors": unit_errors})
        errors.extend(unit_errors)
    local_cache_time_code_hash = single_cache_time_source_hash(CACHE_ROOT, local_config_hash)
    for dataset_id, seed, model in sorted(expected_keys):
        if model == "tabpfn":
            continue
        table, split = _locked_split_and_table(tables, splits, dataset_id, seed)
        ids, labels = _expected_ids_and_labels(table, split)
        provenance = expected_local_cache_provenance(ROOT, table, split, seed, model, config_hash=local_config_hash, code_hash=local_cache_time_code_hash)
        cache_dir = CACHE_ROOT / f"cfg-{local_config_hash[:12]}" / f"code-{local_cache_time_code_hash[:12]}" / dataset_id / f"seed-{seed}" / model
        unit_errors = validate_cache_unit(cache_dir, provenance, ids, labels)
        observed_keys.add((dataset_id, seed, model))
        unit_results.append({"dataset_id": dataset_id, "seed": seed, "model": model, "errors": unit_errors})
        errors.extend(unit_errors)
    if observed_keys != expected_keys:
        errors.append(f"observed cache keys differ from the expected 240-unit coverage: missing={sorted(expected_keys - observed_keys)[:3]} extra={sorted(observed_keys - expected_keys)[:3]}")
    installed: list[str] = []
    combined_errors: list[str] = []
    verdict = "PASS" if not errors else "FAIL"
    if verdict == "PASS":
        installed = install_tabpfn_cache_units(tabpfn_cache_root, CACHE_ROOT, tabpfn_units, tabpfn_config_hash, tabpfn_code_hash)
        combined_entries = {PurePosixPath(path.relative_to(CACHE_ROOT).as_posix()) for path in CACHE_ROOT.rglob("*")}
        combined_errors = validate_combined_cache_relative_entries(combined_entries, local_config_hash, local_cache_time_code_hash, tabpfn_config_hash, tabpfn_code_hash, expected_keys)
        if combined_errors:
            verdict = "FAIL"
    result = {
        "artifact_id": "stage08_v11_cache_intake_audit",
        "stage": "Stage 08 / Task 6",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "verdict": verdict,
        "scope": "D08_003_V11_240_CACHE_INDEPENDENT_INTAKE",
        "preserved_archive": {"path": str(preserved_archive.relative_to(ROOT)), "sha256": archive_sha256, "bytes": archive_bytes},
        "preserved_receipt": {"path": str(preserved_receipt.relative_to(ROOT)), "sha256": _sha256_path(preserved_receipt)},
        "lineage": {
            "tabpfn_config_hash": tabpfn_config_hash,
            "tabpfn_cache_time_code_hash": tabpfn_code_hash,
            "audit_time_src_code_hash": audit_time_code_hash,
            "upload_bundle_receipt": UPLOAD_RECEIPT_PATH,
            "local_config_hash": local_config_hash,
            "local_cache_time_code_hash": local_cache_time_code_hash,
            "environment_hash": ENVIRONMENT_HASH,
            "split_lock_sha256": SPLIT_LOCK_SHA256,
            "local_lock_sha256": LOCAL_LOCK_SHA256,
            "tabpfn_lock_sha256": TABPFN_LOCK_SHA256,
            "d08_003_cache_lock_sha256": TABPFN_LOCK_SHA256,
        },
        "expected_units": AUTHORIZED_TOTAL_UNITS,
        "valid_units": sum(not unit["errors"] for unit in unit_results),
        "model_counts": {
            model: sum(unit["model"] == model and not unit["errors"] for unit in unit_results)
            for model in (*LOCAL_MODELS, "tabpfn")
        },
        "tabpfn_units_audited": len(tabpfn_units),
        "local_units_audited": sum(model != "tabpfn" for _, _, model in expected_keys),
        "installed_tabpfn_units": len(installed),
        "installed_paths": installed,
        "combined_tree_errors": combined_errors,
        "prohibited_members": prohibited,
        "unit_errors": [unit for unit in unit_results if unit["errors"]],
        "errors": errors,
        "cp_evaluated": False,
        "pilot_outputs": False,
        "formal_run_manifest_created": False,
        "full_experiment_executed": False,
        "limitations": [
            "The audit validates returned probability arrays, lineage, and metrics; it does not re-fit TabPFN locally.",
            "The 160 local LR/XGBoost caches are revalidated read-only against their cache-time source hash; no local cache was regenerated or modified.",
        ],
    }
    output_path = intake_dir / "intake_audit.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "valid_units": result["valid_units"], "installed": len(installed), "output": str(output_path)}, ensure_ascii=False))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

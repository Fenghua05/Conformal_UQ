"""Stage 05A train-only LR/XGBoost prediction caching; deliberately excludes CP."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conformal_uq.config import FROZEN_DATASETS, FROZEN_SEEDS, canonical_json, load_config
from conformal_uq.data import load_locked_dataset
from conformal_uq.identity import derive_seed, run_id
from conformal_uq.logging import write_event
from conformal_uq.metrics import binary_predictive_metrics
from conformal_uq.models import fit_predict_locked_pipeline
from conformal_uq.paths import cache_path, create_immutable_run_dir
from conformal_uq.prediction_cache import read_valid_cache, write_prediction_cache
from conformal_uq.provenance import sha256_path
from conformal_uq.split import make_stratified_split
from conformal_uq.stage08_authorization import load_d08_003_authorization


V11_LOCAL_HYPERPARAMETERS = {
    "logistic_regression": {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 2000, "class_weight": None},
    "xgboost": {"objective": "binary:logistic", "eval_metric": "logloss", "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "reg_lambda": 1, "reg_alpha": 0, "scale_pos_weight": 1, "tree_method": "hist", "n_jobs": 1, "early_stopping": False},
}


class V11LocalCacheRunLockError(RuntimeError):
    """Raised when another process already owns this exact v1.1 cache scope."""


def _sha256_bytes(*values: bytes) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value)
    return digest.hexdigest()


def _code_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "src").rglob("*.py")):
        digest.update(str(path.relative_to(root)).replace("\\", "/").encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _read_stage05_config(root: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    base_path = root / "configs" / "stage03_base_v1.0.yaml"
    stage_path = root / "configs" / "stage05_lr_xgboost_v1.0.yaml"
    base = load_config(base_path)
    stage = yaml.safe_load(stage_path.read_text(encoding="utf-8"))
    if stage.get("authorized_local_models") != ["logistic_regression", "xgboost"] or stage.get("formal_execution_enabled") is not True:
        raise ValueError("Stage 05A configuration must authorize only frozen LR/XGBoost local execution.")
    if stage.get("tabpfn_status") != "BLOCKED_PENDING_SEPARATE_PACKAGE_CHECKPOINT_DEVICE_CONTEXT_LOCK":
        raise ValueError("TabPFN gate has been modified without a separate lock.")
    return base, stage, _sha256_bytes(base_path.read_bytes(), stage_path.read_bytes())


def _read_stage05_lock_mapping(path: Path) -> dict[str, Any]:
    """Read a v1.1 local-cache lock; isolated for lock-level test injection."""
    stage = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(stage, dict):
        raise ValueError("An explicit Stage 08 local-cache lock must be a YAML mapping.")
    return stage


def _require_non_authorizing_v11_gate(gate: dict[str, Any]) -> None:
    for key in (
        "split_regeneration_authorized",
        "local_cache_generation_authorized",
        "tabpfn_cache_generation_authorized",
        "pilot_authorized",
        "formal_run_manifest_authorized",
        "full_experiment_authorized",
    ):
        if gate.get(key) is not False:
            raise ValueError(f"The v1.1 local-cache lock must keep {key} false; D08-003 is a separate receipt gate.")


def load_stage05_lock(root: Path, lock_path: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Validate an explicit v1.1 local-cache lock without reading data or writing output."""
    stage_path = lock_path if lock_path.is_absolute() else root / lock_path
    stage = _read_stage05_lock_mapping(stage_path)
    if stage.get("protocol_version") != "v1.1":
        raise ValueError("An explicit Stage 08 local-cache lock must use protocol v1.1.")
    split_path = root / stage.get("split_lock_path", "")
    split_lock = _read_stage05_lock_mapping(split_path)
    if not isinstance(split_lock, dict) or split_lock.get("protocol_version") != stage["protocol_version"]:
        raise ValueError("Local-cache lock must bind to the matching v1.1 split lock.")
    if stage.get("registry", {}).get("locked_primary_ids") != FROZEN_DATASETS or split_lock.get("registry", {}).get("locked_primary_ids") != FROZEN_DATASETS:
        raise ValueError("Local-cache lock registry IDs differ from the frozen eight-dataset registry.")
    if stage.get("seeds") != FROZEN_SEEDS or split_lock.get("seeds") != FROZEN_SEEDS:
        raise ValueError("Local-cache lock seeds differ from the frozen ten-seed protocol.")
    if stage.get("paths", {}).get("split_root") != "artifacts/splits/v1.1" or stage.get("paths", {}).get("cache_root") != "artifacts/caches/v1.1":
        raise ValueError("v1.1 local cache paths must be isolated from v1.0 artifacts.")
    if stage.get("authorized_local_models") != ["logistic_regression", "xgboost"]:
        raise ValueError("The v1.1 local-cache lock may authorize only LR and XGBoost.")
    if stage.get("model_hyperparameters") != V11_LOCAL_HYPERPARAMETERS:
        raise ValueError("The v1.1 local-cache lock hyperparameters differ from the frozen LR/XGBoost contract.")
    gate, output = stage.get("execution_gate", {}), stage.get("output_contract", {})
    _require_non_authorizing_v11_gate(gate)
    if any(output.get(key) is not False for key in ("conformal_prediction_allowed", "pilot_allowed", "formal_outputs_allowed")):
        raise ValueError("The v1.1 local-cache lock must prohibit CP, pilot, and formal outputs.")
    base = {
        "protocol": {"protocol_version_for_seed_derivation": stage["protocol_version"]},
        "datasets": {"primary_ids": stage["registry"]["locked_primary_ids"]},
        "experiment": {"seeds": stage["seeds"]},
    }
    return base, stage, _sha256_bytes(split_path.read_bytes(), stage_path.read_bytes())


def locked_split_manifest_path(root: Path, lock: dict[str, Any], dataset_id: str, base_seed: int) -> Path:
    """Return a v1.1 split path only after checking the requested unit is locked."""
    if dataset_id not in lock["registry"]["locked_primary_ids"] or base_seed not in lock["seeds"]:
        raise ValueError("Split requested outside the explicit v1.1 lock scope.")
    return root / lock["paths"]["split_root"] / dataset_id / f"seed-{base_seed}.json"


def locked_cache_root(root: Path, lock: dict[str, Any]) -> Path:
    """Return the isolated v1.1 cache root without creating it."""
    return root / lock["paths"]["cache_root"]


def _require_canonical_v11_lock_path(root: Path, lock_path: Path) -> None:
    expected = (root / "configs" / "stage05_lr_xgboost_v1.1.yaml").resolve()
    actual = (lock_path if lock_path.is_absolute() else root / lock_path).resolve()
    if actual != expected:
        raise ValueError("Stage 08 v1.1 execution accepts only configs/stage05_lr_xgboost_v1.1.yaml.")


def build_stage05_v11_execution_plan(root: Path, lock_path: Path) -> dict[str, Any]:
    """Validate D08-003 and derive local-only v1.1 cache work without output."""
    _require_canonical_v11_lock_path(root, lock_path)
    authorization = load_d08_003_authorization(root)
    base, lock, lock_hash = load_stage05_lock(root, lock_path)
    gate, output = lock["execution_gate"], lock["output_contract"]
    if gate.get("d08_003_numeric_cache_budget_receipt_required_before_execution") is not True:
        raise ValueError("The v1.1 local-cache lock must require D08-003 before execution.")
    _require_non_authorizing_v11_gate(gate)
    if any(output.get(key) is not False for key in ("conformal_prediction_allowed", "pilot_allowed", "formal_outputs_allowed")):
        raise ValueError("The v1.1 local-cache lock must prohibit CP, pilot, and formal outputs.")
    config_hash = _sha256_bytes(bytes.fromhex(lock_hash), bytes.fromhex(authorization["final_lock_sha256"]))
    local_lock_path = (lock_path if lock_path.is_absolute() else root / lock_path).resolve()
    split_lock_path = (root / lock["split_lock_path"]).resolve()
    return {
        "base": base,
        "lock": lock,
        "authorization": authorization,
        "config_hash": config_hash,
        "split_root": root / lock["paths"]["split_root"],
        "cache_root": locked_cache_root(root, lock),
        "local_lock_path": local_lock_path,
        "split_lock_path": split_lock_path,
        "local_cache_lock_sha256": sha256_path(local_lock_path),
        "split_lock_sha256": sha256_path(split_lock_path),
        "unit_count": len(lock["registry"]["locked_primary_ids"]) * len(lock["seeds"]) * len(lock["authorized_local_models"]),
    }


def v11_local_cache_run_lock_path(root: Path, plan: dict[str, Any]) -> Path:
    """Return the sole transient lock path for this receipt-bound cache scope."""
    return root / "artifacts" / "stage08_v11" / "locks" / (
        "v11_local_cache_"
        f"cfg-{plan['config_hash'][:12]}_"
        f"final-{plan['authorization']['final_lock_sha256'][:12]}.lock"
    )


def v11_local_cache_run_identifier(config_hash: str) -> str:
    """Allocate a unique v1.1 evidence-run identifier without changing v1.0 IDs."""
    return f"{run_id('stage08-v11-local-cache', config_hash)}_{uuid.uuid4().hex[:12]}"


@contextmanager
def exclusive_v11_local_cache_run_lock(root: Path, plan: dict[str, Any]):
    """Acquire the exact v1.1 local-cache scope before any execution side effect.

    A pre-existing lock is deliberately never reclaimed automatically: an
    operator must inspect a possible crashed holder rather than risk allowing
    overlapping cache writers.  The owner releases only its own token in a
    finally block.
    """
    path = v11_local_cache_run_lock_path(root, plan)
    token = uuid.uuid4().hex
    metadata = {
        "artifact_type": "stage08_v11_local_cache_exclusive_run_lock",
        "scope": "D08-003 authorized 160-unit v1.1 local LR/XGBoost cache run",
        "config_hash": plan["config_hash"],
        "d08_003_cache_lock_sha256": plan["authorization"]["final_lock_sha256"],
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
        raise V11LocalCacheRunLockError(
            f"The exact v1.1 local-cache scope is already held: {path}. "
            f"No data/model/cache/run work was started. Holder metadata: {holder}"
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


def reuse_or_write_v11_cache(
    destination: Path,
    provenance: dict[str, Any],
    expected_ids: dict[str, Any],
    expected_y: dict[str, Any],
    table: Any,
    split: Any,
    model_name: str,
    hyperparameters: dict[str, Any],
    derived_seed: int,
) -> tuple[str, dict[str, Any], str]:
    """Safely reuse one complete cache, fitting only when it is truly absent."""
    try:
        cached = read_valid_cache(destination, provenance, expected_ids, expected_y)
        return cached["model_hash"], cached["manifest"]["metrics"], "reused_validated_v11_cache"
    except FileNotFoundError:
        output = fit_predict_locked_pipeline(table, split, model_name, hyperparameters, derived_seed)
        metrics = {
            "calibration_pool": binary_predictive_metrics(output.calibration_y, output.calibration_probabilities),
            "test": binary_predictive_metrics(output.test_y, output.test_probabilities),
        }
        manifest = write_prediction_cache(destination, provenance, expected_ids, {"calibration_pool": output.calibration_y, "test": output.test_y}, {"calibration_pool": output.calibration_probabilities, "test": output.test_probabilities}, output.model_hash, metrics)
        return manifest["model_hash"], metrics, "trained_and_cached_v11"


def _locked_split(root: Path, table: Any, base_seed: int, protocol_version: str) -> Any:
    split = make_stratified_split(table, base_seed, protocol_version=protocol_version)
    path = root / "artifacts" / "splits" / "v1.0" / table.dataset_id / f"seed-{base_seed}.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("split_hash") != split.split_hash or manifest.get("split_ids") != split.ids.as_dict():
        raise ValueError(f"{table.dataset_id}/{base_seed}: regenerated split does not exactly match locked Stage 04 manifest.")
    return split


def _locked_v11_split(root: Path, lock: dict[str, Any], table: Any, base_seed: int) -> Any:
    """Read and verify only the v1.1 split manifest for a local cache unit."""
    split = make_stratified_split(table, base_seed, protocol_version=lock["protocol_version"])
    path = locked_split_manifest_path(root, lock, table.dataset_id, base_seed)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("split_hash") != split.split_hash or manifest.get("split_ids") != split.ids.as_dict():
        raise ValueError(f"{table.dataset_id}/{base_seed}: regenerated v1.1 split does not match its locked manifest.")
    return split


def _run_stage05_v11_local_cache(root: Path, plan: dict[str, Any]) -> int:
    """Generate exactly the receipt-authorized local probability caches, and nothing else."""
    base, lock, config_hash = plan["base"], plan["lock"], plan["config_hash"]
    code_hash = _code_hash(root)
    environment_hash = sha256_path(root / "environment" / "environment_lock_v1.0.json")
    identifier = v11_local_cache_run_identifier(config_hash)
    run_root = create_immutable_run_dir(root / "artifacts" / "runs", identifier)
    event_path = run_root / "events.jsonl"
    completed, rows, failures = 0, [], []
    write_event(event_path, run_id=identifier, stage="Stage 08 / Task 4", level="INFO", event="v11_local_cache_run_started", config_hash=config_hash, message="Starting exactly 160 authorized v1.1 LR/XGBoost cache units; CP, pilot, and formal outputs are prohibited.")
    for dataset_id in base["datasets"]["primary_ids"]:
        table = load_locked_dataset(root, dataset_id)
        for base_seed in base["experiment"]["seeds"]:
            split = _locked_v11_split(root, lock, table, base_seed)
            expected_ids = {"calibration_pool": split.ids.calibration_pool, "test": split.ids.test}
            expected_y = {partition: table.subset_labels(ids).to_numpy(dtype="int8", copy=True) for partition, ids in expected_ids.items()}
            for model_name in lock["authorized_local_models"]:
                canonical_seed, derived_seed = derive_seed(base["protocol"]["protocol_version_for_seed_derivation"], dataset_id, base_seed, model_name)
                provenance = {
                    "config_hash": config_hash, "code_hash": code_hash, "environment_hash": environment_hash,
                    "dataset_hash": table.raw_sha256, "split_hash": split.split_hash, "model_name": model_name,
                    "base_seed": base_seed, "label_mapping": table.label_mapping, "class_labels": [0, 1],
                    "protocol_version": "v1.1",
                    "local_cache_lock_sha256": plan["local_cache_lock_sha256"],
                    "split_lock_sha256": plan["split_lock_sha256"],
                    "d08_003_cache_lock_sha256": plan["authorization"]["final_lock_sha256"],
                }
                destination = cache_path(plan["cache_root"], config_hash, code_hash, dataset_id, base_seed, model_name)
                scope = {"dataset_id": dataset_id, "seed": base_seed, "model": model_name}
                try:
                    model_hash, metrics, action = reuse_or_write_v11_cache(
                        destination, provenance, expected_ids, expected_y, table, split, model_name,
                        lock["model_hyperparameters"][model_name], derived_seed,
                    )
                    completed += 1
                    rows.append({**scope, "action": action, "cache_path": str(destination.relative_to(root)), "model_hash": model_hash, "split_hash": split.split_hash, "test_auroc": metrics["test"]["auroc"], "test_auprc": metrics["test"]["auprc"], "calibration_pool_auroc": metrics["calibration_pool"]["auroc"], "calibration_pool_auprc": metrics["calibration_pool"]["auprc"]})
                    write_event(event_path, run_id=identifier, stage="Stage 08 / Task 4", level="INFO", event="v11_local_cache_unit_complete", config_hash=config_hash, message=action, **scope, model_hash=model_hash)
                except Exception as exc:
                    failure_path = _failure(root, scope, exc, identifier, config_hash, v11_scope=True)
                    failures.append({**scope, "failure_record": str(failure_path.relative_to(root)), "exception_type": type(exc).__name__, "exception": str(exc)})
                    write_event(event_path, run_id=identifier, stage="Stage 08 / Task 4", level="ERROR", event="v11_local_cache_unit_failed", config_hash=config_hash, message=str(exc), **scope)
                # Deriving the canonical seed remains part of the deterministic cache contract.
                if not canonical_seed:
                    raise RuntimeError("The v1.1 local-cache seed derivation unexpectedly produced an empty canonical input.")
    status = "PASS" if not failures and completed == plan["unit_count"] == 160 else "FAIL"
    summary = {
        "artifact_id": f"{identifier}_summary", "stage": "Stage 08 / Task 4", "run_id": identifier,
        "protocol_version": "v1.1", "status": status, "config_hash": config_hash, "code_hash": code_hash,
        "environment_hash": environment_hash, "local_cache_lock_sha256": plan["local_cache_lock_sha256"],
        "split_lock_sha256": plan["split_lock_sha256"], "d08_003_cache_lock_sha256": plan["authorization"]["final_lock_sha256"],
        "authorized_local_cache_units": 160, "completed_local_cache_units": completed,
        "model_counts": {model: sum(row["model"] == model for row in rows) for model in lock["authorized_local_models"]},
        "units": rows, "failures": failures,
        "cp_evaluated": False, "pilot_outputs": False, "formal_run_manifest_created": False, "full_experiment_executed": False,
    }
    (run_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    write_event(event_path, run_id=identifier, stage="Stage 08 / Task 4", level="INFO" if status == "PASS" else "ERROR", event="v11_local_cache_run_finished", config_hash=config_hash, message=status, completed=completed, failures=len(failures))
    print(json.dumps({"status": status, "run_root": str(run_root), "completed": completed, "failures": len(failures), "cp_evaluated": False, "pilot_outputs": False, "formal_outputs": False}))
    return 0 if status == "PASS" else 2


def _failure(
    root: Path, scope: dict[str, Any], exception: Exception, run_identifier: str, config_hash: str, *, v11_scope: bool = False,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if v11_scope:
        path = root / "artifacts" / "failures" / (
            f"{timestamp}_stage08_v11_{run_identifier}_{scope['dataset_id']}_{scope['seed']}_{scope['model']}.json"
        )
    else:
        # Historical v1.0 filename behavior remains unchanged.
        path = root / "artifacts" / "failures" / f"{timestamp}_stage05_{scope['dataset_id']}_{scope['seed']}_{scope['model']}.json"
    payload = {"failure_id": path.stem, "stage": "Stage 05", "run_id": run_identifier, "timestamp_utc": timestamp, "classification": "bug_or_data_or_environment_pending_triage", "retry_count": 0, "scope": scope, "config_hash": config_hash, "exception_type": type(exception).__name__, "exception": str(exception), "action": "No blind retry; inspect this immutable failure record before rerun."}
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if v11_scope:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
    else:
        path.write_text(encoded, encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--lock", type=Path, help="Explicit v1.1 local-cache lock; preparation locks reject execution.")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.lock:
        plan = build_stage05_v11_execution_plan(root, args.lock)
        if args.mode != "formal":
            raise ValueError("The v1.1 local-cache branch executes only the complete 160-unit cache scope; smoke mode is not a D08-003 artifact.")
        # Must occur before a run directory, data load, model fit, or cache
        # access so a second invocation cannot race an in-progress 160-unit run.
        with exclusive_v11_local_cache_run_lock(root, plan):
            return _run_stage05_v11_local_cache(root, plan)
    base, stage, config_hash = _read_stage05_config(root)
    code_hash = _code_hash(root)
    environment_hash = sha256_path(root / "environment" / "environment_lock_v1.0.json")
    identifier = run_id("stage05-smoke" if args.mode == "smoke" else "stage05a", config_hash)
    run_root = create_immutable_run_dir(root / "artifacts" / "runs", identifier)
    event_path = run_root / "events.jsonl"
    datasets = [stage["smoke_selection"]["dataset_id"]] if args.mode == "smoke" else list(base["datasets"]["primary_ids"])
    seeds = [stage["smoke_selection"]["seed"]] if args.mode == "smoke" else list(base["experiment"]["seeds"])
    rows, failures = [], []
    write_event(event_path, run_id=identifier, stage="Stage 05", level="INFO", event="run_started", config_hash=config_hash, message="Starting train-only base-prediction cache generation; no CP cells are evaluated.", mode=args.mode)
    for dataset_id in datasets:
        table = load_locked_dataset(root, dataset_id)
        for base_seed in seeds:
            split = _locked_split(root, table, base_seed, base["protocol"]["protocol_version_for_seed_derivation"])
            expected_ids = {"calibration_pool": split.ids.calibration_pool, "test": split.ids.test}
            expected_y = {partition: table.subset_labels(ids).to_numpy(dtype="int8", copy=True) for partition, ids in expected_ids.items()}
            for model_name in stage["authorized_local_models"]:
                canonical_seed, derived_seed = derive_seed(base["protocol"]["protocol_version_for_seed_derivation"], dataset_id, base_seed, model_name)
                provenance = {"config_hash": config_hash, "code_hash": code_hash, "environment_hash": environment_hash, "dataset_hash": table.raw_sha256, "split_hash": split.split_hash, "model_name": model_name, "base_seed": base_seed, "label_mapping": table.label_mapping, "class_labels": [0, 1]}
                destination = cache_path(root / base["paths"]["cache_root"], config_hash, code_hash, dataset_id, base_seed, model_name)
                scope = {"dataset_id": dataset_id, "seed": base_seed, "model": model_name}
                try:
                    try:
                        cached = read_valid_cache(destination, provenance, expected_ids, expected_y)
                        metrics = cached["manifest"].get("metrics", {})
                        model_hash, action = cached["model_hash"], "reused_validated_cache"
                    except FileNotFoundError:
                        output = fit_predict_locked_pipeline(table, split, model_name, base["models"][model_name]["hyperparameters"], derived_seed)
                        metrics = {"calibration_pool": binary_predictive_metrics(output.calibration_y, output.calibration_probabilities), "test": binary_predictive_metrics(output.test_y, output.test_probabilities)}
                        manifest = write_prediction_cache(destination, provenance, expected_ids, {"calibration_pool": output.calibration_y, "test": output.test_y}, {"calibration_pool": output.calibration_probabilities, "test": output.test_probabilities}, output.model_hash, metrics)
                        model_hash, action = manifest["model_hash"], "trained_and_cached"
                    rows.append({**scope, "derived_seed": derived_seed, "canonical_seed_input": canonical_seed, "action": action, "model_hash": model_hash, "split_hash": split.split_hash, "cache_path": str(destination.relative_to(root)), "test_auroc": metrics["test"]["auroc"], "test_auprc": metrics["test"]["auprc"], "calibration_pool_auroc": metrics["calibration_pool"]["auroc"], "calibration_pool_auprc": metrics["calibration_pool"]["auprc"]})
                    write_event(event_path, run_id=identifier, stage="Stage 05", level="INFO", event="prediction_unit_complete", config_hash=config_hash, message=action, **scope, model_hash=model_hash)
                except Exception as exc:
                    failure_path = _failure(root, scope, exc, identifier, config_hash)
                    failures.append({**scope, "failure_record": str(failure_path.relative_to(root)), "exception": str(exc)})
                    write_event(event_path, run_id=identifier, stage="Stage 05", level="ERROR", event="prediction_unit_failed", config_hash=config_hash, message=str(exc), **scope)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.to_parquet(run_root / "base_predictive_metrics.parquet", index=False)
        frame.to_csv(run_root / "base_predictive_metrics.csv", index=False)
    summary = {"artifact_id": f"{identifier}_summary", "stage": "Stage 05A", "mode": args.mode, "status": "PASS" if not failures else "FAIL", "run_id": identifier, "config_hash": config_hash, "code_hash": code_hash, "environment_hash": environment_hash, "dataset_count": len(datasets), "seed_count": len(seeds), "model_count": len(stage["authorized_local_models"]), "prediction_units_complete": len(rows), "failures": failures, "tabpfn": {"status": stage["tabpfn_status"], "executed": False}, "cp_evaluated": False, "metrics_path": "base_predictive_metrics.parquet"}
    (run_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    write_event(event_path, run_id=identifier, stage="Stage 05", level="INFO" if not failures else "ERROR", event="run_finished", config_hash=config_hash, message=summary["status"], completed=len(rows), failures=len(failures))
    print(json.dumps({"status": summary["status"], "run_root": str(run_root), "completed": len(rows), "failures": len(failures)}))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())

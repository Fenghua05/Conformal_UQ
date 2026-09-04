"""Run the hash-gated, two-dataset Stage 07 CP pilot from immutable caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cloud" / "tabpfn_stage05b"))

from stage05b_common import checked_lock
from conformal_uq.conformal import CP_METHODS, M_MINORITY, evaluate_split_cp, select_nested_calibration_subsets
from conformal_uq.data import load_dataset_registry, registry_record
from conformal_uq.logging import write_event
from conformal_uq.metrics import binary_predictive_metrics
from conformal_uq.paths import create_immutable_run_dir
from conformal_uq.prediction_cache import read_valid_cache
from conformal_uq.provenance import sha256_path
from conformal_uq.results_schema import validate_results_records
from conformal_uq.stage07_qc import diagnostic_figures, independent_qc
from conformal_uq.split import make_stratified_split
from conformal_uq.data import load_locked_dataset


MODELS = ("logistic_regression", "xgboost", "tabpfn")
RESULTS_SCHEMA_VERSION = "v1.1.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def expected_pilot_cells(dataset_ids: tuple[str, ...], seeds: tuple[int, ...]) -> tuple[tuple[str, int, str, str, int], ...]:
    return tuple((dataset_id, int(seed), model, method, m) for dataset_id in dataset_ids for seed in seeds for model in MODELS for method in CP_METHODS for m in M_MINORITY)


def _cell_name(cell: tuple[str, int, str, str, int]) -> str:
    dataset_id, seed, model, method, m = cell
    return f"{dataset_id}__seed-{seed}__{model}__{method}__m-{m}.json"


def _minority_label(root: Path, dataset_id: str, registry_path: Path) -> int:
    record = registry_record(load_dataset_registry(root, registry_path), dataset_id)
    return int(record["label_mapping_to_protocol_binary"][str(record["minority_original_label"])])


def _load_cache(root: Path, dataset_id: str, seed: int, model: str, registry_path: Path) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    table = load_locked_dataset(root, dataset_id, registry_path=registry_path)
    split = make_stratified_split(table, seed, protocol_version="v1.0")
    split_manifest = json.loads((root / "artifacts" / "splits" / "v1.0" / dataset_id / f"seed-{seed}.json").read_text(encoding="utf-8"))
    if split_manifest.get("raw_sha256") != table.raw_sha256 or split_manifest.get("split_hash") != split.split_hash or split_manifest.get("split_ids") != split.ids.as_dict():
        raise ValueError(f"{dataset_id}/{seed}: regenerated split does not match the immutable Stage 04 manifest.")
    candidates = sorted((root / "artifacts" / "caches").glob(f"cfg-*/code-*/{dataset_id}/seed-{seed}/{model}"))
    if len(candidates) != 1:
        raise ValueError(f"{dataset_id}/{seed}/{model}: expected exactly one cache lineage, found {len(candidates)}.")
    manifest = json.loads((candidates[0] / "manifest.json").read_text(encoding="utf-8"))
    provenance = manifest.get("provenance", {})
    if provenance.get("model_name") != model or provenance.get("base_seed") != seed or provenance.get("dataset_hash") != table.raw_sha256 or provenance.get("split_hash") != split.split_hash or provenance.get("label_mapping") != table.label_mapping or provenance.get("class_labels") != [0, 1]:
        raise ValueError(f"{dataset_id}/{seed}/{model}: cache provenance does not match the locked binary table/split.")
    expected_ids = {"calibration_pool": split.ids.calibration_pool, "test": split.ids.test}
    expected_labels = {name: table.subset_labels(ids).to_numpy(dtype="int8", copy=True) for name, ids in expected_ids.items()}
    cached = read_valid_cache(candidates[0], provenance, expected_ids, expected_labels)
    recomputed = binary_predictive_metrics(cached["labels"]["test"], cached["probabilities"]["test"])
    declared = cached["manifest"].get("metrics", {}).get("test", {})
    if any(abs(float(recomputed[key]) - float(declared[key])) > 1e-15 for key in ("auroc", "auprc")):
        raise ValueError(f"{dataset_id}/{seed}/{model}: cache predictive metrics differ from its manifest.")
    return table, split, cached, {"cache_dir": candidates[0], "provenance": provenance, "base_metrics": recomputed}


def _record(*, run_id: str, dataset_id: str, seed: int, model: str, method: str, m: int, table: Any, split: Any, cache: dict[str, Any], context: dict[str, Any], minority_label: int) -> dict[str, Any]:
    subsets = select_nested_calibration_subsets(cache["ids"]["calibration_pool"], cache["labels"]["calibration_pool"], protocol_version="v1.0", dataset_id=dataset_id, base_seed=seed, minority_label=minority_label)
    cell = evaluate_split_cp(cache["ids"]["calibration_pool"], cache["labels"]["calibration_pool"], cache["probabilities"]["calibration_pool"], cache["labels"]["test"], cache["probabilities"]["test"], subsets, m_minority=m, cp_method=method)
    provenance = context["provenance"]
    return {
        "dataset_id": dataset_id, "seed": seed, "model": model, "cp_method": method, "m_minority": m, "m_majority": cell["m_majority"], "alpha": 0.1,
        "protocol_version": "v1.0", "config_hash": provenance["config_hash"], "code_hash": provenance["code_hash"], "run_id": run_id,
        "artifact_id": f"{run_id}_{dataset_id}_{seed}_{model}_{method}_m{m}", "results_schema_version": RESULTS_SCHEMA_VERSION, "status": "PASS", "minority_label": minority_label,
        **cell, "auroc": context["base_metrics"]["auroc"], "auprc": context["base_metrics"]["auprc"],
        "split_hash": split.split_hash, "dataset_hash": table.raw_sha256, "environment_hash": provenance["environment_hash"], "model_hash": cache["model_hash"],
        "prediction_cache_hash": cache["manifest"]["cache_sha256"], "label_mapping_hash": _hash_json(table.label_mapping), "created_utc": utc_now(),
    }


def _run_identity(lock: dict[str, Any]) -> str:
    payload = json.dumps({"datasets": lock["pilot_dataset_ids"], "seeds": lock["seeds"], "models": MODELS, "cp": CP_METHODS, "m": M_MINORITY}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--pilot-decision", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    lock, lock_config_hash, lock_code_hash = checked_lock(ROOT, args.lock, args.pilot_decision)
    dataset_ids, seeds = tuple(lock["pilot_dataset_ids"]), tuple(int(seed) for seed in lock["seeds"])
    expected = expected_pilot_cells(dataset_ids, seeds)
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_stage07-pilot_{_run_identity(lock)}"
    run_root = args.run_dir.resolve() if args.run_dir else create_immutable_run_dir(ROOT / "artifacts" / "stage07", run_id)
    if args.run_dir:
        if not run_root.is_dir() or run_root.parent != (ROOT / "artifacts" / "stage07").resolve():
            raise ValueError("--run-dir must be an existing direct child of artifacts/stage07.")
    run_root.mkdir(parents=True, exist_ok=True)
    event_path, cell_root = run_root / "events.jsonl", run_root / "cells"
    cell_root.mkdir(exist_ok=True)
    manifest_path = run_root / "run_manifest.json"
    manifest = {"artifact_id": f"{run_root.name}_manifest", "stage": "Stage 07", "scope": "LOCKED_TWO_DATASET_PILOT_ONLY", "pilot_dataset_ids": list(dataset_ids), "seeds": list(seeds), "models": list(MODELS), "cp_methods": list(CP_METHODS), "m_minority": list(M_MINORITY), "expected_cells": len(expected), "stage05b_lock_hash": lock_config_hash, "stage05b_guard_code_hash_at_start": lock_code_hash, "created_utc": utc_now()}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        # A returned cache's own manifest is the immutable Stage 05B code
        # identity. The local Stage 07 source tree can grow between resume
        # attempts and must not invalidate a verified Stage 07 cell.
        for key in ("pilot_dataset_ids", "seeds", "models", "cp_methods", "m_minority", "expected_cells", "stage05b_lock_hash"):
            if old.get(key) != manifest[key]:
                raise ValueError("Existing Stage 07 run directory has a different controlled manifest.")
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    write_event(event_path, run_id=run_root.name, stage="Stage 07", level="INFO", event="pilot_started", config_hash=lock_config_hash, message="Hash-gated pilot started or resumed; CP uses only immutable caches.", expected_cells=len(expected))
    cache_context: dict[tuple[str, int, str], tuple[Any, Any, dict[str, Any], dict[str, Any], int]] = {}
    failures: list[dict[str, Any]] = []
    for dataset_id, seed, model, method, m in expected:
        cell = (dataset_id, seed, model, method, m)
        destination = cell_root / _cell_name(cell)
        if destination.exists():
            continue
        try:
            key = (dataset_id, seed, model)
            if key not in cache_context:
                table, split, cache, context = _load_cache(ROOT, dataset_id, seed, model, ROOT / lock["registry_path"])
                cache_context[key] = (table, split, cache, context, _minority_label(ROOT, dataset_id, ROOT / lock["registry_path"]))
            table, split, cache, context, minority_label = cache_context[key]
            record = _record(run_id=run_root.name, dataset_id=dataset_id, seed=seed, model=model, method=method, m=m, table=table, split=split, cache=cache, context=context, minority_label=minority_label)
            errors = validate_results_records([record])
            if errors:
                raise ValueError(errors)
            destination.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            write_event(event_path, run_id=run_root.name, stage="Stage 07", level="INFO", event="cell_complete", config_hash=record["config_hash"], message="CP cell verified and written once.", dataset_id=dataset_id, seed=seed, model=model, cp_method=method, m_minority=m)
        except Exception as exc:
            failure_path = ROOT / "artifacts" / "failures" / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_stage07_{uuid.uuid4().hex[:10]}.json"
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure = {"dataset_id": dataset_id, "seed": seed, "model": model, "cp_method": method, "m_minority": m, "exception_type": type(exc).__name__, "exception": str(exc), "retry_count": 0, "failure_record": str(failure_path.relative_to(ROOT))}
            failure_path.write_text(json.dumps({"stage": "Stage 07", "run_id": run_root.name, "created_utc": utc_now(), "classification": "implementation_or_input_failure_pending_triage", **failure}, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            failures.append(failure)
            write_event(event_path, run_id=run_root.name, stage="Stage 07", level="ERROR", event="cell_failed", config_hash=lock_config_hash, message=str(exc), **failure)
    records = [json.loads((cell_root / _cell_name(cell)).read_text(encoding="utf-8")) for cell in expected if (cell_root / _cell_name(cell)).exists()]
    batch_errors = validate_results_records(records)
    status = "PASS" if len(records) == len(expected) and not failures and not batch_errors else "FAIL"
    status_payload = {"artifact_id": f"{run_root.name}_status", "stage": "Stage 07", "status": status, "expected_cells": len(expected), "verified_cells": len(records), "missing_cells": len(expected) - len(records), "failures": failures, "batch_validation_errors": batch_errors, "updated_utc": utc_now(), "resume_command": f"{sys.executable} {Path(__file__).name} --lock {args.lock} --pilot-decision {args.pilot_decision} --run-dir {run_root}"}
    (run_root / "run_status.json").write_text(json.dumps(status_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if status != "PASS":
        write_event(event_path, run_id=run_root.name, stage="Stage 07", level="ERROR", event="pilot_incomplete", config_hash=lock_config_hash, message="Pilot stopped with preserved cells; inspect run_status and resume only after triage.", completed=len(records), failures=len(failures))
        print(json.dumps({"status": status, "run_dir": str(run_root), "completed": len(records), "failures": len(failures)}))
        return 2
    results_path = run_root / "results_long.parquet"
    if results_path.exists():
        if len(pd.read_parquet(results_path)) != len(expected):
            raise ValueError("Existing results_long is inconsistent; it will not be overwritten.")
    else:
        pd.DataFrame(records).to_parquet(results_path, index=False)
        pd.DataFrame(records).to_csv(run_root / "results_long.csv", index=False)
    qc = independent_qc(ROOT, records, lock)
    figure_map = diagnostic_figures(records, run_root / "figures")
    qc["figures"] = figure_map
    (run_root / "pilot_qc.json").write_text(json.dumps(qc, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if qc["status"] != "PASS":
        raise ValueError("Independent Stage 07 QC failed; preserved results require triage before use.")
    write_event(event_path, run_id=run_root.name, stage="Stage 07", level="WARN" if qc["global_coverage_diagnostic"]["implementation_review_required"] else "INFO", event="pilot_cells_complete", config_hash=lock_config_hash, message="All 480 CP cells and independent QC completed; coverage flags are diagnostic only.", completed=len(records), global_implementation_review_required=qc["global_coverage_diagnostic"]["implementation_review_required"])
    print(json.dumps({"status": "PASS", "run_dir": str(run_root), "results_long": str(results_path), "completed": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

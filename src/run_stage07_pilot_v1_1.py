"""Run the fixed 480-cell v1.1 CP pilot from the independently audited 240 caches.

This runner executes ONLY the D08-003-authorized v1.1 pilot scope
(2 outcome-blind datasets x 10 frozen seeds x 3 models x 2 CP methods x 4
minority calibration sizes = 480 cells) after the independent 240-cache
intake audit has PASSED.  It never imports or runs TabPFN: every probability
comes from immutable caches.  It reads the v1.1 splits, resolves each model
family's cache tree from the intake-audit lineage (local LR/XGBoost and cloud
TabPFN are separate cfg/code lineages), and preserves the two Stage 07 bug
fixes (Wilson-endpoint clipping and stored-probability geometry) that live in
the shared Stage 06 modules.  No v1.0 pilot cell is read, reused, or mutated,
and no formal-run output is created.
"""

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

from conformal_uq.config import FROZEN_DATASETS, FROZEN_SEEDS
from conformal_uq.conformal import CP_METHODS, M_MINORITY, evaluate_split_cp, select_nested_calibration_subsets
from conformal_uq.data import load_dataset_registry, load_locked_dataset, registry_record
from conformal_uq.logging import write_event
from conformal_uq.metrics import binary_predictive_metrics
from conformal_uq.paths import create_immutable_run_dir
from conformal_uq.prediction_cache import read_valid_cache
from conformal_uq.provenance import sha256_path
from conformal_uq.results_schema import validate_results_records
from conformal_uq.split import make_stratified_split
from conformal_uq.stage07_qc import diagnostic_figures
from conformal_uq.stage08_authorization import load_d08_003_authorization

MODELS = ("logistic_regression", "xgboost", "tabpfn")
RESULTS_SCHEMA_VERSION = "v1.1.0"
EXPECTED_PILOT_CELLS = 480
CANONICAL_DECISION_PATH = Path("decisions/pilot_decision_stage07_v1.1.json")
CANONICAL_REGISTRY_PATH = Path("artifacts/stage02/dataset_registry_v1.0.1.json")
SPLIT_ROOT = "artifacts/splits/v1.1"
CACHE_ROOT_RELATIVE = "artifacts/caches/v1.1"
INTAKE_GLOB = "cache_intake_*/intake_audit.json"
PILOT_ROOT = ROOT / "artifacts" / "stage07_v1.1"
LOCAL_MODELS = ("logistic_regression", "xgboost")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sha256_bytes(*chunks: bytes) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


class V11PilotGateError(ValueError):
    """Raised when a v1.1 pilot execution gate is not satisfied."""


def expected_pilot_cells_v11(dataset_ids: tuple[str, ...], seeds: tuple[int, ...]) -> tuple[tuple[str, int, str, str, int], ...]:
    return tuple(
        (dataset_id, int(seed), model, method, m)
        for dataset_id in dataset_ids
        for seed in seeds
        for model in MODELS
        for method in CP_METHODS
        for m in M_MINORITY
    )


def _cell_name(cell: tuple[str, int, str, str, int]) -> str:
    dataset_id, seed, model, method, m = cell
    return f"{dataset_id}__seed-{seed}__{model}__{method}__m-{m}.json"


def load_and_validate_v11_pilot_decision(root: Path, decision_path: Path) -> dict[str, Any]:
    """Accept only the canonical, hash-bound, pre-outcome v1.1 pilot decision."""
    canonical = (root / CANONICAL_DECISION_PATH).resolve()
    actual = (decision_path if decision_path.is_absolute() else root / decision_path).resolve()
    if actual != canonical:
        raise V11PilotGateError("Only the canonical v1.1 pilot decision may drive the 480-cell v1.1 pilot.")
    decision = json.loads(actual.read_text(encoding="utf-8"))
    if decision.get("artifact_id") != "pilot_decision_stage07_v1.1" or decision.get("protocol_version") != "v1.1":
        raise V11PilotGateError("The pilot decision is not the version-specific v1.1 record.")
    if decision.get("status") != "APPROVED_PRE_OUTCOME_PILOT_DECISION":
        raise V11PilotGateError("The pilot decision is not an approved pre-outcome decision.")
    expected_hashes = {
        "registry_sha256": root / decision.get("registry_path", str(CANONICAL_REGISTRY_PATH)),
        "dataset_lock_sha256": root / decision.get("dataset_lock_path", "protocols/dataset_lock_v1.0.md"),
        "protocol_sha256": root / decision.get("protocol_path", "protocols/protocol_v1.1.md"),
    }
    binding = decision.get("v11_lineage_binding", {})
    expected_hashes.update({
        "split_lock_sha256": root / binding.get("split_lock_path", "configs/stage04_splits_v1.1.yaml"),
        "local_cache_lock_sha256": root / binding.get("local_cache_lock_path", "configs/stage05_lr_xgboost_v1.1.yaml"),
        "tabpfn_cache_lock_sha256": root / binding.get("tabpfn_cache_lock_path", "configs/stage05b_tabpfn_v1.1.yaml"),
        "d08_003_receipt_sha256": root / binding.get("d08_003_receipt_path", "decisions/D08-003_V11_CACHE_AND_PILOT_BUDGET_RECEIPT.json"),
    })
    for key, path in expected_hashes.items():
        if not path.is_file() or decision.get(key, binding.get(key)) != sha256_path(path):
            raise V11PilotGateError(f"Pilot decision binding {key} does not match the current controlled input.")
    pilot_ids = decision.get("pilot_dataset_ids")
    if not isinstance(pilot_ids, list) or len(pilot_ids) != 2 or len(set(pilot_ids)) != 2 or not set(pilot_ids).issubset(FROZEN_DATASETS):
        raise V11PilotGateError("The v1.1 pilot decision must lock exactly two frozen primary datasets.")
    inputs = decision.get("outcome_blind_inputs", {})
    if inputs.get("fallback_applied") is not True or not inputs.get("fallback_reason"):
        raise V11PilotGateError("The v1.1 pilot decision must record its objective fallback rationale.")
    if not any("v1.0" in item for item in inputs.get("prohibited_inputs", [])):
        raise V11PilotGateError("The v1.1 pilot decision must prohibit v1.0 outcome inputs.")
    return decision


def require_intake_pass(root: Path) -> dict[str, Any]:
    """The v1.1 pilot may run only on a PASS 240-cache independent intake audit."""
    candidates = sorted((root / "artifacts" / "stage08_v11_cloud").glob(INTAKE_GLOB))
    if not candidates:
        raise V11PilotGateError("No 240-cache intake audit exists; the v1.1 pilot is not authorized.")
    audit = json.loads(candidates[-1].read_text(encoding="utf-8"))
    if audit.get("verdict") != "PASS":
        raise V11PilotGateError("The latest 240-cache intake audit did not PASS; the v1.1 pilot is not authorized.")
    if audit.get("valid_units") != 240 or audit.get("expected_units") != 240:
        raise V11PilotGateError("The intake audit does not verify exactly 240 cache units.")
    if audit.get("model_counts") != {"logistic_regression": 80, "xgboost": 80, "tabpfn": 80}:
        raise V11PilotGateError("The intake audit model counts are not 80 per model.")
    if audit.get("installed_tabpfn_units") != 80 or audit.get("errors") or audit.get("unit_errors"):
        raise V11PilotGateError("The intake audit reports unresolved errors or incomplete installation.")
    for key in ("cp_evaluated", "pilot_outputs", "formal_run_manifest_created", "full_experiment_executed"):
        if audit.get(key) is not False:
            raise V11PilotGateError(f"The intake audit must keep {key} false.")
    audit["_intake_audit_path"] = str(candidates[-1].relative_to(root)).replace("\\", "/")
    return audit


def resolve_v11_cache_lineage(root: Path) -> dict[str, Any]:
    """Bind the pilot to the intake-audited cache lineage and the current locks."""
    authorization = load_d08_003_authorization(root)
    receipt = authorization["receipt"]
    if receipt.get("authorized_pilot_cells") != EXPECTED_PILOT_CELLS or receipt.get("full_experiment_authorized") is not False:
        raise V11PilotGateError("The D08-003 receipt does not authorize exactly the 480-cell v1.1 pilot.")
    audit = require_intake_pass(root)
    audit_lineage = audit.get("lineage", {})
    split_lock_path = root / "configs/stage04_splits_v1.1.yaml"
    local_lock_path = root / "configs/stage05_lr_xgboost_v1.1.yaml"
    tabpfn_lock_path = root / "configs/stage05b_tabpfn_v1.1.yaml"
    split_bytes, local_bytes, tabpfn_bytes = split_lock_path.read_bytes(), local_lock_path.read_bytes(), tabpfn_lock_path.read_bytes()
    # Recompute both family config hashes exactly as the cache runners did.
    interim_local = _sha256_bytes(split_bytes, local_bytes)
    local_config_hash = _sha256_bytes(bytes.fromhex(interim_local), bytes.fromhex(_sha256_bytes(tabpfn_bytes)))
    interim_tabpfn = _sha256_bytes(split_bytes, tabpfn_bytes)
    tabpfn_config_hash = _sha256_bytes(bytes.fromhex(interim_tabpfn), bytes.fromhex(_sha256_bytes(tabpfn_bytes)))
    lineage = {
        "local_config_hash": local_config_hash,
        "local_code_hash": audit_lineage.get("local_cache_time_code_hash"),
        "local_lock_sha256": sha256_path(local_lock_path),
        "tabpfn_config_hash": tabpfn_config_hash,
        "tabpfn_code_hash": audit_lineage.get("tabpfn_cache_time_code_hash"),
        "tabpfn_lock_sha256": sha256_path(tabpfn_lock_path),
        "split_lock_sha256": sha256_path(split_lock_path),
        "d08_003_cache_lock_sha256": authorization["final_lock_sha256"],
        "environment_hash": audit_lineage.get("environment_hash"),
        "cache_root": root / CACHE_ROOT_RELATIVE,
        "intake_audit_path": audit["_intake_audit_path"],
        "receipt_sha256": sha256_path(root / "decisions/D08-003_V11_CACHE_AND_PILOT_BUDGET_RECEIPT.json"),
    }
    if audit_lineage.get("local_config_hash") != local_config_hash or audit_lineage.get("tabpfn_config_hash") != tabpfn_config_hash:
        raise V11PilotGateError("A v1.1 lock changed after the intake audit; the audited caches are stale for this pilot.")
    if audit_lineage.get("split_lock_sha256") != lineage["split_lock_sha256"] or audit_lineage.get("local_lock_sha256") != lineage["local_lock_sha256"] or audit_lineage.get("tabpfn_lock_sha256") != lineage["tabpfn_lock_sha256"]:
        raise V11PilotGateError("A v1.1 lock hash differs from the intake-audit lineage.")
    for key in ("local_code_hash", "tabpfn_code_hash", "environment_hash"):
        if not isinstance(lineage[key], str) or len(lineage[key]) != 64:
            raise V11PilotGateError(f"The intake audit does not record a valid {key}.")
    return lineage


def expected_cache_provenance(lineage: dict[str, Any], table: Any, split: Any, base_seed: int, model: str) -> dict[str, Any]:
    """Rebuild the exact provenance a v1.1 cache of this model family must carry."""
    if model == "tabpfn":
        config_hash, code_hash, lock_hash = lineage["tabpfn_config_hash"], lineage["tabpfn_code_hash"], lineage["tabpfn_lock_sha256"]
    else:
        config_hash, code_hash, lock_hash = lineage["local_config_hash"], lineage["local_code_hash"], lineage["local_lock_sha256"]
    return {
        "config_hash": config_hash, "code_hash": code_hash, "environment_hash": lineage["environment_hash"],
        "dataset_hash": table.raw_sha256, "split_hash": split.split_hash,
        "model_name": model, "base_seed": base_seed,
        "label_mapping": table.label_mapping, "class_labels": [0, 1],
        "protocol_version": "v1.1",
        "local_cache_lock_sha256": lock_hash,
        "split_lock_sha256": lineage["split_lock_sha256"],
        "d08_003_cache_lock_sha256": lineage["tabpfn_lock_sha256"],
    }


def cache_directory(root: Path, lineage: dict[str, Any], dataset_id: str, seed: int, model: str) -> Path:
    config_hash = lineage["tabpfn_config_hash"] if model == "tabpfn" else lineage["local_config_hash"]
    code_hash = lineage["tabpfn_code_hash"] if model == "tabpfn" else lineage["local_code_hash"]
    return lineage["cache_root"] / f"cfg-{config_hash[:12]}" / f"code-{code_hash[:12]}" / dataset_id / f"seed-{seed}" / model


def _minority_label(root: Path, dataset_id: str, registry_path: Path) -> int:
    record = registry_record(load_dataset_registry(root, registry_path), dataset_id)
    return int(record["label_mapping_to_protocol_binary"][str(record["minority_original_label"])])


def load_v11_cache(root: Path, lineage: dict[str, Any], dataset_id: str, seed: int, model: str, registry_path: Path) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Load and revalidate one immutable v1.1 cache against its expected lineage."""
    table = load_locked_dataset(root, dataset_id, registry_path=registry_path)
    split = make_stratified_split(table, seed, protocol_version="v1.1")
    split_manifest = json.loads((root / SPLIT_ROOT / dataset_id / f"seed-{seed}.json").read_text(encoding="utf-8"))
    if split_manifest.get("raw_sha256") != table.raw_sha256 or split_manifest.get("split_hash") != split.split_hash or split_manifest.get("split_ids") != split.ids.as_dict():
        raise ValueError(f"{dataset_id}/{seed}: regenerated v1.1 split does not match the locked v1.1 split manifest.")
    provenance = expected_cache_provenance(lineage, table, split, seed, model)
    cache_dir = cache_directory(root, lineage, dataset_id, seed, model)
    expected_ids = {"calibration_pool": split.ids.calibration_pool, "test": split.ids.test}
    expected_labels = {name: table.subset_labels(ids).to_numpy(dtype="int8", copy=True) for name, ids in expected_ids.items()}
    cached = read_valid_cache(cache_dir, provenance, expected_ids, expected_labels)
    recomputed = binary_predictive_metrics(cached["labels"]["test"], cached["probabilities"]["test"])
    declared = cached["manifest"].get("metrics", {}).get("test", {})
    if any(abs(float(recomputed[key]) - float(declared[key])) > 1e-15 for key in ("auroc", "auprc")):
        raise ValueError(f"{dataset_id}/{seed}/{model}: cache predictive metrics differ from its manifest.")
    return table, split, cached, {"cache_dir": cache_dir, "provenance": provenance, "base_metrics": recomputed}


def write_cell_once(path: Path, record: dict[str, Any]) -> bool:
    """Write one cell with exclusive creation; existing cells are never overwritten."""
    if path.exists():
        return False
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
    return True


def build_v11_cell_record(*, run_identifier: str, dataset_id: str, seed: int, model: str, method: str, m: int, table: Any, split: Any, cache: dict[str, Any], context: dict[str, Any], minority_label: int) -> dict[str, Any]:
    subsets = select_nested_calibration_subsets(cache["ids"]["calibration_pool"], cache["labels"]["calibration_pool"], protocol_version="v1.1", dataset_id=dataset_id, base_seed=seed, minority_label=minority_label)
    cell = evaluate_split_cp(cache["ids"]["calibration_pool"], cache["labels"]["calibration_pool"], cache["probabilities"]["calibration_pool"], cache["labels"]["test"], cache["probabilities"]["test"], subsets, m_minority=m, cp_method=method)
    provenance = context["provenance"]
    return {
        "dataset_id": dataset_id, "seed": seed, "model": model, "cp_method": method, "m_minority": m, "m_majority": cell["m_majority"], "alpha": 0.1,
        "protocol_version": "v1.1", "config_hash": provenance["config_hash"], "code_hash": provenance["code_hash"], "run_id": run_identifier,
        "artifact_id": f"{run_identifier}_{dataset_id}_{seed}_{model}_{method}_m{m}", "results_schema_version": RESULTS_SCHEMA_VERSION, "status": "PASS", "minority_label": minority_label,
        **cell, "auroc": context["base_metrics"]["auroc"], "auprc": context["base_metrics"]["auprc"],
        "split_hash": split.split_hash, "dataset_hash": table.raw_sha256, "environment_hash": provenance["environment_hash"], "model_hash": cache["model_hash"],
        "prediction_cache_hash": cache["manifest"]["cache_sha256"], "label_mapping_hash": _hash_json(table.label_mapping), "created_utc": utc_now(),
    }


def independent_qc_v11(root: Path, records: list[dict[str, Any]], decision: dict[str, Any], lineage: dict[str, Any]) -> dict[str, Any]:
    """Independent v1.1 QC: schema, subset identity, ranks, nesting, and full recomputation."""
    import numpy as np

    errors = validate_results_records(records)
    dataset_ids, seeds = list(decision["pilot_dataset_ids"]), list(FROZEN_SEEDS)
    expected = len(dataset_ids) * len(seeds) * len(MODELS) * len(CP_METHODS) * len(M_MINORITY)
    pair_groups: dict[tuple[str, int, str, int], list[dict[str, Any]]] = {}
    for record in records:
        pair_groups.setdefault((record["dataset_id"], record["seed"], record["model"], record["m_minority"]), []).append(record)
    subset_identity = all(len(group) == 2 and len({row["subset_hash"] for row in group}) == 1 for group in pair_groups.values())
    rank_m10 = all(row["rank_minority"] == 10 for row in records if row["cp_method"] == "class_conditional_cp" and row["m_minority"] == 10)
    rank_m20 = all(row["rank_minority"] == 19 for row in records if row["cp_method"] == "class_conditional_cp" and row["m_minority"] == 20)
    registry_path = root / decision["registry_path"]
    nested_subset, recomputation = True, True
    for dataset_id in dataset_ids:
        minority = _minority_label(root, dataset_id, registry_path)
        for seed in seeds:
            for model in MODELS:
                matching = [row for row in records if row["dataset_id"] == dataset_id and row["seed"] == seed and row["model"] == model]
                if len(matching) != 8:
                    recomputation = False
                    continue
                table, split, cache, _context = load_v11_cache(root, lineage, dataset_id, seed, model, registry_path)
                subsets = select_nested_calibration_subsets(cache["ids"]["calibration_pool"], cache["labels"]["calibration_pool"], protocol_version="v1.1", dataset_id=dataset_id, base_seed=seed, minority_label=minority)
                nested_subset = nested_subset and all(set(subsets.minority_ids_by_m[a]).issubset(subsets.minority_ids_by_m[b]) for a, b in zip(M_MINORITY, M_MINORITY[1:]))
                for row in matching:
                    direct = evaluate_split_cp(cache["ids"]["calibration_pool"], cache["labels"]["calibration_pool"], cache["probabilities"]["calibration_pool"], cache["labels"]["test"], cache["probabilities"]["test"], subsets, m_minority=row["m_minority"], cp_method=row["cp_method"])
                    for field in ("subset_hash", "rank_global", "rank_minority", "rank_majority", "covered_count_overall", "covered_count_minority", "covered_count_majority"):
                        recomputation = recomputation and direct[field] == row[field]
                    for field in ("q_global", "q_minority", "q_majority", "threshold_gap", "threshold_sum", "coverage_overall", "coverage_minority", "coverage_majority", "singleton_rate", "empty_rate", "doubleton_rate", "average_set_size"):
                        a, b = direct[field], row[field]
                        recomputation = recomputation and ((a is None and b is None) or (a is not None and b is not None and abs(float(a) - float(b)) <= 1e-12))
    global_rows = [row for row in records if row["cp_method"] == "global_split_cp"]
    systematic: list[dict[str, Any]] = []
    for (model, m), rows in pd.DataFrame(global_rows).groupby(["model", "m_minority"], sort=True):
        flags = [not (row["coverage_overall_wilson_low"] <= 0.9 <= row["coverage_overall_wilson_high"]) for _, row in rows.iterrows()]
        systematic.append({"model": model, "m_minority": int(m), "flagged_cells": int(sum(flags)), "total_cells": int(len(flags)), "flagged_fraction": float(np.mean(flags)), "implementation_review_required": bool(np.mean(flags) > 0.5)})
    return {
        "status": "PASS" if not errors and len(records) == expected and subset_identity and rank_m10 and rank_m20 and nested_subset and recomputation else "FAIL",
        "protocol_version": "v1.1",
        "expected_cells": expected, "validated_cells": len(records),
        "schema_errors": errors,
        "checks": {
            "m10_max_score_rank": rank_m10, "m20_19th_order_statistic_rank": rank_m20,
            "nested_subset": nested_subset, "cp_subset_identity": subset_identity,
            "q_threshold_sum_and_geometry": recomputation,
            "classwise_coverage_and_wilson": not errors,
            "probability_mapping_from_validated_caches": True,
        },
        "global_coverage_diagnostic": {
            "nominal": 0.9, "groups": systematic,
            "implementation_review_required": any(item["implementation_review_required"] for item in systematic),
            "interpretation_rule": "A flag requires implementation review first; it is not a scientific finding.",
        },
    }


def _run_identity(decision: dict[str, Any]) -> str:
    payload = json.dumps({"datasets": decision["pilot_dataset_ids"], "seeds": list(FROZEN_SEEDS), "models": MODELS, "cp": CP_METHODS, "m": M_MINORITY}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the fixed 480-cell v1.1 CP pilot from the audited 240 caches.")
    parser.add_argument("--decision", type=Path, default=CANONICAL_DECISION_PATH)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    decision = load_and_validate_v11_pilot_decision(ROOT, args.decision)
    lineage = resolve_v11_cache_lineage(ROOT)
    dataset_ids = tuple(decision["pilot_dataset_ids"])
    seeds = tuple(int(seed) for seed in FROZEN_SEEDS)
    expected = expected_pilot_cells_v11(dataset_ids, seeds)
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_stage07-v11-pilot_{_run_identity(decision)}"
    run_root = args.run_dir.resolve() if args.run_dir else create_immutable_run_dir(PILOT_ROOT, run_id)
    if args.run_dir:
        if not run_root.is_dir() or run_root.parent != PILOT_ROOT.resolve():
            raise ValueError("--run-dir must be an existing direct child of artifacts/stage07_v1.1.")
    run_root.mkdir(parents=True, exist_ok=True)
    event_path, cell_root = run_root / "events.jsonl", run_root / "cells"
    cell_root.mkdir(exist_ok=True)
    manifest_path = run_root / "run_manifest.json"
    manifest = {
        "artifact_id": f"{run_root.name}_manifest", "stage": "Stage 08 / Task 7",
        "scope": "V11_FIXED_TWO_DATASET_PILOT_ONLY_D08_003",
        "protocol_version": "v1.1",
        "pilot_dataset_ids": list(dataset_ids), "seeds": list(seeds),
        "models": list(MODELS), "cp_methods": list(CP_METHODS), "m_minority": list(M_MINORITY),
        "expected_cells": len(expected),
        "pilot_decision_path": str(CANONICAL_DECISION_PATH).replace("\\", "/"),
        "pilot_decision_sha256": sha256_path(ROOT / CANONICAL_DECISION_PATH),
        "intake_audit_path": lineage["intake_audit_path"],
        "lineage": {key: value for key, value in lineage.items() if key != "cache_root"},
        "created_utc": utc_now(),
        "formal_run_manifest_authorized": False, "full_experiment_authorized": False,
    }
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("pilot_dataset_ids", "seeds", "models", "cp_methods", "m_minority", "expected_cells", "pilot_decision_sha256"):
            if old.get(key) != manifest[key]:
                raise ValueError("Existing v1.1 pilot run directory has a different controlled manifest.")
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    write_event(event_path, run_id=run_root.name, stage="Stage 08 / Task 7", level="INFO", event="v11_pilot_started", config_hash=lineage["local_config_hash"], message="Hash-gated v1.1 pilot started or resumed; CP uses only the independently audited v1.1 caches.", expected_cells=len(expected))
    registry_path = ROOT / decision["registry_path"]
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
                table, split, cache, context = load_v11_cache(ROOT, lineage, dataset_id, seed, model, registry_path)
                cache_context[key] = (table, split, cache, context, _minority_label(ROOT, dataset_id, registry_path))
            table, split, cache, context, minority_label = cache_context[key]
            record = build_v11_cell_record(run_identifier=run_root.name, dataset_id=dataset_id, seed=seed, model=model, method=method, m=m, table=table, split=split, cache=cache, context=context, minority_label=minority_label)
            errors = validate_results_records([record])
            if errors:
                raise ValueError(errors)
            if not write_cell_once(destination, record):
                raise FileExistsError(f"Cell already exists and cannot be overwritten: {destination}")
            write_event(event_path, run_id=run_root.name, stage="Stage 08 / Task 7", level="INFO", event="cell_complete", config_hash=record["config_hash"], message="v1.1 CP cell verified and written once.", dataset_id=dataset_id, seed=seed, model=model, cp_method=method, m_minority=m)
        except Exception as exc:  # noqa: BLE001 - every cell failure is preserved immutably
            failure_path = ROOT / "artifacts" / "failures" / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_stage07_v11_{uuid.uuid4().hex[:10]}.json"
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure = {"dataset_id": dataset_id, "seed": seed, "model": model, "cp_method": method, "m_minority": m, "exception_type": type(exc).__name__, "exception": str(exc), "retry_count": 0, "failure_record": str(failure_path.relative_to(ROOT))}
            failure_path.write_text(json.dumps({"stage": "Stage 08 / Task 7", "run_id": run_root.name, "created_utc": utc_now(), "classification": "implementation_or_input_failure_pending_triage", **failure}, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            failures.append(failure)
            write_event(event_path, run_id=run_root.name, stage="Stage 08 / Task 7", level="ERROR", event="cell_failed", config_hash=lineage["local_config_hash"], message=str(exc), **failure)
    records = [json.loads((cell_root / _cell_name(cell)).read_text(encoding="utf-8")) for cell in expected if (cell_root / _cell_name(cell)).exists()]
    batch_errors = validate_results_records(records)
    status = "PASS" if len(records) == len(expected) and not failures and not batch_errors else "FAIL"
    status_payload = {
        "artifact_id": f"{run_root.name}_status", "stage": "Stage 08 / Task 7", "protocol_version": "v1.1",
        "status": status, "expected_cells": len(expected), "verified_cells": len(records),
        "missing_cells": len(expected) - len(records), "failures": failures,
        "batch_validation_errors": batch_errors, "updated_utc": utc_now(),
        "resume_command": f"{sys.executable} src/run_stage07_pilot_v1_1.py --decision {CANONICAL_DECISION_PATH} --run-dir {run_root}",
        "formal_run_manifest_created": False, "full_experiment_executed": False,
    }
    (run_root / "run_status.json").write_text(json.dumps(status_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if status != "PASS":
        write_event(event_path, run_id=run_root.name, stage="Stage 08 / Task 7", level="ERROR", event="v11_pilot_incomplete", config_hash=lineage["local_config_hash"], message="v1.1 pilot stopped with preserved cells; inspect run_status and resume only after triage.", completed=len(records), failures=len(failures))
        print(json.dumps({"status": status, "run_dir": str(run_root), "completed": len(records), "failures": len(failures)}))
        return 2
    results_path = run_root / "results_long.parquet"
    if results_path.exists():
        if len(pd.read_parquet(results_path)) != len(expected):
            raise ValueError("Existing results_long is inconsistent; it will not be overwritten.")
    else:
        pd.DataFrame(records).to_parquet(results_path, index=False)
        pd.DataFrame(records).to_csv(run_root / "results_long.csv", index=False)
    qc = independent_qc_v11(ROOT, records, decision, lineage)
    figure_map = diagnostic_figures(records, run_root / "figures")
    qc["figures"] = figure_map
    qc["lineage"] = {key: value for key, value in lineage.items() if key != "cache_root"}
    (run_root / "pilot_qc.json").write_text(json.dumps(qc, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if qc["status"] != "PASS":
        raise ValueError("Independent v1.1 pilot QC failed; preserved results require triage before use.")
    write_event(event_path, run_id=run_root.name, stage="Stage 08 / Task 7", level="WARN" if qc["global_coverage_diagnostic"]["implementation_review_required"] else "INFO", event="v11_pilot_cells_complete", config_hash=lineage["local_config_hash"], message="All 480 v1.1 CP cells and independent QC completed; coverage flags are diagnostic only.", completed=len(records), global_implementation_review_required=qc["global_coverage_diagnostic"]["implementation_review_required"])
    print(json.dumps({"status": "PASS", "run_dir": str(run_root), "results_long": str(results_path), "completed": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

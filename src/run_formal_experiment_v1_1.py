"""Execute the frozen 1,920-cell formal experiment from the audited v1.1 caches.

This runner executes ONLY the D08-004-authorized formal scope (8 locked
datasets x 10 frozen seeds x 3 models x 2 CP methods x 4 minority calibration
sizes = 1,920 cells) against the frozen formal run manifest
``configs/formal_run_manifest_v1.1.yaml``.  Every probability comes from the
240 existing, independently intake-audited v1.1 caches; no model is fitted,
no cache is regenerated, no cloud command is run, and no v1.0 artifact is
read.  Cells are written once with exclusive creation, failures are preserved
immutably, and the run finishes with a full-recomputation QC over all 240
cache units.  No scientific interpretation is performed here.
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

MODELS = ("logistic_regression", "xgboost", "tabpfn")
RESULTS_SCHEMA_VERSION = "v1.1.0"
EXPECTED_FORMAL_CELLS = 1920
CANONICAL_RECEIPT_PATH = Path("decisions/D08-004_FORMAL_RUN_GO_RECEIPT.json")
CANONICAL_MANIFEST_PATH = Path("configs/formal_run_manifest_v1.1.yaml")
CANONICAL_REGISTRY_PATH = Path("artifacts/stage02/dataset_registry_v1.0.1.json")
INTAKE_GLOB = "cache_intake_*/intake_audit.json"
RUNS_ROOT = ROOT / "artifacts" / "runs"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sha256_bytes(*chunks: bytes) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


class FormalRunGateError(ValueError):
    """Raised when a formal-run execution gate is not satisfied."""


def load_validated_go_receipt(root: Path, receipt_path: Path) -> dict[str, Any]:
    """Accept only the canonical D08-004 user-go receipt with live bindings."""
    canonical = (root / CANONICAL_RECEIPT_PATH).resolve()
    actual = (receipt_path if receipt_path.is_absolute() else root / receipt_path).resolve()
    if actual != canonical:
        raise FormalRunGateError("Only the canonical D08-004 go receipt may authorize the formal run.")
    receipt = json.loads(actual.read_text(encoding="utf-8"))
    if receipt.get("artifact_id") != "D08-004_stage09_formal_run_go":
        raise FormalRunGateError("The go receipt is not the D08-004 formal-run authorization.")
    if receipt.get("status") != "APPROVED_FOR_FORMAL_MANIFEST_FREEZE_AND_1920_CELL_EXPERIMENT":
        raise FormalRunGateError("The go receipt status is not the approved formal-run scope.")
    if receipt.get("explicit_go") != "go" or receipt.get("approver") != "user":
        raise FormalRunGateError("The go receipt does not record the user's explicit go.")
    if receipt.get("protocol_version") != "v1.1" or receipt.get("expected_formal_cells") != EXPECTED_FORMAL_CELLS:
        raise FormalRunGateError("The go receipt does not bound the v1.1 1,920-cell formal scope.")
    if receipt.get("formal_run_manifest_authorized") is not True or receipt.get("full_experiment_authorized") is not True:
        raise FormalRunGateError("The go receipt does not authorize the manifest freeze and full experiment.")
    basis = receipt.get("authorization_basis", {})
    audit_path = root / str(basis.get("stage08_v11_independent_audit_path", "")).replace("\\", "/")
    if not audit_path.is_file() or basis.get("stage08_v11_independent_audit_sha256") != sha256_path(audit_path):
        raise FormalRunGateError("The go receipt's Stage 08 v1.1 audit binding does not match the current audit file.")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("verdict") != "PASS" or audit.get("gate_recommendation", {}).get("recommendation") != "CONDITIONAL-GO":
        raise FormalRunGateError("The bound Stage 08 v1.1 audit is not a PASS with a CONDITIONAL-GO recommendation.")
    if receipt.get("formal_run_manifest", {}).get("path") != str(CANONICAL_MANIFEST_PATH).replace("\\", "/"):
        raise FormalRunGateError("The go receipt does not name the canonical frozen formal manifest path.")
    return receipt


def load_validated_formal_manifest(root: Path, manifest_path: Path) -> dict[str, Any]:
    """Accept only the canonical frozen manifest whose bindings all hold now."""
    import yaml

    canonical = (root / CANONICAL_MANIFEST_PATH).resolve()
    actual = (manifest_path if manifest_path.is_absolute() else root / manifest_path).resolve()
    if actual != canonical:
        raise FormalRunGateError("Only the canonical frozen formal run manifest may drive the formal experiment.")
    manifest = yaml.safe_load(actual.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("artifact_status") != "FROZEN_FORMAL_RUN_MANIFEST":
        raise FormalRunGateError("The formal manifest is not the frozen v1.1 record.")
    if manifest.get("protocol_version") != "v1.1" or manifest.get("grid", {}).get("expected_cells") != EXPECTED_FORMAL_CELLS:
        raise FormalRunGateError("The formal manifest does not bound the v1.1 1,920-cell grid.")
    if manifest.get("registry", {}).get("locked_primary_ids") != FROZEN_DATASETS or manifest.get("seeds") != FROZEN_SEEDS:
        raise FormalRunGateError("The formal manifest registry or seeds differ from the frozen protocol.")
    authorization = manifest.get("authorization", {})
    if authorization.get("d08_004_receipt_sha256") != sha256_path(root / CANONICAL_RECEIPT_PATH):
        raise FormalRunGateError("The formal manifest is not bound to the current D08-004 go receipt.")
    input_hashes = manifest.get("input_hashes", {})
    binding_files = {
        "protocol_sha256": "protocols/protocol_v1.1.md",
        "dataset_lock_sha256": "protocols/dataset_lock_v1.0.md",
        "split_lock_sha256": "configs/stage04_splits_v1.1.yaml",
        "local_cache_lock_sha256": "configs/stage05_lr_xgboost_v1.1.yaml",
        "tabpfn_cache_lock_sha256": "configs/stage05b_tabpfn_v1.1.yaml",
        "pilot_decision_sha256": "decisions/pilot_decision_stage07_v1.1.json",
        "environment_sha256": "environment/environment_lock_v1.0.json",
        "results_schema_sha256": "configs/results_long.schema.json",
        "stage08_v11_audit_sha256": str((root / "artifacts/stage08_v11/20260831T085012Z_pilot_independent_audit/independent_audit.json").relative_to(root)).replace("\\", "/"),
    }
    for key, relative in binding_files.items():
        path = root / relative
        if not path.is_file() or input_hashes.get(key) != sha256_path(path):
            raise FormalRunGateError(f"Formal manifest binding {key} does not match the current controlled input.")
    gate = manifest.get("execution_gate", {})
    if gate.get("d08_004_go_receipt_required_before_execution") is not True:
        raise FormalRunGateError("The formal manifest must require the D08-004 go receipt before execution.")
    for key in ("model_fitting_allowed", "cloud_execution_allowed", "cache_regeneration_allowed", "protocol_change_allowed"):
        if gate.get(key) is not False:
            raise FormalRunGateError(f"The formal manifest must keep {key} false.")
    return manifest


def require_intake_pass(root: Path) -> dict[str, Any]:
    """The formal run consumes only a PASS 240-cache intake audit."""
    candidates = sorted((root / "artifacts" / "stage08_v11_cloud").glob(INTAKE_GLOB))
    if not candidates:
        raise FormalRunGateError("No 240-cache intake audit exists; the formal run is not authorized.")
    audit = json.loads(candidates[-1].read_text(encoding="utf-8"))
    if audit.get("verdict") != "PASS" or audit.get("valid_units") != 240 or audit.get("model_counts") != {"logistic_regression": 80, "xgboost": 80, "tabpfn": 80}:
        raise FormalRunGateError("The latest intake audit is not a PASS over exactly 240 units.")
    if audit.get("installed_tabpfn_units") != 80 or audit.get("errors") or audit.get("unit_errors"):
        raise FormalRunGateError("The intake audit reports unresolved errors or incomplete installation.")
    return audit


def formal_config_hash(root: Path) -> str:
    """Deterministic formal-run config hash over the frozen manifest and go receipt."""
    return _sha256_bytes((root / CANONICAL_MANIFEST_PATH).read_bytes(), (root / CANONICAL_RECEIPT_PATH).read_bytes())


def resolve_formal_lineage(root: Path) -> dict[str, Any]:
    """Resolve the two cache-family lineages and verify them against current locks."""
    manifest = load_validated_formal_manifest(root, CANONICAL_MANIFEST_PATH)
    intake = require_intake_pass(root)
    intake_lineage = intake.get("lineage", {})
    manifest_lineage = manifest.get("cache_lineage", {})
    if manifest_lineage.get("intake_audit_sha256") != sha256_path(root / str(manifest_lineage.get("intake_audit_path", "")).replace("\\", "/")):
        raise FormalRunGateError("The frozen manifest's intake-audit binding does not match the current intake audit.")
    split_bytes = (root / "configs/stage04_splits_v1.1.yaml").read_bytes()
    local_bytes = (root / "configs/stage05_lr_xgboost_v1.1.yaml").read_bytes()
    tabpfn_bytes = (root / "configs/stage05b_tabpfn_v1.1.yaml").read_bytes()
    local_config_hash = _sha256_bytes(bytes.fromhex(_sha256_bytes(split_bytes, local_bytes)), bytes.fromhex(_sha256_bytes(tabpfn_bytes)))
    tabpfn_config_hash = _sha256_bytes(bytes.fromhex(_sha256_bytes(split_bytes, tabpfn_bytes)), bytes.fromhex(_sha256_bytes(tabpfn_bytes)))
    lineage = {
        "local_config_hash": local_config_hash,
        "local_code_hash": manifest_lineage.get("local_cache_time_code_hash"),
        "local_lock_sha256": sha256_path(root / "configs/stage05_lr_xgboost_v1.1.yaml"),
        "tabpfn_config_hash": tabpfn_config_hash,
        "tabpfn_code_hash": manifest_lineage.get("tabpfn_cache_time_code_hash"),
        "tabpfn_lock_sha256": sha256_path(root / "configs/stage05b_tabpfn_v1.1.yaml"),
        "split_lock_sha256": sha256_path(root / "configs/stage04_splits_v1.1.yaml"),
        "d08_003_cache_lock_sha256": sha256_path(root / "configs/stage05b_tabpfn_v1.1.yaml"),
        "environment_hash": manifest_lineage.get("environment_hash"),
        "cache_root": root / "artifacts" / "caches" / "v1.1",
        "intake_audit_path": str(manifest_lineage.get("intake_audit_path", "")).replace("\\", "/"),
    }
    if intake_lineage.get("local_config_hash") != local_config_hash or intake_lineage.get("tabpfn_config_hash") != tabpfn_config_hash:
        raise FormalRunGateError("A v1.1 lock changed after the intake audit; the audited caches are stale for the formal run.")
    for key, manifest_key in (("local_code_hash", "local_cache_time_code_hash"), ("tabpfn_code_hash", "tabpfn_cache_time_code_hash"), ("environment_hash", "environment_hash")):
        if lineage[key] != manifest_lineage.get(manifest_key) or not isinstance(lineage[key], str) or len(lineage[key]) != 64:
            raise FormalRunGateError(f"The frozen manifest lineage field {manifest_key} is invalid or drifted.")
    return lineage


def expected_formal_cells() -> tuple[tuple[str, int, str, str, int], ...]:
    """Exactly 8 datasets x 10 seeds x 3 models x 2 methods x 4 m = 1,920 cells."""
    return tuple(
        (dataset_id, int(seed), model, method, m)
        for dataset_id in FROZEN_DATASETS
        for seed in FROZEN_SEEDS
        for model in MODELS
        for method in CP_METHODS
        for m in M_MINORITY
    )


def _cell_name(cell: tuple[str, int, str, str, int]) -> str:
    dataset_id, seed, model, method, m = cell
    return f"{dataset_id}__seed-{seed}__{model}__{method}__m-{m}.json"


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
    """Load and revalidate one immutable v1.1 cache against the frozen lineage."""
    table = load_locked_dataset(root, dataset_id, registry_path=registry_path)
    split = make_stratified_split(table, seed, protocol_version="v1.1")
    split_manifest = json.loads((root / "artifacts" / "splits" / "v1.1" / dataset_id / f"seed-{seed}.json").read_text(encoding="utf-8"))
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


def build_formal_cell_record(*, run_identifier: str, formal_manifest_sha256: str, dataset_id: str, seed: int, model: str, method: str, m: int, table: Any, split: Any, cache: dict[str, Any], context: dict[str, Any], minority_label: int) -> dict[str, Any]:
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
        "formal_run_manifest_sha256": formal_manifest_sha256,
    }


def formal_full_recomputation_qc(root: Path, records: list[dict[str, Any]], lineage: dict[str, Any], registry_path: Path) -> dict[str, Any]:
    """Full independent recomputation of every cell from its immutable cache."""
    import numpy as np

    errors = validate_results_records(records)
    expected = len(FROZEN_DATASETS) * len(FROZEN_SEEDS) * len(MODELS) * len(CP_METHODS) * len(M_MINORITY)
    pair_groups: dict[tuple[str, int, str, int], list[dict[str, Any]]] = {}
    for record in records:
        pair_groups.setdefault((record["dataset_id"], record["seed"], record["model"], record["m_minority"]), []).append(record)
    subset_identity = all(len(group) == 2 and len({row["subset_hash"] for row in group}) == 1 for group in pair_groups.values())
    rank_m10 = all(row["rank_minority"] == 10 for row in records if row["cp_method"] == "class_conditional_cp" and row["m_minority"] == 10)
    rank_m20 = all(row["rank_minority"] == 19 for row in records if row["cp_method"] == "class_conditional_cp" and row["m_minority"] == 20)
    nested_subset, recomputation = True, True
    cache_tables: dict[str, Any] = {}
    for dataset_id in FROZEN_DATASETS:
        minority = _minority_label(root, dataset_id, registry_path)
        for seed in FROZEN_SEEDS:
            for model in MODELS:
                matching = [row for row in records if row["dataset_id"] == dataset_id and row["seed"] == seed and row["model"] == model]
                if len(matching) != 8:
                    recomputation = False
                    continue
                table, split, cache, _context = load_v11_cache(root, lineage, dataset_id, seed, model, registry_path)
                cache_tables[dataset_id] = table
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
        "scope": "FULL_RECOMPUTATION_OF_ALL_1920_CELLS_FROM_IMMUTABLE_CACHES",
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
            "interpretation_rule": "A flag requires implementation review first; it is diagnostic-only and not a scientific finding.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute the frozen 1,920-cell formal experiment from the audited v1.1 caches.")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    receipt = load_validated_go_receipt(ROOT, CANONICAL_RECEIPT_PATH)
    manifest = load_validated_formal_manifest(ROOT, CANONICAL_MANIFEST_PATH)
    lineage = resolve_formal_lineage(ROOT)
    registry_path = ROOT / CANONICAL_REGISTRY_PATH
    formal_manifest_sha256 = sha256_path(ROOT / CANONICAL_MANIFEST_PATH)
    receipt_sha256 = sha256_path(ROOT / CANONICAL_RECEIPT_PATH)
    config_hash = formal_config_hash(ROOT)
    expected = expected_formal_cells()
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_stage09-formal_{config_hash[:12]}"
    run_root = args.run_dir.resolve() if args.run_dir else create_immutable_run_dir(RUNS_ROOT, run_id)
    if args.run_dir:
        if not run_root.is_dir() or run_root.parent != RUNS_ROOT.resolve():
            raise ValueError("--run-dir must be an existing direct child of artifacts/runs.")
    run_root.mkdir(parents=True, exist_ok=True)
    event_path, cell_root = run_root / "events.jsonl", run_root / "cells"
    cell_root.mkdir(exist_ok=True)
    run_manifest_path = run_root / "run_manifest.json"
    run_manifest = {
        "artifact_id": f"{run_root.name}_manifest", "stage": "Stage 09 formal",
        "scope": "FROZEN_1920_CELL_FORMAL_EXPERIMENT_D08_004",
        "protocol_version": "v1.1",
        "datasets": list(FROZEN_DATASETS), "seeds": list(FROZEN_SEEDS),
        "models": list(MODELS), "cp_methods": list(CP_METHODS), "m_minority": list(M_MINORITY),
        "expected_cells": len(expected),
        "formal_run_manifest_path": str(CANONICAL_MANIFEST_PATH).replace("\\", "/"),
        "formal_run_manifest_sha256": formal_manifest_sha256,
        "d08_004_receipt_path": str(CANONICAL_RECEIPT_PATH).replace("\\", "/"),
        "d08_004_receipt_sha256": receipt_sha256,
        "formal_config_hash": config_hash,
        "intake_audit_path": lineage["intake_audit_path"],
        "lineage": {key: value for key, value in lineage.items() if key != "cache_root"},
        "created_utc": utc_now(),
        "model_fitting_performed": False, "cloud_execution_performed": False,
        "scientific_interpretation_performed": False,
    }
    if run_manifest_path.exists():
        old = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        for key in ("datasets", "seeds", "models", "cp_methods", "m_minority", "expected_cells", "formal_run_manifest_sha256", "d08_004_receipt_sha256"):
            if old.get(key) != run_manifest[key]:
                raise ValueError("Existing formal run directory has a different controlled manifest.")
    else:
        run_manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    write_event(event_path, run_id=run_root.name, stage="Stage 09 formal", level="INFO", event="formal_run_started", config_hash=config_hash, message="Hash-gated formal experiment started or resumed; CP uses only the 240 independently audited v1.1 caches.", expected_cells=len(expected))
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
            record = build_formal_cell_record(run_identifier=run_root.name, formal_manifest_sha256=formal_manifest_sha256, dataset_id=dataset_id, seed=seed, model=model, method=method, m=m, table=table, split=split, cache=cache, context=context, minority_label=minority_label)
            cell_errors = validate_results_records([record])
            if cell_errors:
                raise ValueError(cell_errors)
            if not write_cell_once(destination, record):
                raise FileExistsError(f"Cell already exists and cannot be overwritten: {destination}")
            write_event(event_path, run_id=run_root.name, stage="Stage 09 formal", level="INFO", event="cell_complete", config_hash=config_hash, message="Formal CP cell verified and written once.", dataset_id=dataset_id, seed=seed, model=model, cp_method=method, m_minority=m)
        except Exception as exc:  # noqa: BLE001 - every cell failure is preserved immutably
            failure_path = ROOT / "artifacts" / "failures" / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_stage09_formal_{uuid.uuid4().hex[:10]}.json"
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure = {"dataset_id": dataset_id, "seed": seed, "model": model, "cp_method": method, "m_minority": m, "exception_type": type(exc).__name__, "exception": str(exc), "retry_count": 0, "failure_record": str(failure_path.relative_to(ROOT))}
            failure_path.write_text(json.dumps({"stage": "Stage 09 formal", "run_id": run_root.name, "created_utc": utc_now(), "classification": "implementation_or_input_failure_pending_triage", **failure}, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            failures.append(failure)
            write_event(event_path, run_id=run_root.name, stage="Stage 09 formal", level="ERROR", event="cell_failed", config_hash=config_hash, message=str(exc), **failure)
    records = [json.loads((cell_root / _cell_name(cell)).read_text(encoding="utf-8")) for cell in expected if (cell_root / _cell_name(cell)).exists()]
    batch_errors = validate_results_records(records)
    status = "PASS" if len(records) == len(expected) and not failures and not batch_errors else "FAIL"
    status_payload = {
        "artifact_id": f"{run_root.name}_status", "stage": "Stage 09 formal", "protocol_version": "v1.1",
        "status": status, "expected_cells": len(expected), "verified_cells": len(records),
        "missing_cells": len(expected) - len(records), "failures": failures,
        "batch_validation_errors": batch_errors, "updated_utc": utc_now(),
        "resume_command": f"{sys.executable} src/run_formal_experiment_v1_1.py --run-dir {run_root}",
        "model_fitting_performed": False, "cloud_execution_performed": False,
        "scientific_interpretation_performed": False,
    }
    (run_root / "run_status.json").write_text(json.dumps(status_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if status != "PASS":
        write_event(event_path, run_id=run_root.name, stage="Stage 09 formal", level="ERROR", event="formal_run_incomplete", config_hash=config_hash, message="Formal run stopped with preserved cells; inspect run_status and resume only after triage.", completed=len(records), failures=len(failures))
        print(json.dumps({"status": status, "run_dir": str(run_root), "completed": len(records), "failures": len(failures)}))
        return 2
    results_path = run_root / "results_long.parquet"
    if results_path.exists():
        if len(pd.read_parquet(results_path)) != len(expected):
            raise ValueError("Existing results_long is inconsistent; it will not be overwritten.")
    else:
        pd.DataFrame(records).to_parquet(results_path, index=False)
        pd.DataFrame(records).to_csv(run_root / "results_long.csv", index=False)
    qc = formal_full_recomputation_qc(ROOT, records, lineage, registry_path)
    figure_map = diagnostic_figures(records, run_root / "figures")
    qc["figures"] = figure_map
    qc["lineage"] = {key: value for key, value in lineage.items() if key != "cache_root"}
    (run_root / "qc.json").write_text(json.dumps(qc, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if qc["status"] != "PASS":
        raise ValueError("Formal full-recomputation QC failed; preserved results require triage before use.")
    write_event(event_path, run_id=run_root.name, stage="Stage 09 formal", level="WARN" if qc["global_coverage_diagnostic"]["implementation_review_required"] else "INFO", event="formal_run_complete", config_hash=config_hash, message="All 1,920 formal cells and full-recomputation QC completed; coverage flags are diagnostic only.", completed=len(records), global_implementation_review_required=qc["global_coverage_diagnostic"]["implementation_review_required"])
    print(json.dumps({"status": "PASS", "run_dir": str(run_root), "results_long": str(results_path), "completed": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Materialize Stage 04 raw-data, split, and train-only preprocessing evidence.

This script does not fit a predictive model or produce conformal results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conformal_uq.data import audit_table, load_dataset_registry, load_locked_dataset, locked_primary_records
from conformal_uq.config import FROZEN_DATASETS, FROZEN_SEEDS
from conformal_uq.preprocessing import TrainOnlyPreprocessor
from conformal_uq.split import make_stratified_split, write_split_manifest
from conformal_uq.stage08_authorization import load_d08_003_authorization


V11_STAGE04_LOCK_PROFILES = {
    "stage04_splits_v1.1.yaml": {
        "stage04_evidence_root": "artifacts/stage08_v11/stage04_split_preparation",
        "output_identifiers": {
            "artifact_id": "stage04_data_preparation_v1.1",
            "version": "v1.1.0",
            "summary_filename": "stage04_summary_v1.1.json",
        },
    },
    "stage04_splits_v1.1.1.yaml": {
        "stage04_evidence_root": "artifacts/stage08_v11/stage04_split_preparation_v1.1.1",
        "output_identifiers": {
            "artifact_id": "stage04_data_preparation_v1.1.1_byte_hash_repair",
            "version": "v1.1.1",
            "summary_filename": "stage04_summary_v1.1.1.json",
        },
    },
}


def _canonical(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _write_immutable_json(path: Path, payload: dict[str, Any], *, byte_stable_lf: bool = False) -> str:
    encoded = _canonical(payload)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"Controlled Stage 04 artifact differs and cannot be overwritten: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        if byte_stable_lf:
            path.write_bytes(encoded.encode("utf-8"))
        else:
            path.write_text(encoded, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest() if byte_stable_lf else hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_split_lock_mapping(path: Path) -> dict[str, Any]:
    """Read the explicit split lock; isolated for lock-level test injection."""
    lock = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(lock, dict):
        raise ValueError("Stage 08 split lock must be a YAML mapping.")
    return lock


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
            raise ValueError(f"The v1.1 split lock must keep {key} false; D08-003 is a separate receipt gate.")


def load_split_lock(root: Path, lock_path: Path) -> dict[str, Any]:
    """Read and validate a v1.1 split lock without loading data or writing output."""
    path = lock_path if lock_path.is_absolute() else root / lock_path
    lock = _read_split_lock_mapping(path)
    profile = V11_STAGE04_LOCK_PROFILES.get(path.name)
    if profile is None:
        raise ValueError("Stage 08 v1.1 split-lock filename is not approved.")
    if lock.get("protocol_version") != "v1.1":
        raise ValueError("An explicit Stage 08 split lock must use protocol v1.1.")
    protocol = lock.get("protocol", {})
    if protocol.get("protocol_version_for_seed_derivation") != lock["protocol_version"]:
        raise ValueError("Split lock protocol and seed-derivation protocol must match.")
    if protocol.get("seed_derivation_algorithm") != "sha256_first_32_bits_unsigned_big_endian":
        raise ValueError("Split lock seed derivation algorithm is not frozen.")
    if lock.get("registry", {}).get("locked_primary_ids") != FROZEN_DATASETS:
        raise ValueError("Split lock registry IDs differ from the frozen eight-dataset registry.")
    if lock.get("seeds") != FROZEN_SEEDS:
        raise ValueError("Split lock seeds differ from the frozen ten-seed protocol.")
    if lock.get("split_fractions") != {"train": 0.6, "calibration_pool": 0.2, "test": 0.2}:
        raise ValueError("Split lock fractions differ from the frozen 60/20/20 protocol.")
    if lock.get("paths", {}).get("split_root") != "artifacts/splits/v1.1":
        raise ValueError("Stage 08 v1.1 splits must use the isolated artifacts/splits/v1.1 root.")
    identifiers = lock.get("output_identifiers", {})
    if lock.get("paths", {}).get("stage04_evidence_root") != profile["stage04_evidence_root"] or identifiers != profile["output_identifiers"]:
        raise ValueError("Stage 08 v1.1 split-lock filename, evidence root, and output identifiers must match one exact approved profile.")
    gate, output = lock.get("execution_gate", {}), lock.get("output_contract", {})
    _require_non_authorizing_v11_gate(gate)
    if any(output.get(key) is not False for key in ("conformal_prediction_allowed", "pilot_allowed", "formal_outputs_allowed")):
        raise ValueError("The split preparation lock must prohibit CP, pilot, and formal outputs.")
    return lock


def split_manifest_path(root: Path, lock: dict[str, Any], dataset_id: str, base_seed: int) -> Path:
    """Derive the version-isolated split path; this function never creates it."""
    if dataset_id not in lock["registry"]["locked_primary_ids"] or base_seed not in lock["seeds"]:
        raise ValueError("Split path requested outside the explicit v1.1 lock scope.")
    return root / lock["paths"]["split_root"] / dataset_id / f"seed-{base_seed}.json"


def stage04_v11_output_paths(root: Path, lock: dict[str, Any]) -> dict[str, Path | str]:
    """Derive v1.1-only evidence destinations without creating any output."""
    return {
        "stage_root": root / lock["paths"]["stage04_evidence_root"],
        "summary_path": root / lock["paths"]["stage04_evidence_root"] / lock["output_identifiers"]["summary_filename"],
        "artifact_id": lock["output_identifiers"]["artifact_id"],
        "version": lock["output_identifiers"]["version"],
    }


def _require_canonical_v11_lock_path(root: Path, lock_path: Path) -> None:
    actual = (lock_path if lock_path.is_absolute() else root / lock_path).resolve()
    allowed = {(root / "configs" / filename).resolve() for filename in V11_STAGE04_LOCK_PROFILES}
    if actual not in allowed:
        raise ValueError("Stage 08 v1.1 execution accepts only the canonical v1.1 lock or its v1.1.1 byte-hash-repair lock.")


def build_stage04_v11_execution_plan(root: Path, lock_path: Path) -> dict[str, Any]:
    """Validate authorization and derive v1.1-only destinations without side effects.

    Receipt validation happens before registry, raw-data, split, model, cache, or
    run-directory access.  The immutable lock intentionally retains its false
    execution flags: D08-003 is the separate, receipt-bound authorization layer.
    """
    _require_canonical_v11_lock_path(root, lock_path)
    authorization = load_d08_003_authorization(root)
    lock = load_split_lock(root, lock_path)
    if lock["execution_gate"].get("d08_003_numeric_cache_budget_receipt_required_before_execution") is not True:
        raise ValueError("The v1.1 split lock must require D08-003 before execution.")
    _require_non_authorizing_v11_gate(lock["execution_gate"])
    if any(lock["output_contract"].get(key) is not False for key in ("conformal_prediction_allowed", "pilot_allowed", "formal_outputs_allowed")):
        raise ValueError("The v1.1 split lock must prohibit CP, pilot, and formal outputs.")
    return {
        "lock": lock,
        "authorization": authorization,
        "output_paths": stage04_v11_output_paths(root, lock),
        "split_root": root / lock["paths"]["split_root"],
        "unit_count": len(lock["registry"]["locked_primary_ids"]) * len(lock["seeds"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--allow-download", action="store_true", help="Only use if a locked raw file is missing.")
    parser.add_argument("--lock", type=Path, help="Explicit v1.1 split lock; preparation locks reject execution.")
    args = parser.parse_args()
    root = args.root.resolve()
    explicit_plan = build_stage04_v11_execution_plan(root, args.lock) if args.lock else None
    explicit_lock = explicit_plan["lock"] if explicit_plan is not None else None
    if explicit_lock is not None:
        config = {
            "datasets": {"primary_ids": explicit_lock["registry"]["locked_primary_ids"]},
            "experiment": {"seeds": explicit_lock["seeds"]},
            "protocol": {
                "protocol_id": explicit_lock["protocol"]["protocol_id"],
                "protocol_version_for_seed_derivation": explicit_lock["protocol_version"],
            },
            "paths": {"split_root": explicit_lock["paths"]["split_root"]},
        }
        registry = load_dataset_registry(root, root / explicit_lock["registry"]["path"])
    else:
        config = yaml.safe_load((root / "configs" / "stage03_base_v1.0.yaml").read_text(encoding="utf-8"))
        registry = load_dataset_registry(root)
    primary = locked_primary_records(registry)
    configured_ids = config["datasets"]["primary_ids"]
    registry_ids = [record["dataset_id"] for record in primary]
    if registry_ids != configured_ids:
        raise RuntimeError(f"Dataset lock/config mismatch; no replacement or split change is permitted: {registry_ids} != {configured_ids}")

    output_paths = (explicit_plan["output_paths"] if explicit_plan is not None else {
        "stage_root": root / "artifacts" / "stage04",
        "summary_path": root / "artifacts" / "stage04" / "stage04_summary_v1.0.json",
        "artifact_id": "stage04_data_preparation_v1.0",
        "version": "v1.0.0",
    })
    stage_root = output_paths["stage_root"]
    evidence: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for record in primary:
        dataset_id = record["dataset_id"]
        table = load_locked_dataset(root, dataset_id, allow_download=args.allow_download)
        data_report = audit_table(table, known_leakage_note=record.get("integrity_audit", {}).get("known_time_or_target_leakage"))
        data_report_hash = _write_immutable_json(
            stage_root / "data_reports" / f"{dataset_id}.json", data_report, byte_stable_lf=explicit_lock is not None
        )
        seed_reports: list[dict[str, Any]] = []
        for base_seed in config["experiment"]["seeds"]:
            split = make_stratified_split(table, base_seed, protocol_version=config["protocol"]["protocol_version_for_seed_derivation"])
            split_path = (split_manifest_path(root, explicit_lock, dataset_id, base_seed) if explicit_lock is not None
                          else root / config["paths"]["split_root"] / "v1.0" / dataset_id / f"seed-{base_seed}.json")
            write_split_manifest(split_path, split, raw_sha256=table.raw_sha256)
            preprocessing = {}
            for model_name in ("logistic_regression", "xgboost"):
                processor = TrainOnlyPreprocessor(model_name).fit(table, split)
                preprocessing[model_name] = processor.report()
                if preprocessing[model_name]["transformed_feature_count"] > 500:
                    failures.append({"dataset_id": dataset_id, "base_seed": base_seed, "issue": "post_train_transform_feature_count_above_500", "value": preprocessing[model_name]["transformed_feature_count"]})
            manifest = split.manifest(raw_sha256=table.raw_sha256)
            seed_report = {
                "dataset_id": dataset_id, "base_seed": base_seed, "split_hash": split.split_hash,
                "split_manifest_path": str(split_path.relative_to(root)), "class_counts": split.class_counts,
                "feasibility": manifest["feasibility"], "preprocessing": preprocessing,
            }
            if not manifest["feasibility"]["pass"]:
                failures.append({"dataset_id": dataset_id, "base_seed": base_seed, "issue": "locked_split_infeasible", "detail": manifest["feasibility"]})
            seed_reports.append(seed_report)
        evidence.append({"dataset_id": dataset_id, "raw_sha256": table.raw_sha256, "data_report_path": str((stage_root / "data_reports" / f"{dataset_id}.json").relative_to(root)), "data_report_sha256": data_report_hash, "seeds": seed_reports})

    summary = {
        "artifact_id": output_paths["artifact_id"], "version": output_paths["version"], "producer_stage": "Stage 04",
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "PASS" if not failures else "FAIL_PROTOCOL_DATASET", "formal_model_training": "NOT_RUN",
        "protocol_id": config["protocol"]["protocol_id"], "dataset_ids": registry_ids,
        "seed_count": len(config["experiment"]["seeds"]), "evidence": evidence, "failures": failures,
        "replacement_action": "NONE; this stage never changes the locked dataset list, split fractions, seeds, or feasibility thresholds.",
    }
    output_hash = _write_immutable_json(output_paths["summary_path"], summary, byte_stable_lf=explicit_lock is not None)
    print(json.dumps({"status": summary["status"], "summary": str(output_paths["summary_path"].relative_to(root)), "summary_sha256": output_hash, "failures": len(failures)}))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())

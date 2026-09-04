"""Independent, read-only verification of the authorized Stage 08 v1.1 splits.

This auditor intentionally does not import the Stage 04 runner or its split
writer.  It recomputes seed provenance, split hashes, partition counts, raw-data
hash bindings, and historical-v1.0 integrity snapshots directly from controlled
files before writing one immutable audit record.
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

from conformal_uq.config import FROZEN_DATASETS, FROZEN_SEEDS
from conformal_uq.data import load_dataset_registry, load_locked_dataset


AUDIT_RELATIVE_PATH = Path("artifacts/stage08_v11/split_regeneration_audit.json")
REPAIR_AUDIT_RELATIVE_PATH = Path("artifacts/stage08_v11/split_regeneration_audit_v1.1.1.json")
SPLIT_LOCK_RELATIVE_PATH = Path("configs/stage04_splits_v1.1.yaml")
REPAIR_SPLIT_LOCK_RELATIVE_PATH = Path("configs/stage04_splits_v1.1.1.yaml")
RECEIPT_RELATIVE_PATH = Path("decisions/D08-003_V11_CACHE_AND_PILOT_BUDGET_RECEIPT.json")
V11_SPLIT_ROOT_RELATIVE_PATH = Path("artifacts/splits/v1.1")
V10_SPLIT_ROOT_RELATIVE_PATH = Path("artifacts/splits/v1.0")


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _derive_seed(protocol_version: str, dataset_id: str, base_seed: int, purpose: str) -> tuple[str, int]:
    canonical = f"{protocol_version}|{dataset_id}|{base_seed}|{purpose}"
    derived = int.from_bytes(hashlib.sha256(canonical.encode("utf-8")).digest()[:4], "big", signed=False)
    return canonical, derived


def _split_hash(dataset_id: str, base_seed: int, fractions: dict[str, float], split_ids: dict[str, list[str]]) -> str:
    payload = {
        "dataset_id": dataset_id,
        "base_seed": base_seed,
        "fractions": fractions,
        "split_ids": {
            "train": split_ids["train"],
            "calibration_pool": split_ids["calibration_pool"],
            "test": split_ids["test"],
        },
    }
    return _sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _snapshot_files(root: Path, relative_root: Path) -> list[dict[str, Any]]:
    directory = root / relative_root
    if not directory.exists():
        return []
    return [
        {
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]


def _immutable_write(path: Path, payload: dict[str, Any]) -> str:
    encoded = _canonical_json(payload)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"Independent split audit is immutable and differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded.encode("utf-8"))
    return _sha256_file(path)


def _audit_stage04_evidence(root: Path, lock: dict[str, Any]) -> dict[str, Any]:
    paths = lock.get("paths", {})
    identifiers = lock.get("output_identifiers", {})
    evidence_root = root / paths["stage04_evidence_root"]
    summary_path = evidence_root / identifiers["summary_filename"]
    errors: list[str] = []
    reports_checked = 0
    if not summary_path.is_file():
        return {
            "status": "FAIL",
            "summary_path": str(summary_path.relative_to(root)).replace("\\", "/"),
            "errors": ["summary_missing"],
            "data_report_count": reports_checked,
        }
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "status": "FAIL",
            "summary_path": str(summary_path.relative_to(root)).replace("\\", "/"),
            "summary_sha256": _sha256_file(summary_path),
            "errors": [f"summary_invalid_json:{exc}"],
            "data_report_count": reports_checked,
        }
    if summary.get("artifact_id") != identifiers["artifact_id"]:
        errors.append("summary_artifact_id")
    if summary.get("version") != identifiers["version"]:
        errors.append("summary_version")
    if summary.get("status") != "PASS" or summary.get("formal_model_training") != "NOT_RUN":
        errors.append("summary_status_or_training_scope")
    if summary.get("dataset_ids") != FROZEN_DATASETS or summary.get("seed_count") != len(FROZEN_SEEDS):
        errors.append("summary_dataset_or_seed_scope")
    evidence = summary.get("evidence", [])
    if not isinstance(evidence, list) or len(evidence) != len(FROZEN_DATASETS):
        errors.append("summary_evidence_count")
        evidence = []
    seed_records = 0
    for item in evidence:
        report_relative_path = Path(str(item.get("data_report_path", "").replace("\\", "/")))
        report_path = root / report_relative_path
        if not report_path.is_file():
            errors.append(f"data_report_missing:{report_relative_path.as_posix()}")
            continue
        reports_checked += 1
        if item.get("data_report_sha256") != _sha256_file(report_path):
            errors.append(f"data_report_byte_hash:{report_relative_path.as_posix()}")
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            errors.append(f"data_report_invalid_json:{report_relative_path.as_posix()}")
            continue
        if report.get("dataset_id") != item.get("dataset_id") or report.get("raw_sha256") != item.get("raw_sha256"):
            errors.append(f"data_report_provenance:{report_relative_path.as_posix()}")
        seeds = item.get("seeds", [])
        if not isinstance(seeds, list) or len(seeds) != len(FROZEN_SEEDS):
            errors.append(f"data_report_seed_count:{item.get('dataset_id')}")
        else:
            seed_records += len(seeds)
    if seed_records != len(FROZEN_DATASETS) * len(FROZEN_SEEDS):
        errors.append("summary_seed_record_count")
    return {
        "status": "PASS" if not errors else "FAIL",
        "summary_path": str(summary_path.relative_to(root)).replace("\\", "/"),
        "summary_sha256": _sha256_file(summary_path),
        "data_report_count": reports_checked,
        "seed_record_count": seed_records,
        "errors": errors,
    }


def audit(root: Path, *, split_lock_relative_path: Path = SPLIT_LOCK_RELATIVE_PATH) -> dict[str, Any]:
    split_lock_path = root / split_lock_relative_path
    receipt_path = root / RECEIPT_RELATIVE_PATH
    lock = yaml.safe_load(split_lock_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_fractions = {"train": 0.6, "calibration_pool": 0.2, "test": 0.2}
    expected_keys = [(dataset_id, base_seed) for dataset_id in FROZEN_DATASETS for base_seed in FROZEN_SEEDS]
    errors: list[dict[str, Any]] = []

    if lock.get("protocol_version") != "v1.1":
        errors.append({"scope": "split_lock", "issue": "protocol_version", "observed": lock.get("protocol_version")})
    if lock.get("registry", {}).get("locked_primary_ids") != FROZEN_DATASETS:
        errors.append({"scope": "split_lock", "issue": "registry_ids"})
    if lock.get("seeds") != FROZEN_SEEDS:
        errors.append({"scope": "split_lock", "issue": "seeds"})
    if lock.get("split_fractions") != expected_fractions:
        errors.append({"scope": "split_lock", "issue": "fractions", "observed": lock.get("split_fractions")})
    if receipt.get("status") != "APPROVED_FOR_STAGE08_V11_CACHE_AND_PILOT_ONLY":
        errors.append({"scope": "d08_003", "issue": "unapproved_receipt_status", "observed": receipt.get("status")})
    if receipt.get("protocol_version") != "v1.1":
        errors.append({"scope": "d08_003", "issue": "protocol_version", "observed": receipt.get("protocol_version")})

    stage04_repair_evidence = _audit_stage04_evidence(root, lock)
    if stage04_repair_evidence["status"] != "PASS":
        errors.append({"scope": "stage04_repair_evidence", "issue": "byte_hash_or_scope_validation", "detail": stage04_repair_evidence["errors"]})

    v11_root = root / V11_SPLIT_ROOT_RELATIVE_PATH
    observed_paths = sorted(v11_root.glob("*/seed-*.json")) if v11_root.exists() else []
    observed_keys: set[tuple[str, int]] = set()
    for path in observed_paths:
        try:
            observed_keys.add((path.parent.name, int(path.stem.removeprefix("seed-"))))
        except ValueError:
            errors.append({"scope": "manifest_inventory", "issue": "invalid_filename", "path": str(path.relative_to(root))})
    expected_key_set = set(expected_keys)
    if observed_keys != expected_key_set or len(observed_paths) != len(expected_keys):
        errors.append({
            "scope": "manifest_inventory",
            "issue": "expected_exactly_80_manifests",
            "observed_count": len(observed_paths),
            "missing_keys": [list(key) for key in sorted(expected_key_set - observed_keys)],
            "extra_keys": [list(key) for key in sorted(observed_keys - expected_key_set)],
        })

    registry = load_dataset_registry(root, root / lock["registry"]["path"])
    raw_hashes = {record["dataset_id"]: record["source"]["raw_sha256"] for record in registry["records"]}
    unit_reports: list[dict[str, Any]] = []
    for dataset_id in FROZEN_DATASETS:
        table = load_locked_dataset(root, dataset_id, allow_download=False, registry_path=root / lock["registry"]["path"])
        sample_id_set = set(table.sample_ids)
        labels_by_id = dict(zip(table.sample_ids, table.labels.tolist(), strict=True))
        for base_seed in FROZEN_SEEDS:
            path = v11_root / dataset_id / f"seed-{base_seed}.json"
            unit_errors: list[str] = []
            if not path.exists():
                unit_reports.append({"dataset_id": dataset_id, "base_seed": base_seed, "status": "FAIL", "errors": ["manifest_missing"]})
                continue
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
                ids = manifest["split_ids"]
                groups = {name: list(ids[name]) for name in ("train", "calibration_pool", "test")}
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                unit_reports.append({"dataset_id": dataset_id, "base_seed": base_seed, "status": "FAIL", "errors": [f"invalid_manifest:{exc}"]})
                continue

            if manifest.get("artifact_type") != "stage04_stratified_split":
                unit_errors.append("artifact_type")
            if manifest.get("dataset_id") != dataset_id or manifest.get("base_seed") != base_seed:
                unit_errors.append("identity")
            if manifest.get("raw_sha256") != raw_hashes.get(dataset_id) or table.raw_sha256 != raw_hashes.get(dataset_id):
                unit_errors.append("raw_sha256")
            if manifest.get("fractions") != expected_fractions:
                unit_errors.append("fractions")

            group_sets = {name: set(values) for name, values in groups.items()}
            if any(len(group_sets[name]) != len(groups[name]) or not group_sets[name] for name in groups):
                unit_errors.append("empty_or_duplicate_ids")
            if group_sets["train"] & group_sets["calibration_pool"] or group_sets["train"] & group_sets["test"] or group_sets["calibration_pool"] & group_sets["test"]:
                unit_errors.append("overlapping_ids")
            if set().union(*group_sets.values()) != sample_id_set:
                unit_errors.append("partition_does_not_cover_raw_sample_ids")

            expected_hash = _split_hash(dataset_id, base_seed, expected_fractions, groups)
            if manifest.get("split_hash") != expected_hash:
                unit_errors.append("split_hash")
            for purpose in ("stratified_test_split", "stratified_calibration_split"):
                canonical, seed = _derive_seed("v1.1", dataset_id, base_seed, purpose)
                if manifest.get("seed_provenance", {}).get(purpose) != {"canonical_input": canonical, "derived_seed": seed}:
                    unit_errors.append(f"seed_provenance:{purpose}")

            actual_counts = {
                name: {
                    "majority": sum(labels_by_id[sample_id] == 0 for sample_id in values),
                    "minority": sum(labels_by_id[sample_id] == 1 for sample_id in values),
                }
                for name, values in groups.items()
            }
            if manifest.get("class_counts") != actual_counts:
                unit_errors.append("class_counts")

            v10_path = root / V10_SPLIT_ROOT_RELATIVE_PATH / dataset_id / f"seed-{base_seed}.json"
            v10_split_hash = None
            if not v10_path.exists():
                unit_errors.append("v1_0_comparator_missing")
            else:
                v10_split_hash = json.loads(v10_path.read_text(encoding="utf-8")).get("split_hash")
                if manifest.get("split_hash") == v10_split_hash:
                    unit_errors.append("v1_0_split_hash_reuse")

            unit_reports.append({
                "dataset_id": dataset_id,
                "base_seed": base_seed,
                "status": "PASS" if not unit_errors else "FAIL",
                "manifest_path": str(path.relative_to(root)).replace("\\", "/"),
                "manifest_sha256": _sha256_file(path),
                "split_hash": manifest.get("split_hash"),
                "v1_0_split_hash": v10_split_hash,
                "counts": actual_counts,
                "errors": unit_errors,
            })

    v10_split_snapshot = _snapshot_files(root, V10_SPLIT_ROOT_RELATIVE_PATH)
    v10_stage04_snapshot = _snapshot_files(root, Path("artifacts/stage04"))
    pass_units = sum(unit["status"] == "PASS" for unit in unit_reports)
    audit_version = lock["output_identifiers"]["version"]
    payload = {
        "artifact_id": "stage08_v11_split_regeneration_independent_audit" if audit_version == "v1.1.0" else "stage08_v11_split_regeneration_independent_audit_byte_hash_repair",
        "version": audit_version,
        "producer_stage": "Stage 08 independent split-regeneration audit",
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "PASS" if not errors and pass_units == 80 else "FAIL",
        "scope": "Only the 80 authorized v1.1 split manifests and their isolated Stage 04 lineage; no model, cache, TabPFN, conformal, pilot, formal-manifest, or formal-experiment output is audited or created.",
        "controlled_inputs": {
            "split_lock_path": str(split_lock_relative_path).replace("\\", "/"),
            "split_lock_sha256": _sha256_file(split_lock_path),
            "d08_003_receipt_path": str(RECEIPT_RELATIVE_PATH).replace("\\", "/"),
            "d08_003_receipt_sha256": _sha256_file(receipt_path),
            "auditor_path": "src/audit_stage08_v11_split_regeneration.py",
            "auditor_sha256": _sha256_file(Path(__file__)),
        },
        "expected_manifest_count": 80,
        "observed_manifest_count": len(observed_paths),
        "passed_manifest_count": pass_units,
        "errors": errors,
        "units": unit_reports,
        "stage04_repair_evidence": stage04_repair_evidence,
        "v1_0_integrity_snapshot": {
            "purpose": "Historical-only SHA-256 inventory recorded after v1.1 regeneration; this audit writes no v1.0 path.",
            "split_manifest_files": v10_split_snapshot,
            "stage04_evidence_files": v10_stage04_snapshot,
            "total_files": len(v10_split_snapshot) + len(v10_stage04_snapshot),
        },
        "prohibited_artifact_scan": {
            "checked_roots": ["artifacts/splits/v1.1", "artifacts/stage08_v11"],
            "prohibited_names": ["predictions.npz", "cache_manifest.json", "results_long.parquet", "results_long.csv", "formal_run_manifest.json"],
            "status": "NOT_APPLICABLE_TO_SPLIT_REGENERATION",
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--split-lock", type=Path, default=SPLIT_LOCK_RELATIVE_PATH)
    parser.add_argument("--output", type=Path, default=AUDIT_RELATIVE_PATH)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.split_lock not in {SPLIT_LOCK_RELATIVE_PATH, REPAIR_SPLIT_LOCK_RELATIVE_PATH}:
        raise ValueError("Independent split audit accepts only the canonical v1.1 lock or the v1.1.1 byte-hash-repair lock.")
    expected_output = AUDIT_RELATIVE_PATH if args.split_lock == SPLIT_LOCK_RELATIVE_PATH else REPAIR_AUDIT_RELATIVE_PATH
    if args.output != expected_output:
        raise ValueError("Independent split audit output path must match its canonical lock-specific immutable destination.")
    output = root / args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable independent split audit: {output}")
    payload = audit(root, split_lock_relative_path=args.split_lock)
    digest = _immutable_write(output, payload)
    print(json.dumps({
        "status": payload["status"],
        "passed_manifest_count": payload["passed_manifest_count"],
        "observed_manifest_count": payload["observed_manifest_count"],
        "audit": str(args.output).replace("\\", "/"),
        "audit_sha256": digest,
    }, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

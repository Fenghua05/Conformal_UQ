"""Create a non-destructive status reconciliation for the user-approved Stage 02 lock."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


SOURCE_SHA256 = "33d80dd85066d669b7404ceb209e1754f6cb80f0819e5c48e6bb1edde9fddfb3"
PRIMARY_IDS = (
    "openml_3_kr_vs_kp", "openml_24_mushroom", "openml_1486_nomao", "openml_1489_phoneme",
    "openml_1590_adult", "openml_4534_phishingwebsite", "openml_23512_higgs", "openml_23517_numerai28_6",
)
REPLACEMENT_IDS = ("openml_40701_churn", "openml_41143_jasmine", "openml_41146_sylvine", "openml_41150_miniboone")


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconcile_registry(source_registry: dict) -> dict:
    """Reconcile an interleaved candidate catalogue without changing source/data fields."""
    registry = copy.deepcopy(source_registry)
    if registry.get("status") != "PROPOSED_PENDING_USER_CONFIRMATION":
        raise ValueError("Source registry does not have the expected unresolved status.")
    records = registry.get("records", [])
    by_id = {record.get("dataset_id"): record for record in records}
    locked_ids = set(PRIMARY_IDS).union(REPLACEMENT_IDS)
    if len(by_id) != len(records) or not locked_ids.issubset(by_id):
        raise ValueError("Source registry is missing an approved primary or ordered replacement ID.")
    if any(by_id[dataset_id].get("selection_status") != "PROPOSED_PENDING_USER_CONFIRMATION" for dataset_id in locked_ids):
        raise ValueError("An approved source record has an unexpected selection status.")
    registry.update({
        "artifact_id": "dataset_registry_v1.0.1",
        "version": "v1.0.1",
        "status": "LOCKED_BY_USER_CONFIRMATION",
        "supersedes": "artifacts/stage02/dataset_registry_v1.0.json",
        "superseded_source_sha256": SOURCE_SHA256,
        "status_reconciliation": {
            "decision_id": "D02-001",
            "decision_time": "2026-08-30T00:00:00+08:00",
            "reason": "The v1.0 registry retained a proposed label after the user-approved dataset lock. This revision changes status metadata only; source hashes, data metadata, protocol, ranks, and replacement order are unchanged.",
        },
    })
    for dataset_id in PRIMARY_IDS:
        by_id[dataset_id]["status"] = "eligible_locked"
        by_id[dataset_id]["selection_status"] = "LOCKED_BY_USER_CONFIRMATION"
    for dataset_id in REPLACEMENT_IDS:
        by_id[dataset_id]["status"] = "eligible_locked_replacement"
        by_id[dataset_id]["selection_status"] = "LOCKED_ORDERED_REPLACEMENT"
    return registry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite superseding registry: {args.output}")
    if sha256_path(args.source) != SOURCE_SHA256:
        raise ValueError("Stage 02 v1.0 registry SHA-256 differs from the reviewed input.")
    registry = reconcile_registry(json.loads(args.source.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(registry, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(sha256_path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

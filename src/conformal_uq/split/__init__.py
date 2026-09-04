"""Deterministic Stage 04 60/20/20 stratified split materialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from ..data import BinaryTable
from ..identity import derive_seed, sha256_text

FRACTIONS = {"train": 0.6, "calibration_pool": 0.2, "test": 0.2}
FEASIBILITY_REQUIREMENTS = {"calibration_minority": 100, "calibration_majority": 200, "test_minority": 75}


@dataclass(frozen=True)
class SplitIDs:
    train: tuple[str, ...]
    calibration_pool: tuple[str, ...]
    test: tuple[str, ...]

    def validate(self) -> None:
        groups = (set(self.train), set(self.calibration_pool), set(self.test))
        if not all(groups):
            raise ValueError("Every split group must be non-empty.")
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("Train, calibration pool, and test IDs must be disjoint.")

    def as_dict(self) -> dict[str, list[str]]:
        return {"train": list(self.train), "calibration_pool": list(self.calibration_pool), "test": list(self.test)}


@dataclass(frozen=True)
class StratifiedSplit:
    dataset_id: str
    base_seed: int
    ids: SplitIDs
    split_hash: str
    class_counts: dict[str, dict[str, int]]
    seed_provenance: dict[str, dict[str, Any]]

    def manifest(self, *, raw_sha256: str) -> dict[str, Any]:
        return {
            "artifact_type": "stage04_stratified_split", "dataset_id": self.dataset_id, "base_seed": self.base_seed,
            "fractions": FRACTIONS, "raw_sha256": raw_sha256, "seed_provenance": self.seed_provenance,
            "split_ids": self.ids.as_dict(), "split_hash": self.split_hash, "class_counts": self.class_counts,
            "feasibility": split_feasibility_from_split(self),
        }


def _counts(table: BinaryTable, ids: tuple[str, ...]) -> dict[str, int]:
    labels = table.subset_labels(ids)
    return {"majority": int((labels == 0).sum()), "minority": int((labels == 1).sum())}


def _hash_ids(dataset_id: str, base_seed: int, ids: SplitIDs) -> str:
    value = json.dumps({"dataset_id": dataset_id, "base_seed": base_seed, "fractions": FRACTIONS, "split_ids": ids.as_dict()}, sort_keys=True, separators=(",", ":"))
    return sha256_text(value)


def make_stratified_split(table: BinaryTable, base_seed: int, *, protocol_version: str = "v1.0") -> StratifiedSplit:
    """Use the frozen two-stage randomization, then serialize IDs in stable order."""
    if not table.dataset_id:
        raise ValueError("Dataset ID is required for protocol seed routing.")
    positions = np.arange(len(table.labels))
    test_canonical, test_seed = derive_seed(protocol_version, table.dataset_id, base_seed, "stratified_test_split")
    train_cal, test = train_test_split(positions, test_size=FRACTIONS["test"], stratify=table.labels, random_state=test_seed)
    cal_canonical, cal_seed = derive_seed(protocol_version, table.dataset_id, base_seed, "stratified_calibration_split")
    train, calibration_pool = train_test_split(train_cal, test_size=0.25, stratify=table.labels.iloc[train_cal], random_state=cal_seed)
    ids = SplitIDs(
        train=tuple(sorted(table.sample_ids[int(position)] for position in train)),
        calibration_pool=tuple(sorted(table.sample_ids[int(position)] for position in calibration_pool)),
        test=tuple(sorted(table.sample_ids[int(position)] for position in test)),
    )
    ids.validate()
    return StratifiedSplit(
        dataset_id=table.dataset_id, base_seed=base_seed, ids=ids,
        split_hash=_hash_ids(table.dataset_id, base_seed, ids),
        class_counts={"train": _counts(table, ids.train), "calibration_pool": _counts(table, ids.calibration_pool), "test": _counts(table, ids.test)},
        seed_provenance={
            "stratified_test_split": {"canonical_input": test_canonical, "derived_seed": test_seed},
            "stratified_calibration_split": {"canonical_input": cal_canonical, "derived_seed": cal_seed},
        },
    )


def split_feasibility_from_split(split: StratifiedSplit) -> dict[str, Any]:
    counts = split.class_counts
    checks = {
        "calibration_minority": counts["calibration_pool"]["minority"] >= FEASIBILITY_REQUIREMENTS["calibration_minority"],
        "calibration_majority": counts["calibration_pool"]["majority"] >= FEASIBILITY_REQUIREMENTS["calibration_majority"],
        "test_minority": counts["test"]["minority"] >= FEASIBILITY_REQUIREMENTS["test_minority"],
    }
    return {"requirements": FEASIBILITY_REQUIREMENTS, "checks": checks, "pass": all(checks.values())}


def split_feasibility(table: BinaryTable, base_seed: int, *, protocol_version: str = "v1.0") -> dict[str, Any]:
    return split_feasibility_from_split(make_stratified_split(table, base_seed, protocol_version=protocol_version))


def write_split_manifest(path: Path, split: StratifiedSplit, *, raw_sha256: str) -> Path:
    """Create one immutable split manifest or prove a prior one is byte-identical."""
    payload = split.manifest(raw_sha256=raw_sha256)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"Existing split manifest differs and cannot be overwritten: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")
    return path

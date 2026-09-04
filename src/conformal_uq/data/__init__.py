"""Stage 04 immutable raw-data loading and integrity evidence."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from scipy.io import arff

from ..provenance import sha256_path


@dataclass(frozen=True)
class BinaryTableContract:
    sample_ids: Sequence[str]
    labels: Sequence[int]

    def validate(self) -> None:
        if len(self.sample_ids) != len(self.labels) or not self.sample_ids:
            raise ValueError("Sample IDs and labels must be non-empty and equal in length.")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("Sample IDs must be unique.")
        if set(self.labels).difference({0, 1}):
            raise ValueError("Labels must be binary protocol labels {0, 1}.")


@dataclass(frozen=True)
class BinaryTable:
    """A derived in-memory table; its raw ARFF source is never mutated."""

    dataset_id: str
    features: pd.DataFrame
    labels: pd.Series
    sample_ids: tuple[str, ...]
    raw_path: Path
    raw_sha256: str
    target_column: str
    label_mapping: dict[str, int]

    def __post_init__(self) -> None:
        if len(self.features) != len(self.labels) or len(self.features) != len(self.sample_ids):
            raise ValueError("Features, labels, and sample IDs must have equal row counts.")
        BinaryTableContract(self.sample_ids, self.labels.tolist()).validate()
        if len(self.raw_sha256) != 64:
            raise ValueError("raw_sha256 must be a SHA-256 hex digest.")

    def row_positions(self, ids: Sequence[str]) -> list[int]:
        positions_by_id = {sample_id: position for position, sample_id in enumerate(self.sample_ids)}
        try:
            positions = [positions_by_id[sample_id] for sample_id in ids]
        except KeyError as exc:
            raise ValueError(f"Unknown sample ID: {exc.args[0]}") from exc
        if len(set(positions)) != len(positions):
            raise ValueError("Operation IDs must be unique.")
        return positions

    def subset_features(self, ids: Sequence[str]) -> pd.DataFrame:
        return self.features.iloc[self.row_positions(ids)].copy()

    def subset_labels(self, ids: Sequence[str]) -> pd.Series:
        return self.labels.iloc[self.row_positions(ids)].copy()


def _decode(value: Any) -> Any:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def _load_arff(path: Path) -> pd.DataFrame:
    records, _ = arff.loadarff(str(path))
    frame = pd.DataFrame(records)
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].map(_decode)
    return frame.replace("?", pd.NA)


def load_dataset_registry(root: Path, registry_path: Path | None = None) -> dict[str, Any]:
    path = registry_path if registry_path is not None else root / "artifacts" / "stage02" / "dataset_registry_v1.0.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("records"), list):
        raise ValueError("Stage 02 registry has no records list.")
    return payload


def registry_record(registry: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    matches = [record for record in registry["records"] if record.get("dataset_id") == dataset_id]
    if len(matches) != 1:
        raise KeyError(f"Dataset {dataset_id!r} is not uniquely recorded.")
    return matches[0]


def locked_primary_records(registry: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        (record for record in registry["records"] if record.get("proposed_role") == "primary"),
        key=lambda record: int(record.get("frozen_selection_rank", 999999)),
    )


def materialize_raw_file(record: dict[str, Any], root: Path, *, allow_download: bool = False) -> Path:
    """Return a hash-verified raw file, downloading only if explicitly requested.

    Existing sources are never overwritten. A download is staged in a temporary
    file, hash verified against Stage 02 evidence, then moved into the raw path.
    """
    source = record["source"]
    raw_path = root / Path(str(source["raw_local_path"]).replace("\\", "/"))
    expected = str(source["raw_sha256"])
    if raw_path.exists():
        observed = sha256_path(raw_path)
        if observed != expected:
            raise ValueError(f"{record['dataset_id']}: raw SHA-256 mismatch ({observed} != {expected}).")
        return raw_path
    if not allow_download:
        raise FileNotFoundError(f"Missing locked raw source: {raw_path}")
    file_id = source.get("download_file_id")
    if not file_id:
        raise ValueError(f"{record['dataset_id']}: registry has no immutable OpenML file ID.")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://www.openml.org/data/v1/download/{file_id}"
    with tempfile.NamedTemporaryFile(dir=raw_path.parent, prefix=f".{raw_path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                shutil.copyfileobj(response, handle)
            observed = sha256_path(temporary)
            if observed != expected:
                raise ValueError(f"{record['dataset_id']}: downloaded SHA-256 mismatch ({observed} != {expected}).")
            temporary.replace(raw_path)
        finally:
            temporary.unlink(missing_ok=True)
    return raw_path


def load_locked_dataset(root: Path, dataset_id: str, *, allow_download: bool = False, registry_path: Path | None = None) -> BinaryTable:
    """Load the Stage 02 source with its locked target mapping and stable row IDs."""
    record = registry_record(load_dataset_registry(root, registry_path), dataset_id)
    raw_path = materialize_raw_file(record, root, allow_download=allow_download)
    frame = _load_arff(raw_path)
    target_candidates = [name for name in frame.columns if str(name).casefold() == str(record["target"]).casefold()]
    if len(target_candidates) != 1:
        raise KeyError(f"{dataset_id}: locked target {record['target']!r} is absent or ambiguous.")
    target = target_candidates[0]
    mapping = {str(key): int(value) for key, value in record["label_mapping_to_protocol_binary"].items()}
    raw_labels = frame[target].map(_decode).astype("string").str.strip()
    unexpected = sorted(set(raw_labels.dropna().astype(str)).difference(mapping))
    if raw_labels.isna().any() or unexpected:
        raise ValueError(f"{dataset_id}: locked binary label mapping cannot be applied; unexpected={unexpected}.")
    labels = raw_labels.map(mapping).astype("int8").reset_index(drop=True)
    return BinaryTable(
        dataset_id=dataset_id,
        features=frame.drop(columns=[target]).copy(),
        labels=labels,
        sample_ids=tuple(f"{dataset_id}:{row:08d}" for row in range(len(frame))),
        raw_path=raw_path,
        raw_sha256=sha256_path(raw_path),
        target_column=str(target),
        label_mapping=mapping,
    )


def audit_table(table: BinaryTable, *, known_leakage_note: str | None = None) -> dict[str, Any]:
    """Audit the derived table without filtering data or changing raw sources."""
    features = table.features
    schema, identifier_like = [], []
    for name in features.columns:
        series = features[name]
        name_string = str(name)
        near_unique = series.nunique(dropna=False) / max(1, len(series)) > 0.995
        id_name = bool(re.search(r"(^|[_ -])(id|identifier|index|unnamed)([_ -]|$)", name_string, re.I))
        if id_name or near_unique:
            identifier_like.append(name_string)
        schema.append({
            "name": name_string,
            "dtype": str(series.dtype),
            "semantic_type": "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical",
            "missing_values": int(series.isna().sum()),
            "unique_nonmissing_values": int(series.nunique(dropna=True)),
        })
    joined = pd.concat([features.reset_index(drop=True), table.labels.rename("__protocol_label__")], axis=1)
    return {
        "dataset_id": table.dataset_id,
        "raw_path": str(table.raw_path),
        "raw_sha256": table.raw_sha256,
        "n_rows": int(len(features)), "n_features": int(features.shape[1]),
        "class_counts": {"majority": int((table.labels == 0).sum()), "minority": int((table.labels == 1).sum())},
        "target_column": table.target_column, "label_mapping": table.label_mapping,
        "missing_values": int(features.isna().sum().sum()),
        "rows_with_missing_values": int(features.isna().any(axis=1).sum()),
        "exact_duplicate_rows_including_target": int(joined.duplicated().sum()),
        "duplicate_feature_rows": int(features.duplicated().sum()),
        "identifier_like_columns": identifier_like,
        "lexical_target_or_outcome_flags": [str(name) for name in features.columns if re.search(r"(target|label|outcome|response|result|class)", str(name), re.I)],
        "known_leakage_note": known_leakage_note or "No source-documented leakage field in the locked registry; flags are reported, never silently dropped.",
        "feature_schema": schema,
    }


def table_hash(table: BinaryTable) -> str:
    value = json.dumps({"dataset_id": table.dataset_id, "raw_sha256": table.raw_sha256, "sample_ids": table.sample_ids}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

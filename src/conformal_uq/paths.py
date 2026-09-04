"""Immutable artifact path allocation."""

from __future__ import annotations

from pathlib import Path


class ImmutablePathError(FileExistsError):
    """Raised when a controlled run directory would overwrite prior evidence."""


def create_immutable_run_dir(root: Path, run_id: str) -> Path:
    target = root / run_id
    try:
        target.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ImmutablePathError(f"Controlled run directory already exists: {target}") from exc
    return target


def cache_path(root: Path, config_hash: str, code_hash: str, dataset_id: str, seed: int, model: str) -> Path:
    if not all((config_hash, code_hash, dataset_id, model)):
        raise ValueError("Cache paths require config, code, dataset, seed, and model provenance.")
    return root / f"cfg-{config_hash[:12]}" / f"code-{code_hash[:12]}" / dataset_id / f"seed-{seed}" / model

"""Deterministic seed and run identity functions shared by every module."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone


def canonical_seed_input(protocol_version: str, dataset_id: str, base_seed: int, purpose: str) -> str:
    if not all(isinstance(value, str) and value for value in (protocol_version, dataset_id, purpose)):
        raise ValueError("protocol_version, dataset_id, and purpose must be non-empty strings.")
    if not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a non-negative integer.")
    return f"{protocol_version}|{dataset_id}|{base_seed}|{purpose}"


def derive_seed(protocol_version: str, dataset_id: str, base_seed: int, purpose: str) -> tuple[str, int]:
    canonical = canonical_seed_input(protocol_version, dataset_id, base_seed, purpose)
    seed = int.from_bytes(hashlib.sha256(canonical.encode("utf-8")).digest()[:4], "big", signed=False)
    return canonical, seed


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_id(stage: str, config_hash: str, now: datetime | None = None) -> str:
    if not stage or len(config_hash) < 12:
        raise ValueError("stage and a SHA-256 config hash are required.")
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{stage}_{config_hash[:12]}"

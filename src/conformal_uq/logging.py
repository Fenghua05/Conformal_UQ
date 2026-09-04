"""Structured JSONL logging with protocol-required common fields."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REQUIRED_EVENT_FIELDS = ("timestamp_utc", "run_id", "stage", "level", "event", "config_hash", "message")


def write_event(path: Path, *, run_id: str, stage: str, level: str, event: str, config_hash: str, message: str, **scope: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "run_id": run_id,
        "stage": stage,
        "level": level,
        "event": event,
        "config_hash": config_hash,
        "message": message,
    }
    payload.update({key: value for key, value in scope.items() if value is not None})
    missing = [key for key in REQUIRED_EVENT_FIELDS if not payload.get(key)]
    if missing:
        raise ValueError(f"Incomplete structured log event: {missing}")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return payload

"""Data contracts only; real-data acquisition is deliberately deferred to Stage 04."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


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

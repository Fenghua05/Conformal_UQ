"""Split identity contracts; actual stratified splitting is a Stage 04 task."""

from __future__ import annotations

from dataclasses import dataclass


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

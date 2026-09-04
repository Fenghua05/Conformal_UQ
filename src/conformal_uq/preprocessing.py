"""Train-only preprocessing boundary contract; no transformer fitting in Stage 03."""

from __future__ import annotations


def assert_train_only_fit(fit_partition: str) -> None:
    if fit_partition != "train":
        raise ValueError("Preprocessing may fit only on the train partition.")

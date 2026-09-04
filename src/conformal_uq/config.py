"""Configuration loading and frozen-protocol validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

FROZEN_SEEDS = [104729, 130363, 155921, 196613, 262147, 318281, 374209, 419893, 481517, 552721]
FROZEN_DATASETS = [
    "openml_3_kr_vs_kp", "openml_24_mushroom", "openml_1486_nomao",
    "openml_1489_phoneme", "openml_1590_adult", "openml_4534_phishingwebsite",
    "openml_23512_higgs", "openml_23517_numerai28_6",
]
FROZEN_M_MINORITY = [10, 20, 50, 100]


class ConfigError(ValueError):
    """Raised when a configuration breaks a frozen Stage 01 protocol choice."""


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ConfigError("Top-level YAML content must be a mapping.")
    validate_config(payload)
    return payload


def validate_config(config: dict[str, Any]) -> None:
    required = {"config_version", "protocol", "experiment", "datasets", "models", "paths", "outputs", "seed_derivation"}
    missing = required.difference(config)
    if missing:
        raise ConfigError(f"Missing top-level configuration keys: {sorted(missing)}")
    experiment = config["experiment"]
    if experiment.get("alpha") != 0.1:
        raise ConfigError("alpha must remain frozen at 0.1.")
    if experiment.get("m_minority") != FROZEN_M_MINORITY:
        raise ConfigError("m_minority must remain [10, 20, 50, 100].")
    if experiment.get("m_majority") != 200:
        raise ConfigError("m_majority must remain 200.")
    if experiment.get("seeds") != FROZEN_SEEDS:
        raise ConfigError("The ten top-level seeds differ from the frozen protocol.")
    if config["datasets"].get("primary_ids") != FROZEN_DATASETS:
        raise ConfigError("Primary dataset IDs differ from the user-locked dataset lock.")
    if config["seed_derivation"].get("algorithm") != "sha256_first_32_bits_unsigned_big_endian":
        raise ConfigError("Seed routing must use the protocol SHA-256 derivation rule.")
    tabpfn = config["models"].get("tabpfn", {})
    if tabpfn.get("availability") != "PENDING_EXPLICIT_AUTHORIZATION":
        raise ConfigError("TabPFN must remain explicitly authorization-gated in Stage 03.")


def assert_formal_run_allowed(config: dict[str, Any]) -> None:
    """Prevent a formal run while its mandatory TabPFN environment is unresolved."""
    validate_config(config)
    raise ConfigError(
        "Formal execution is disabled in Stage 03: TabPFN package/checkpoint/device and compute authority are unresolved."
    )

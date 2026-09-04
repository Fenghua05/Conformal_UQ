"""Receipt-bound authorization checks for Stage 08 v1.1 cache preparation.

This module is deliberately limited to immutable configuration and receipt
validation.  It never imports TabPFN, loads a dataset, or creates an artifact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .config import FROZEN_DATASETS, FROZEN_SEEDS


D08_003_RECEIPT_PATH = "decisions/D08-003_V11_CACHE_AND_PILOT_BUDGET_RECEIPT.json"
FINAL_TABPFN_LOCK_PATH = "configs/stage05b_tabpfn_v1.1.yaml"
EXPECTED_STATUS = "APPROVED_FOR_STAGE08_V11_CACHE_AND_PILOT_ONLY"
EXPECTED_COUNTS = {
    "authorized_tabpfn_units": 80,
    "authorized_local_lr_xgboost_units": 160,
    "authorized_cache_intake_units": 240,
    "authorized_pilot_cells": 480,
}
EXPECTED_RUNTIME = {
    "provider": "AutoDL",
    "os": "Ubuntu 22.04",
    "device": "cuda",
    "gpu_name": "NVIDIA GeForce RTX 4090",
    "gpu_memory_gb": 24,
    "tabpfn_version": "8.5.0",
    "checkpoint_path": "/root/autodl-fs/tabpfn-model-cache/tabpfn-v3-classifier-v3_default.ckpt",
    "checkpoint_sha256": "d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988",
    "train_partition_contract": "full_fixed_train_partition",
    "no_truncation_or_subsampling": True,
    "ignore_pretraining_limits": False,
    "max_train_rows": 100000,
    "max_transformed_features": 2000,
}


def sha256_file(path: Path) -> str:
    """Return the digest of an immutable input without modifying it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_exact(mapping: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    for key, value in expected.items():
        if mapping.get(key) != value:
            raise ValueError(f"D08-003 {label} field {key} differs from the immutable v1.1 authorization.")


def _require_non_authorizing_gate(gate: dict[str, Any], label: str) -> None:
    for key in (
        "split_regeneration_authorized",
        "local_cache_generation_authorized",
        "tabpfn_cache_generation_authorized",
        "pilot_authorized",
        "formal_run_manifest_authorized",
        "full_experiment_authorized",
    ):
        if gate.get(key) is not False:
            raise ValueError(f"The {label} must keep {key} false.")


def validate_d08_003_receipt(
    receipt: dict[str, Any],
    final_lock_path: Path,
    final_lock: dict[str, Any],
) -> None:
    """Validate the D08-003 receipt against the current final TabPFN lock.

    The receipt is an execution gate, not a mutable replacement for any lock.
    Both its lock digest and its fixed cloud runtime must agree with the file
    presently in the workspace.
    """
    if receipt.get("artifact_id") != "D08-003_stage08_v11_cache_and_pilot_budget":
        raise ValueError("D08-003 artifact identity is invalid.")
    if receipt.get("status") != EXPECTED_STATUS:
        raise ValueError("D08-003 status is not the approved v1.1 cache-and-pilot-only receipt.")
    if receipt.get("protocol_version") != "v1.1":
        raise ValueError("D08-003 protocol_version must be v1.1.")
    _require_exact(receipt, {"maximum_wall_clock_hours": 12, "maximum_cloud_storage_gb": 50}, "numeric budget")
    _require_exact(receipt, EXPECTED_COUNTS, "unit counts")
    if receipt.get("formal_run_manifest_authorized") is not False or receipt.get("full_experiment_authorized") is not False:
        raise ValueError("D08-003 must not authorize a formal manifest or full experiment.")
    if receipt.get("cache_lock_path") != FINAL_TABPFN_LOCK_PATH:
        raise ValueError("D08-003 must bind the canonical final v1.1 TabPFN cache lock path.")
    current_hash = sha256_file(final_lock_path)
    if receipt.get("cache_lock_sha256") != current_hash:
        raise ValueError("D08-003 cache-lock hash binding does not match the current final lock.")

    binding = receipt.get("receipt_to_lock_binding")
    if not isinstance(binding, dict) or binding.get("receipt_path") != D08_003_RECEIPT_PATH:
        raise ValueError("D08-003 requires the canonical receipt-to-lock binding.")
    if binding.get("cache_lock_path") != FINAL_TABPFN_LOCK_PATH or binding.get("cache_lock_sha256") != current_hash:
        raise ValueError("D08-003 receipt-to-lock binding does not match the current final lock.")
    if binding.get("receipt_content_is_not_an_input_to_cache_lock_hash") is not True:
        raise ValueError("D08-003 receipt-to-lock binding provenance is invalid.")

    runtime = receipt.get("immutable_runtime_inputs")
    if not isinstance(runtime, dict):
        raise ValueError("D08-003 immutable runtime inputs are missing.")
    _require_exact(runtime, EXPECTED_RUNTIME, "immutable runtime inputs")
    if not isinstance(final_lock, dict) or final_lock.get("protocol_version") != "v1.1":
        raise ValueError("The final TabPFN cache lock is not a v1.1 lock.")
    if final_lock.get("split_lock_path") != "configs/stage04_splits_v1.1.yaml":
        raise ValueError("The final TabPFN cache lock must link the canonical v1.1 split lock.")
    if final_lock.get("registry", {}).get("locked_primary_ids") != FROZEN_DATASETS or final_lock.get("seeds") != FROZEN_SEEDS:
        raise ValueError("The final TabPFN cache lock registry or seeds differ from the frozen v1.1 protocol.")
    canonical_split_path = final_lock_path.resolve().parents[1] / "configs" / "stage04_splits_v1.1.yaml"
    if not canonical_split_path.is_file():
        raise FileNotFoundError(f"The final TabPFN cache lock requires its canonical split lock: {canonical_split_path}")
    split_lock = yaml.safe_load(canonical_split_path.read_text(encoding="utf-8"))
    if not isinstance(split_lock, dict) or split_lock.get("protocol_version") != "v1.1":
        raise ValueError("The canonical v1.1 split lock is invalid.")
    if split_lock.get("registry", {}).get("locked_primary_ids") != FROZEN_DATASETS or split_lock.get("seeds") != FROZEN_SEEDS:
        raise ValueError("The canonical v1.1 split lock registry or seeds differ from the frozen protocol.")
    split_protocol = split_lock.get("protocol", {})
    if split_protocol.get("protocol_version_for_seed_derivation") != "v1.1" or split_protocol.get("seed_derivation_algorithm") != "sha256_first_32_bits_unsigned_big_endian":
        raise ValueError("The canonical v1.1 split lock does not bind the frozen seed derivation.")
    if split_lock.get("paths", {}).get("split_root") != "artifacts/splits/v1.1":
        raise ValueError("The canonical v1.1 split lock must use the isolated v1.1 split root.")
    lock_runtime = final_lock.get("runtime")
    if not isinstance(lock_runtime, dict):
        raise ValueError("The final TabPFN cache lock runtime is missing.")
    _require_exact(lock_runtime, {key: value for key, value in EXPECTED_RUNTIME.items() if key not in {"max_train_rows", "max_transformed_features"}}, "final-lock runtime")
    _require_exact(final_lock.get("safety_limits", {}), {"max_train_rows": 100000, "max_transformed_features": 2000}, "final-lock safety limits")
    if final_lock.get("paths", {}).get("split_root") != "artifacts/splits/v1.1" or final_lock.get("paths", {}).get("cache_root") != "artifacts/caches/v1.1":
        raise ValueError("The final TabPFN cache lock must use isolated v1.1 split/cache roots.")
    authorization = final_lock.get("authorization", {})
    if authorization.get("d08_003_budget_receipt_path") != D08_003_RECEIPT_PATH:
        raise ValueError("The final TabPFN cache lock does not name the canonical D08-003 receipt.")
    if authorization.get("receipt_to_lock_binding") != "receipt_must_match_this_exact_lock_sha256":
        raise ValueError("The final TabPFN cache lock has no exact receipt binding contract.")
    expected_cache_scope = {
        "authorized_tabpfn_units": 80,
        "datasets": 8,
        "seeds_per_dataset": 10,
        "probability_cache_only": True,
        "conformal_prediction_allowed": False,
        "pilot_output_allowed": False,
        "formal_output_allowed": False,
    }
    _require_exact(authorization.get("cache_only_scope", {}), expected_cache_scope, "final-lock cache-only scope")
    gate, output = final_lock.get("execution_gate", {}), final_lock.get("output_contract", {})
    if gate.get("d08_003_numeric_cache_budget_receipt_required_before_execution") is not True:
        raise ValueError("The final TabPFN cache lock must require D08-003 before execution.")
    _require_non_authorizing_gate(gate, "final TabPFN cache lock")
    if any(output.get(key) is not False for key in ("conformal_prediction_allowed", "pilot_allowed", "formal_outputs_allowed")):
        raise ValueError("The final TabPFN cache lock must prohibit CP, pilot, and formal outputs.")


def load_d08_003_authorization(root: Path) -> dict[str, Any]:
    """Load the only authorized Stage 08 v1.1 cache receipt and final lock."""
    receipt_path = root / D08_003_RECEIPT_PATH
    final_lock_path = root / FINAL_TABPFN_LOCK_PATH
    if not receipt_path.is_file():
        raise FileNotFoundError(f"D08-003 budget receipt is required before v1.1 execution: {receipt_path}")
    if not final_lock_path.is_file():
        raise FileNotFoundError(f"Final v1.1 TabPFN cache lock is required before v1.1 execution: {final_lock_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    final_lock = yaml.safe_load(final_lock_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError("D08-003 receipt must be a JSON object.")
    validate_d08_003_receipt(receipt, final_lock_path, final_lock)
    return {
        "receipt": receipt,
        "receipt_path": receipt_path,
        "final_lock": final_lock,
        "final_lock_path": final_lock_path,
        "final_lock_sha256": sha256_file(final_lock_path),
    }

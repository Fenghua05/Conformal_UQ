"""Deterministic toy-only data used solely for Stage 03 smoke validation."""

from __future__ import annotations

from .data import BinaryTableContract
from .split import SplitIDs


def make_toy_contract() -> tuple[BinaryTableContract, SplitIDs]:
    ids = tuple(f"toy-{index:02d}" for index in range(12))
    labels = tuple(index % 2 for index in range(12))
    table = BinaryTableContract(ids, labels)
    split = SplitIDs(train=ids[:6], calibration_pool=ids[6:9], test=ids[9:])
    table.validate()
    split.validate()
    return table, split


def toy_results_record(config_hash: str, code_hash: str, run_id: str) -> dict[str, object]:
    return {
        "dataset_id": "TOY_ONLY_STAGE03", "seed": 104729, "model": "TOY_NO_MODEL_FIT",
        "cp_method": "global_split_cp", "m_minority": 10, "m_majority": 200, "alpha": 0.1,
        "protocol_version": "v1.0", "config_hash": config_hash, "code_hash": code_hash,
        "run_id": run_id, "artifact_id": "stage03_toy_smoke", "results_schema_version": "v1.1.0",
        "status": "TOY_ONLY_NOT_RESEARCH_RESULT", "minority_label": 1,
        "n_cal_total": 210, "n_cal_minority": 10, "n_cal_majority": 200,
        "rank_global": 190, "rank_minority": None, "rank_majority": None,
        "auroc": 0.5, "auprc": 0.5,
        "coverage_overall": 1.0, "coverage_minority": 1.0, "coverage_majority": 1.0,
        "covered_count_overall": 2, "covered_count_minority": 1, "covered_count_majority": 1,
        "coverage_overall_wilson_low": 0.34238022750665303, "coverage_overall_wilson_high": 1.0,
        "coverage_minority_wilson_low": 0.20654931437723742, "coverage_minority_wilson_high": 1.0,
        "coverage_majority_wilson_low": 0.20654931437723742, "coverage_majority_wilson_high": 1.0,
        "coverage_disparity": 0.0, "singleton_rate": 1.0, "average_set_size": 1.0,
        "empty_rate": 0.0, "doubleton_rate": 0.0, "q_global": 0.5, "q_minority": None,
        "q_majority": None, "threshold_gap": None, "threshold_sum": None,
        "n_test": 2, "n_test_minority": 1, "n_test_majority": 1, "subset_hash": "TOY_ONLY_NO_CALIBRATION_SUBSET",
        "split_hash": "TOY_ONLY_SPLIT", "dataset_hash": "TOY_ONLY_DATASET", "environment_hash": "TOY_ONLY_ENVIRONMENT",
        "model_hash": "TOY_ONLY_MODEL", "prediction_cache_hash": "TOY_ONLY_CACHE", "label_mapping_hash": "TOY_ONLY_LABEL_MAPPING",
        "created_utc": "2026-08-30T00:00:00Z",
    }

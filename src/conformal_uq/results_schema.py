"""Validation for provenance-complete Stage 06 ``results_long`` records."""

from __future__ import annotations

import math
from typing import Any, Iterable

from .metrics import wilson_interval

UNIQUE_KEY = (
    "dataset_id", "seed", "model", "cp_method", "m_minority", "m_majority",
    "alpha", "protocol_version", "config_hash", "code_hash",
)
REQUIRED_COLUMNS = (
    *UNIQUE_KEY, "run_id", "artifact_id", "results_schema_version", "status",
    "minority_label", "n_cal_total", "n_cal_minority", "n_cal_majority",
    "rank_global", "rank_minority", "rank_majority", "auroc", "auprc",
    "coverage_overall", "coverage_minority", "coverage_majority", "coverage_disparity",
    "covered_count_overall", "covered_count_minority", "covered_count_majority",
    "coverage_overall_wilson_low", "coverage_overall_wilson_high",
    "coverage_minority_wilson_low", "coverage_minority_wilson_high",
    "coverage_majority_wilson_low", "coverage_majority_wilson_high",
    "singleton_rate", "average_set_size", "empty_rate", "doubleton_rate",
    "q_global", "q_minority", "q_majority", "threshold_gap", "threshold_sum",
    "n_test", "n_test_minority", "n_test_majority", "subset_hash",
    "split_hash", "dataset_hash", "environment_hash", "model_hash",
    "prediction_cache_hash", "label_mapping_hash", "created_utc",
)
_PROPORTIONS = ("coverage_overall", "coverage_minority", "coverage_majority", "singleton_rate", "empty_rate", "doubleton_rate")


def _is_number(value: Any) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(float(value))


def _coverage_errors(record: dict[str, Any], scope: str, total: int) -> list[str]:
    errors: list[str] = []
    count = record[f"covered_count_{scope}"]
    coverage = record[f"coverage_{scope}"]
    lower = record[f"coverage_{scope}_wilson_low"]
    upper = record[f"coverage_{scope}_wilson_high"]
    if not isinstance(count, int) or not 0 <= count <= total:
        return [f"invalid:covered_count_{scope}"]
    if not _is_number(coverage) or abs(float(coverage) - count / total) > 1e-12:
        errors.append(f"invalid:coverage_{scope}")
    if not _is_number(lower) or not _is_number(upper):
        errors.append(f"invalid:wilson_{scope}")
    else:
        expected_lower, expected_upper = wilson_interval(count, total)
        if not 0.0 <= float(lower) <= float(upper) <= 1.0 or abs(float(lower) - expected_lower) > 1e-12 or abs(float(upper) - expected_upper) > 1e-12:
            errors.append(f"invalid:wilson_{scope}")
    return errors


def validate_results_record(record: dict[str, Any]) -> list[str]:
    errors = [f"missing:{column}" for column in REQUIRED_COLUMNS if column not in record]
    if errors:
        return errors
    if record["cp_method"] not in {"global_split_cp", "class_conditional_cp"}:
        errors.append("invalid:cp_method")
    if record["m_minority"] not in {10, 20, 50, 100} or record["m_majority"] != 200:
        errors.append("invalid:calibration_sizes")
    if record["alpha"] != 0.1:
        errors.append("invalid:alpha")
    if record["status"] not in {"PASS", "FAIL", "WARN", "SKIPPED", "NOT_RUN", "TOY_ONLY_NOT_RESEARCH_RESULT"}:
        errors.append("invalid:status")
    if record["minority_label"] not in {0, 1}:
        errors.append("invalid:minority_label")
    if record["n_cal_total"] != record["n_cal_minority"] + record["n_cal_majority"] or record["n_cal_minority"] != record["m_minority"] or record["n_cal_majority"] != record["m_majority"]:
        errors.append("invalid:calibration_class_counts")
    if record["n_test"] != record["n_test_minority"] + record["n_test_majority"] or record["n_test_minority"] <= 0 or record["n_test_majority"] <= 0:
        errors.append("invalid:test_class_counts")
    for column in _PROPORTIONS:
        if not _is_number(record[column]) or not 0.0 <= float(record[column]) <= 1.0:
            errors.append(f"invalid:proportion:{column}")
    if _is_number(record.get("coverage_disparity")) and abs(float(record["coverage_disparity"]) - abs(float(record["coverage_minority"]) - float(record["coverage_majority"]))) > 1e-12:
        errors.append("invalid:coverage_disparity")
    if abs((float(record["singleton_rate"]) + float(record["empty_rate"]) + float(record["doubleton_rate"])) - 1.0) > 1e-12:
        errors.append("invalid:binary_set_geometry_rate_sum")
    errors.extend(_coverage_errors(record, "overall", int(record["n_test"])))
    errors.extend(_coverage_errors(record, "minority", int(record["n_test_minority"])))
    errors.extend(_coverage_errors(record, "majority", int(record["n_test_majority"])))
    if record["cp_method"] == "global_split_cp":
        if not _is_number(record["q_global"]) or record["q_minority"] is not None or record["q_majority"] is not None or record["threshold_gap"] is not None or record["threshold_sum"] is not None or not isinstance(record["rank_global"], int) or record["rank_minority"] is not None or record["rank_majority"] is not None:
            errors.append("invalid:global_threshold_fields")
    elif record["cp_method"] == "class_conditional_cp":
        if record["q_global"] is not None or not all(_is_number(record[field]) for field in ("q_minority", "q_majority", "threshold_gap", "threshold_sum")) or record["rank_global"] is not None or not isinstance(record["rank_minority"], int) or not isinstance(record["rank_majority"], int):
            errors.append("invalid:class_conditional_threshold_fields")
        elif abs(float(record["threshold_gap"]) - abs(float(record["q_minority"]) - float(record["q_majority"]))) > 1e-12 or abs(float(record["threshold_sum"]) - (float(record["q_minority"]) + float(record["q_majority"]))) > 1e-12:
            errors.append("invalid:threshold_geometry_fields")
    for column in ("subset_hash", "split_hash", "dataset_hash", "environment_hash", "model_hash", "prediction_cache_hash", "label_mapping_hash", "created_utc"):
        if not isinstance(record[column], str) or not record[column]:
            errors.append(f"invalid:provenance:{column}")
    return errors


def unique_key(record: dict[str, Any]) -> tuple[Any, ...]:
    errors = validate_results_record(record)
    if errors:
        raise ValueError(f"Invalid results_long record: {errors}")
    return tuple(record[column] for column in UNIQUE_KEY)


def validate_results_records(records: Iterable[dict[str, Any]]) -> list[str]:
    """Validate every row and reject a duplicated protocol unique key."""
    seen: set[tuple[Any, ...]] = set()
    errors: list[str] = []
    for index, record in enumerate(records):
        row_errors = validate_results_record(record)
        errors.extend(f"row:{index}:{error}" for error in row_errors)
        if not row_errors:
            key = tuple(record[column] for column in UNIQUE_KEY)
            if key in seen:
                errors.append(f"row:{index}:duplicate:unique_key")
            seen.add(key)
    return errors

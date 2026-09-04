"""Base predictive and Stage 06 conformal metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

WILSON_Z_95 = 1.959963984540054


def binary_predictive_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(y, dtype=np.int8)
    proba = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or proba.shape != (len(labels), 2):
        raise ValueError("Binary metrics require n labels and an n×2 probability array.")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("AUROC/AUPRC require both protocol classes.")
    if not np.isfinite(proba).all() or (proba < 0).any() or (proba > 1).any() or not np.allclose(proba.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError("Metrics require valid aligned class probabilities.")
    return {"auroc": float(roc_auc_score(labels, proba[:, 1])), "auprc": float(average_precision_score(labels, proba[:, 1])), "n_rows": int(len(labels)), "positive_count": int(labels.sum())}


def wilson_interval(covered: int, total: int, z: float = WILSON_Z_95) -> tuple[float, float]:
    """Two-sided Wilson score interval without continuity correction."""
    if not isinstance(covered, (int, np.integer)) or not isinstance(total, (int, np.integer)) or not 0 <= covered <= total or total <= 0:
        raise ValueError("Wilson interval requires integer 0 <= covered <= total and total > 0.")
    p = covered / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    # Floating arithmetic can place an exact endpoint (for example 783/783)
    # infinitesimally outside [0, 1]; Wilson intervals are probabilities.
    return float(max(0.0, center - radius)), float(min(1.0, center + radius))


def global_coverage_sanity(covered: int, total: int, *, nominal: float = 0.9) -> dict[str, float | bool | int]:
    """Report, but do not over-interpret, a Global-CP nominal-coverage check."""
    if not 0.0 < nominal < 1.0:
        raise ValueError("Nominal coverage must be strictly between zero and one.")
    lower, upper = wilson_interval(covered, total)
    observed = covered / total
    return {
        "covered_count": covered, "total": total, "observed_coverage": observed,
        "nominal_coverage": nominal, "absolute_deviation": abs(observed - nominal),
        "wilson_low": lower, "wilson_high": upper, "nominal_within_wilson": lower <= nominal <= upper,
    }


def _coverage_fields(name: str, covered: np.ndarray) -> dict[str, float | int]:
    total = int(len(covered))
    if total == 0:
        raise ValueError(f"Coverage subgroup {name} has zero test rows.")
    count = int(covered.sum())
    lower, upper = wilson_interval(count, total)
    return {f"covered_count_{name}": count, f"coverage_{name}": count / total, f"coverage_{name}_wilson_low": lower, f"coverage_{name}_wilson_high": upper}


def binary_cp_metrics(labels: np.ndarray, included: np.ndarray, *, minority_label: int) -> dict[str, float | int]:
    """Coverage/count/Wilson and binary prediction-set efficiency metrics."""
    y = np.asarray(labels, dtype=np.int8)
    sets = np.asarray(included, dtype=bool)
    if y.ndim != 1 or sets.shape != (len(y), 2) or set(y.tolist()).difference({0, 1}) or minority_label not in {0, 1}:
        raise ValueError("Binary CP metrics require aligned labels, 2-label sets, and explicit minority identity.")
    covered = sets[np.arange(len(y)), y]
    minority, majority = y == minority_label, y != minority_label
    result: dict[str, float | int] = {}
    result.update(_coverage_fields("overall", covered))
    result.update(_coverage_fields("minority", covered[minority]))
    result.update(_coverage_fields("majority", covered[majority]))
    sizes = sets.sum(axis=1)
    result.update({
        "coverage_disparity": abs(float(result["coverage_minority"]) - float(result["coverage_majority"])),
        "singleton_rate": float((sizes == 1).mean()), "average_set_size": float(sizes.mean()),
        "empty_rate": float((sizes == 0).mean()), "doubleton_rate": float((sizes == 2).mean()),
        "n_test": int(len(y)), "n_test_minority": int(minority.sum()), "n_test_majority": int(majority.sum()),
    })
    if abs(result["empty_rate"] + result["singleton_rate"] + result["doubleton_rate"] - 1.0) > 1e-12:
        raise AssertionError("Binary prediction-set decomposition does not sum to one.")
    return result

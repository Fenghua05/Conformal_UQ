"""Protocol-locked binary split-conformal prediction primitives for Stage 06."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable, Mapping, Sequence

import numpy as np

from .identity import derive_seed
from .metrics import binary_cp_metrics

ALPHA = 0.1
M_MAJORITY = 200
M_MINORITY = (10, 20, 50, 100)
CP_METHODS = ("global_split_cp", "class_conditional_cp")


@dataclass(frozen=True)
class ExactQuantile:
    """A finite-sample conformal order statistic and its one-based rank."""

    n: int
    rank: int
    threshold: float


@dataclass(frozen=True)
class CalibrationSubsets:
    """One fixed majority sample and nested minority prefixes for a seed."""

    majority_ids: tuple[str, ...]
    minority_ids_by_m: Mapping[int, tuple[str, ...]]
    majority_label: int
    minority_label: int

    def subset_ids(self, m_minority: int) -> tuple[str, ...]:
        if m_minority not in self.minority_ids_by_m:
            raise ValueError(f"Unsupported minority calibration size: {m_minority}.")
        return self.majority_ids + self.minority_ids_by_m[m_minority]

    def subset_hash(self, m_minority: int) -> str:
        """Hash membership canonically, so the hash represents sample identity."""
        payload = {
            "majority_ids": sorted(self.majority_ids),
            "minority_ids": sorted(self.minority_ids_by_m[m_minority]),
            "m_majority": len(self.majority_ids),
            "m_minority": m_minority,
            "majority_label": self.majority_label,
            "minority_label": self.minority_label,
        }
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_binary_arrays(labels: Sequence[int] | np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int8)
    proba = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or proba.shape != (len(y), 2):
        raise ValueError("Expected n binary labels and an aligned n×2 probability array.")
    if set(y.tolist()).difference({0, 1}):
        raise ValueError("Only protocol binary labels 0 and 1 are valid.")
    if not np.isfinite(proba).all() or (proba < 0).any() or (proba > 1).any() or not np.allclose(proba.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError("Probabilities must be finite, aligned [0,1] rows summing to one.")
    return y, proba


def exact_conformal_quantile(scores: Iterable[float], alpha: float = ALPHA) -> ExactQuantile:
    """Return the required non-interpolated finite-sample order statistic."""
    values = np.asarray(list(scores), dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Conformal scores must be a non-empty finite one-dimensional array.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between zero and one.")
    rank = int(math.ceil((len(values) + 1) * (1.0 - alpha)))
    sorted_scores = np.sort(values)
    threshold = float(sorted_scores[rank - 1]) if rank <= len(values) else math.inf
    return ExactQuantile(n=len(values), rank=rank, threshold=threshold)


def nonconformity_scores(labels: Sequence[int] | np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Compute the only permitted score: ``1 - p_y(x)`` in [0,1] order."""
    y, proba = _validate_binary_arrays(labels, probabilities)
    return 1.0 - proba[np.arange(len(y)), y]


def select_nested_calibration_subsets(
    sample_ids: Sequence[str], labels: Sequence[int] | np.ndarray, *, protocol_version: str,
    dataset_id: str, base_seed: int, minority_label: int, m_majority: int = M_MAJORITY,
    m_minority: Sequence[int] = M_MINORITY,
) -> CalibrationSubsets:
    """Sample a fixed majority set plus deterministic nested minority prefixes."""
    ids = tuple(str(value) for value in sample_ids)
    y = np.asarray(labels, dtype=np.int8)
    if len(ids) != len(y) or len(ids) != len(set(ids)) or not ids:
        raise ValueError("Calibration sample IDs must be non-empty, unique, and label-aligned.")
    if minority_label not in {0, 1} or set(y.tolist()).difference({0, 1}):
        raise ValueError("minority_label and all calibration labels must be binary 0/1.")
    sizes = tuple(int(value) for value in m_minority)
    if sizes != tuple(sorted(set(sizes))) or not sizes or sizes[0] <= 0:
        raise ValueError("Minority sizes must be strictly increasing positive values.")
    majority_label = 1 - minority_label
    majority_indices, minority_indices = np.flatnonzero(y == majority_label), np.flatnonzero(y == minority_label)
    if len(majority_indices) < m_majority or len(minority_indices) < sizes[-1]:
        raise ValueError("Calibration pool cannot meet the fixed class-specific sample requirements.")
    majority_seed = derive_seed(protocol_version, dataset_id, base_seed, "calibration_majority_subset")[1]
    minority_seed = derive_seed(protocol_version, dataset_id, base_seed, "calibration_minority_nested_subset")[1]
    majority_order = np.random.default_rng(majority_seed).permutation(majority_indices)
    minority_order = np.random.default_rng(minority_seed).permutation(minority_indices)
    majority_ids = tuple(ids[index] for index in majority_order[:m_majority])
    nested = {size: tuple(ids[index] for index in minority_order[:size]) for size in sizes}
    return CalibrationSubsets(majority_ids, nested, majority_label, minority_label)


def prediction_sets(probabilities: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    """Return boolean label inclusions under the required ``score <= threshold``."""
    proba = np.asarray(probabilities, dtype=np.float64)
    _, proba = _validate_binary_arrays(np.zeros(len(proba), dtype=np.int8), proba)
    q = np.asarray(thresholds, dtype=np.float64)
    if q.shape != (2,) or np.isnan(q).any() or (q < 0).any():
        raise ValueError("Binary inclusion thresholds must be two non-negative values.")
    return (1.0 - proba) <= q[None, :]


def binary_geometry_categories(probabilities: np.ndarray, q0: float, q1: float) -> np.ndarray:
    """Classify the explicit empty/doubleton intervals in binary threshold geometry."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim not in {1, 2} or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or (p.ndim == 2 and p.shape[1] != 2):
        raise ValueError("Probabilities must be finite binary values in [0,1].")
    if q0 < 0 or q1 < 0:
        raise ValueError("Thresholds must be non-negative.")
    if p.ndim == 2:
        # Cache rows are permitted a small row-sum tolerance. Use their two
        # stored columns directly so geometry agrees with the actual set test,
        # rather than assuming p0 is exactly 1-p1.
        included0, included1 = (1.0 - p[:, 0]) <= q0, (1.0 - p[:, 1]) <= q1
        return np.where(included0 & included1, "doubleton", np.where(~included0 & ~included1, "empty", "singleton"))
    categories = np.full(len(p), "singleton", dtype="U9")
    if q0 + q1 < 1.0:
        categories[(q0 < p) & (p < 1.0 - q1)] = "empty"
    else:
        categories[(1.0 - q1 <= p) & (p <= q0)] = "doubleton"
    return categories


def evaluate_split_cp(
    calibration_ids: Sequence[str], calibration_labels: Sequence[int] | np.ndarray,
    calibration_probabilities: np.ndarray, test_labels: Sequence[int] | np.ndarray,
    test_probabilities: np.ndarray, subsets: CalibrationSubsets, *, m_minority: int,
    cp_method: str, alpha: float = ALPHA,
) -> dict[str, float | int | str | None]:
    """Evaluate one Global or Class-Conditional CP cell from an existing cache."""
    if cp_method not in CP_METHODS:
        raise ValueError(f"Unsupported CP method: {cp_method}.")
    cal_y, cal_proba = _validate_binary_arrays(calibration_labels, calibration_probabilities)
    test_y, test_proba = _validate_binary_arrays(test_labels, test_probabilities)
    ids = tuple(str(value) for value in calibration_ids)
    if len(ids) != len(cal_y) or len(set(ids)) != len(ids):
        raise ValueError("Calibration IDs must be unique and aligned to labels/probabilities.")
    positions = {sample_id: index for index, sample_id in enumerate(ids)}
    selected_ids = subsets.subset_ids(m_minority)
    if any(sample_id not in positions for sample_id in selected_ids):
        raise ValueError("Calibration subset contains an ID absent from the cache.")
    selected = np.asarray([positions[sample_id] for sample_id in selected_ids], dtype=int)
    selected_y, selected_proba = cal_y[selected], cal_proba[selected]
    selected_scores = nonconformity_scores(selected_y, selected_proba)
    if cp_method == "global_split_cp":
        global_q = exact_conformal_quantile(selected_scores, alpha)
        thresholds = (global_q.threshold, global_q.threshold)
        result: dict[str, float | int | str | None] = {
            "q_global": global_q.threshold, "q_minority": None, "q_majority": None, "threshold_gap": None, "threshold_sum": None,
            "rank_global": global_q.rank, "rank_minority": None, "rank_majority": None,
        }
    else:
        q_minority = exact_conformal_quantile(selected_scores[selected_y == subsets.minority_label], alpha)
        q_majority = exact_conformal_quantile(selected_scores[selected_y == subsets.majority_label], alpha)
        threshold_values = [0.0, 0.0]
        threshold_values[subsets.minority_label] = q_minority.threshold
        threshold_values[subsets.majority_label] = q_majority.threshold
        thresholds = tuple(threshold_values)
        result = {
            "q_global": None, "q_minority": q_minority.threshold, "q_majority": q_majority.threshold,
            "threshold_gap": abs(q_minority.threshold - q_majority.threshold), "threshold_sum": q_minority.threshold + q_majority.threshold,
            "rank_global": None, "rank_minority": q_minority.rank, "rank_majority": q_majority.rank,
        }
    included = prediction_sets(test_proba, thresholds)
    result.update(binary_cp_metrics(test_y, included, minority_label=subsets.minority_label))
    result.update({
        "cp_method": cp_method, "m_minority": m_minority, "m_majority": len(subsets.majority_ids),
        "n_cal_total": len(selected), "n_cal_minority": int((selected_y == subsets.minority_label).sum()),
        "n_cal_majority": int((selected_y == subsets.majority_label).sum()), "subset_hash": subsets.subset_hash(m_minority),
    })
    if cp_method == "class_conditional_cp":
        actual = np.where(included.sum(axis=1) == 0, "empty", np.where(included.sum(axis=1) == 2, "doubleton", "singleton"))
        expected = binary_geometry_categories(test_proba, thresholds[0], thresholds[1])
        if not np.array_equal(actual, expected):
            raise AssertionError("Binary threshold geometry disagrees with actual prediction sets.")
    return result

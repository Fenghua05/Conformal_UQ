"""Frozen Stage 05 predictive pipelines; no calibration/test fitting or tuning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from .data import BinaryTable
from .preprocessing import TrainOnlyPreprocessor
from .split import StratifiedSplit

CLASS_LABELS = (0, 1)


@dataclass(frozen=True)
class PipelinePrediction:
    model_name: str
    class_labels: tuple[int, int]
    calibration_probabilities: np.ndarray
    test_probabilities: np.ndarray
    calibration_y: np.ndarray
    test_y: np.ndarray
    preprocessor_report: dict[str, Any]
    model_hash: str
    estimator_version: str


def model_spec(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    try:
        spec = config["models"][model_name]
    except KeyError as exc:
        raise KeyError(f"Unknown configured model: {model_name}") from exc
    return dict(spec)


def assert_model_execution_allowed(config: dict[str, Any], model_name: str) -> None:
    if model_name == "tabpfn":
        raise RuntimeError("TabPFN execution is pending explicit package/checkpoint/device authorization.")
    if model_name not in {"logistic_regression", "xgboost"}:
        raise ValueError(f"Unsupported Stage 05 model: {model_name}")


def _make_estimator(model_name: str, hyperparameters: dict[str, Any], derived_seed: int) -> Any:
    if model_name == "logistic_regression":
        expected = {"C", "penalty", "solver", "max_iter", "class_weight"}
        if set(hyperparameters) != expected:
            raise ValueError("Logistic Regression hyperparameters differ from the frozen configuration.")
        return LogisticRegression(**hyperparameters, random_state=derived_seed)
    if model_name == "xgboost":
        from xgboost import XGBClassifier
        params = dict(hyperparameters)
        early_stopping = params.pop("early_stopping", None)
        if early_stopping is not False:
            raise ValueError("XGBoost early stopping is prohibited by the frozen protocol.")
        required = {"objective", "eval_metric", "n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight", "reg_lambda", "reg_alpha", "scale_pos_weight", "tree_method", "n_jobs"}
        if set(params) != required:
            raise ValueError("XGBoost hyperparameters differ from the frozen configuration.")
        return XGBClassifier(**params, random_state=derived_seed, verbosity=0)
    raise ValueError(f"Stage 05 has no authorized estimator for {model_name!r}.")


def _align_binary_probabilities(estimator: Any, probabilities: Any) -> np.ndarray:
    raw = np.asarray(probabilities, dtype=np.float64)
    classes = tuple(int(value) for value in np.asarray(estimator.classes_).tolist())
    if raw.ndim != 2 or raw.shape[1] != len(classes) or set(classes) != set(CLASS_LABELS):
        raise ValueError(f"Estimator class/probability columns do not represent protocol labels [0, 1]: {classes}.")
    aligned = np.empty((raw.shape[0], 2), dtype=np.float64)
    for column, label in enumerate(CLASS_LABELS):
        aligned[:, column] = raw[:, classes.index(label)]
    if not np.isfinite(aligned).all() or (aligned < 0).any() or (aligned > 1).any() or not np.allclose(aligned.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError("Estimator emitted invalid class probabilities.")
    return aligned


def _fingerprint_fitted_estimator(estimator: Any, model_name: str, derived_seed: int, preprocessor_report: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps({"model_name": model_name, "derived_seed": derived_seed, "preprocessor": preprocessor_report}, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(np.asarray(estimator.classes_).tobytes())
    if model_name == "logistic_regression":
        digest.update(np.asarray(estimator.coef_, dtype=np.float64).tobytes())
        digest.update(np.asarray(estimator.intercept_, dtype=np.float64).tobytes())
        digest.update(str(estimator.n_iter_).encode("ascii"))
    elif model_name == "xgboost":
        digest.update(bytes(estimator.get_booster().save_raw(raw_format="json")))
    else:
        raise ValueError(model_name)
    return digest.hexdigest()


def fit_predict_locked_pipeline(table: BinaryTable, split: StratifiedSplit, model_name: str, hyperparameters: dict[str, Any], derived_seed: int) -> PipelinePrediction:
    """Fit exactly once on locked train IDs and predict held-out fixed ID orders."""
    if split.dataset_id != table.dataset_id:
        raise ValueError("Table/split dataset IDs do not match.")
    if model_name not in {"logistic_regression", "xgboost"}:
        raise RuntimeError("Only LR and XGBoost are authorized in the local Stage 05 runner.")
    if not isinstance(derived_seed, int) or not 0 <= derived_seed <= 0xFFFFFFFF:
        raise ValueError("Model random seed must be an unsigned 32-bit derived seed.")
    processor = TrainOnlyPreprocessor(model_name).fit(table, split)
    train_x = processor.transform(table, split.ids.train, partition="train")
    calibration_x = processor.transform(table, split.ids.calibration_pool, partition="calibration_pool")
    test_x = processor.transform(table, split.ids.test, partition="test")
    train_y = table.subset_labels(split.ids.train).to_numpy(dtype=np.int8, copy=True)
    calibration_y = table.subset_labels(split.ids.calibration_pool).to_numpy(dtype=np.int8, copy=True)
    test_y = table.subset_labels(split.ids.test).to_numpy(dtype=np.int8, copy=True)
    if set(train_y.tolist()) != set(CLASS_LABELS):
        raise ValueError("Locked train split lacks a protocol class.")
    estimator = _make_estimator(model_name, hyperparameters, derived_seed)
    estimator.fit(train_x, train_y)
    report = processor.report()
    return PipelinePrediction(
        model_name=model_name, class_labels=CLASS_LABELS,
        calibration_probabilities=_align_binary_probabilities(estimator, estimator.predict_proba(calibration_x)),
        test_probabilities=_align_binary_probabilities(estimator, estimator.predict_proba(test_x)),
        calibration_y=calibration_y, test_y=test_y, preprocessor_report=report,
        model_hash=_fingerprint_fitted_estimator(estimator, model_name, derived_seed, report),
        estimator_version=f"{estimator.__class__.__module__}.{estimator.__class__.__name__}",
    )

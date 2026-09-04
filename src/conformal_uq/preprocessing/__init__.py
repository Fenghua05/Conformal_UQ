"""Train-only preprocessing with explicit leakage guards for Stage 04."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..data import BinaryTable
from ..identity import sha256_text
from ..split import StratifiedSplit


def assert_train_only_fit(fit_partition: str) -> None:
    if fit_partition != "train":
        raise ValueError("Preprocessing may fit only on the train partition.")


def _ids_hash(ids: Sequence[str]) -> str:
    return sha256_text(json.dumps(sorted(ids), separators=(",", ":")))


def _normalise_categorical_missing_values(features: pd.DataFrame) -> pd.DataFrame:
    """Adapt pandas missing sentinels to sklearn's object-column imputer input."""
    normalised = features.copy()
    for column in normalised.columns:
        if not pd.api.types.is_numeric_dtype(normalised[column]):
            values = normalised[column].astype(object)
            normalised[column] = values.where(values.notna(), np.nan)
    return normalised


@dataclass
class TrainOnlyPreprocessor:
    """A model-appropriate transformer that can be fit only through a split's train IDs."""

    model_name: str
    transformer: ColumnTransformer | None = None
    fit_audit: list[dict[str, Any]] = field(default_factory=list)
    train_ids_hash: str | None = None
    feature_schema: list[dict[str, Any]] = field(default_factory=list)
    transformed_feature_names: list[str] = field(default_factory=list)

    def _build_transformer(self, features: pd.DataFrame) -> ColumnTransformer:
        numeric = [str(column) for column in features.columns if pd.api.types.is_numeric_dtype(features[column])]
        categorical = [str(column) for column in features.columns if str(column) not in numeric]
        if self.model_name == "logistic_regression":
            numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        elif self.model_name == "xgboost":
            numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        else:
            raise ValueError("Stage 04 preprocessing supports only logistic_regression and xgboost; TabPFN remains authorization-gated.")
        numeric_pipeline = Pipeline(numeric_steps)
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ])
        return ColumnTransformer(
            transformers=[("numeric", numeric_pipeline, numeric), ("categorical", categorical_pipeline, categorical)],
            remainder="drop", sparse_threshold=1.0,
        )

    def fit(self, table: BinaryTable, split: StratifiedSplit, *, fit_ids: Sequence[str] | None = None, fit_partition: str = "train") -> "TrainOnlyPreprocessor":
        assert_train_only_fit(fit_partition)
        expected = tuple(split.ids.train)
        provided = tuple(fit_ids) if fit_ids is not None else expected
        if set(provided) != set(expected) or len(provided) != len(expected):
            raise ValueError("Transformer fit IDs must be exactly the split train IDs.")
        held_out = set(split.ids.calibration_pool) | set(split.ids.test)
        if set(provided) & held_out:
            raise ValueError("Transformer fit IDs intersect calibration or test IDs.")
        train_features = _normalise_categorical_missing_values(table.subset_features(expected))
        train_labels = table.subset_labels(expected)
        self.transformer = self._build_transformer(train_features)
        # y is intentionally omitted: imputers/scalers/encoders must not access labels.
        self.transformer.fit(train_features)
        self.train_ids_hash = _ids_hash(expected)
        self.feature_schema = [
            {"name": str(column), "dtype": str(train_features[column].dtype), "semantic_type": "numeric" if pd.api.types.is_numeric_dtype(train_features[column]) else "categorical"}
            for column in train_features.columns
        ]
        self.transformed_feature_names = [str(value) for value in self.transformer.get_feature_names_out()]
        base = {"fit_scope": "train", "fit_ids_hash": self.train_ids_hash, "labels_passed_to_fit": False, "model_name": self.model_name}
        self.fit_audit = [
            {**base, "transformer": "numeric.imputer", "strategy": "median"},
            *([{**base, "transformer": "numeric.scaler", "strategy": "standard"}] if self.model_name == "logistic_regression" else []),
            {**base, "transformer": "categorical.imputer", "strategy": "most_frequent"},
            {**base, "transformer": "categorical.encoder", "strategy": "onehot_handle_unknown_ignore"},
            {**base, "transformer": "column_transformer", "strategy": "semantic_columns"},
        ]
        # Guard retained as an executable invariant: labels may be read only to
        # establish split identity, never supplied to sklearn's fit call.
        if len(train_labels) != len(expected):
            raise AssertionError("Train label identity changed during preprocessing setup.")
        return self

    def _check_fitted(self) -> ColumnTransformer:
        if self.transformer is None or self.train_ids_hash is None:
            raise RuntimeError("Preprocessor must be fit on train IDs before transform.")
        return self.transformer

    def transform(self, table: BinaryTable, ids: Sequence[str], *, partition: str) -> Any:
        if partition not in {"train", "calibration_pool", "test"}:
            raise ValueError("partition must be train, calibration_pool, or test.")
        return self.transform_features(table.subset_features(ids), ids, partition=partition)

    def transform_features(self, features: pd.DataFrame, ids: Sequence[str], *, partition: str) -> Any:
        transformer = self._check_fitted()
        if len(features) != len(ids):
            raise ValueError("Transform features and IDs must have equal row counts.")
        if partition != "train" and _ids_hash(ids) == self.train_ids_hash:
            raise ValueError("Train IDs cannot be relabeled as held-out data.")
        return transformer.transform(_normalise_categorical_missing_values(features))

    def report(self) -> dict[str, Any]:
        self._check_fitted()
        return {
            "model_name": self.model_name, "train_ids_hash": self.train_ids_hash,
            "fit_audit": self.fit_audit, "feature_schema": self.feature_schema,
            "transformed_feature_count": len(self.transformed_feature_names),
            "transformed_feature_names": self.transformed_feature_names,
        }

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OCCUPANCY_BANDS = ("0-2", "3-10", "11-20", "21-35", "36-45")
OCCUPANCY_BAND_TO_INDEX = {label: index for index, label in enumerate(OCCUPANCY_BANDS)}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Pipeline
    target_mode: str = "direct"


def occupancy_band(value: object) -> str | float:
    if value is None or pd.isna(value):
        return np.nan

    numeric = float(value)
    if numeric <= 2:
        return "0-2"
    if numeric <= 10:
        return "3-10"
    if numeric <= 20:
        return "11-20"
    if numeric <= 35:
        return "21-35"
    return "36-45"


def build_regression_model_specs(*, random_state: int = 42) -> dict[str, ModelSpec]:
    return {
        "ridge_residual": ModelSpec(
            name="ridge_residual",
            target_mode="residual",
            estimator=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=2.0)),
                ]
            ),
        ),
        "random_forest_residual": ModelSpec(
            name="random_forest_residual",
            target_mode="residual",
            estimator=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=300,
                            max_depth=6,
                            min_samples_leaf=2,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
        "gradient_boosting_residual": ModelSpec(
            name="gradient_boosting_residual",
            target_mode="residual",
            estimator=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        GradientBoostingRegressor(
                            n_estimators=200,
                            learning_rate=0.05,
                            max_depth=2,
                            min_samples_leaf=2,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
    }


def build_classification_model_specs(*, random_state: int = 42) -> dict[str, ModelSpec]:
    return {
        "logistic_band_classifier": ModelSpec(
            name="logistic_band_classifier",
            estimator=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=4000,
                            class_weight="balanced",
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
        "random_forest_band_classifier": ModelSpec(
            name="random_forest_band_classifier",
            estimator=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=300,
                            max_depth=6,
                            min_samples_leaf=2,
                            class_weight="balanced_subsample",
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
        "gradient_boosting_band_classifier": ModelSpec(
            name="gradient_boosting_band_classifier",
            estimator=Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        GradientBoostingClassifier(
                            n_estimators=200,
                            learning_rate=0.05,
                            max_depth=2,
                            min_samples_leaf=2,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        ),
    }


def select_usable_feature_columns(
    frame: pd.DataFrame,
    candidate_feature_columns: list[str],
) -> list[str]:
    return [
        column
        for column in candidate_feature_columns
        if column in frame.columns and frame[column].notna().any()
    ]


def fit_weighted_regression_model(
    train_df: pd.DataFrame,
    *,
    feature_columns: list[str],
    model_spec: ModelSpec,
    target_col: str = "occ_mid",
    baseline_col: str = "occ_physics_est_smooth",
    min_training_rows: int = 3,
) -> dict[str, object]:
    if len(train_df) < min_training_rows:
        raise ValueError(
            f"Need at least {min_training_rows} rows to fit {model_spec.name}; got {len(train_df)}."
        )

    usable_feature_columns = select_usable_feature_columns(train_df, feature_columns)
    if not usable_feature_columns:
        raise ValueError(f"No usable feature columns were available for {model_spec.name}.")

    y_train = pd.to_numeric(train_df[target_col], errors="coerce")
    if model_spec.target_mode == "residual":
        y_train = y_train - pd.to_numeric(train_df[baseline_col], errors="coerce")

    estimator = clone(model_spec.estimator)
    sample_weight = (
        pd.to_numeric(train_df.get("confidence_weight", pd.Series(1.0, index=train_df.index)), errors="coerce")
        .fillna(1.0)
        .to_numpy(dtype=float)
    )

    estimator.fit(
        train_df[usable_feature_columns],
        y_train,
        model__sample_weight=sample_weight,
    )

    return {
        "spec": model_spec,
        "kind": "pipeline",
        "estimator": estimator,
        "feature_columns": usable_feature_columns,
    }


def predict_weighted_regression_model(
    fitted_model: dict[str, object],
    feature_df: pd.DataFrame,
    *,
    baseline_col: str = "occ_physics_est_smooth",
    min_occupancy: float = 0.0,
    max_occupancy: float = 45.0,
) -> pd.Series:
    estimator: Pipeline = fitted_model["estimator"]  # type: ignore[assignment]
    feature_columns: list[str] = fitted_model["feature_columns"]  # type: ignore[assignment]
    spec: ModelSpec = fitted_model["spec"]  # type: ignore[assignment]

    prediction = estimator.predict(feature_df[feature_columns])
    prediction = pd.Series(prediction, index=feature_df.index, dtype=float)

    if spec.target_mode == "residual":
        baseline = pd.to_numeric(feature_df[baseline_col], errors="coerce").fillna(0.0)
        prediction = baseline + prediction

    return prediction.clip(lower=min_occupancy, upper=max_occupancy)


def fit_weighted_classifier_model(
    train_df: pd.DataFrame,
    *,
    feature_columns: list[str],
    model_spec: ModelSpec,
    target_col: str = "occ_band",
) -> dict[str, object]:
    usable_feature_columns = select_usable_feature_columns(train_df, feature_columns)
    if not usable_feature_columns:
        raise ValueError(f"No usable feature columns were available for {model_spec.name}.")

    y_train = train_df[target_col].astype("string").fillna(pd.NA)
    y_train = y_train.dropna()
    if y_train.empty:
        raise ValueError(f"No classification labels were available for {model_spec.name}.")

    aligned_train = train_df.loc[y_train.index]
    unique_classes = sorted(y_train.unique().tolist(), key=lambda label: OCCUPANCY_BAND_TO_INDEX.get(label, 999))
    if len(unique_classes) == 1:
        return {
            "spec": model_spec,
            "kind": "constant",
            "label": unique_classes[0],
            "feature_columns": usable_feature_columns,
            "classes_": unique_classes,
        }

    estimator = clone(model_spec.estimator)
    sample_weight = (
        pd.to_numeric(
            aligned_train.get("confidence_weight", pd.Series(1.0, index=aligned_train.index)),
            errors="coerce",
        )
        .fillna(1.0)
        .to_numpy(dtype=float)
    )

    estimator.fit(
        aligned_train[usable_feature_columns],
        y_train.to_numpy(),
        model__sample_weight=sample_weight,
    )

    return {
        "spec": model_spec,
        "kind": "pipeline",
        "estimator": estimator,
        "feature_columns": usable_feature_columns,
        "classes_": list(estimator.named_steps["model"].classes_),
    }


def predict_weighted_classifier_model(
    fitted_model: dict[str, object],
    feature_df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    feature_columns: list[str] = fitted_model["feature_columns"]  # type: ignore[assignment]

    if fitted_model["kind"] == "constant":
        label = str(fitted_model["label"])
        prediction = pd.Series(label, index=feature_df.index, dtype="string")
        confidence = pd.Series(1.0, index=feature_df.index, dtype=float)
        return prediction, confidence

    estimator: Pipeline = fitted_model["estimator"]  # type: ignore[assignment]
    raw_prediction = estimator.predict(feature_df[feature_columns])
    prediction = pd.Series(raw_prediction, index=feature_df.index, dtype="string")

    if hasattr(estimator, "predict_proba"):
        probability = estimator.predict_proba(feature_df[feature_columns])
        confidence = pd.Series(probability.max(axis=1), index=feature_df.index, dtype=float)
    else:
        confidence = pd.Series(np.nan, index=feature_df.index, dtype=float)

    return prediction, confidence


def extract_model_importance(
    fitted_model: dict[str, object],
) -> pd.DataFrame:
    if fitted_model["kind"] != "pipeline":
        return pd.DataFrame(columns=["feature", "importance"])

    feature_columns: list[str] = fitted_model["feature_columns"]  # type: ignore[assignment]
    estimator: Pipeline = fitted_model["estimator"]  # type: ignore[assignment]
    model = estimator.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        importance = np.asarray(model.coef_, dtype=float)
        if importance.ndim > 1:
            importance = np.abs(importance).mean(axis=0)
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    frame = pd.DataFrame({"feature": feature_columns, "importance": importance})
    return frame.sort_values("importance", key=lambda values: values.abs(), ascending=False).reset_index(
        drop=True
    )


def bootstrap_anchor_resampled_predictions(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    *,
    feature_columns: list[str],
    model_spec: ModelSpec,
    anchor_ids: list[str],
    baseline_col: str = "occ_physics_est_smooth",
    target_col: str = "occ_mid",
    min_occupancy: float = 0.0,
    max_occupancy: float = 45.0,
    n_bootstraps: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    boot_predictions: list[np.ndarray] = []

    if not anchor_ids:
        raise ValueError("Bootstrap uncertainty requires at least one anchor id.")

    source_sets = parse_source_anchor_sets(train_df.get("source_anchor_ids", pd.Series("", index=train_df.index)))

    for _ in range(n_bootstraps):
        sampled_anchor_ids = rng.choice(anchor_ids, size=len(anchor_ids), replace=True)
        boot_train = _bootstrap_training_frame(
            train_df=train_df,
            source_sets=source_sets,
            sampled_anchor_ids=sampled_anchor_ids.tolist(),
        )
        if boot_train.empty:
            continue

        try:
            fitted = fit_weighted_regression_model(
                boot_train,
                feature_columns=feature_columns,
                model_spec=model_spec,
                target_col=target_col,
                baseline_col=baseline_col,
            )
        except ValueError:
            continue

        prediction = predict_weighted_regression_model(
            fitted,
            predict_df,
            baseline_col=baseline_col,
            min_occupancy=min_occupancy,
            max_occupancy=max_occupancy,
        )
        boot_predictions.append(prediction.to_numpy(dtype=float))

    if not boot_predictions:
        raise ValueError("Bootstrap uncertainty failed to produce any valid fitted models.")

    prediction_matrix = np.vstack(boot_predictions)
    lower = np.nanpercentile(prediction_matrix, 10, axis=0)
    median = np.nanmedian(prediction_matrix, axis=0)
    upper = np.nanpercentile(prediction_matrix, 90, axis=0)
    interval_width = upper - lower
    confidence_score = 1.0 - np.clip(interval_width / max(max_occupancy - min_occupancy, 1.0), 0.0, 1.0)

    return pd.DataFrame(
        {
            "prediction_median": median,
            "prediction_lower": lower,
            "prediction_upper": upper,
            "prediction_interval_width": interval_width,
            "prediction_confidence_score": confidence_score,
        },
        index=predict_df.index,
    )


def parse_source_anchor_sets(source_anchor_ids: pd.Series) -> pd.Series:
    return source_anchor_ids.fillna("").astype(str).apply(
        lambda value: tuple(sorted({part.strip() for part in value.split(",") if part.strip()}))
    )


def band_distance(predicted_band: object, true_band: object) -> float:
    if predicted_band is None or pd.isna(predicted_band) or true_band is None or pd.isna(true_band):
        return float("nan")
    return float(
        abs(
            OCCUPANCY_BAND_TO_INDEX[str(predicted_band)]
            - OCCUPANCY_BAND_TO_INDEX[str(true_band)]
        )
    )


def _bootstrap_training_frame(
    *,
    train_df: pd.DataFrame,
    source_sets: pd.Series,
    sampled_anchor_ids: list[str],
) -> pd.DataFrame:
    selected_frames: list[pd.DataFrame] = []
    for anchor_id, repeat_count in Counter(sampled_anchor_ids).items():
        mask = source_sets.apply(lambda anchor_set: anchor_id in anchor_set)
        anchor_rows = train_df.loc[mask]
        if anchor_rows.empty:
            continue
        for _ in range(repeat_count):
            selected_frames.append(anchor_rows.copy())

    if not selected_frames:
        return pd.DataFrame(columns=train_df.columns)
    return pd.concat(selected_frames, axis=0)

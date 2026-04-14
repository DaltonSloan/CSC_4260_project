from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_RESIDUAL_FEATURES = [
    "occ_physics_est_smooth",
    "co2_ppm",
    "co2_slope_ppm",
    "flow_cfm",
    "flow_pct_change",
    "humidity_pct",
    "humidity_delta_pct",
    "temp_f",
    "temp_delta_f",
]


def prepare_residual_training_frame(
    labeled_df: pd.DataFrame,
    *,
    target_col: str = "occ_mid",
    baseline_col: str = "occ_physics_est_smooth",
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Build the residual-learning frame from labeled pseudo-label rows."""

    feature_cols = feature_cols or DEFAULT_RESIDUAL_FEATURES
    train = labeled_df.loc[labeled_df["has_pseudo_label"]].copy()
    train["occ_formula_residual"] = train[target_col] - train[baseline_col]
    train["occ_formula_abs_error"] = train["occ_formula_residual"].abs()

    keep_cols = [
        "confidence",
        "confidence_weight",
        target_col,
        baseline_col,
        "occ_formula_residual",
        "occ_formula_abs_error",
    ] + feature_cols

    existing_cols = list(dict.fromkeys(column for column in keep_cols if column in train.columns))
    return train[existing_cols].copy()


def fit_residual_correction_model(
    labeled_df: pd.DataFrame,
    *,
    feature_cols: list[str] | None = None,
    target_col: str = "occ_mid",
    baseline_col: str = "occ_physics_est_smooth",
    alpha: float = 1.0,
    min_occupancy: float = 0.0,
    max_occupancy: float = 45.0,
) -> tuple[pd.DataFrame, pd.DataFrame, Pipeline, list[str]]:
    """
    Fit a simple residual model where ML predicts the error on top of the CO2 baseline.

    Returns:
    - augmented full dataframe
    - training subset with residual targets and fitted values
    - fitted sklearn pipeline
    - feature column list used by the model
    """

    feature_cols = feature_cols or DEFAULT_RESIDUAL_FEATURES
    training = prepare_residual_training_frame(
        labeled_df,
        target_col=target_col,
        baseline_col=baseline_col,
        feature_cols=feature_cols,
    )
    usable_feature_cols = [
        column
        for column in feature_cols
        if column in training.columns and training[column].notna().any()
    ]

    if len(training) < 3:
        raise ValueError(
            "Need at least 3 pseudo-labeled rows before fitting a residual correction model."
        )
    if not usable_feature_cols:
        raise ValueError("No residual-model features had usable non-null training values.")

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )

    sample_weight = training.get("confidence_weight", pd.Series(1.0, index=training.index)).fillna(1.0)
    model.fit(
        training[usable_feature_cols],
        training["occ_formula_residual"],
        ridge__sample_weight=sample_weight,
    )

    augmented = labeled_df.copy()
    augmented["occ_formula_residual"] = np.where(
        augmented["has_pseudo_label"],
        augmented[target_col] - augmented[baseline_col],
        np.nan,
    )
    augmented["occ_formula_abs_error"] = augmented["occ_formula_residual"].abs()
    augmented["occ_residual_pred"] = model.predict(augmented[usable_feature_cols])
    augmented["occ_hybrid_pred"] = (
        augmented[baseline_col] + augmented["occ_residual_pred"]
    ).clip(lower=min_occupancy, upper=max_occupancy)
    augmented["occ_hybrid_pred_rounded"] = augmented["occ_hybrid_pred"].round(1)

    training = training.copy()
    training["occ_residual_pred"] = model.predict(training[usable_feature_cols])
    training["occ_hybrid_fitted"] = (
        training[baseline_col] + training["occ_residual_pred"]
    ).clip(lower=min_occupancy, upper=max_occupancy)
    training["occ_hybrid_fitted_error"] = training[target_col] - training["occ_hybrid_fitted"]
    training["occ_hybrid_fitted_abs_error"] = training["occ_hybrid_fitted_error"].abs()

    return augmented, training, model, usable_feature_cols


def residual_model_coefficients(model: Pipeline, feature_cols: list[str]) -> pd.DataFrame:
    """Return standardized Ridge coefficients for quick interpretation."""

    ridge = model.named_steps["ridge"]
    return (
        pd.DataFrame({"feature": feature_cols, "coefficient": ridge.coef_})
        .sort_values("coefficient", key=lambda col: col.abs(), ascending=False)
        .reset_index(drop=True)
    )

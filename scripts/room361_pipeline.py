from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scripts.occupancy_features import (
    DEFAULT_RESAMPLE_MINUTES,
    candidate_temporal_feature_columns,
    engineer_temporal_features,
)
from scripts.occupancy_modeling import (
    OCCUPANCY_BANDS,
    ModelSpec,
    band_distance,
    bootstrap_anchor_resampled_predictions,
    build_classification_model_specs,
    build_regression_model_specs,
    extract_model_importance,
    fit_weighted_classifier_model,
    fit_weighted_regression_model,
    occupancy_band,
    parse_source_anchor_sets,
    predict_weighted_classifier_model,
    predict_weighted_regression_model,
)
from scripts.occupancy_pseudo_labels import (
    DEFAULT_CO2_GENERATION_LPS_PER_PERSON,
    DEFAULT_MAX_OCCUPANCY,
    DEFAULT_MIN_OCCUPANCY,
    DEFAULT_OUTDOOR_CO2_PPM,
    generate_pseudo_labels,
    load_room_fpb_timeseries,
    summarize_pseudo_labels,
)

sns.set_theme(style="whitegrid", context="talk")


@dataclass(frozen=True)
class Room361PipelineConfig:
    input_files: tuple[str, ...] = ("data/4_3To4_9.csv",)
    anchor_file: str = "data/room361_manual_anchors.csv"
    schedule_file: str | None = None
    output_dir: str = "reports/room361_pipeline"
    time_col: str = "dateTime"
    resample_rule: str = "5min"
    resample_minutes: int = DEFAULT_RESAMPLE_MINUTES
    min_occupancy: float = DEFAULT_MIN_OCCUPANCY
    max_occupancy: float = DEFAULT_MAX_OCCUPANCY
    outdoor_co2_ppm: float = DEFAULT_OUTDOOR_CO2_PPM
    co2_generation_lps_per_person: float = DEFAULT_CO2_GENERATION_LPS_PER_PERSON
    interpolation_max_gap_min: int = 120
    random_state: int = 42
    bootstrap_iterations: int = 150


def run_room361_pipeline(config: Room361PipelineConfig | None = None) -> dict[str, Any]:
    config = config or Room361PipelineConfig()

    output_dir = Path(config.output_dir)
    figure_dir = output_dir / "figures"
    model_dir = output_dir / "models"
    for directory in (output_dir, figure_dir, model_dir):
        directory.mkdir(parents=True, exist_ok=True)

    anchors_raw = pd.read_csv(config.anchor_file)
    schedule_df = _load_optional_schedule(config.schedule_file)

    sensor = load_room_fpb_timeseries(
        list(config.input_files),
        time_col=config.time_col,
        resample_rule=config.resample_rule,
        outdoor_co2_ppm=config.outdoor_co2_ppm,
        co2_generation_lps_per_person=config.co2_generation_lps_per_person,
        min_occupancy=config.min_occupancy,
        max_occupancy=config.max_occupancy,
    )
    labeled, label_rows, normalized_anchors = generate_pseudo_labels(
        sensor,
        anchors_raw,
        interpolation_max_gap_min=config.interpolation_max_gap_min,
        min_occupancy=config.min_occupancy,
        max_occupancy=config.max_occupancy,
    )
    pseudo_label_summary = summarize_pseudo_labels(labeled, normalized_anchors)

    feature_frame = engineer_temporal_features(
        labeled,
        resample_minutes=config.resample_minutes,
        schedule_df=schedule_df,
    )
    feature_frame["source_anchor_tuple"] = parse_source_anchor_sets(feature_frame.get("source_anchor_ids"))
    feature_frame["true_occ_band"] = feature_frame["occ_mid"].apply(occupancy_band)

    candidate_features = candidate_temporal_feature_columns(feature_frame)
    feature_manifest = pd.DataFrame({"feature": candidate_features}).sort_values("feature").reset_index(drop=True)

    regression_specs = build_regression_model_specs(random_state=config.random_state)
    classification_specs = build_classification_model_specs(random_state=config.random_state)

    forward_regression = _run_forward_chaining_regression_validation(
        feature_frame,
        normalized_anchors,
        candidate_features,
        regression_specs,
        min_occupancy=config.min_occupancy,
        max_occupancy=config.max_occupancy,
    )
    forward_classification = _run_forward_chaining_classification_validation(
        feature_frame,
        normalized_anchors,
        candidate_features,
        classification_specs,
    )

    anchor_regression = _run_anchor_window_regression_evaluation(
        feature_frame,
        normalized_anchors,
        candidate_features,
        regression_specs,
        min_occupancy=config.min_occupancy,
        max_occupancy=config.max_occupancy,
    )
    anchor_classification = _run_anchor_window_classification_evaluation(
        feature_frame,
        normalized_anchors,
        candidate_features,
        classification_specs,
    )

    regression_comparison = _summarize_regression_models(
        anchor_regression,
        forward_regression,
    )
    classification_comparison = _summarize_classification_models(
        anchor_classification,
        forward_classification,
    )

    best_regression_name = _choose_best_regression_model(regression_comparison)
    best_classifier_name = _choose_best_classifier_model(classification_comparison)
    best_regression_spec = regression_specs[best_regression_name]
    best_classifier_spec = classification_specs[best_classifier_name]

    best_anchor_regression = _run_anchor_window_regression_evaluation(
        feature_frame,
        normalized_anchors,
        candidate_features,
        {best_regression_name: best_regression_spec},
        min_occupancy=config.min_occupancy,
        max_occupancy=config.max_occupancy,
        bootstrap_iterations=config.bootstrap_iterations,
        bootstrap_intervals=True,
        random_state=config.random_state,
    )
    best_anchor_classification = _run_anchor_window_classification_evaluation(
        feature_frame,
        normalized_anchors,
        candidate_features,
        {best_classifier_name: best_classifier_spec},
    )

    labeled_training = feature_frame.loc[feature_frame["has_pseudo_label"]].copy()
    full_regression_predictions = _fit_full_regression_models(
        feature_frame,
        labeled_training,
        candidate_features,
        regression_specs,
        min_occupancy=config.min_occupancy,
        max_occupancy=config.max_occupancy,
    )
    full_classification_predictions = _fit_full_classification_models(
        feature_frame,
        labeled_training,
        candidate_features,
        classification_specs,
    )

    best_regression_bundle = full_regression_predictions["models"][best_regression_name]
    best_classifier_bundle = full_classification_predictions["models"][best_classifier_name]

    bootstrap_predictions = bootstrap_anchor_resampled_predictions(
        labeled_training,
        feature_frame,
        feature_columns=candidate_features,
        model_spec=best_regression_spec,
        anchor_ids=normalized_anchors["anchor_id"].tolist(),
        baseline_col="occ_physics_est_smooth",
        target_col="occ_mid",
        min_occupancy=config.min_occupancy,
        max_occupancy=config.max_occupancy,
        n_bootstraps=config.bootstrap_iterations,
        random_state=config.random_state,
    )

    predictions = feature_frame.copy()
    for model_name, prediction in full_regression_predictions["predictions"].items():
        predictions[f"{model_name}_pred"] = prediction
    for model_name, prediction in full_classification_predictions["predictions"].items():
        predictions[f"{model_name}_band"] = prediction["band"]
        predictions[f"{model_name}_band_confidence"] = prediction["confidence"]

    predictions["best_regression_model"] = best_regression_name
    predictions["best_occ_pred"] = full_regression_predictions["predictions"][best_regression_name]
    predictions = predictions.join(bootstrap_predictions)
    predictions["best_occ_lower"] = predictions["prediction_lower"]
    predictions["best_occ_upper"] = predictions["prediction_upper"]
    predictions["best_occ_confidence"] = predictions["prediction_confidence_score"]
    predictions["best_classifier_model"] = best_classifier_name
    predictions["predicted_occupancy_band"] = full_classification_predictions["predictions"][best_classifier_name][
        "band"
    ]
    predictions["predicted_occupancy_band_confidence"] = full_classification_predictions["predictions"][
        best_classifier_name
    ]["confidence"]

    anchor_summary = normalized_anchors.copy()
    anchor_summary["true_occ_band"] = anchor_summary["count"].apply(occupancy_band)
    pseudo_label_summary_df = pd.DataFrame(
        [{"metric": key, "value": json.dumps(value) if isinstance(value, dict) else value} for key, value in pseudo_label_summary.items()]
    )
    best_anchor_table = _merge_best_anchor_tables(
        best_anchor_regression,
        best_anchor_classification,
        model_name=best_regression_name,
        classifier_name=best_classifier_name,
    )
    regression_importance = extract_model_importance(best_regression_bundle)
    classification_importance = extract_model_importance(best_classifier_bundle)

    physics_diagnostic_table, physics_diagnostic_text = _build_physics_bias_diagnostics(
        predictions,
        normalized_anchors,
        best_anchor_table,
        outdoor_co2_ppm=config.outdoor_co2_ppm,
        co2_generation_lps_per_person=config.co2_generation_lps_per_person,
        max_occupancy=config.max_occupancy,
    )

    findings_summary = _build_findings_summary(
        config=config,
        pseudo_label_summary=pseudo_label_summary,
        regression_comparison=regression_comparison,
        classification_comparison=classification_comparison,
        best_anchor_table=best_anchor_table,
        physics_diagnostic_text=physics_diagnostic_text,
    )

    manifest = {
        "config": asdict(config),
        "best_regression_model": best_regression_name,
        "best_classifier_model": best_classifier_name,
        "candidate_feature_count": len(candidate_features),
        "pseudo_labeled_rows": int(feature_frame["has_pseudo_label"].sum()),
        "anchor_rows": int(len(normalized_anchors)),
    }

    _write_dataframe(feature_frame.reset_index(), output_dir / "feature_dataset.csv")
    _write_dataframe(predictions.reset_index(), output_dir / "predictions.csv")
    _write_dataframe(
        predictions.loc[predictions["has_pseudo_label"]].reset_index(),
        output_dir / "pseudo_labeled_dataset.csv",
    )
    _write_dataframe(label_rows, output_dir / "pseudo_label_rows.csv")
    _write_dataframe(anchor_summary, output_dir / "anchor_summary.csv")
    _write_dataframe(pseudo_label_summary_df, output_dir / "pseudo_label_coverage_summary.csv")
    _write_dataframe(feature_manifest, output_dir / "feature_manifest.csv")
    _write_dataframe(forward_regression, output_dir / "time_blocked_regression_validation.csv")
    _write_dataframe(forward_classification, output_dir / "time_blocked_classification_validation.csv")
    _write_dataframe(anchor_regression, output_dir / "anchor_window_regression_evaluation.csv")
    _write_dataframe(anchor_classification, output_dir / "anchor_window_classification_evaluation.csv")
    _write_dataframe(regression_comparison, output_dir / "model_comparison_regression.csv")
    _write_dataframe(classification_comparison, output_dir / "model_comparison_classification.csv")
    _write_dataframe(best_anchor_table, output_dir / "best_anchor_window_evaluation.csv")
    _write_dataframe(regression_importance, output_dir / "best_regression_feature_importance.csv")
    _write_dataframe(classification_importance, output_dir / "best_classification_feature_importance.csv")
    _write_dataframe(physics_diagnostic_table, output_dir / "physics_bias_diagnostic_table.csv")

    _write_markdown_table(anchor_summary, output_dir / "anchor_summary.md", title="Anchor Summary")
    _write_markdown_table(
        pseudo_label_summary_df,
        output_dir / "pseudo_label_coverage_summary.md",
        title="Pseudo-Label Coverage Summary",
        note="Counts below describe weakly supervised labels, not verified occupancy truth.",
    )
    _write_markdown_table(
        regression_comparison,
        output_dir / "model_comparison_regression.md",
        title="Regression Model Comparison",
        note=(
            "Anchor-window metrics are the reportable accuracy numbers. "
            "Forward-chaining metrics are pseudo-label diagnostics only."
        ),
    )
    _write_markdown_table(
        classification_comparison,
        output_dir / "model_comparison_classification.md",
        title="Classification Model Comparison",
        note=(
            "Band metrics are limited by sparse labels and low class diversity in some folds."
        ),
    )
    _write_markdown_table(
        best_anchor_table,
        output_dir / "best_anchor_window_evaluation.md",
        title="Best Model Anchor-Window Evaluation",
        note="Each row is a real manual anchor window held out from training for that fold.",
    )
    _write_markdown_table(
        regression_importance.head(15),
        output_dir / "best_regression_feature_importance.md",
        title="Top Regression Features",
    )
    _write_markdown_table(
        classification_importance.head(15),
        output_dir / "best_classification_feature_importance.md",
        title="Top Classification Features",
    )
    (output_dir / "findings_summary.md").write_text(findings_summary, encoding="utf-8")
    (output_dir / "physics_bias_diagnostics.md").write_text(physics_diagnostic_text, encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    joblib.dump(best_regression_bundle, model_dir / "best_regression_model.joblib")
    joblib.dump(best_classifier_bundle, model_dir / "best_classifier_model.joblib")

    _save_time_series_figure(
        predictions,
        normalized_anchors,
        figure_dir / "room361_co2_flow_occupancy_timeseries.png",
    )
    _save_anchor_comparison_figure(
        best_anchor_table,
        figure_dir / "room361_model_predictions_vs_anchors.png",
    )
    _save_residual_error_figure(
        best_anchor_table,
        figure_dir / "room361_anchor_residual_error.png",
    )
    _save_feature_importance_figure(
        regression_importance,
        figure_dir / "room361_best_regression_feature_importance.png",
        title="Best Regression Model Feature Importance",
        value_column="importance",
    )
    _save_feature_importance_figure(
        classification_importance,
        figure_dir / "room361_best_classification_feature_importance.png",
        title="Best Classification Model Feature Importance",
        value_column="importance",
    )
    _save_correlation_heatmap(
        feature_frame,
        candidate_features,
        figure_dir / "room361_engineered_feature_correlation_heatmap.png",
    )

    return {
        "config": config,
        "feature_frame": feature_frame,
        "predictions": predictions,
        "normalized_anchors": normalized_anchors,
        "regression_comparison": regression_comparison,
        "classification_comparison": classification_comparison,
        "best_anchor_table": best_anchor_table,
        "physics_diagnostic_table": physics_diagnostic_table,
        "output_dir": output_dir,
    }


def _run_forward_chaining_regression_validation(
    feature_frame: pd.DataFrame,
    anchors: pd.DataFrame,
    candidate_features: list[str],
    model_specs: dict[str, ModelSpec],
    *,
    min_occupancy: float,
    max_occupancy: float,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    ordered_anchors = anchors.sort_values("anchor_time").reset_index(drop=True)

    for split_index in range(1, len(ordered_anchors)):
        train_anchor_ids = ordered_anchors.loc[: split_index - 1, "anchor_id"].tolist()
        validation_anchor = ordered_anchors.iloc[split_index]

        train_df = feature_frame.loc[
            feature_frame["has_pseudo_label"]
            & feature_frame["source_anchor_tuple"].apply(
                lambda anchor_ids: bool(anchor_ids) and set(anchor_ids).issubset(set(train_anchor_ids))
            )
        ].copy()
        validation_df = feature_frame.loc[
            feature_frame["has_pseudo_label"]
            & feature_frame["source_anchor_tuple"].apply(lambda anchor_ids: validation_anchor.anchor_id in anchor_ids)
        ].copy()
        if train_df.empty or validation_df.empty:
            continue

        baseline_pred = pd.to_numeric(validation_df["occ_physics_est_smooth"], errors="coerce").clip(
            lower=min_occupancy,
            upper=max_occupancy,
        )
        records.append(
            _regression_metric_record(
                model_name="physics_baseline",
                split_name=f"forward_split_{split_index}",
                evaluation_scope="pseudo_label_diagnostic",
                y_true=validation_df["occ_mid"],
                y_pred=baseline_pred,
                weights=validation_df["confidence_weight"],
            )
        )

        for model_name, model_spec in model_specs.items():
            try:
                fitted = fit_weighted_regression_model(
                    train_df,
                    feature_columns=candidate_features,
                    model_spec=model_spec,
                )
                prediction = predict_weighted_regression_model(
                    fitted,
                    validation_df,
                    min_occupancy=min_occupancy,
                    max_occupancy=max_occupancy,
                )
                records.append(
                    _regression_metric_record(
                        model_name=model_name,
                        split_name=f"forward_split_{split_index}",
                        evaluation_scope="pseudo_label_diagnostic",
                        y_true=validation_df["occ_mid"],
                        y_pred=prediction,
                        weights=validation_df["confidence_weight"],
                    )
                )
            except ValueError as exc:
                records.append(
                    {
                        "model_name": model_name,
                        "split_name": f"forward_split_{split_index}",
                        "evaluation_scope": "pseudo_label_diagnostic",
                        "weighted_mae": np.nan,
                        "weighted_rmse": np.nan,
                        "mean_error": np.nan,
                        "rows": int(len(validation_df)),
                        "status": f"skipped: {exc}",
                    }
                )

    return pd.DataFrame.from_records(records)


def _run_forward_chaining_classification_validation(
    feature_frame: pd.DataFrame,
    anchors: pd.DataFrame,
    candidate_features: list[str],
    model_specs: dict[str, ModelSpec],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    ordered_anchors = anchors.sort_values("anchor_time").reset_index(drop=True)

    for split_index in range(1, len(ordered_anchors)):
        train_anchor_ids = ordered_anchors.loc[: split_index - 1, "anchor_id"].tolist()
        validation_anchor = ordered_anchors.iloc[split_index]

        train_df = feature_frame.loc[
            feature_frame["has_pseudo_label"]
            & feature_frame["source_anchor_tuple"].apply(
                lambda anchor_ids: bool(anchor_ids) and set(anchor_ids).issubset(set(train_anchor_ids))
            )
        ].copy()
        validation_df = feature_frame.loc[
            feature_frame["has_pseudo_label"]
            & feature_frame["source_anchor_tuple"].apply(lambda anchor_ids: validation_anchor.anchor_id in anchor_ids)
        ].copy()
        if train_df.empty or validation_df.empty:
            continue

        baseline_band = validation_df["occ_physics_est_smooth"].apply(occupancy_band)
        records.append(
            _classification_metric_record(
                model_name="physics_baseline_band",
                split_name=f"forward_split_{split_index}",
                evaluation_scope="pseudo_label_diagnostic",
                y_true=validation_df["true_occ_band"],
                y_pred=baseline_band,
                confidence=pd.Series(np.nan, index=validation_df.index),
                weights=validation_df["confidence_weight"],
            )
        )

        for model_name, model_spec in model_specs.items():
            try:
                fitted = fit_weighted_classifier_model(
                    train_df,
                    feature_columns=candidate_features,
                    model_spec=model_spec,
                )
                prediction, confidence = predict_weighted_classifier_model(fitted, validation_df)
                records.append(
                    _classification_metric_record(
                        model_name=model_name,
                        split_name=f"forward_split_{split_index}",
                        evaluation_scope="pseudo_label_diagnostic",
                        y_true=validation_df["true_occ_band"],
                        y_pred=prediction,
                        confidence=confidence,
                        weights=validation_df["confidence_weight"],
                    )
                )
            except ValueError as exc:
                records.append(
                    {
                        "model_name": model_name,
                        "split_name": f"forward_split_{split_index}",
                        "evaluation_scope": "pseudo_label_diagnostic",
                        "accuracy": np.nan,
                        "mean_band_distance": np.nan,
                        "mean_confidence": np.nan,
                        "rows": int(len(validation_df)),
                        "status": f"skipped: {exc}",
                    }
                )

    return pd.DataFrame.from_records(records)


def _run_anchor_window_regression_evaluation(
    feature_frame: pd.DataFrame,
    anchors: pd.DataFrame,
    candidate_features: list[str],
    model_specs: dict[str, ModelSpec],
    *,
    min_occupancy: float,
    max_occupancy: float,
    bootstrap_intervals: bool = False,
    bootstrap_iterations: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for anchor in anchors.itertuples(index=False):
        train_df = feature_frame.loc[
            feature_frame["has_pseudo_label"]
            & feature_frame["source_anchor_tuple"].apply(lambda anchor_ids: anchor.anchor_id not in anchor_ids)
        ].copy()
        evaluation_df = feature_frame.loc[
            (feature_frame.index >= anchor.window_start) & (feature_frame.index <= anchor.window_end)
        ].copy()
        if train_df.empty or evaluation_df.empty:
            continue

        anchor_timestamp = _nearest_timestamp(evaluation_df.index, anchor.anchor_time)
        baseline_prediction = pd.to_numeric(evaluation_df["occ_physics_est_smooth"], errors="coerce").clip(
            lower=min_occupancy,
            upper=max_occupancy,
        )
        records.append(
            _anchor_regression_record(
                model_name="physics_baseline",
                anchor=anchor,
                evaluation_df=evaluation_df,
                prediction=baseline_prediction,
                anchor_timestamp=anchor_timestamp,
                prediction_lower=None,
                prediction_upper=None,
                confidence_score=None,
            )
        )

        for model_name, model_spec in model_specs.items():
            try:
                fitted = fit_weighted_regression_model(
                    train_df,
                    feature_columns=candidate_features,
                    model_spec=model_spec,
                )
                prediction = predict_weighted_regression_model(
                    fitted,
                    evaluation_df,
                    min_occupancy=min_occupancy,
                    max_occupancy=max_occupancy,
                )

                prediction_lower = None
                prediction_upper = None
                confidence_score = None
                if bootstrap_intervals:
                    interval_df = bootstrap_anchor_resampled_predictions(
                        train_df,
                        evaluation_df,
                        feature_columns=candidate_features,
                        model_spec=model_spec,
                        anchor_ids=train_df["source_anchor_tuple"]
                        .explode()
                        .dropna()
                        .astype(str)
                        .drop_duplicates()
                        .tolist(),
                        min_occupancy=min_occupancy,
                        max_occupancy=max_occupancy,
                        n_bootstraps=bootstrap_iterations,
                        random_state=random_state,
                    )
                    prediction_lower = interval_df["prediction_lower"]
                    prediction_upper = interval_df["prediction_upper"]
                    confidence_score = interval_df["prediction_confidence_score"]

                records.append(
                    _anchor_regression_record(
                        model_name=model_name,
                        anchor=anchor,
                        evaluation_df=evaluation_df,
                        prediction=prediction,
                        anchor_timestamp=anchor_timestamp,
                        prediction_lower=prediction_lower,
                        prediction_upper=prediction_upper,
                        confidence_score=confidence_score,
                    )
                )
            except ValueError as exc:
                records.append(
                    {
                        "model_name": model_name,
                        "anchor_id": anchor.anchor_id,
                        "anchor_time": anchor.anchor_time,
                        "anchor_timestamp_used": anchor_timestamp,
                        "true_count": anchor.count,
                        "occ_low": anchor.occ_low,
                        "occ_high": anchor.occ_high,
                        "pred_at_anchor": np.nan,
                        "pred_window_mean": np.nan,
                        "pred_window_median": np.nan,
                        "pred_abs_error": np.nan,
                        "within_anchor_range": np.nan,
                        "prediction_lower_at_anchor": np.nan,
                        "prediction_upper_at_anchor": np.nan,
                        "prediction_confidence_at_anchor": np.nan,
                        "prediction_interval_contains_true": np.nan,
                        "status": f"skipped: {exc}",
                    }
                )

    return pd.DataFrame.from_records(records)


def _run_anchor_window_classification_evaluation(
    feature_frame: pd.DataFrame,
    anchors: pd.DataFrame,
    candidate_features: list[str],
    model_specs: dict[str, ModelSpec],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for anchor in anchors.itertuples(index=False):
        train_df = feature_frame.loc[
            feature_frame["has_pseudo_label"]
            & feature_frame["source_anchor_tuple"].apply(lambda anchor_ids: anchor.anchor_id not in anchor_ids)
        ].copy()
        evaluation_df = feature_frame.loc[
            (feature_frame.index >= anchor.window_start) & (feature_frame.index <= anchor.window_end)
        ].copy()
        if train_df.empty or evaluation_df.empty:
            continue

        anchor_timestamp = _nearest_timestamp(evaluation_df.index, anchor.anchor_time)
        true_band = occupancy_band(anchor.count)
        baseline_band = evaluation_df["occ_physics_est_smooth"].apply(occupancy_band)
        records.append(
            _anchor_classification_record(
                model_name="physics_baseline_band",
                anchor=anchor,
                true_band=true_band,
                prediction=baseline_band,
                confidence=pd.Series(np.nan, index=evaluation_df.index),
                anchor_timestamp=anchor_timestamp,
            )
        )

        for model_name, model_spec in model_specs.items():
            try:
                fitted = fit_weighted_classifier_model(
                    train_df,
                    feature_columns=candidate_features,
                    model_spec=model_spec,
                )
                prediction, confidence = predict_weighted_classifier_model(fitted, evaluation_df)
                records.append(
                    _anchor_classification_record(
                        model_name=model_name,
                        anchor=anchor,
                        true_band=true_band,
                        prediction=prediction,
                        confidence=confidence,
                        anchor_timestamp=anchor_timestamp,
                    )
                )
            except ValueError as exc:
                records.append(
                    {
                        "model_name": model_name,
                        "anchor_id": anchor.anchor_id,
                        "anchor_time": anchor.anchor_time,
                        "anchor_timestamp_used": anchor_timestamp,
                        "true_band": true_band,
                        "pred_band_at_anchor": pd.NA,
                        "pred_band_window_mode": pd.NA,
                        "band_correct": np.nan,
                        "band_distance": np.nan,
                        "band_confidence_at_anchor": np.nan,
                        "status": f"skipped: {exc}",
                    }
                )

    return pd.DataFrame.from_records(records)


def _fit_full_regression_models(
    feature_frame: pd.DataFrame,
    labeled_training: pd.DataFrame,
    candidate_features: list[str],
    model_specs: dict[str, ModelSpec],
    *,
    min_occupancy: float,
    max_occupancy: float,
) -> dict[str, Any]:
    predictions: dict[str, pd.Series] = {}
    models: dict[str, dict[str, object]] = {}

    for model_name, model_spec in model_specs.items():
        fitted = fit_weighted_regression_model(
            labeled_training,
            feature_columns=candidate_features,
            model_spec=model_spec,
        )
        predictions[model_name] = predict_weighted_regression_model(
            fitted,
            feature_frame,
            min_occupancy=min_occupancy,
            max_occupancy=max_occupancy,
        )
        models[model_name] = fitted

    return {"predictions": predictions, "models": models}


def _fit_full_classification_models(
    feature_frame: pd.DataFrame,
    labeled_training: pd.DataFrame,
    candidate_features: list[str],
    model_specs: dict[str, ModelSpec],
) -> dict[str, Any]:
    predictions: dict[str, dict[str, pd.Series]] = {}
    models: dict[str, dict[str, object]] = {}

    for model_name, model_spec in model_specs.items():
        fitted = fit_weighted_classifier_model(
            labeled_training,
            feature_columns=candidate_features,
            model_spec=model_spec,
        )
        band_prediction, confidence_prediction = predict_weighted_classifier_model(fitted, feature_frame)
        predictions[model_name] = {
            "band": band_prediction,
            "confidence": confidence_prediction,
        }
        models[model_name] = fitted

    return {"predictions": predictions, "models": models}


def _summarize_regression_models(
    anchor_evaluation: pd.DataFrame,
    forward_validation: pd.DataFrame,
) -> pd.DataFrame:
    if anchor_evaluation.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "anchor_eval_windows",
                "anchor_mae",
                "anchor_rmse",
                "anchor_within_range_rate",
                "anchor_mean_bias",
                "diagnostic_splits",
                "diagnostic_weighted_mae",
                "diagnostic_weighted_rmse",
            ]
        )

    anchor_summary = (
        anchor_evaluation.groupby("model_name", dropna=False)
        .agg(
            anchor_eval_windows=("anchor_id", "count"),
            anchor_mae=("pred_abs_error", "mean"),
            anchor_rmse=("pred_abs_error", lambda values: float(np.sqrt(np.mean(np.square(values.dropna())))) if len(values.dropna()) else np.nan),
            anchor_within_range_rate=("within_anchor_range", "mean"),
            anchor_mean_bias=("pred_error", "mean"),
        )
        .reset_index()
    )
    if forward_validation.empty:
        forward_summary = pd.DataFrame(columns=["model_name", "diagnostic_splits", "diagnostic_weighted_mae", "diagnostic_weighted_rmse"])
    else:
        forward_summary = (
            forward_validation.groupby("model_name", dropna=False)
            .agg(
                diagnostic_splits=("split_name", "count"),
                diagnostic_weighted_mae=("weighted_mae", "mean"),
                diagnostic_weighted_rmse=("weighted_rmse", "mean"),
            )
            .reset_index()
        )
    return anchor_summary.merge(forward_summary, on="model_name", how="left").sort_values(
        ["anchor_mae", "anchor_rmse"],
        na_position="last",
    ).reset_index(drop=True)


def _summarize_classification_models(
    anchor_evaluation: pd.DataFrame,
    forward_validation: pd.DataFrame,
) -> pd.DataFrame:
    if anchor_evaluation.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "anchor_eval_windows",
                "anchor_band_accuracy",
                "anchor_mean_band_distance",
                "anchor_mean_confidence",
                "diagnostic_splits",
                "diagnostic_accuracy",
                "diagnostic_mean_band_distance",
            ]
        )

    anchor_summary = (
        anchor_evaluation.groupby("model_name", dropna=False)
        .agg(
            anchor_eval_windows=("anchor_id", "count"),
            anchor_band_accuracy=("band_correct", "mean"),
            anchor_mean_band_distance=("band_distance", "mean"),
            anchor_mean_confidence=("band_confidence_at_anchor", "mean"),
        )
        .reset_index()
    )
    if forward_validation.empty:
        forward_summary = pd.DataFrame(columns=["model_name", "diagnostic_splits", "diagnostic_accuracy", "diagnostic_mean_band_distance"])
    else:
        forward_summary = (
            forward_validation.groupby("model_name", dropna=False)
            .agg(
                diagnostic_splits=("split_name", "count"),
                diagnostic_accuracy=("accuracy", "mean"),
                diagnostic_mean_band_distance=("mean_band_distance", "mean"),
            )
            .reset_index()
        )
    return anchor_summary.merge(forward_summary, on="model_name", how="left").sort_values(
        ["anchor_band_accuracy", "anchor_mean_band_distance"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)


def _choose_best_regression_model(summary_df: pd.DataFrame) -> str:
    valid = summary_df.loc[summary_df["model_name"] != "physics_baseline"].copy()
    valid = valid.sort_values(["anchor_mae", "anchor_rmse", "anchor_within_range_rate"], ascending=[True, True, False])
    if valid.empty:
        raise ValueError("No fitted regression models were available for comparison.")
    return str(valid.iloc[0]["model_name"])


def _choose_best_classifier_model(summary_df: pd.DataFrame) -> str:
    valid = summary_df.loc[summary_df["model_name"] != "physics_baseline_band"].copy()
    valid = valid.sort_values(
        ["anchor_band_accuracy", "anchor_mean_band_distance", "anchor_mean_confidence"],
        ascending=[False, True, False],
    )
    if valid.empty:
        raise ValueError("No fitted classification models were available for comparison.")
    return str(valid.iloc[0]["model_name"])


def _merge_best_anchor_tables(
    best_anchor_regression: pd.DataFrame,
    best_anchor_classification: pd.DataFrame,
    *,
    model_name: str,
    classifier_name: str,
) -> pd.DataFrame:
    regression_best = best_anchor_regression.loc[best_anchor_regression["model_name"] == model_name].copy()
    physics = best_anchor_regression.loc[best_anchor_regression["model_name"] == "physics_baseline"].copy()
    classification_best = best_anchor_classification.loc[
        best_anchor_classification["model_name"] == classifier_name
    ].copy()

    regression_best = regression_best.rename(
        columns={
            "pred_at_anchor": "best_pred_at_anchor",
            "pred_window_mean": "best_pred_window_mean",
            "pred_window_median": "best_pred_window_median",
            "pred_abs_error": "best_pred_abs_error",
            "pred_error": "best_pred_error",
            "prediction_lower_at_anchor": "best_pred_lower_at_anchor",
            "prediction_upper_at_anchor": "best_pred_upper_at_anchor",
            "prediction_confidence_at_anchor": "best_pred_confidence_at_anchor",
            "prediction_interval_contains_true": "best_interval_contains_true",
        }
    )
    physics = physics.rename(
        columns={
            "pred_at_anchor": "physics_pred_at_anchor",
            "pred_window_mean": "physics_pred_window_mean",
            "pred_window_median": "physics_pred_window_median",
            "pred_abs_error": "physics_pred_abs_error",
            "pred_error": "physics_pred_error",
            "within_anchor_range": "physics_within_anchor_range",
        }
    )
    classification_best = classification_best.rename(
        columns={
            "pred_band_at_anchor": "pred_band_at_anchor",
            "pred_band_window_mode": "pred_band_window_mode",
            "band_correct": "pred_band_correct",
            "band_distance": "pred_band_distance",
            "band_confidence_at_anchor": "pred_band_confidence_at_anchor",
        }
    )

    merged = regression_best.merge(
        physics[
            [
                "anchor_id",
                "physics_pred_at_anchor",
                "physics_pred_window_mean",
                "physics_pred_window_median",
                "physics_pred_abs_error",
                "physics_pred_error",
                "physics_within_anchor_range",
            ]
        ],
        on="anchor_id",
        how="left",
    ).merge(
        classification_best[
            [
                "anchor_id",
                "true_band",
                "pred_band_at_anchor",
                "pred_band_window_mode",
                "pred_band_correct",
                "pred_band_distance",
                "pred_band_confidence_at_anchor",
            ]
        ],
        on="anchor_id",
        how="left",
    )
    merged["best_model_name"] = model_name
    merged["best_classifier_name"] = classifier_name
    return merged.sort_values("anchor_time").reset_index(drop=True)


def _build_physics_bias_diagnostics(
    predictions: pd.DataFrame,
    anchors: pd.DataFrame,
    best_anchor_table: pd.DataFrame,
    *,
    outdoor_co2_ppm: float,
    co2_generation_lps_per_person: float,
    max_occupancy: float,
) -> tuple[pd.DataFrame, str]:
    records: list[dict[str, Any]] = []

    for anchor in anchors.itertuples(index=False):
        anchor_eval = best_anchor_table.loc[best_anchor_table["anchor_id"] == anchor.anchor_id]
        anchor_timestamp = _nearest_timestamp(
            predictions.loc[(predictions.index >= anchor.window_start) & (predictions.index <= anchor.window_end)].index,
            anchor.anchor_time,
        )
        row = predictions.loc[anchor_timestamp]
        future_window = predictions.loc[
            (predictions.index >= anchor.anchor_time) & (predictions.index <= anchor.anchor_time + pd.Timedelta(minutes=30))
        ]
        peak_timestamp = future_window["occ_physics_est_smooth"].idxmax() if not future_window.empty else anchor_timestamp
        peak_value = future_window["occ_physics_est_smooth"].max() if not future_window.empty else row["occ_physics_est_smooth"]

        baseline_400 = _physics_estimate_from_row(
            row,
            outdoor_co2_ppm=400.0,
            co2_generation_lps_per_person=co2_generation_lps_per_person,
            max_occupancy=max_occupancy,
        )
        baseline_default = _physics_estimate_from_row(
            row,
            outdoor_co2_ppm=outdoor_co2_ppm,
            co2_generation_lps_per_person=co2_generation_lps_per_person,
            max_occupancy=max_occupancy,
        )
        baseline_430 = _physics_estimate_from_row(
            row,
            outdoor_co2_ppm=430.0,
            co2_generation_lps_per_person=co2_generation_lps_per_person,
            max_occupancy=max_occupancy,
        )

        eval_row = anchor_eval.iloc[0] if not anchor_eval.empty else None
        records.append(
            {
                "anchor_id": anchor.anchor_id,
                "anchor_time": anchor.anchor_time,
                "true_count": anchor.count,
                "physics_pred_at_anchor": row["occ_physics_est_smooth"],
                "physics_abs_error": abs(float(row["occ_physics_est_smooth"]) - float(anchor.count)),
                "best_model_pred_at_anchor": eval_row["best_pred_at_anchor"] if eval_row is not None else np.nan,
                "best_model_abs_error": eval_row["best_pred_abs_error"] if eval_row is not None else np.nan,
                "residual_correction": (
                    eval_row["best_pred_at_anchor"] - row["occ_physics_est_smooth"]
                    if eval_row is not None
                    else np.nan
                ),
                "co2_ppm_at_anchor": row.get("co2_ppm", np.nan),
                "flow_cfm_at_anchor": row.get("flow_cfm", np.nan),
                "flow_window_mean": predictions.loc[
                    (predictions.index >= anchor.window_start) & (predictions.index <= anchor.window_end),
                    "flow_cfm",
                ].mean(),
                "physics_future_30m_peak": peak_value,
                "physics_future_30m_peak_time": peak_timestamp,
                "physics_minutes_to_peak": (
                    (peak_timestamp - anchor.anchor_time).total_seconds() / 60.0
                    if isinstance(peak_timestamp, pd.Timestamp)
                    else np.nan
                ),
                "physics_est_outdoor_400ppm": baseline_400,
                "physics_est_outdoor_default": baseline_default,
                "physics_est_outdoor_430ppm": baseline_430,
            }
        )

    diagnostic_df = pd.DataFrame.from_records(records).sort_values("anchor_time").reset_index(drop=True)

    if diagnostic_df.empty:
        return diagnostic_df, "# Physics Bias Diagnostics\n\nNo anchor diagnostics were available.\n"

    mean_physics_error = diagnostic_df["physics_abs_error"].mean()
    mean_best_error = diagnostic_df["best_model_abs_error"].mean()
    worst_row = diagnostic_df.sort_values("physics_abs_error", ascending=False).iloc[0]
    lag_note = (
        f"The largest physics miss occurred at {worst_row['anchor_time']:%Y-%m-%d %H:%M}, "
        f"where the smoothed physics estimate was {worst_row['physics_pred_at_anchor']:.1f} "
        f"against a manual count of {worst_row['true_count']:.0f}. "
        f"The strongest physics response within the next 30 minutes reached "
        f"{worst_row['physics_future_30m_peak']:.1f} after about {worst_row['physics_minutes_to_peak']:.0f} minutes."
    )
    outdoor_note = (
        f"Changing the outdoor CO2 assumption at that timestamp from 400 ppm to 430 ppm shifts the estimate only from "
        f"{worst_row['physics_est_outdoor_400ppm']:.1f} to {worst_row['physics_est_outdoor_430ppm']:.1f}, "
        "which is not enough to close the full gap by itself."
    )
    flow_note = (
        f"Mean anchor-window flow near the worst miss was {worst_row['flow_window_mean']:.1f} cfm, "
        "which supports the interpretation that active ventilation diluted the immediate CO2 signal."
    )
    if mean_best_error < mean_physics_error:
        model_note = (
            f"Across the available anchor windows, the hybrid model lowered mean absolute error from "
            f"{mean_physics_error:.2f} people for the physics baseline to {mean_best_error:.2f} people."
        )
    else:
        model_note = (
            f"Across the available anchor windows, the hybrid model did not beat the physics baseline yet: "
            f"the baseline MAE was {mean_physics_error:.2f} people versus {mean_best_error:.2f} people for the hybrid model."
        )

    markdown = "\n".join(
        [
            "# Physics Bias Diagnostics",
            "",
            "This diagnostic checks where the airflow-aware CO2 physics baseline underpredicts the manual anchors and where the hybrid model corrects that bias.",
            "",
            f"- {lag_note}",
            f"- {outdoor_note}",
            f"- {flow_note}",
            f"- {model_note}",
            "",
            "The interpretation should still be treated as preliminary because only three manual anchor windows are available.",
            "",
            "## Diagnostic Table",
            "",
            _dataframe_to_markdown(diagnostic_df),
            "",
        ]
    )
    return diagnostic_df, markdown


def _build_findings_summary(
    *,
    config: Room361PipelineConfig,
    pseudo_label_summary: dict[str, Any],
    regression_comparison: pd.DataFrame,
    classification_comparison: pd.DataFrame,
    best_anchor_table: pd.DataFrame,
    physics_diagnostic_text: str,
) -> str:
    best_regression_candidates = regression_comparison.loc[
        regression_comparison["model_name"] != "physics_baseline"
    ]
    best_classification_candidates = classification_comparison.loc[
        classification_comparison["model_name"] != "physics_baseline_band"
    ]
    best_regression = (
        best_regression_candidates.iloc[0] if not best_regression_candidates.empty else None
    )
    best_classification = (
        best_classification_candidates.iloc[0] if not best_classification_candidates.empty else None
    )
    baseline_regression = regression_comparison.loc[
        regression_comparison["model_name"] == "physics_baseline"
    ]
    baseline_classification = classification_comparison.loc[
        classification_comparison["model_name"] == "physics_baseline_band"
    ]
    pseudo_rows = int(pseudo_label_summary.get("label_rows", 0))
    anchor_rows = int(pseudo_label_summary.get("anchor_rows", 0))

    summary_lines = [
        "# Room 361 Weak-Supervision Findings",
        "",
        "This pipeline uses weak supervision and pseudo-labeling rather than classical bootstrap resampling.",
        "",
        "## What The Pipeline Does",
        "",
        "- Manual occupancy counts provide the only real ground-truth anchors.",
        "- Door-closed anchor windows receive higher confidence weights than lower-confidence anchor windows.",
        "- Pseudo-label ranges are preserved as `occ_low`, `occ_high`, `occ_mid`, `confidence`, and `confidence_weight` instead of being promoted to exact truth.",
        "- A hybrid physics-plus-ML approach is used, with the airflow-aware CO2 physics estimate acting as the baseline and the ML regressors learning a residual correction.",
        "- Final reportable evaluation is restricted to held-out manual anchor windows. Pseudo-label validation is saved only as a diagnostic check.",
        "",
        "## Data Limits",
        "",
        f"- The current Room 361 run uses {anchor_rows} manual anchor windows and {pseudo_rows} pseudo-labeled rows after weak supervision.",
        "- VOC support is implemented in the feature pipeline, but the current Room 361 FPB source file does not contain VOC measurements, so VOC-based features remain unavailable in this run.",
        "- Because the anchor set is sparse, any accuracy claims should be framed as preliminary and illustrative rather than fully validated.",
        "",
        "## Current Model Snapshot",
        "",
    ]

    if not baseline_regression.empty:
        baseline_row = baseline_regression.iloc[0]
        summary_lines.append(
            f"- Physics baseline held-out anchor MAE: {baseline_row['anchor_mae']:.2f} people "
            f"(RMSE {baseline_row['anchor_rmse']:.2f})."
        )
    if best_regression is not None:
        summary_lines.append(
            f"- Best trainable regression model by held-out anchor MAE: `{best_regression['model_name']}` "
            f"(MAE {best_regression['anchor_mae']:.2f}, RMSE {best_regression['anchor_rmse']:.2f}, "
            f"within-range rate {best_regression['anchor_within_range_rate']:.2%})."
        )
    if not baseline_classification.empty:
        baseline_band_row = baseline_classification.iloc[0]
        summary_lines.append(
            f"- Physics baseline band accuracy on held-out anchors: "
            f"{baseline_band_row['anchor_band_accuracy']:.2%}."
        )
    if best_classification is not None:
        summary_lines.append(
            f"- Best trainable occupancy-band classifier by held-out anchor accuracy: `{best_classification['model_name']}` "
            f"(accuracy {best_classification['anchor_band_accuracy']:.2%}, "
            f"mean band distance {best_classification['anchor_mean_band_distance']:.2f})."
        )

    if not best_anchor_table.empty:
        coverage = best_anchor_table["best_interval_contains_true"].mean()
        summary_lines.append(
            f"- Bootstrap interval coverage on held-out anchors for the best regression model was {coverage:.2%}."
        )

    summary_lines.extend(
        [
            "",
            "## Reporting Guidance",
            "",
            "- Use the anchor-window evaluation table as the main accuracy table in the capstone report.",
            "- Describe forward-chaining validation as an internal pseudo-label diagnostic, not as final accuracy.",
            "- Be explicit that additional direct occupancy counts would be required for stronger validation, calibration, and uncertainty estimates.",
            "",
            "## Physics Bias Note",
            "",
            "The detailed bias discussion is saved separately in `physics_bias_diagnostics.md`. A short version is included below for convenience.",
            "",
            physics_diagnostic_text,
        ]
    )

    return "\n".join(summary_lines)


def _save_time_series_figure(
    predictions: pd.DataFrame,
    anchors: pd.DataFrame,
    output_path: Path,
) -> None:
    plot_df = predictions.reset_index().rename(columns={"index": "ts"})

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.5]},
    )

    axes[0].plot(plot_df["ts"], plot_df["co2_ppm"], color="#1b5e20", linewidth=1.8, label="CO2")
    axes[0].set_ylabel("ppm")
    axes[0].set_title("Room 361 CO2, Ventilation, and Occupancy Estimates")
    axes[0].legend(loc="upper left")

    axes[1].plot(plot_df["ts"], plot_df["flow_cfm"], color="#0d47a1", linewidth=1.8, label="Discharge air flow")
    axes[1].set_ylabel("cfm")
    axes[1].legend(loc="upper left")

    mask = plot_df["has_pseudo_label"].fillna(False)
    axes[2].plot(
        plot_df["ts"],
        plot_df["occ_physics_est_smooth"],
        color="0.5",
        linewidth=1.5,
        label="Physics baseline",
    )
    axes[2].fill_between(
        plot_df["ts"],
        plot_df["occ_low"],
        plot_df["occ_high"],
        where=mask,
        color="#f39c12",
        alpha=0.25,
        label="Pseudo-label range",
    )
    axes[2].plot(
        plot_df["ts"],
        plot_df["best_occ_pred"],
        color="#6a1b9a",
        linewidth=2.0,
        label="Best hybrid prediction",
    )
    axes[2].fill_between(
        plot_df["ts"],
        plot_df["best_occ_lower"],
        plot_df["best_occ_upper"],
        color="#ab47bc",
        alpha=0.15,
        label="Bootstrap interval",
    )
    axes[2].scatter(
        anchors["anchor_time"],
        anchors["count"],
        color="black",
        s=50,
        zorder=4,
        label="Manual anchors",
    )
    axes[2].set_ylabel("people")
    axes[2].set_xlabel("timestamp")
    axes[2].legend(loc="upper left")

    for axis in axes:
        for anchor in anchors.itertuples(index=False):
            axis.axvspan(anchor.window_start, anchor.window_end, color="#f7dc6f", alpha=0.08)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_anchor_comparison_figure(best_anchor_table: pd.DataFrame, output_path: Path) -> None:
    if best_anchor_table.empty:
        return

    plot_df = best_anchor_table.copy()
    labels = plot_df["anchor_id"] + " (" + plot_df["anchor_time"].dt.strftime("%m-%d %H:%M") + ")"
    x = np.arange(len(plot_df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, plot_df["true_count"], width=width, label="True count", color="#263238")
    ax.bar(x, plot_df["physics_pred_at_anchor"], width=width, label="Physics baseline", color="#90a4ae")
    ax.bar(x + width, plot_df["best_pred_at_anchor"], width=width, label="Best hybrid", color="#8e24aa")

    if plot_df["best_pred_lower_at_anchor"].notna().any():
        yerr = np.vstack(
            [
                plot_df["best_pred_at_anchor"] - plot_df["best_pred_lower_at_anchor"],
                plot_df["best_pred_upper_at_anchor"] - plot_df["best_pred_at_anchor"],
            ]
        )
        ax.errorbar(
            x + width,
            plot_df["best_pred_at_anchor"],
            yerr=yerr,
            fmt="none",
            ecolor="#4a148c",
            elinewidth=1.5,
            capsize=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("people")
    ax.set_title("Held-Out Anchor Predictions")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_residual_error_figure(best_anchor_table: pd.DataFrame, output_path: Path) -> None:
    if best_anchor_table.empty:
        return

    plot_df = best_anchor_table.sort_values("anchor_time").copy()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        plot_df["anchor_time"],
        plot_df["physics_pred_abs_error"],
        marker="o",
        linewidth=2,
        color="#607d8b",
        label="Physics baseline abs error",
    )
    ax.plot(
        plot_df["anchor_time"],
        plot_df["best_pred_abs_error"],
        marker="o",
        linewidth=2,
        color="#8e24aa",
        label="Best hybrid abs error",
    )
    ax.set_ylabel("absolute error (people)")
    ax.set_xlabel("anchor timestamp")
    ax.set_title("Anchor-Window Absolute Error Over Time")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_feature_importance_figure(
    importance_df: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    value_column: str,
) -> None:
    if importance_df.empty:
        return

    plot_df = importance_df.head(15).iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(plot_df["feature"], plot_df[value_column], color="#00897b")
    ax.set_title(title)
    ax.set_xlabel("importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_correlation_heatmap(
    feature_frame: pd.DataFrame,
    candidate_features: list[str],
    output_path: Path,
) -> None:
    training = feature_frame.loc[feature_frame["has_pseudo_label"]].copy()
    if training.empty:
        return

    valid_features = []
    for column in candidate_features:
        if training[column].notna().sum() < 5:
            continue
        std_value = training[column].std(skipna=True)
        if pd.isna(std_value) or std_value == 0:
            continue
        valid_features.append(column)

    correlations = training[valid_features].corrwith(training["occ_mid"]).dropna().abs()
    selected_features = correlations.sort_values(ascending=False).head(12).index.tolist()
    if not selected_features:
        return

    heatmap_source = training[selected_features + ["occ_mid"]].copy()
    non_constant = []
    for column in heatmap_source.columns:
        std_value = heatmap_source[column].std(skipna=True)
        if pd.isna(std_value) or std_value == 0:
            continue
        non_constant.append(column)
    if len(non_constant) < 2:
        return

    heatmap_df = heatmap_source[non_constant].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(heatmap_df, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Engineered Feature Correlation Heatmap (Pseudo-Labeled Subset)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _regression_metric_record(
    *,
    model_name: str,
    split_name: str,
    evaluation_scope: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    weights: pd.Series,
) -> dict[str, Any]:
    y_true = pd.to_numeric(y_true, errors="coerce")
    y_pred = pd.to_numeric(y_pred, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce").fillna(1.0)
    mask = y_true.notna() & y_pred.notna() & weights.notna()

    if not mask.any():
        return {
            "model_name": model_name,
            "split_name": split_name,
            "evaluation_scope": evaluation_scope,
            "weighted_mae": np.nan,
            "weighted_rmse": np.nan,
            "mean_error": np.nan,
            "rows": 0,
            "status": "no_valid_rows",
        }

    y_true = y_true.loc[mask]
    y_pred = y_pred.loc[mask]
    weights = weights.loc[mask]
    error = y_pred - y_true
    weighted_mae = np.average(np.abs(error), weights=weights)
    weighted_rmse = math.sqrt(np.average(np.square(error), weights=weights))
    mean_error = np.average(error, weights=weights)

    return {
        "model_name": model_name,
        "split_name": split_name,
        "evaluation_scope": evaluation_scope,
        "weighted_mae": float(weighted_mae),
        "weighted_rmse": float(weighted_rmse),
        "mean_error": float(mean_error),
        "rows": int(mask.sum()),
        "status": "ok",
    }


def _classification_metric_record(
    *,
    model_name: str,
    split_name: str,
    evaluation_scope: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    confidence: pd.Series,
    weights: pd.Series,
) -> dict[str, Any]:
    y_true = y_true.astype("string")
    y_pred = y_pred.astype("string")
    confidence = pd.to_numeric(confidence, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce").fillna(1.0)
    mask = y_true.notna() & y_pred.notna() & weights.notna()

    if not mask.any():
        return {
            "model_name": model_name,
            "split_name": split_name,
            "evaluation_scope": evaluation_scope,
            "accuracy": np.nan,
            "mean_band_distance": np.nan,
            "mean_confidence": np.nan,
            "rows": 0,
            "status": "no_valid_rows",
        }

    y_true = y_true.loc[mask]
    y_pred = y_pred.loc[mask]
    confidence = confidence.loc[mask]
    weights = weights.loc[mask]

    accuracy = np.average((y_true == y_pred).astype(float), weights=weights)
    distances = pd.Series(
        [band_distance(pred_band, true_band) for pred_band, true_band in zip(y_pred, y_true, strict=False)],
        index=y_true.index,
        dtype=float,
    )
    mean_distance = np.average(distances.fillna(0.0), weights=weights)

    return {
        "model_name": model_name,
        "split_name": split_name,
        "evaluation_scope": evaluation_scope,
        "accuracy": float(accuracy),
        "mean_band_distance": float(mean_distance),
        "mean_confidence": float(confidence.mean()) if confidence.notna().any() else np.nan,
        "rows": int(mask.sum()),
        "status": "ok",
    }


def _anchor_regression_record(
    *,
    model_name: str,
    anchor: Any,
    evaluation_df: pd.DataFrame,
    prediction: pd.Series,
    anchor_timestamp: pd.Timestamp,
    prediction_lower: pd.Series | None,
    prediction_upper: pd.Series | None,
    confidence_score: pd.Series | None,
) -> dict[str, Any]:
    pred_at_anchor = float(prediction.loc[anchor_timestamp])
    pred_window_mean = float(prediction.mean())
    pred_window_median = float(prediction.median())
    pred_error = pred_at_anchor - float(anchor.count)

    lower_at_anchor = float(prediction_lower.loc[anchor_timestamp]) if prediction_lower is not None else np.nan
    upper_at_anchor = float(prediction_upper.loc[anchor_timestamp]) if prediction_upper is not None else np.nan
    confidence_at_anchor = float(confidence_score.loc[anchor_timestamp]) if confidence_score is not None else np.nan
    interval_contains_true = (
        lower_at_anchor <= float(anchor.count) <= upper_at_anchor
        if not np.isnan(lower_at_anchor) and not np.isnan(upper_at_anchor)
        else np.nan
    )

    return {
        "model_name": model_name,
        "anchor_id": anchor.anchor_id,
        "anchor_time": anchor.anchor_time,
        "anchor_timestamp_used": anchor_timestamp,
        "true_count": float(anchor.count),
        "occ_low": float(anchor.occ_low),
        "occ_high": float(anchor.occ_high),
        "pred_at_anchor": pred_at_anchor,
        "pred_window_mean": pred_window_mean,
        "pred_window_median": pred_window_median,
        "pred_error": pred_error,
        "pred_abs_error": abs(pred_error),
        "within_anchor_range": float(anchor.occ_low) <= pred_at_anchor <= float(anchor.occ_high),
        "prediction_lower_at_anchor": lower_at_anchor,
        "prediction_upper_at_anchor": upper_at_anchor,
        "prediction_confidence_at_anchor": confidence_at_anchor,
        "prediction_interval_contains_true": interval_contains_true,
        "status": "ok",
    }


def _anchor_classification_record(
    *,
    model_name: str,
    anchor: Any,
    true_band: str,
    prediction: pd.Series,
    confidence: pd.Series,
    anchor_timestamp: pd.Timestamp,
) -> dict[str, Any]:
    pred_band_at_anchor = str(prediction.loc[anchor_timestamp])
    pred_band_window_mode = _mode_or_missing(prediction)
    band_confidence = float(confidence.loc[anchor_timestamp]) if confidence.notna().any() else np.nan
    correct = pred_band_at_anchor == true_band

    return {
        "model_name": model_name,
        "anchor_id": anchor.anchor_id,
        "anchor_time": anchor.anchor_time,
        "anchor_timestamp_used": anchor_timestamp,
        "true_band": true_band,
        "pred_band_at_anchor": pred_band_at_anchor,
        "pred_band_window_mode": pred_band_window_mode,
        "band_correct": correct,
        "band_distance": band_distance(pred_band_at_anchor, true_band),
        "band_confidence_at_anchor": band_confidence,
        "status": "ok",
    }


def _physics_estimate_from_row(
    row: pd.Series,
    *,
    outdoor_co2_ppm: float,
    co2_generation_lps_per_person: float,
    max_occupancy: float,
) -> float:
    co2_ppm = float(row.get("co2_ppm", np.nan))
    flow_lps = float(row.get("flow_lps", np.nan))
    if np.isnan(co2_ppm) or np.isnan(flow_lps):
        return float("nan")
    delta = max(co2_ppm - outdoor_co2_ppm, 0.0)
    estimate = flow_lps * delta / (1e6 * co2_generation_lps_per_person)
    return float(np.clip(estimate, 0.0, max_occupancy))


def _nearest_timestamp(index: pd.DatetimeIndex, target: pd.Timestamp) -> pd.Timestamp:
    if len(index) == 0:
        raise ValueError("Cannot select a nearest timestamp from an empty index.")
    deltas = np.abs((index - target).asi8)
    return index[int(np.argmin(deltas))]


def _mode_or_missing(values: pd.Series) -> str | float:
    mode = values.mode(dropna=True)
    if mode.empty:
        return np.nan
    return str(mode.iloc[0])


def _load_optional_schedule(schedule_file: str | None) -> pd.DataFrame | None:
    if schedule_file is None:
        return None
    schedule_path = Path(schedule_file)
    if not schedule_path.exists():
        return None
    return pd.read_csv(schedule_path)


def _write_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def _write_markdown_table(
    df: pd.DataFrame,
    path: Path,
    *,
    title: str,
    note: str | None = None,
) -> None:
    lines = [f"# {title}", ""]
    if note:
        lines.extend([note, ""])
    lines.append(_dataframe_to_markdown(df))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"

    display_df = df.copy()
    for column in display_df.columns:
        if pd.api.types.is_datetime64_any_dtype(display_df[column]):
            display_df[column] = display_df[column].dt.strftime("%Y-%m-%d %H:%M:%S")

    headers = [str(column) for column in display_df.columns]
    rows = [[_format_markdown_value(value) for value in row] for row in display_df.itertuples(index=False, name=None)]
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    header_line = "| " + " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)) + " |"
    separator_line = "| " + " | ".join("-" * widths[index] for index in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator_line, *row_lines])


def _format_markdown_value(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Room 361 weak-supervision occupancy pipeline.")
    parser.add_argument("--input", nargs="+", default=list(Room361PipelineConfig.input_files), help="FPB CSV input paths.")
    parser.add_argument("--anchors", default=Room361PipelineConfig.anchor_file, help="Manual anchor CSV.")
    parser.add_argument("--schedule", default=None, help="Optional class schedule CSV.")
    parser.add_argument("--output-dir", default=Room361PipelineConfig.output_dir, help="Output directory.")
    parser.add_argument("--bootstrap-iterations", type=int, default=Room361PipelineConfig.bootstrap_iterations)
    parser.add_argument("--random-state", type=int, default=Room361PipelineConfig.random_state)
    args = parser.parse_args()

    config = Room361PipelineConfig(
        input_files=tuple(args.input),
        anchor_file=args.anchors,
        schedule_file=args.schedule,
        output_dir=args.output_dir,
        bootstrap_iterations=args.bootstrap_iterations,
        random_state=args.random_state,
    )
    result = run_room361_pipeline(config)
    print(f"Saved Room 361 pipeline artifacts to {result['output_dir']}")


if __name__ == "__main__":
    main()

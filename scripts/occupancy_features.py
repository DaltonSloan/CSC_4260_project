from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

DEFAULT_RESAMPLE_MINUTES = 5
DEFAULT_LAG_MINUTES = (5, 10, 15, 20, 30)
DEFAULT_ROLLING_MINUTES = (10, 15, 30)

BASE_SENSOR_COLUMNS = (
    "co2_ppm",
    "voc",
    "temp_f",
    "humidity_pct",
    "flow_cfm",
)
PHYSICS_CONTEXT_COLUMNS = (
    "occ_physics_est",
    "occ_physics_est_smooth",
    "delta_co2_ppm",
    "flow_lps",
)


def ensure_optional_sensor_columns(sensor_df: pd.DataFrame) -> pd.DataFrame:
    """Add optional-but-expected sensor columns so downstream code can degrade cleanly."""

    frame = sensor_df.copy()
    for column in BASE_SENSOR_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan

    if "incoming_air_cfm" not in frame.columns and "flow_cfm" in frame.columns:
        frame["incoming_air_cfm"] = frame["flow_cfm"]

    return frame


def engineer_temporal_features(
    sensor_df: pd.DataFrame,
    *,
    resample_minutes: int = DEFAULT_RESAMPLE_MINUTES,
    lag_minutes: Iterable[int] = DEFAULT_LAG_MINUTES,
    rolling_minutes: Iterable[int] = DEFAULT_ROLLING_MINUTES,
    schedule_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Add past-looking temporal features without using future information.

    Notes:
    - Lagged, rolling, delta, and slope features only use the current and previous rows.
    - VOC-derived features are created only when a VOC column exists or is supplied later.
    - `is_class_time` is optional and remains missing if no schedule is supplied.
    """

    frame = ensure_optional_sensor_columns(sensor_df)
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Temporal feature engineering requires a DatetimeIndex.")

    lag_minutes = tuple(sorted(set(int(value) for value in lag_minutes)))
    rolling_minutes = tuple(sorted(set(int(value) for value in rolling_minutes)))

    temporal_columns = [
        column
        for column in (*BASE_SENSOR_COLUMNS, *PHYSICS_CONTEXT_COLUMNS, "incoming_air_cfm")
        if column in frame.columns
    ]

    new_columns: dict[str, pd.Series] = {}

    for column in temporal_columns:
        series = pd.to_numeric(frame[column], errors="coerce")

        new_columns[f"{column}_delta_5m"] = series.diff()
        pct_change = series.pct_change(fill_method=None).replace(
            [np.inf, -np.inf],
            np.nan,
        )
        new_columns[f"{column}_pct_change_5m"] = pct_change

        for minutes in lag_minutes:
            lag_steps = max(int(round(minutes / resample_minutes)), 1)
            new_columns[f"{column}_lag_{minutes}m"] = series.shift(lag_steps)

        for minutes in rolling_minutes:
            window_steps = max(int(round(minutes / resample_minutes)), 1)
            new_columns[f"{column}_roll_mean_{minutes}m"] = series.rolling(
                window_steps,
                min_periods=1,
            ).mean()
            new_columns[f"{column}_roll_std_{minutes}m"] = series.rolling(
                window_steps,
                min_periods=2,
            ).std()
            new_columns[f"{column}_slope_{minutes}m"] = _rolling_slope(
                series,
                window_steps=window_steps,
                step_minutes=resample_minutes,
            )

    if {"co2_ppm", "flow_cfm"}.issubset(frame.columns):
        new_columns["co2_flow_interaction"] = frame["co2_ppm"] * frame["flow_cfm"]
    if {"humidity_pct", "temp_f"}.issubset(frame.columns):
        new_columns["humidity_temp_interaction"] = frame["humidity_pct"] * frame["temp_f"]
    if {"voc", "flow_cfm"}.issubset(frame.columns):
        new_columns["voc_flow_interaction"] = frame["voc"] * frame["flow_cfm"]

    new_columns["hour_of_day"] = pd.Series(frame.index.hour, index=frame.index, dtype=float)
    new_columns["day_of_week"] = pd.Series(frame.index.dayofweek, index=frame.index, dtype=float)
    new_columns["is_weekend"] = pd.Series(frame.index.dayofweek, index=frame.index).isin([5, 6]).astype(float)

    hour_fraction = frame.index.hour + (frame.index.minute / 60.0)
    new_columns["hour_sin"] = pd.Series(np.sin(2.0 * np.pi * hour_fraction / 24.0), index=frame.index)
    new_columns["hour_cos"] = pd.Series(np.cos(2.0 * np.pi * hour_fraction / 24.0), index=frame.index)

    new_columns["is_class_time"] = _build_class_time_flag(frame.index, schedule_df=schedule_df)

    return pd.concat([frame, pd.DataFrame(new_columns, index=frame.index)], axis=1)


def candidate_temporal_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns that are safe to consider for model fitting."""

    numeric_columns = feature_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    excluded = {
        "occ_low",
        "occ_high",
        "occ_mid",
        "confidence_weight",
        "confidence_rank",
        "interval_width",
        "source_priority",
        "has_pseudo_label",
        "occ_formula_residual",
        "occ_formula_abs_error",
        "occ_residual_pred",
        "occ_hybrid_pred",
        "occ_hybrid_pred_rounded",
        "occ_hybrid_fitted",
        "occ_hybrid_fitted_error",
        "occ_hybrid_fitted_abs_error",
        "target_occupancy_band_index",
        "predicted_occupancy_band_index",
    }
    return [column for column in numeric_columns if column not in excluded]


def _rolling_slope(
    series: pd.Series,
    *,
    window_steps: int,
    step_minutes: int,
) -> pd.Series:
    return series.rolling(window_steps, min_periods=2).apply(
        lambda values: _linear_slope(values, step_minutes=step_minutes),
        raw=True,
    )


def _linear_slope(values: np.ndarray, *, step_minutes: int) -> float:
    mask = np.isfinite(values)
    if mask.sum() < 2:
        return float("nan")

    x = np.arange(len(values), dtype=float) * step_minutes
    x = x[mask]
    y = values[mask].astype(float)

    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = float(np.dot(x_centered, x_centered))
    if denominator == 0:
        return float("nan")
    return float(np.dot(x_centered, y_centered) / denominator)


def _build_class_time_flag(
    index: pd.DatetimeIndex,
    *,
    schedule_df: pd.DataFrame | None,
) -> pd.Series:
    if schedule_df is None or schedule_df.empty:
        return pd.Series(np.nan, index=index, dtype=float)

    required = {"day_of_week", "start_time", "end_time"}
    missing = required.difference(schedule_df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(
            "Schedule dataframe is missing required columns for `is_class_time`: "
            f"{missing_list}"
        )

    schedule = schedule_df.copy()
    schedule["day_of_week"] = pd.to_numeric(schedule["day_of_week"], errors="coerce")
    schedule["start_minutes"] = schedule["start_time"].apply(_time_to_minutes)
    schedule["end_minutes"] = schedule["end_time"].apply(_time_to_minutes)
    schedule = schedule.dropna(subset=["day_of_week", "start_minutes", "end_minutes"])

    if schedule.empty:
        return pd.Series(np.nan, index=index, dtype=float)

    timestamp_minutes = index.hour * 60 + index.minute
    result = pd.Series(False, index=index, dtype=bool)

    for row in schedule.itertuples(index=False):
        active = (
            (index.dayofweek == int(row.day_of_week))
            & (timestamp_minutes >= row.start_minutes)
            & (timestamp_minutes < row.end_minutes)
        )
        result |= active

    return result.astype(float)


def _time_to_minutes(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")

    parsed = pd.to_datetime(str(value), errors="coerce")
    if pd.isna(parsed):
        return float("nan")
    return float(parsed.hour * 60 + parsed.minute)

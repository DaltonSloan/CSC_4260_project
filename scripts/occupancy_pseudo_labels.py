from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

POINT_NAME_MAP = {
    "Zone CO2": "co2_ppm",
    "Discharge Air Flow": "flow_cfm",
    "Zone Air Humid": "humidity_pct",
    "Zone Temp": "temp_f",
}

CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
CONFIDENCE_WEIGHT = {"low": 0.35, "medium": 0.65, "high": 1.0}
BASE_MARGIN_FACTOR = {"low": 0.14, "medium": 0.10, "high": 0.08}
BASE_MARGIN_MIN = {"low": 3, "medium": 2, "high": 1}
DISTANCE_MARGIN_MAX = {"low": 4, "medium": 3, "high": 1}
SENSOR_STABILITY_THRESHOLDS = {
    "co2_slope_ppm": 25.0,
    "flow_pct_change": 0.12,
    "temp_delta_f": 1.0,
    "humidity_delta_pct": 3.0,
}
DEFAULT_OUTDOOR_CO2_PPM = 415.0
DEFAULT_CO2_GENERATION_LPS_PER_PERSON = 0.005
CFM_TO_LPS = 0.471947
DEFAULT_INTERPOLATION_MAX_GAP_MIN = 120
DEFAULT_MIN_OCCUPANCY = 0.0
DEFAULT_MAX_OCCUPANCY = 45.0


def load_room_fpb_timeseries(
    file_path: str | Path | list[str | Path] | tuple[str | Path, ...],
    *,
    time_col: str = "dateTime",
    resample_rule: str = "5min",
    outdoor_co2_ppm: float = DEFAULT_OUTDOOR_CO2_PPM,
    co2_generation_lps_per_person: float = DEFAULT_CO2_GENERATION_LPS_PER_PERSON,
    min_occupancy: float = DEFAULT_MIN_OCCUPANCY,
    max_occupancy: float = DEFAULT_MAX_OCCUPANCY,
) -> pd.DataFrame:
    """Load one or more room FPB exports and reshape them into a feature frame."""

    file_paths = _coerce_file_paths(file_path)
    frames: list[pd.DataFrame] = []

    for current_path in file_paths:
        df = pd.read_csv(current_path)
        required = {time_col, "pointDisplayName", "value"}
        missing = required.difference(df.columns)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Missing required columns in {current_path}: {missing_list}")

        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col])
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    dedupe_cols = [
        column
        for column in [time_col, "pointDisplayName", "value", "deviceName", "pointField"]
        if column in df.columns
    ]
    if dedupe_cols:
        df = df.drop_duplicates(subset=dedupe_cols)

    wide = (
        df.pivot_table(
            index=time_col,
            columns="pointDisplayName",
            values="value",
            aggfunc="mean",
        )
        .sort_index()
        .resample(resample_rule)
        .mean()
        .rename(columns=POINT_NAME_MAP)
    )

    sensor = wide.rename_axis("ts").copy()
    sensor["flow_lps"] = sensor["flow_cfm"] * CFM_TO_LPS
    sensor["delta_co2_ppm"] = (sensor["co2_ppm"] - outdoor_co2_ppm).clip(lower=0)
    sensor["occ_physics_est"] = (
        sensor["flow_lps"] * sensor["delta_co2_ppm"] / (1e6 * co2_generation_lps_per_person)
    ).clip(lower=min_occupancy, upper=max_occupancy)
    sensor["occ_physics_est_smooth"] = sensor["occ_physics_est"].rolling(3, min_periods=1).mean()
    sensor["occ_physics_est_smooth"] = sensor["occ_physics_est_smooth"].clip(
        lower=min_occupancy,
        upper=max_occupancy,
    )

    sensor["co2_slope_ppm"] = sensor["co2_ppm"].diff()
    sensor["flow_pct_change"] = sensor["flow_cfm"].pct_change(fill_method=None)
    sensor["flow_pct_change"] = sensor["flow_pct_change"].replace([np.inf, -np.inf], np.nan)
    sensor["temp_delta_f"] = sensor["temp_f"].diff()
    sensor["humidity_delta_pct"] = sensor["humidity_pct"].diff()
    sensor["sensor_stable"] = _sensor_stability_mask(sensor)

    return sensor


def normalize_anchor_table(anchor_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize user-supplied anchor rows into explicit windows and ranges."""

    if anchor_df.empty:
        return pd.DataFrame(
            columns=[
                "anchor_id",
                "anchor_time",
                "count",
                "occ_low",
                "occ_high",
                "window_before_min",
                "window_after_min",
                "door_closed_minutes",
                "confidence",
                "reason",
                "window_start",
                "window_end",
                "base_margin",
            ]
        )

    anchors = anchor_df.copy()
    required = {"anchor_time", "count"}
    missing = required.difference(anchors.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Anchor table is missing required columns: {missing_list}")

    anchors["anchor_time"] = pd.to_datetime(anchors["anchor_time"], errors="coerce")
    anchors = anchors.dropna(subset=["anchor_time", "count"]).copy()
    if anchors.empty:
        raise ValueError("Anchor table did not contain any valid timestamp/count rows.")

    anchors["count"] = pd.to_numeric(anchors["count"], errors="coerce")
    anchors = anchors.dropna(subset=["count"]).copy()
    anchors["count"] = anchors["count"].clip(
        lower=DEFAULT_MIN_OCCUPANCY,
        upper=DEFAULT_MAX_OCCUPANCY,
    )

    if "door_closed_minutes" not in anchors.columns:
        anchors["door_closed_minutes"] = 0
    anchors["door_closed_minutes"] = pd.to_numeric(
        anchors["door_closed_minutes"], errors="coerce"
    ).fillna(0)

    half_window_from_doors = anchors["door_closed_minutes"] / 2.0

    if "window_before_min" not in anchors.columns:
        anchors["window_before_min"] = np.nan
    if "window_after_min" not in anchors.columns:
        anchors["window_after_min"] = np.nan

    anchors["window_before_min"] = pd.to_numeric(
        anchors["window_before_min"], errors="coerce"
    )
    anchors["window_after_min"] = pd.to_numeric(
        anchors["window_after_min"], errors="coerce"
    )

    default_before = pd.Series(
        np.where(anchors["door_closed_minutes"] > 0, half_window_from_doors, 5),
        index=anchors.index,
    )
    default_after = pd.Series(
        np.where(anchors["door_closed_minutes"] > 0, half_window_from_doors, 5),
        index=anchors.index,
    )
    anchors["window_before_min"] = anchors["window_before_min"].fillna(default_before)
    anchors["window_after_min"] = anchors["window_after_min"].fillna(default_after)

    if "confidence" not in anchors.columns:
        anchors["confidence"] = np.nan
    anchors["confidence"] = anchors["confidence"].map(_normalize_confidence)
    default_confidence = pd.Series(
        np.where(anchors["door_closed_minutes"] >= 15, "high", "medium"),
        index=anchors.index,
    )
    anchors["confidence"] = anchors["confidence"].fillna(default_confidence)

    if "reason" not in anchors.columns:
        anchors["reason"] = ""
    anchors["reason"] = anchors["reason"].fillna("").astype(str)
    anchors.loc[anchors["reason"].str.strip() == "", "reason"] = np.where(
        anchors["door_closed_minutes"] > 0,
        "direct count with stable door-closed window",
        "direct count only",
    )

    explicit_low = "occ_low" in anchors.columns
    explicit_high = "occ_high" in anchors.columns
    if not explicit_low:
        anchors["occ_low"] = np.nan
    if not explicit_high:
        anchors["occ_high"] = np.nan

    anchors["occ_low"] = pd.to_numeric(anchors["occ_low"], errors="coerce")
    anchors["occ_high"] = pd.to_numeric(anchors["occ_high"], errors="coerce")

    auto_margin = anchors.apply(
        lambda row: _default_margin(row["count"], row["confidence"]), axis=1
    )
    anchors["occ_low"] = anchors["occ_low"].fillna(anchors["count"] - auto_margin)
    anchors["occ_high"] = anchors["occ_high"].fillna(anchors["count"] + auto_margin)
    anchors["occ_low"] = anchors["occ_low"].clip(
        lower=DEFAULT_MIN_OCCUPANCY,
        upper=DEFAULT_MAX_OCCUPANCY,
    )
    anchors["occ_high"] = anchors["occ_high"].clip(
        lower=DEFAULT_MIN_OCCUPANCY,
        upper=DEFAULT_MAX_OCCUPANCY,
    )
    anchors["occ_low"] = np.minimum(anchors["occ_low"], anchors["count"])
    anchors["occ_high"] = np.maximum(anchors["occ_high"], anchors["count"])

    anchors["base_margin"] = (
        pd.concat(
            [
                (anchors["count"] - anchors["occ_low"]).abs(),
                (anchors["occ_high"] - anchors["count"]).abs(),
            ],
            axis=1,
        )
        .max(axis=1)
        .round()
        .astype(int)
    )

    anchors["window_start"] = anchors["anchor_time"] - pd.to_timedelta(
        anchors["window_before_min"], unit="m"
    )
    anchors["window_end"] = anchors["anchor_time"] + pd.to_timedelta(
        anchors["window_after_min"], unit="m"
    )

    anchors = anchors.sort_values("anchor_time").reset_index(drop=True)
    anchors["anchor_id"] = [f"A{i + 1}" for i in range(len(anchors))]

    return anchors[
        [
            "anchor_id",
            "anchor_time",
            "count",
            "occ_low",
            "occ_high",
            "window_before_min",
            "window_after_min",
            "door_closed_minutes",
            "confidence",
            "reason",
            "window_start",
            "window_end",
            "base_margin",
        ]
    ]


def generate_pseudo_labels(
    sensor_df: pd.DataFrame,
    anchor_df: pd.DataFrame,
    *,
    interpolation_max_gap_min: int = DEFAULT_INTERPOLATION_MAX_GAP_MIN,
    min_occupancy: float = DEFAULT_MIN_OCCUPANCY,
    max_occupancy: float = DEFAULT_MAX_OCCUPANCY,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Attach occupancy pseudo-labels to a time-indexed sensor frame."""

    if sensor_df.index.name != "ts":
        sensor = sensor_df.copy().rename_axis("ts")
    else:
        sensor = sensor_df.copy()

    anchors = normalize_anchor_table(anchor_df)
    anchor_labels = _materialize_anchor_labels(
        sensor,
        anchors,
        min_occupancy=min_occupancy,
        max_occupancy=max_occupancy,
    )
    interpolated_labels = _materialize_interpolated_labels(
        sensor,
        anchors,
        interpolation_max_gap_min=interpolation_max_gap_min,
        min_occupancy=min_occupancy,
        max_occupancy=max_occupancy,
    )

    label_rows = pd.concat([anchor_labels, interpolated_labels], ignore_index=True)
    if label_rows.empty:
        labeled = sensor.copy()
        for column in [
            "occ_low",
            "occ_high",
            "occ_mid",
            "confidence",
            "confidence_weight",
            "label_source",
            "label_reason",
            "source_anchor_ids",
            "sensor_stable_for_label",
            "occ_band",
            "has_pseudo_label",
        ]:
            labeled[column] = np.nan
        labeled["has_pseudo_label"] = False
        return labeled, label_rows, anchors

    label_rows["confidence_rank"] = label_rows["confidence"].map(CONFIDENCE_RANK)
    label_rows["interval_width"] = label_rows["occ_high"] - label_rows["occ_low"]
    label_rows["source_priority"] = label_rows["label_source"].map({"anchor": 0, "interpolated": 1})

    best_rows = (
        label_rows.sort_values(
            ["ts", "confidence_rank", "source_priority", "interval_width"],
            ascending=[True, False, True, True],
        )
        .drop_duplicates(subset=["ts"], keep="first")
        .set_index("ts")
    )

    labeled = sensor.join(
        best_rows[
            [
                "occ_low",
                "occ_high",
                "occ_mid",
                "confidence",
                "label_source",
                "label_reason",
                "source_anchor_ids",
                "sensor_stable_for_label",
            ]
        ],
        how="left",
    )
    labeled["confidence_weight"] = labeled["confidence"].map(CONFIDENCE_WEIGHT)
    labeled["occ_band"] = labeled["occ_mid"].apply(_occupancy_band)
    labeled["has_pseudo_label"] = labeled["occ_mid"].notna()

    return labeled, label_rows.sort_values("ts").reset_index(drop=True), anchors


def summarize_pseudo_labels(labeled_df: pd.DataFrame, anchors_df: pd.DataFrame) -> dict[str, object]:
    """Provide a compact summary that is easy to display in notebooks/CLI."""

    labeled_count = int(labeled_df["has_pseudo_label"].sum()) if "has_pseudo_label" in labeled_df else 0
    overlap_anchor_count = 0
    if not anchors_df.empty and not labeled_df.empty:
        overlap_anchor_count = int(
            (
                (anchors_df["window_end"] >= labeled_df.index.min())
                & (anchors_df["window_start"] <= labeled_df.index.max())
            ).sum()
        )

    summary = {
        "sensor_rows": int(len(labeled_df)),
        "label_rows": labeled_count,
        "label_coverage_pct": round(100 * labeled_count / max(len(labeled_df), 1), 2),
        "anchor_rows": int(len(anchors_df)),
        "overlapping_anchor_rows": overlap_anchor_count,
        "sensor_start": labeled_df.index.min().isoformat() if len(labeled_df) else None,
        "sensor_end": labeled_df.index.max().isoformat() if len(labeled_df) else None,
    }

    if labeled_count:
        summary["confidence_counts"] = (
            labeled_df.loc[labeled_df["has_pseudo_label"], "confidence"]
            .value_counts(dropna=True)
            .to_dict()
        )
        summary["label_source_counts"] = (
            labeled_df.loc[labeled_df["has_pseudo_label"], "label_source"]
            .value_counts(dropna=True)
            .to_dict()
        )

    return summary


def _materialize_anchor_labels(
    sensor: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    min_occupancy: float,
    max_occupancy: float,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for anchor in anchors.itertuples(index=False):
        window = sensor.loc[(sensor.index >= anchor.window_start) & (sensor.index <= anchor.window_end)]
        if window.empty:
            continue

        span_min = max(float(anchor.window_before_min), float(anchor.window_after_min), 1.0)
        for ts, row in window.iterrows():
            distance_min = abs((ts - anchor.anchor_time).total_seconds()) / 60.0
            distance_ratio = distance_min / span_min
            extra_margin = math.ceil(distance_ratio * DISTANCE_MARGIN_MAX[anchor.confidence])
            sensor_stable = bool(row.get("sensor_stable", True))
            sensor_penalty = 0 if sensor_stable else 1
            margin = int(anchor.base_margin + extra_margin + sensor_penalty)

            records.append(
                {
                    "ts": ts,
                    "occ_low": _clip_occ_value(
                        math.floor(anchor.count - margin),
                        min_occupancy=min_occupancy,
                        max_occupancy=max_occupancy,
                    ),
                    "occ_high": _clip_occ_value(
                        math.ceil(anchor.count + margin),
                        min_occupancy=min_occupancy,
                        max_occupancy=max_occupancy,
                    ),
                    "occ_mid": _clip_occ_value(
                        float(anchor.count),
                        min_occupancy=min_occupancy,
                        max_occupancy=max_occupancy,
                    ),
                    "confidence": anchor.confidence,
                    "label_source": "anchor",
                    "label_reason": anchor.reason,
                    "source_anchor_ids": anchor.anchor_id,
                    "sensor_stable_for_label": sensor_stable,
                }
            )

    return pd.DataFrame.from_records(records)


def _materialize_interpolated_labels(
    sensor: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    interpolation_max_gap_min: int,
    min_occupancy: float,
    max_occupancy: float,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    if len(anchors) < 2:
        return pd.DataFrame.from_records(records)

    for left, right in zip(anchors.itertuples(index=False), anchors.iloc[1:].itertuples(index=False)):
        if left.anchor_time.date() != right.anchor_time.date():
            continue

        gap_start = left.window_end
        gap_end = right.window_start
        gap_minutes = (gap_end - gap_start).total_seconds() / 60.0
        if gap_minutes <= 0 or gap_minutes > interpolation_max_gap_min:
            continue

        gap_window = sensor.loc[(sensor.index > gap_start) & (sensor.index < gap_end)]
        if gap_window.empty:
            continue

        count_delta = float(right.count) - float(left.count)
        median_co2_slope = gap_window["co2_slope_ppm"].median(skipna=True)
        base_confidence = "low"
        if (
            gap_minutes <= 60
            and not pd.isna(median_co2_slope)
            and (
                abs(count_delta) <= 1
                or np.sign(count_delta) == np.sign(median_co2_slope)
            )
        ):
            base_confidence = "medium"

        for ts, row in gap_window.iterrows():
            position = (ts - gap_start).total_seconds() / max((gap_end - gap_start).total_seconds(), 1.0)
            middle_boost = 1.0 - abs(2.0 * position - 1.0)
            sensor_stable = bool(row.get("sensor_stable", True))
            sensor_penalty = 0 if sensor_stable else 1
            base_margin = max(int(left.base_margin), int(right.base_margin))
            bridge_margin = math.ceil(gap_minutes / 30.0)
            margin = int(base_margin + bridge_margin + math.ceil(2 * middle_boost) + sensor_penalty)
            occ_mid = float(left.count) + position * count_delta

            records.append(
                {
                    "ts": ts,
                    "occ_low": _clip_occ_value(
                        math.floor(occ_mid - margin),
                        min_occupancy=min_occupancy,
                        max_occupancy=max_occupancy,
                    ),
                    "occ_high": _clip_occ_value(
                        math.ceil(occ_mid + margin),
                        min_occupancy=min_occupancy,
                        max_occupancy=max_occupancy,
                    ),
                    "occ_mid": _clip_occ_value(
                        round(occ_mid, 1),
                        min_occupancy=min_occupancy,
                        max_occupancy=max_occupancy,
                    ),
                    "confidence": base_confidence,
                    "label_source": "interpolated",
                    "label_reason": (
                        f"interpolated between {left.anchor_id} ({left.anchor_time:%Y-%m-%d %H:%M}) "
                        f"and {right.anchor_id} ({right.anchor_time:%Y-%m-%d %H:%M})"
                    ),
                    "source_anchor_ids": f"{left.anchor_id},{right.anchor_id}",
                    "sensor_stable_for_label": sensor_stable,
                }
            )

    return pd.DataFrame.from_records(records)


def _sensor_stability_mask(sensor: pd.DataFrame) -> pd.Series:
    stable = pd.Series(True, index=sensor.index, dtype=bool)
    for column, threshold in SENSOR_STABILITY_THRESHOLDS.items():
        if column not in sensor.columns:
            continue
        within_threshold = sensor[column].isna() | (sensor[column].abs() <= threshold)
        stable &= within_threshold
    return stable


def _coerce_file_paths(
    file_path: str | Path | list[str | Path] | tuple[str | Path, ...]
) -> list[Path]:
    if isinstance(file_path, (str, Path)):
        return [Path(file_path)]
    return [Path(path) for path in file_path]


def _normalize_confidence(value: object) -> str | float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    text = str(value).strip().lower()
    return text if text in CONFIDENCE_RANK else np.nan


def _default_margin(count: float, confidence: str) -> int:
    return max(
        BASE_MARGIN_MIN[confidence],
        int(math.ceil(float(count) * BASE_MARGIN_FACTOR[confidence])),
    )


def _clip_occ_value(
    value: float,
    *,
    min_occupancy: float = DEFAULT_MIN_OCCUPANCY,
    max_occupancy: float = DEFAULT_MAX_OCCUPANCY,
) -> float:
    return float(np.clip(value, min_occupancy, max_occupancy))


def _occupancy_band(value: object) -> str | float:
    if value is None or pd.isna(value):
        return np.nan
    value = float(value)
    if value <= 2:
        return "0-2"
    if value <= 10:
        return "3-10"
    if value <= 20:
        return "11-20"
    if value <= 35:
        return "21-35"
    return "36-45"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate uncertainty-bounded occupancy pseudo-labels for room sensor data."
    )
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="One or more FPB CSV paths.",
    )
    parser.add_argument("--anchors", required=True, help="Path to the anchor CSV.")
    parser.add_argument("--output", required=True, help="Path to the labeled output CSV.")
    parser.add_argument(
        "--time-col",
        default="dateTime",
        help="Timestamp column to use from the FPB export.",
    )
    parser.add_argument(
        "--resample-rule",
        default="5min",
        help="Pandas resample rule for the sensor timeseries.",
    )
    parser.add_argument(
        "--interpolation-max-gap-min",
        type=int,
        default=DEFAULT_INTERPOLATION_MAX_GAP_MIN,
        help="Maximum gap between anchors to fill with interpolated pseudo-labels.",
    )
    args = parser.parse_args()

    sensor = load_room_fpb_timeseries(
        args.input,
        time_col=args.time_col,
        resample_rule=args.resample_rule,
    )
    anchors = pd.read_csv(args.anchors)
    labeled, label_rows, normalized_anchors = generate_pseudo_labels(
        sensor,
        anchors,
        interpolation_max_gap_min=args.interpolation_max_gap_min,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.reset_index().to_csv(output_path, index=False)

    interval_output = output_path.with_name(f"{output_path.stem}_label_rows.csv")
    anchors_output = output_path.with_name(f"{output_path.stem}_anchors_normalized.csv")
    label_rows.to_csv(interval_output, index=False)
    normalized_anchors.to_csv(anchors_output, index=False)

    summary = summarize_pseudo_labels(labeled, normalized_anchors)
    print(f"Saved labeled rows: {output_path}")
    print(f"Saved interval rows: {interval_output}")
    print(f"Saved normalized anchors: {anchors_output}")
    print(summary)


if __name__ == "__main__":
    main()

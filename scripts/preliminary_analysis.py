#!/usr/bin/env python3
"""Generate preliminary building analysis summaries and figures."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


@dataclass
class RoomSummary:
    room: str
    start: str
    end: str
    interval_count: int
    co2_mean: float | None
    co2_max: float | None
    pct_co2_above_800: float | None
    pct_co2_above_1000: float | None
    temp_mean: float | None
    temp_work_mean: float | None
    temp_off_mean: float | None
    temp_min: float | None
    temp_max: float | None
    voc_mean: float | None
    voc_work_mean: float | None
    voc_off_mean: float | None
    voc_max: float | None
    humidity_mean: float | None
    humidity_work_mean: float | None
    humidity_off_mean: float | None
    humidity_max: float | None
    noise_mean: float | None
    noise_work_mean: float | None
    noise_off_mean: float | None
    noise_max: float | None


@dataclass
class AhuSummary:
    ahu: str
    start: str
    end: str
    interval_count: int
    discharge_temp_mean: float | None
    discharge_temp_max: float | None
    outside_temp_mean: float | None
    zone_co2_mean: float | None
    zone_co2_max: float | None


def round_or_none(value: float | int | None, digits: int = 2) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_whole_building() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "whole_building_consumption.csv", skiprows=2)
    df.columns = ["hour", "day_start", "consumption_kwh"]
    df["timestamp"] = pd.to_datetime(df["day_start"]) + pd.to_timedelta(df["hour"], unit="h")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["weekday"] = df["timestamp"].dt.day_name()
    df["day_type"] = df["timestamp"].dt.dayofweek.map(lambda day: "Weekend" if day >= 5 else "Weekday")
    df["work_hours"] = df["timestamp"].dt.hour.between(6, 17)
    return df


def load_room(room: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{room}.csv")
    df["time"] = pd.to_datetime(df["time"])
    value_columns = [column for column in df.columns if column.endswith("_value")]
    room_df = (
        df[["time"] + value_columns]
        .set_index("time")
        .sort_index()
        .resample("15min")
        .mean(numeric_only=True)
    )
    room_df["hour"] = room_df.index.hour
    room_df["day_type"] = room_df.index.dayofweek.map(lambda day: "Weekend" if day >= 5 else "Weekday")
    room_df["work_hours"] = room_df["hour"].between(6, 17)
    return room_df


def load_ahu(pattern: str, ahu_name: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(DATA_DIR.glob(pattern)):
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"])
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No files matched {pattern}")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    value_columns = [column for column in combined.columns if column.endswith("_value")]
    ahu_df = (
        combined[["time"] + value_columns]
        .set_index("time")
        .sort_index()
        .resample("15min")
        .mean(numeric_only=True)
    )
    ahu_df["ahu"] = ahu_name
    ahu_df["hour"] = ahu_df.index.hour
    ahu_df["day_type"] = ahu_df.index.dayofweek.map(lambda day: "Weekend" if day >= 5 else "Weekday")
    return ahu_df


def summarize_whole_building(df: pd.DataFrame) -> Dict[str, object]:
    work_group = df.groupby("work_hours")["consumption_kwh"]
    day_group = df.groupby("day_type")["consumption_kwh"]
    peak_row = df.loc[df["consumption_kwh"].idxmax()]
    return {
        "start": df["timestamp"].min().isoformat(),
        "end": df["timestamp"].max().isoformat(),
        "records": int(len(df)),
        "work_hours_mean_kwh": round_or_none(work_group.mean().get(True)),
        "off_hours_mean_kwh": round_or_none(work_group.mean().get(False)),
        "weekday_mean_kwh": round_or_none(day_group.mean().get("Weekday")),
        "weekend_mean_kwh": round_or_none(day_group.mean().get("Weekend")),
        "peak_timestamp": peak_row["timestamp"].isoformat(),
        "peak_kwh": round_or_none(peak_row["consumption_kwh"]),
    }


def summarize_room(room: str, df: pd.DataFrame) -> RoomSummary:
    co2_column = next((column for column in df.columns if "CO2_value" in column), None)
    temp_column = next((column for column in df.columns if "Temp_value" in column), None)
    voc_column = next((column for column in df.columns if "VOC_value" in column), None)
    humidity_column = next((column for column in df.columns if "Humid_value" in column), None)
    noise_column = next((column for column in df.columns if "Noise Level_value" in column), None)

    co2 = df[co2_column].dropna() if co2_column else pd.Series(dtype=float)
    temp = df[temp_column].dropna() if temp_column else pd.Series(dtype=float)
    temp_work = df.loc[df["work_hours"], temp_column].dropna() if temp_column else pd.Series(dtype=float)
    temp_off = df.loc[~df["work_hours"], temp_column].dropna() if temp_column else pd.Series(dtype=float)
    voc = df[voc_column].dropna() if voc_column else pd.Series(dtype=float)
    voc_work = df.loc[df["work_hours"], voc_column].dropna() if voc_column else pd.Series(dtype=float)
    voc_off = df.loc[~df["work_hours"], voc_column].dropna() if voc_column else pd.Series(dtype=float)
    humidity = df[humidity_column].dropna() if humidity_column else pd.Series(dtype=float)
    humidity_work = (
        df.loc[df["work_hours"], humidity_column].dropna() if humidity_column else pd.Series(dtype=float)
    )
    humidity_off = (
        df.loc[~df["work_hours"], humidity_column].dropna() if humidity_column else pd.Series(dtype=float)
    )
    noise = df[noise_column].dropna() if noise_column else pd.Series(dtype=float)
    noise_work = df.loc[df["work_hours"], noise_column].dropna() if noise_column else pd.Series(dtype=float)
    noise_off = df.loc[~df["work_hours"], noise_column].dropna() if noise_column else pd.Series(dtype=float)

    return RoomSummary(
        room=room,
        start=df.index.min().isoformat(),
        end=df.index.max().isoformat(),
        interval_count=int(len(df)),
        co2_mean=round_or_none(co2.mean() if not co2.empty else None),
        co2_max=round_or_none(co2.max() if not co2.empty else None),
        pct_co2_above_800=round_or_none((co2.gt(800).mean() * 100) if not co2.empty else None),
        pct_co2_above_1000=round_or_none((co2.gt(1000).mean() * 100) if not co2.empty else None),
        temp_mean=round_or_none(temp.mean() if not temp.empty else None),
        temp_work_mean=round_or_none(temp_work.mean() if not temp_work.empty else None),
        temp_off_mean=round_or_none(temp_off.mean() if not temp_off.empty else None),
        temp_min=round_or_none(temp.min() if not temp.empty else None),
        temp_max=round_or_none(temp.max() if not temp.empty else None),
        voc_mean=round_or_none(voc.mean() if not voc.empty else None),
        voc_work_mean=round_or_none(voc_work.mean() if not voc_work.empty else None),
        voc_off_mean=round_or_none(voc_off.mean() if not voc_off.empty else None),
        voc_max=round_or_none(voc.max() if not voc.empty else None),
        humidity_mean=round_or_none(humidity.mean() if not humidity.empty else None),
        humidity_work_mean=round_or_none(humidity_work.mean() if not humidity_work.empty else None),
        humidity_off_mean=round_or_none(humidity_off.mean() if not humidity_off.empty else None),
        humidity_max=round_or_none(humidity.max() if not humidity.empty else None),
        noise_mean=round_or_none(noise.mean() if not noise.empty else None),
        noise_work_mean=round_or_none(noise_work.mean() if not noise_work.empty else None),
        noise_off_mean=round_or_none(noise_off.mean() if not noise_off.empty else None),
        noise_max=round_or_none(noise.max() if not noise.empty else None),
    )


def summarize_ahu(name: str, df: pd.DataFrame) -> AhuSummary:
    discharge = df.get("Discharge Air Temp_value", pd.Series(dtype=float)).dropna()
    outside = df.get("Outside Air Temp_value", pd.Series(dtype=float)).dropna()
    zone_co2 = df.get("Zone CO2_value", pd.Series(dtype=float)).dropna()
    return AhuSummary(
        ahu=name,
        start=df.index.min().isoformat(),
        end=df.index.max().isoformat(),
        interval_count=int(len(df)),
        discharge_temp_mean=round_or_none(discharge.mean() if not discharge.empty else None),
        discharge_temp_max=round_or_none(discharge.max() if not discharge.empty else None),
        outside_temp_mean=round_or_none(outside.mean() if not outside.empty else None),
        zone_co2_mean=round_or_none(zone_co2.mean() if not zone_co2.empty else None),
        zone_co2_max=round_or_none(zone_co2.max() if not zone_co2.empty else None),
    )


def plot_whole_building(df: pd.DataFrame, output_dir: Path) -> None:
    ensure_parent(output_dir / "placeholder")
    hourly = (
        df.groupby(["day_type", "hour"], as_index=False)["consumption_kwh"]
        .mean()
        .rename(columns={"consumption_kwh": "avg_consumption_kwh"})
    )
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=hourly, x="hour", y="avg_consumption_kwh", hue="day_type", marker="o")
    plt.axvspan(6, 17, color="gold", alpha=0.08, label="Working hours")
    plt.title("Average Whole-Building Energy Use by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Consumption (kWh)")
    plt.tight_layout()
    plt.savefig(output_dir / "whole_building_hourly_profile.png", dpi=180)
    plt.close()


def plot_rooms(rooms: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    ensure_parent(output_dir / "placeholder")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    for room, df in rooms.items():
        temp_column = next((column for column in df.columns if "Temp_value" in column), None)
        if temp_column:
            subset = df[[temp_column]].dropna().reset_index()
            subset["room"] = room
            sns.lineplot(data=subset, x="time", y=temp_column, ax=axes[0], label=f"Room {room}")
    axes[0].set_title("Zone Temperature Trends")
    axes[0].set_ylabel("Temperature (F)")

    for room, df in rooms.items():
        co2_column = next((column for column in df.columns if "CO2_value" in column), None)
        if co2_column:
            subset = df[[co2_column]].dropna().reset_index()
            subset["room"] = room
            sns.lineplot(data=subset, x="time", y=co2_column, ax=axes[1], label=f"Room {room}")
    axes[1].axhline(800, color="crimson", linestyle="--", linewidth=1, label="800 ppm threshold")
    axes[1].set_title("CO2 Trends by Room")
    axes[1].set_ylabel("CO2 (ppm)")
    axes[1].set_xlabel("Timestamp")
    for axis in axes:
        axis.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "room_temperature_co2_timeseries.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=False)

    for room, df in rooms.items():
        temp_column = next((column for column in df.columns if "Temp_value" in column), None)
        if temp_column:
            subset = df[[temp_column]].dropna().reset_index()
            sns.lineplot(data=subset, x="time", y=temp_column, ax=axes[0], label=f"Room {room}")
    axes[0].set_title("Zone Temperature Signals")
    axes[0].set_ylabel("Temperature (F)")

    room_354 = rooms.get("354")
    if room_354 is not None and "VOC_value" in room_354:
        subset = room_354[["VOC_value"]].dropna().reset_index()
        sns.lineplot(data=subset, x="time", y="VOC_value", ax=axes[1], label="Room 354 VOC")
    axes[1].set_title("Room 354 VOC Trend")
    axes[1].set_ylabel("VOC")

    room_361 = rooms.get("361")
    if room_361 is not None and "Zone Air Humid_value" in room_361:
        subset = room_361[["Zone Air Humid_value"]].dropna().reset_index()
        sns.lineplot(
            data=subset,
            x="time",
            y="Zone Air Humid_value",
            ax=axes[2],
            label="Room 361 Humidity",
        )
    axes[2].set_title("Room 361 Humidity Trend")
    axes[2].set_ylabel("Humidity (%)")

    if room_354 is not None and "Noise Level_value" in room_354:
        subset = room_354[["Noise Level_value"]].dropna().reset_index()
        sns.lineplot(
            data=subset,
            x="time",
            y="Noise Level_value",
            ax=axes[3],
            label="Room 354 Vibration/Noise Proxy",
        )
    axes[3].set_title("Room 354 Floor-Vibration Proxy Trend")
    axes[3].set_ylabel("Noise Level")
    axes[3].set_xlabel("Timestamp")

    for axis in axes:
        if axis.lines:
            axis.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "room_occupancy_sensor_timeseries.png", dpi=180)
    plt.close()


def plot_ahu(ahus: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    ensure_parent(output_dir / "placeholder")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    for name, df in ahus.items():
        if "Discharge Air Temp_value" in df:
            subset = df[["Discharge Air Temp_value"]].dropna().reset_index()
            sns.lineplot(data=subset, x="time", y="Discharge Air Temp_value", ax=axes[0], label=name)
    axes[0].set_title("AHU Discharge Air Temperature")
    axes[0].set_ylabel("Temperature (F)")

    for name, df in ahus.items():
        if "Zone CO2_value" in df:
            subset = df[["Zone CO2_value"]].dropna().reset_index()
            sns.lineplot(data=subset, x="time", y="Zone CO2_value", ax=axes[1], label=name)
    axes[1].axhline(800, color="crimson", linestyle="--", linewidth=1, label="800 ppm threshold")
    axes[1].set_title("AHU Zone CO2 Trend")
    axes[1].set_ylabel("CO2 (ppm)")
    axes[1].set_xlabel("Timestamp")
    for axis in axes:
        axis.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "ahu_discharge_temp_and_co2.png", dpi=180)
    plt.close()


def build_summary() -> Dict[str, object]:
    sns.set_theme(style="whitegrid")

    whole_building = load_whole_building()
    rooms = {room: load_room(room) for room in ("354", "361")}
    ahus = {
        "AHU_01": load_ahu("AHU_01_26121_26219_*.csv", "AHU_01"),
        "AHU_02": load_ahu("AHU_02_26121_26219_*.csv", "AHU_02"),
        "AHU_03": load_ahu("AHU_03_26121_26219_*.csv", "AHU_03"),
    }

    return {
        "whole_building": summarize_whole_building(whole_building),
        "rooms": [asdict(summarize_room(room, rooms[room])) for room in rooms],
        "ahus": [asdict(summarize_ahu(name, ahus[name])) for name in ahus],
    }


def create_outputs(output_dir: Path, summary_path: Path | None) -> Dict[str, object]:
    sns.set_theme(style="whitegrid")

    whole_building = load_whole_building()
    rooms = {room: load_room(room) for room in ("354", "361")}
    ahus = {
        "AHU_01": load_ahu("AHU_01_26121_26219_*.csv", "AHU_01"),
        "AHU_02": load_ahu("AHU_02_26121_26219_*.csv", "AHU_02"),
        "AHU_03": load_ahu("AHU_03_26121_26219_*.csv", "AHU_03"),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_whole_building(whole_building, output_dir)
    plot_rooms(rooms, output_dir)
    plot_ahu(ahus, output_dir)

    summary = {
        "whole_building": summarize_whole_building(whole_building),
        "rooms": [asdict(summarize_room(room, rooms[room])) for room in rooms],
        "ahus": [asdict(summarize_ahu(name, ahus[name])) for name in ahus],
    }
    if summary_path is not None:
        ensure_parent(summary_path)
        summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HVAC preliminary analysis figures and summary.")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "reports" / "figures"),
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--summary-json",
        default=str(ROOT / "reports" / "analysis_summary.json"),
        help="Path for the generated summary JSON.",
    )
    args = parser.parse_args()

    summary = create_outputs(Path(args.output_dir), Path(args.summary_json))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

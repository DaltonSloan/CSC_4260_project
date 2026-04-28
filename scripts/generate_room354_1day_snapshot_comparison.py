from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "reports" / "room354_1day_people_estimated_each_measurement.csv"
OUTPUT_FIGURE = ROOT / "reports" / "figures" / "room354_1day_snapshot_comparison.png"
OUTPUT_IMAGE = ROOT / "image.png"
OUTPUT_TABLE = ROOT / "reports" / "room354_1day_snapshot_comparison.csv"

LOCAL_TZ = "America/Chicago"
DAY_START_HOUR = 7
DAY_END_HOUR = 18
MIN_GAP_MINUTES = 90
WINDOW_MINUTES = 10
NUM_SNAPSHOTS = 3

ROOM_VOLUME_FT3 = 50 * 30 * 15
ROOM_VOLUME_M3 = ROOM_VOLUME_FT3 * 0.0283168
C_OUT = 420.0
G_PERSON = 0.018

MEASUREMENT_COLOR = "#263238"
PHYSICS_COLOR = "#9CB1BC"
BLEND_COLOR = "#8E24AA"


def load_room354_1day() -> pd.DataFrame:
    df = pd.read_csv(INPUT_FILE, parse_dates=["ts"]).sort_values("ts").reset_index(drop=True)
    df["local_ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(LOCAL_TZ)
    airflow_ach = df["flow_cfm"] * 60.0 / ROOM_VOLUME_FT3
    q_m3_h = airflow_ach * ROOM_VOLUME_M3
    co2_delta = (df["co2_ppm"] - C_OUT).clip(lower=0)
    df["people_physics"] = q_m3_h * (co2_delta * 1e-6) / G_PERSON
    return df


def select_snapshot_rows(df: pd.DataFrame) -> pd.DataFrame:
    local_day = df["local_ts"].dt.date.mode().iat[0]
    daytime = df.loc[
        (df["local_ts"].dt.date == local_day)
        & (df["local_ts"].dt.hour >= DAY_START_HOUR)
        & (df["local_ts"].dt.hour <= DAY_END_HOUR)
    ].copy()

    values = daytime["people_est_smooth_5min"].to_numpy()
    peak_indices: list[int] = []
    for idx in range(1, len(daytime) - 1):
        if values[idx] >= values[idx - 1] and values[idx] >= values[idx + 1]:
            peak_indices.append(daytime.index[idx])

    peaks = daytime.loc[peak_indices].sort_values("people_est_smooth_5min", ascending=False)
    selected: list[pd.Series] = []
    min_gap = pd.Timedelta(minutes=MIN_GAP_MINUTES)
    for row in peaks.itertuples(index=False):
        if all(abs(row.local_ts - picked["local_ts"]) >= min_gap for picked in selected):
            selected.append(pd.Series(row._asdict()))
        if len(selected) == NUM_SNAPSHOTS:
            break

    if len(selected) < NUM_SNAPSHOTS:
        raise RuntimeError("Not enough distinct Room 354 daytime peaks were found.")

    selected_df = pd.DataFrame(selected).sort_values("local_ts").reset_index(drop=True)
    return selected_df


def build_comparison_table(df: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for idx, row in selected.iterrows():
        window = df.loc[
            (df["local_ts"] >= row["local_ts"] - pd.Timedelta(minutes=WINDOW_MINUTES))
            & (df["local_ts"] <= row["local_ts"] + pd.Timedelta(minutes=WINDOW_MINUTES))
        ].copy()

        records.append(
            {
                "snapshot_id": f"S{idx + 1}",
                "utc_timestamp": row["ts"].isoformat(),
                "local_timestamp": row["local_ts"].strftime("%Y-%m-%d %H:%M:%S %Z"),
                "label": f"S{idx + 1} ({row['local_ts'].strftime('%m-%d %H:%M %Z')})",
                "co2_ppm": float(row["co2_ppm"]),
                "flow_cfm": float(row["flow_cfm"]),
                "measurement_people": float(row["people_est_smooth_5min"]),
                "measurement_window_std": float(window["people_est_smooth_5min"].std(ddof=1)),
                "physics_people": float(row["people_physics"]),
                "physics_window_std": float(window["people_physics"].std(ddof=1)),
                "blended_people": float(row["people_est"]),
                "blended_window_std": float(window["people_est"].std(ddof=1)),
            }
        )

    comparison = pd.DataFrame.from_records(records)
    comparison.to_csv(OUTPUT_TABLE, index=False)
    return comparison


def add_value_labels(ax: plt.Axes, bars, values: np.ndarray) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.75,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#1f1f1f",
        )


def plot_comparison(comparison: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    labels = comparison["label"].to_numpy()
    measured = comparison["measurement_people"].to_numpy()
    measured_err = comparison["measurement_window_std"].to_numpy()
    physics = comparison["physics_people"].to_numpy()
    physics_err = comparison["physics_window_std"].to_numpy()
    blended = comparison["blended_people"].to_numpy()
    blended_err = comparison["blended_window_std"].to_numpy()

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12.5, 7.0), dpi=200)
    bars_measurement = ax.bar(
        x - width,
        measured,
        width,
        yerr=measured_err,
        capsize=4,
        color=MEASUREMENT_COLOR,
        edgecolor="white",
        linewidth=1.2,
        label="5-min smoothed measurement",
    )
    bars_physics = ax.bar(
        x,
        physics,
        width,
        yerr=physics_err,
        capsize=4,
        color=PHYSICS_COLOR,
        edgecolor="white",
        linewidth=1.2,
        label="Physics baseline",
    )
    bars_blended = ax.bar(
        x + width,
        blended,
        width,
        yerr=blended_err,
        capsize=4,
        color=BLEND_COLOR,
        edgecolor="white",
        linewidth=1.2,
        label="Blended estimate",
    )

    ax.set_title("Room 354 One-Day Snapshot Comparison", fontsize=20, pad=16)
    ax.set_ylabel("people", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(loc="upper left", fontsize=11, frameon=True)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(measured.max(), physics.max(), blended.max()) + 8)

    add_value_labels(ax, bars_measurement, measured)
    add_value_labels(ax, bars_physics, physics)
    add_value_labels(ax, bars_blended, blended)

    footnote = (
        "Snapshots are the top three daytime local peaks on 2026-03-31 in Room 354, "
        "selected with at least 90 minutes between peaks. Error bars show +/-1 std "
        "within a +/-10 minute local window."
    )
    fig.text(0.01, 0.01, footnote, fontsize=10, color="#4b5563")

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIGURE, bbox_inches="tight")
    fig.savefig(OUTPUT_IMAGE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_room354_1day()
    selected = select_snapshot_rows(df)
    comparison = build_comparison_table(df, selected)
    plot_comparison(comparison)

    print(comparison.to_string(index=False))
    print(f"Saved {OUTPUT_FIGURE}")
    print(f"Saved {OUTPUT_IMAGE}")
    print(f"Saved {OUTPUT_TABLE}")


if __name__ == "__main__":
    main()

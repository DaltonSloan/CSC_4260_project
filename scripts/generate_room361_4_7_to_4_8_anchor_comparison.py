from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.occupancy_pseudo_labels import load_room_fpb_timeseries, normalize_anchor_table

INPUT_FILE = ROOT / "data" / "4_7To4_8.csv"
ANCHOR_FILE = ROOT / "data" / "room361_manual_anchors.csv"
HYBRID_FILE = ROOT / "reports" / "room361_pipeline" / "best_anchor_window_evaluation.csv"
OUTPUT_FIGURE = ROOT / "reports" / "figures" / "room361_4_7_to_4_8_anchor_comparison.png"
OUTPUT_TABLE = ROOT / "reports" / "room361_4_7_to_4_8_anchor_comparison.csv"
OUTPUT_IMAGE = ROOT / "image.png"

TRUE_COLOR = "#263238"
PHYSICS_COLOR = "#9CB1BC"
HYBRID_COLOR = "#8E24AA"
HYBRID_ERR_COLOR = "#4A148C"


def nearest_timestamp(index: pd.Index, target: pd.Timestamp) -> pd.Timestamp:
    if len(index) == 0:
        raise ValueError("Cannot find nearest timestamp in an empty index.")
    return index[index.get_indexer([pd.Timestamp(target)], method="nearest")[0]]


def build_comparison_table() -> pd.DataFrame:
    sensor = load_room_fpb_timeseries(INPUT_FILE)
    anchors = normalize_anchor_table(pd.read_csv(ANCHOR_FILE))
    hybrid = pd.read_csv(HYBRID_FILE, parse_dates=["anchor_time", "anchor_timestamp_used"])

    records: list[dict[str, object]] = []
    for anchor in anchors.itertuples(index=False):
        window = sensor.loc[(sensor.index >= anchor.window_start) & (sensor.index <= anchor.window_end)].copy()
        anchor_timestamp = nearest_timestamp(window.index, anchor.anchor_time)
        hybrid_row = hybrid.loc[hybrid["anchor_id"] == anchor.anchor_id]
        if hybrid_row.empty:
            raise ValueError(f"Missing hybrid prediction row for {anchor.anchor_id}.")
        hybrid_row = hybrid_row.iloc[0]

        records.append(
            {
                "anchor_id": anchor.anchor_id,
                "anchor_time": anchor.anchor_time,
                "anchor_timestamp_used": anchor_timestamp,
                "label": f"{anchor.anchor_id} ({anchor.anchor_time:%m-%d %H:%M})",
                "true_count": float(anchor.count),
                "true_low": float(anchor.occ_low),
                "true_high": float(anchor.occ_high),
                "physics_pred_at_anchor": float(sensor.loc[anchor_timestamp, "occ_physics_est_smooth"]),
                "physics_window_std": float(window["occ_physics_est_smooth"].std(ddof=1)),
                "physics_window_mean": float(window["occ_physics_est_smooth"].mean()),
                "best_pred_at_anchor": float(hybrid_row["best_pred_at_anchor"]),
                "best_pred_lower_at_anchor": float(hybrid_row["best_pred_lower_at_anchor"]),
                "best_pred_upper_at_anchor": float(hybrid_row["best_pred_upper_at_anchor"]),
                "best_pred_confidence_at_anchor": float(hybrid_row["best_pred_confidence_at_anchor"]),
            }
        )

    comparison = pd.DataFrame.from_records(records).sort_values("anchor_time").reset_index(drop=True)
    comparison.to_csv(OUTPUT_TABLE, index=False)
    return comparison


def add_labels(ax: plt.Axes, bars, values: np.ndarray) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.75,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#212121",
        )


def plot_comparison(comparison: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    x = np.arange(len(comparison))
    width = 0.25

    true_values = comparison["true_count"].to_numpy()
    physics_values = comparison["physics_pred_at_anchor"].to_numpy()
    hybrid_values = comparison["best_pred_at_anchor"].to_numpy()
    hybrid_yerr = np.vstack(
        [
            hybrid_values - comparison["best_pred_lower_at_anchor"].to_numpy(),
            comparison["best_pred_upper_at_anchor"].to_numpy() - hybrid_values,
        ]
    )

    fig, ax = plt.subplots(figsize=(12.5, 6.8), dpi=200)
    true_bars = ax.bar(
        x - width,
        true_values,
        width=width,
        label="True count",
        color=TRUE_COLOR,
        edgecolor="white",
        linewidth=1.2,
    )
    physics_bars = ax.bar(
        x,
        physics_values,
        width=width,
        label="Physics baseline",
        color=PHYSICS_COLOR,
        edgecolor="white",
        linewidth=1.2,
    )
    hybrid_bars = ax.bar(
        x + width,
        hybrid_values,
        width=width,
        label="Best hybrid",
        color=HYBRID_COLOR,
        edgecolor="white",
        linewidth=1.2,
    )

    ax.errorbar(
        x + width,
        hybrid_values,
        yerr=hybrid_yerr,
        fmt="none",
        ecolor=HYBRID_ERR_COLOR,
        elinewidth=1.6,
        capsize=4,
    )

    ax.set_title("Held-Out Anchor Predictions from 4/7 to 4/8 Room 361 Data", fontsize=20, pad=16)
    ax.set_ylabel("people", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison["label"], rotation=15, ha="right", fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(loc="upper left", fontsize=11, frameon=True)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(true_values.max(), hybrid_values.max()) + 8)

    add_labels(ax, true_bars, true_values)
    add_labels(ax, physics_bars, physics_values)
    add_labels(ax, hybrid_bars, hybrid_values)

    footnote = (
        "Physics bars were recalculated from the raw Room 361 FPB slice in data/4_7To4_8.csv. "
        "Hybrid bars and intervals come from the current held-out Room 361 pipeline outputs for those same anchors."
    )
    fig.text(0.01, 0.01, footnote, fontsize=10, color="#4B5563")

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIGURE, bbox_inches="tight")
    fig.savefig(OUTPUT_IMAGE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    comparison = build_comparison_table()
    plot_comparison(comparison)
    print(comparison.to_string(index=False))
    print(f"Saved {OUTPUT_FIGURE}")
    print(f"Saved {OUTPUT_IMAGE}")
    print(f"Saved {OUTPUT_TABLE}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the updated project report PDF and embed Room 354 notebook figures."""

from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer


ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = ROOT / "reports" / "updated_project_report.md"
REPORT_PDF = ROOT / "reports" / "updated_project_report.pdf"
FIG_DIR = ROOT / "reports" / "figures"
IMAGE_PATTERN = re.compile(r"^!\[(.*?)\]\((.*?)\)$")
ROOM_VOLUME_FT3 = 50 * 30 * 15
ROOM_VOLUME_M3 = ROOM_VOLUME_FT3 * 0.0283168
BASELINE_ACH = 4.0


def load_point_history(path: Path, rename_map: dict[str, str]) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw["time"] = pd.to_datetime(raw["dateTimeUtc"], errors="coerce")
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")

    wide = (
        raw.loc[raw["pointDisplayName"].isin(rename_map), ["time", "pointDisplayName", "value"]]
        .dropna(subset=["time"])
        .pivot_table(index="time", columns="pointDisplayName", values="value", aggfunc="mean")
        .sort_index()
        .rename(columns=rename_map)
        .resample("5min")
        .mean()
    )
    wide.columns.name = None
    return wide


def load_room354_data() -> pd.DataFrame:
    iaq = load_point_history(
        ROOT / "data" / "354_IAQ_30day(03-28-2026).csv",
        {
            "CO2": "co2_iaq",
            "VOC": "voc",
            "Zone Air Humid": "humidity_iaq",
            "Zone Temp": "temp_iaq",
        },
    )

    fpb = load_point_history(
        ROOT / "data" / "354_FPB_30day(03-28-2026).csv",
        {
            "Zone CO2": "co2_fpb",
            "Zone Air Humid": "humidity_fpb",
            "Zone Temp": "temp_fpb",
            "Discharge Air Flow": "airflow_cfm",
        },
    )

    merged = iaq.join(fpb, how="outer")
    merged["co2"] = merged[["co2_iaq", "co2_fpb"]].mean(axis=1)
    merged["humidity"] = merged[["humidity_iaq", "humidity_fpb"]].mean(axis=1)
    merged["temperature"] = merged[["temp_iaq", "temp_fpb"]].mean(axis=1)
    merged["airflow_ach"] = merged["airflow_cfm"] * 60 / ROOM_VOLUME_FT3
    merged = merged[["voc", "co2", "humidity", "temperature", "airflow_cfm", "airflow_ach"]].copy()
    return merged


def minmax_robust(series: pd.Series, q_low: float = 0.05, q_high: float = 0.95) -> pd.Series:
    lo = series.quantile(q_low)
    hi = series.quantile(q_high)
    if pd.isna(lo) or pd.isna(hi) or hi <= lo:
        return pd.Series(0.0, index=series.index)
    return ((series - lo) / (hi - lo)).clip(0, 1)


def add_occupancy_estimate(merged: pd.DataFrame) -> pd.DataFrame:
    merged = merged.copy()
    merged["ach_effective"] = merged["airflow_ach"].fillna(BASELINE_ACH)
    q_m3_h = merged["ach_effective"] * ROOM_VOLUME_M3
    co2_delta = (merged["co2"] - 420.0).clip(lower=0)
    merged["people_co2_anchor"] = q_m3_h * (co2_delta * 1e-6) / 0.018

    feature_index = (
        0.45 * minmax_robust(merged["co2"]).fillna(0)
        + 0.30 * minmax_robust(merged["voc"]).fillna(0)
        + 0.15 * minmax_robust(merged["humidity"]).fillna(0)
        + 0.10 * minmax_robust(merged["temperature"]).fillna(0)
    )
    anchor_p95 = merged["people_co2_anchor"].quantile(0.95)
    merged["people_feature_scaled"] = feature_index * anchor_p95
    merged["people_estimated"] = (
        0.70 * merged["people_co2_anchor"].fillna(0)
        + 0.30 * merged["people_feature_scaled"].fillna(0)
    ).rolling(3, min_periods=1).mean()
    return merged


def export_room354_figures() -> None:
    sns.set_theme(style="whitegrid")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    merged = add_occupancy_estimate(load_room354_data())
    plot_df = merged.reset_index()

    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    sns.lineplot(data=plot_df, x="time", y="voc", ax=axes[0], label="VOC")
    sns.lineplot(data=plot_df, x="time", y="co2", ax=axes[0], label="CO2 (ppm)")
    axes[0].set_title("VOC and CO2 Over Time (Room 354)")
    axes[0].set_ylabel("VOC / ppm")
    axes[0].legend(loc="upper left")

    sns.lineplot(data=plot_df, x="time", y="humidity", ax=axes[1], color="teal", label="Humidity")
    axes[1].set_title("Humidity Over Time (Room 354)")
    axes[1].set_ylabel("Humidity (%)")
    axes[1].legend(loc="upper left")

    sns.lineplot(
        data=plot_df,
        x="time",
        y="temperature",
        ax=axes[2],
        color="orange",
        label="Temperature",
    )
    axes[2].set_title("Temperature Over Time (Room 354)")
    axes[2].set_ylabel("Temperature (F)")
    axes[2].legend(loc="upper left")

    sns.lineplot(
        data=plot_df,
        x="time",
        y="airflow_cfm",
        ax=axes[3],
        color="purple",
        label="Discharge Air Flow (cfm)",
    )
    axes[3].set_title("Discharge Air Flow Over Time (Room 354)")
    axes[3].set_ylabel("Air Flow (cfm)")
    axes[3].set_xlabel("Time")
    axes[3].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "room354_feature_comparison.png", dpi=180)
    plt.close(fig)

    corr_df = merged[
        ["voc", "co2", "humidity", "temperature", "airflow_cfm", "people_estimated"]
    ].rename(
        columns={
            "voc": "VOC",
            "co2": "CO2",
            "humidity": "Humidity",
            "temperature": "Temperature",
            "airflow_cfm": "Air Flow (cfm)",
            "people_estimated": "Estimated People",
        }
    )
    corr_matrix = corr_df.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Correlation Matrix for Room 354 Features")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "room354_correlation_matrix.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 4))
    sns.lineplot(
        x=merged.index,
        y=merged["people_estimated"].rolling(3, min_periods=1).mean(),
        color="black",
        label="Estimated People",
        ax=ax,
    )
    ax.set_title("Estimated Occupancy Over Time (Airflow-Aware CO2 Anchor)")
    ax.set_ylabel("Estimated People")
    ax.set_xlabel("Time")

    ax2 = ax.twinx()
    sns.lineplot(
        x=merged.index,
        y=merged["airflow_cfm"],
        color="purple",
        alpha=0.3,
        label="Discharge Air Flow (cfm)",
        ax=ax2,
    )
    ax2.set_ylabel("Air Flow (cfm)")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "room354_estimated_occupancy.png", dpi=180)
    plt.close(fig)


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#1f2937"),
            spaceAfter=12,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13.5,
            leading=17,
            textColor=colors.HexColor("#111827"),
            spaceBefore=9,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.2,
            leading=13.5,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BulletBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.2,
            leading=13.5,
            leftIndent=14,
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="NumberBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.2,
            leading=13.5,
            leftIndent=14,
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=9.5,
            leading=12,
            textColor=colors.HexColor("#374151"),
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=8,
        )
    )
    return styles


def fmt_inline(text: str) -> str:
    text = escape(text)
    return re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)


def is_special(line: str) -> bool:
    stripped = line.strip()
    return (
        stripped.startswith("# ")
        or stripped.startswith("## ")
        or stripped.startswith("- ")
        or IMAGE_PATTERN.match(stripped) is not None
        or re.match(r"^\d+\.\s+", stripped) is not None
    )


def add_page_number(canvas, doc) -> None:
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawRightString(7.5 * inch, 0.5 * inch, f"Page {doc.page}")


def build_pdf() -> None:
    export_room354_figures()

    styles = build_styles()
    lines = REPORT_MD.read_text(encoding="utf-8").splitlines()
    story = []
    i = 0

    while i < len(lines):
        stripped = lines[i].strip()

        if not stripped:
            story.append(Spacer(1, 0.08 * inch))
            i += 1
            continue

        if stripped.startswith("# "):
            story.append(Paragraph(fmt_inline(stripped[2:]), styles["ReportTitle"]))
            i += 1
            continue

        if stripped.startswith("## "):
            story.append(Paragraph(fmt_inline(stripped[3:]), styles["Section"]))
            i += 1
            continue

        if stripped.startswith("- "):
            story.append(Paragraph(fmt_inline(stripped[2:]), styles["BulletBody"], bulletText="•"))
            i += 1
            continue

        image_match = IMAGE_PATTERN.match(stripped)
        if image_match:
            caption, raw_path = image_match.groups()
            image_path = Path(raw_path)
            if not image_path.is_absolute():
                image_path = (REPORT_MD.parent / image_path).resolve()
            if image_path.exists():
                width, height = ImageReader(str(image_path)).getSize()
                max_width = 6.7 * inch
                max_height = 5.5 * inch
                scale = min(max_width / width, max_height / height)
                story.append(Image(str(image_path), width=width * scale, height=height * scale))
                story.append(Paragraph(fmt_inline(caption), styles["Caption"]))
            i += 1
            continue

        number_match = re.match(r"^(\d+)\.\s+(.*)", stripped)
        if number_match:
            num, body = number_match.groups()
            story.append(Paragraph(fmt_inline(body), styles["NumberBody"], bulletText=f"{num}."))
            i += 1
            continue

        para_lines = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if not nxt or is_special(lines[i]):
                break
            para_lines.append(nxt)
            i += 1
        story.append(Paragraph(fmt_inline(" ".join(para_lines)), styles["Body"]))

    doc = SimpleDocTemplate(
        str(REPORT_PDF),
        pagesize=letter,
        leftMargin=0.8 * inch,
        rightMargin=0.8 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="Room 354 Occupancy Report",
    )
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


if __name__ == "__main__":
    build_pdf()

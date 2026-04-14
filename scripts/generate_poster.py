from __future__ import annotations

from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/workspaces/CSC_4260_project")
OUTPUT_DIR = ROOT / "reports" / "poster"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANVAS_W = 5600
CANVAS_H = 3150
MARGIN_X = 90
MARGIN_Y = 70
GUTTER = 60
HEADER_H = 230

TT_PURPLE = (92, 52, 142)
TT_GOLD = (194, 145, 55)
INK = (24, 24, 28)
MUTED = (95, 99, 104)
PANEL = (248, 246, 252)
PANEL_BORDER = (219, 210, 235)
WHITE = (255, 255, 255)


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    font_name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{font_name}", size=size)


FONT_TITLE = load_font(54, bold=True)
FONT_SUBTITLE = load_font(32, bold=False)
FONT_SECTION = load_font(34, bold=True)
FONT_BODY = load_font(24, bold=False)
FONT_BODY_BOLD = load_font(24, bold=True)
FONT_SMALL = load_font(18, bold=False)
FONT_CAPTION = load_font(20, bold=False)
FONT_METRIC = load_font(26, bold=True)


def draw_wrapped_text(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], font, fill, width_px: int, line_gap: int = 8):
    x, y = xy
    paragraphs = text.split("\n")
    current_y = y
    for para in paragraphs:
        if not para.strip():
            current_y += font.size + line_gap
            continue
        avg_char_width = max(font.size * 0.56, 1)
        max_chars = max(int(width_px / avg_char_width), 12)
        lines = wrap(para, width=max_chars, break_long_words=False, break_on_hyphens=False) or [""]
        for line in lines:
            draw.text((x, current_y), line, font=font, fill=fill)
            bbox = draw.textbbox((x, current_y), line, font=font)
            current_y = bbox[3] + line_gap
    return current_y


def draw_bullets(draw: ImageDraw.ImageDraw, items: list[str], xy: tuple[int, int], width_px: int, bullet_indent: int = 30, font=FONT_BODY, fill=INK, line_gap: int = 7):
    x, y = xy
    current_y = y
    for item in items:
        bullet_x = x
        text_x = x + bullet_indent
        draw.text((bullet_x, current_y), "•", font=FONT_BODY_BOLD, fill=TT_PURPLE)
        current_y = draw_wrapped_text(draw, item, (text_x, current_y), font, fill, width_px - bullet_indent, line_gap=line_gap)
        current_y += 6
    return current_y


def fit_image(path: Path, target_w: int, target_h: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    scale = min(target_w / image.width, target_h / image.height)
    resized = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    background = Image.new("RGB", (target_w, target_h), WHITE)
    offset = ((target_w - resized.width) // 2, (target_h - resized.height) // 2)
    background.paste(resized, offset)
    return background


def draw_section_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=26, fill=PANEL, outline=PANEL_BORDER, width=3)
    draw.rounded_rectangle((x0, y0, x1, y0 + 62), radius=26, fill=(238, 233, 247), outline=PANEL_BORDER, width=0)
    draw.rectangle((x0, y0 + 40, x1, y0 + 62), fill=(238, 233, 247))
    draw.text((x0 + 26, y0 + 14), title, font=FONT_SECTION, fill=TT_PURPLE)


def paste_image(canvas: Image.Image, image: Image.Image, xy: tuple[int, int]):
    canvas.paste(image, xy)


def draw_metric_card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], label: str, value: str):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=20, fill=WHITE, outline=PANEL_BORDER, width=2)
    draw.text((x0 + 18, y0 + 16), value, font=FONT_METRIC, fill=TT_PURPLE)
    draw_wrapped_text(draw, label, (x0 + 18, y0 + 56), FONT_SMALL, MUTED, width_px=(x1 - x0 - 36), line_gap=4)


def main():
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), WHITE)
    draw = ImageDraw.Draw(canvas)

    draw.rounded_rectangle((0, 0, CANVAS_W, HEADER_H), radius=0, fill=WHITE)
    draw.rectangle((0, HEADER_H - 12, CANVAS_W, HEADER_H), fill=TT_PURPLE)

    logo_source = Image.open(ROOT / "Digital Twin.png").convert("RGB")
    logo_crop = logo_source.crop((0, 0, 450, 135))
    logo_crop.thumbnail((420, 120), Image.Resampling.LANCZOS)
    canvas.paste(logo_crop, (MARGIN_X, 42))

    title = "Passive Occupancy Estimation in Smart Classrooms Using CO2, VOC, Airflow, and Weak Supervision"
    authors = "Jun Han, Samuel Hartmann, Dalton Sloan, Garrett Green"
    subtitle = "Tennessee Tech University • Ashraf Islam Engineering Building (Rooms 354 and 361)"
    draw_wrapped_text(draw, title, (520, 34), FONT_TITLE, INK, width_px=CANVAS_W - 610, line_gap=4)
    draw.text((520, 132), authors, font=FONT_SUBTITLE, fill=INK)
    draw.text((520, 172), subtitle, font=FONT_SMALL, fill=MUTED)

    col_w = (CANVAS_W - 2 * MARGIN_X - 3 * GUTTER) // 4
    col_x = [MARGIN_X + i * (col_w + GUTTER) for i in range(4)]
    top_y = HEADER_H + 32

    problem_box = (col_x[0], top_y, col_x[0] + col_w, 960)
    dataset_box = (col_x[0], 990, col_x[0] + col_w, 1920)
    methods_box = (col_x[1], top_y, col_x[1] + col_w, 1170)
    story_box = (col_x[2], top_y, col_x[2] + col_w, 1920)
    results_box = (col_x[3], top_y, col_x[3] + col_w, 1440)
    conclusion_box = (col_x[3], 1470, col_x[3] + col_w, 2200)
    signal_box = (col_x[0], 1950, col_x[1] + col_w, CANVAS_H - 80)
    refs_box = (col_x[2], 1950, col_x[3] + col_w, CANVAS_H - 80)

    for box, title_text in [
        (problem_box, "Problem Statement"),
        (dataset_box, "Dataset"),
        (methods_box, "Methods"),
        (story_box, "Story"),
        (results_box, "Results"),
        (conclusion_box, "Conclusion"),
        (signal_box, "Signal Relationships"),
        (refs_box, "References"),
    ]:
        draw_section_panel(draw, box, title_text)

    # Problem statement
    px0, py0, px1, py1 = problem_box
    problem_text = (
        "Accurately estimating room occupancy without cameras or badge scanners remains difficult. "
        "HVAC systems often run on fixed schedules instead of actual room use, which wastes energy "
        "during low-occupancy periods.\n\n"
        "This project asks whether CO2, VOC, humidity, temperature, and airflow signals already present "
        "in smart buildings can be used to continuously estimate classroom occupancy in AIEB Rooms 354 and 361."
    )
    draw_wrapped_text(draw, problem_text, (px0 + 26, py0 + 88), FONT_BODY, INK, width_px=col_w - 52)
    draw_bullets(
        draw,
        [
            "Target classrooms hold roughly 35–60 students.",
            "No cameras or badge scans are required.",
            "Goal: infer occupancy states that can support occupancy-aware HVAC control.",
        ],
        (px0 + 26, py0 + 380),
        width_px=col_w - 52,
    )

    # Signal relationships
    gx0, gy0, gx1, gy1 = signal_box
    corr_plot = fit_image(ROOT / "reports" / "figures" / "room354_correlation_matrix.png", (gx1 - gx0) - 52, 620)
    paste_image(canvas, corr_plot, (gx0 + 26, gy0 + 88))
    draw.text((gx0 + 26, gy0 + 724), "Figure 3. Correlation structure across Room 354 sensor signals.", font=FONT_CAPTION, fill=MUTED)
    signal_text = (
        "CO2 carried the strongest relationship with estimated occupancy (r = 0.978), followed by VOC (r = 0.748). "
        "Humidity and temperature behaved more like supporting context variables, while discharge airflow had weak direct "
        "correlation because it mainly changes the dilution rate rather than the number of people in the room."
    )
    draw_wrapped_text(draw, signal_text, (gx0 + 26, gy0 + 770), FONT_BODY, INK, width_px=(gx1 - gx0 - 52))

    # Dataset
    dx0, dy0, dx1, dy1 = dataset_box
    draw_bullets(
        draw,
        [
            "Room 354 (primary): 30-day IAQ + FPB export from 2026-02-27 to 2026-03-29, resampled to 5-minute intervals, about 8,600 timesteps.",
            "Sensors: CO2 (ppm), VOC, humidity (%), temperature (F), and discharge airflow (cfm).",
            "Room 361 (validation): Apr 7–9, 2026 FPB export with 3 manual headcount anchors.",
            "Tennessee Tech enrollment records were parsed into class schedule windows for an is_class_time feature.",
        ],
        (dx0 + 26, dy0 + 92),
        width_px=col_w - 52,
    )
    dataset_para = (
        "Both exports were delivered as long-format point-history CSVs. We filtered the relevant points, "
        "pivoted them into room-level columns, resampled them to 5-minute intervals, and merged IAQ and FPB streams. "
        "Discharge airflow was converted to ACH using room volume. Missing humidity and temperature values were retained "
        "instead of imputed so we did not fabricate occupancy evidence."
    )
    draw_wrapped_text(draw, dataset_para, (dx0 + 26, dy0 + 488), FONT_BODY, INK, width_px=col_w - 52)

    # Methods
    mx0, my0, mx1, my1 = methods_box
    draw_bullets(
        draw,
        [
            "Physics-based CO2 mass-balance occupancy anchor using airflow-aware ACH.",
            "Blended estimate: 70% CO2 anchor + 30% VOC, humidity, and temperature index.",
            "Temporal features: 5–30 minute lags, rolling means, slopes, and interaction terms.",
            "Weak supervision: manual counts expanded into pseudo-labels with confidence weights.",
            "Hybrid physics + ML residual models: Ridge, Random Forest, and Gradient Boosting.",
        ],
        (mx0 + 26, my0 + 92),
        width_px=col_w - 52,
    )

    # Story
    sx0, sy0, sx1, sy1 = story_box
    story_plot = fit_image(ROOT / "reports" / "figures" / "room354_estimated_occupancy(1 month).png", col_w - 52, 620)
    paste_image(canvas, story_plot, (sx0 + 26, sy0 + 88))
    draw.text((sx0 + 26, sy0 + 724), "Figure 1. Room 354 airflow-aware occupancy estimate over 30 days.", font=FONT_CAPTION, fill=MUTED)

    story_text = (
        "CO2 and VOC rise together when students arrive. Their Pearson correlation is r = 0.751, while airflow acts as "
        "a ventilation diluter instead of a direct headcount signal (r = -0.064 with CO2). Replacing a fixed 4 ACH assumption "
        "with the measured median of 2.96 ACH lowered the mean estimate from 5.2 to 4.2 people and reduced the most extreme peaks by 35%."
    )
    draw_wrapped_text(draw, story_text, (sx0 + 26, sy0 + 770), FONT_BODY, INK, width_px=col_w - 52)

    story_para = (
        "The largest miss occurred at the 2026-04-07 14:38 anchor in Room 361: the true count was 33, but the physics model predicted about 11 "
        "because active ventilation near 1,130 cfm diluted CO2 faster than occupants could build it up. Outdoor CO2 sensitivity checks showed that "
        "tuning the background concentration alone could not close that gap."
    )
    draw_wrapped_text(draw, story_para, (sx0 + 26, sy0 + 1006), FONT_BODY, INK, width_px=col_w - 52)

    # Results
    rx0, ry0, rx1, ry1 = results_box
    result_plot = fit_image(ROOT / "reports" / "room361_pipeline" / "figures" / "room361_co2_flow_occupancy_timeseries.png", col_w - 52, 420)
    paste_image(canvas, result_plot, (rx0 + 26, ry0 + 88))
    draw.text((rx0 + 26, ry0 + 520), "Figure 2. Held-out Room 361 anchor-window comparison.", font=FONT_CAPTION, fill=MUTED)

    card_y = ry0 + 570
    card_w = (col_w - 70) // 2
    card_h = 122
    metric_cards = [
        ("CO2 ↔ estimated occupancy", "r = 0.978"),
        ("VOC ↔ estimated occupancy", "r = 0.748"),
        ("Blended model percentiles", "P90 ≈ 13, P99 ≈ 30"),
        ("Extreme spikes > 30 people", "150 → 85 (-43%)"),
        ("Physics baseline on 3 anchors", "MAE 7.3, RMSE 12.5"),
        ("Gradient boosting residual", "MAE 8.5, RMSE 12.6"),
    ]
    for idx, (label, value) in enumerate(metric_cards):
        row = idx // 2
        col = idx % 2
        x0 = rx0 + 26 + col * (card_w + 18)
        y0 = card_y + row * (card_h + 16)
        draw_metric_card(draw, (x0, y0, x0 + card_w, y0 + card_h), label, value)

    # Conclusion
    cx0, cy0, cx1, cy1 = conclusion_box
    draw_bullets(
        draw,
        [
            "CO2 and VOC were the strongest passive occupancy indicators in this classroom setting.",
            "Measured airflow improved the estimates by correcting for real ventilation rates rather than assuming a fixed ACH.",
            "The hybrid physics + ML workflow transfers to new rooms with minimal labeled data.",
            "Three manual anchors are not enough for robust ML training, so more ground-truth counts are still needed.",
            "A validated occupancy model could support occupancy-aware HVAC scheduling and reduce energy waste during low-use periods.",
        ],
        (cx0 + 26, cy0 + 92),
        width_px=col_w - 52,
    )

    # References
    fx0, fy0, fx1, fy1 = refs_box
    refs = [
        "ASHRAE 62.1. Ventilation for Acceptable Indoor Air Quality.",
        "Fisk, W. J., et al. (2011). CO2-based occupancy estimation in commercial buildings.",
        "Tennessee Tech AIEB point-history sensor exports (FPB and IAQ, 2026).",
        "Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12:2825–2830.",
        "Room 361 manual anchor counts and weak-supervision pipeline outputs in this repository.",
    ]
    ref_text = (
        "Poster takeaway: passive indoor environmental signals are strong enough to estimate coarse occupancy states, "
        "but robust validation still depends on a larger set of direct headcount anchors."
    )
    draw_wrapped_text(draw, ref_text, (fx0 + 26, fy0 + 92), FONT_BODY_BOLD, INK, width_px=(fx1 - fx0 - 52))
    draw_bullets(draw, refs, (fx0 + 26, fy0 + 180), width_px=(fx1 - fx0 - 52), font=FONT_BODY)

    output_png = OUTPUT_DIR / "aieb_occupancy_poster.png"
    output_jpg = OUTPUT_DIR / "aieb_occupancy_poster.jpg"
    canvas.save(output_png)
    canvas.save(output_jpg, quality=95)
    print(f"Saved {output_png}")
    print(f"Saved {output_jpg}")


if __name__ == "__main__":
    main()

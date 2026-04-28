from __future__ import annotations

from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/workspaces/CSC_4260_project")
OUT_DIR = ROOT / "reports" / "poster"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANVAS_W = 5760
CANVAS_H = 4320
MARGIN = 120
GUTTER = 60
HEADER_H = 300

WHITE = (255, 255, 255)
INK = (28, 28, 34)
MUTED = (92, 96, 104)
PURPLE = (94, 53, 177)
PURPLE_DARK = (68, 39, 128)
LAVENDER = (244, 240, 251)
BORDER = (214, 204, 235)
GOLD = (189, 145, 59)


def font(size: int, *, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size=size)


TITLE_FONT = font(74, bold=True)
AUTHOR_FONT = font(34, bold=False)
SECTION_FONT = font(40, bold=True)
BODY_FONT = font(28, bold=False)
BODY_BOLD = font(28, bold=True)
SMALL_FONT = font(22, bold=False)
CAPTION_FONT = font(20, bold=False)
METRIC_FONT = font(34, bold=True)


def draw_wrapped(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], *, font_obj, fill, width, gap=8):
    x, y = xy
    avg_char = max(font_obj.size * 0.52, 1)
    paragraphs = text.split("\n")
    current_y = y
    for para in paragraphs:
        if not para.strip():
            current_y += font_obj.size + gap
            continue
        lines = wrap(para, width=max(int(width / avg_char), 12), break_long_words=False, break_on_hyphens=False) or [""]
        for line in lines:
            draw.text((x, current_y), line, font=font_obj, fill=fill)
            bbox = draw.textbbox((x, current_y), line, font=font_obj)
            current_y = bbox[3] + gap
    return current_y


def draw_bullets(draw: ImageDraw.ImageDraw, items: list[str], xy: tuple[int, int], *, width, font_obj=BODY_FONT):
    x, y = xy
    current_y = y
    for item in items:
        draw.text((x, current_y), "•", font=BODY_BOLD, fill=PURPLE)
        current_y = draw_wrapped(draw, item, (x + 30, current_y), font_obj=font_obj, fill=INK, width=width - 30, gap=7)
        current_y += 8
    return current_y


def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=30, fill=LAVENDER, outline=BORDER, width=3)
    draw.rounded_rectangle((x0, y0, x1, y0 + 72), radius=30, fill=(236, 230, 248), outline=BORDER, width=0)
    draw.rectangle((x0, y0 + 42, x1, y0 + 72), fill=(236, 230, 248))
    draw.text((x0 + 28, y0 + 16), title, font=SECTION_FONT, fill=PURPLE_DARK)


def fit_image(path: Path, target_w: int, target_h: int):
    image = Image.open(path).convert("RGB")
    scale = min(target_w / image.width, target_h / image.height)
    resized = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    background = Image.new("RGB", (target_w, target_h), WHITE)
    background.paste(resized, ((target_w - resized.width) // 2, (target_h - resized.height) // 2))
    return background


def metric_card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], value: str, label: str):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=22, fill=WHITE, outline=BORDER, width=2)
    draw.text((x0 + 18, y0 + 14), value, font=METRIC_FONT, fill=PURPLE)
    draw_wrapped(draw, label, (x0 + 18, y0 + 62), font_obj=SMALL_FONT, fill=MUTED, width=(x1 - x0 - 36), gap=4)


def paste(canvas: Image.Image, image: Image.Image, xy: tuple[int, int]):
    canvas.paste(image, xy)


def main():
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), WHITE)
    draw = ImageDraw.Draw(canvas)

    # Header
    draw.rectangle((0, 0, CANVAS_W, HEADER_H), fill=WHITE)
    draw.rectangle((0, HEADER_H - 14, CANVAS_W, HEADER_H), fill=PURPLE)

    logo = Image.open(ROOT / "Digital Twin.png").convert("RGB")
    logo_crop = logo.crop((0, 0, 450, 135))
    logo_crop.thumbnail((420, 130), Image.Resampling.LANCZOS)
    canvas.paste(logo_crop, (MARGIN, 50))

    title = "Passive Occupancy Estimation in Smart Classrooms Using CO2, VOC, Airflow, and Weak Supervision"
    authors = "Fengjun Han, Samuel Hartmann, Dalton Sloan, Garrett Green"
    affiliation = "Tennessee Technological University • CSC 4260 • Ashraf Islam Engineering Building"
    draw_wrapped(draw, title, (560, 44), font_obj=TITLE_FONT, fill=INK, width=CANVAS_W - 700, gap=6)
    draw.text((560, 164), authors, font=AUTHOR_FONT, fill=INK)
    draw.text((560, 214), affiliation, font=SMALL_FONT, fill=MUTED)

    # Layout
    top = HEADER_H + 40
    col_w = (CANVAS_W - 2 * MARGIN - 2 * GUTTER) // 3
    x1 = MARGIN
    x2 = MARGIN + col_w + GUTTER
    x3 = MARGIN + 2 * (col_w + GUTTER)

    intro_box = (x1, top, x1 + col_w, 1160)
    data_box = (x1, 1200, x1 + col_w, 2240)
    methods_box = (x1, 2280, x1 + col_w, CANVAS_H - 120)

    story_box = (x2, top, x2 + col_w, 2450)
    discussion_box = (x2, 2490, x2 + col_w, CANVAS_H - 120)

    results_box = (x3, top, x3 + col_w, 2020)
    conclusion_box = (x3, 2060, x3 + col_w, 3100)
    refs_box = (x3, 3140, x3 + col_w, CANVAS_H - 120)

    for box, title_text in [
        (intro_box, "Intro / Significance / Research Question"),
        (data_box, "Dataset"),
        (methods_box, "Methods"),
        (story_box, "Results"),
        (discussion_box, "Discussion"),
        (results_box, "Validation & Key Metrics"),
        (conclusion_box, "Conclusions / Recommendations"),
        (refs_box, "References & Acknowledgements"),
    ]:
        panel(draw, box, title_text)

    # Intro
    ix0, iy0, ix1, iy1 = intro_box
    intro_text = (
        "HVAC systems often follow fixed schedules instead of actual room usage, which wastes energy during low-occupancy periods. "
        "This project investigates whether passive smart-building signals can estimate classroom occupancy without cameras or badge scanners."
    )
    draw_wrapped(draw, intro_text, (ix0 + 28, iy0 + 96), font_obj=BODY_FONT, fill=INK, width=col_w - 56)
    draw.text((ix0 + 28, iy0 + 356), "Research question", font=BODY_BOLD, fill=PURPLE_DARK)
    draw_bullets(
        draw,
        [
            "Can CO2, VOC, humidity, temperature, and airflow estimate occupancy continuously?",
            "Can a hybrid physics + machine learning workflow remain useful with limited labels?",
            "Can the result support occupancy-aware HVAC control in university classrooms?",
        ],
        (ix0 + 28, iy0 + 404),
        width=col_w - 56,
    )

    # Dataset
    dx0, dy0, dx1, dy1 = data_box
    draw_bullets(
        draw,
        [
            "Room 354 primary dataset: 30-day IAQ + FPB export from 2026-02-27 to 2026-03-29, resampled to 5-minute intervals (~8,600 rows).",
            "Sensors: CO2 (ppm), VOC, humidity (%), temperature (F), and discharge airflow (cfm).",
            "Room 361 validation dataset: Apr 7–9, 2026 FPB export with 3 manual headcount anchors.",
            "Enrollment records were converted into class schedule windows for weak supervision experiments.",
        ],
        (dx0 + 28, dy0 + 96),
        width=col_w - 56,
    )
    data_para = (
        "Both room exports were long-format point-history CSVs. We filtered the required points, pivoted them into room-level columns, "
        "resampled to 5-minute intervals, and merged IAQ and FPB streams on time. Discharge airflow was converted to ACH using room volume. "
        "Missing humidity and temperature values were retained rather than imputed."
    )
    draw_wrapped(draw, data_para, (dx0 + 28, dy0 + 520), font_obj=BODY_FONT, fill=INK, width=col_w - 56)

    # Methods
    mx0, my0, mx1, my1 = methods_box
    draw_bullets(
        draw,
        [
            "Physics baseline: airflow-aware CO2 mass-balance occupancy anchor.",
            "Blended estimate: 70% CO2 anchor + 30% VOC, humidity, and temperature index.",
            "Temporal features: 5–30 minute lags, rolling means, slopes, and interaction terms.",
            "Weak supervision: manual anchor counts expanded into confidence-weighted pseudo-labels.",
            "Hybrid residual models: Ridge, Random Forest, and Gradient Boosting.",
        ],
        (mx0 + 28, my0 + 96),
        width=col_w - 56,
    )
    draw.text((mx0 + 28, my0 + 610), "Model assumptions", font=BODY_BOLD, fill=PURPLE_DARK)
    draw_bullets(
        draw,
        [
            "Room volume ≈ 637.1 m^3",
            "Outdoor CO2 baseline = 420 ppm",
            "Per-person CO2 generation = 0.018 m^3/h/person",
            "Measured airflow is treated as a ventilation proxy",
        ],
        (mx0 + 28, my0 + 658),
        width=col_w - 56,
    )

    # Main results / story figure
    sx0, sy0, sx1, sy1 = story_box
    occ_img = fit_image(ROOT / "reports" / "figures" / "room354_estimated_occupancy(1 month).png", col_w - 56, 920)
    paste(canvas, occ_img, (sx0 + 28, sy0 + 96))
    draw.text((sx0 + 28, sy0 + 1035), "Figure 1. Room 354 airflow-aware occupancy estimate across the 30-day study window.", font=CAPTION_FONT, fill=MUTED)

    corr_img = fit_image(ROOT / "reports" / "figures" / "room354_correlation_matrix.png", col_w - 56, 620)
    paste(canvas, corr_img, (sx0 + 28, sy0 + 1080))
    draw.text((sx0 + 28, sy0 + 1715), "Figure 2. CO2 and VOC carry the strongest occupancy signal; airflow is mainly a dilution context variable.", font=CAPTION_FONT, fill=MUTED)

    results_para = (
        "Using measured airflow instead of a fixed 4 ACH assumption lowered the mean estimate from 5.22 to 4.17 people. "
        "The blended model reduced volatility by 16% and cut extreme spikes above 30 estimated people from 150 to 85."
    )
    draw_wrapped(draw, results_para, (sx0 + 28, sy0 + 1760), font_obj=BODY_FONT, fill=INK, width=col_w - 56)

    # Discussion
    qx0, qy0, qx1, qy1 = discussion_box
    discussion_text = (
        "CO2 and VOC were the strongest passive occupancy indicators in this classroom setting. Airflow did not behave like a direct headcount signal; "
        "instead, it improved the estimate by correcting how quickly indoor CO2 was diluted. That explains why the fixed-ACH baseline overstated occupancy during many periods.\n\n"
        "The largest miss occurred at the Room 361 anchor on 2026-04-07 14:38. The true count was 33, while the physics baseline predicted about 11 with airflow near 1,130 cfm. "
        "This indicates a limitation of pure CO2 physics under high-flow conditions: ventilation diluted the signal faster than occupants could build it up at the sensor.\n\n"
        "Because only 3 manual anchors were available, the machine-learning results should be treated as preliminary. The current evidence supports coarse occupancy estimation much more strongly than exact headcount prediction."
    )
    draw_wrapped(draw, discussion_text, (qx0 + 28, qy0 + 96), font_obj=BODY_FONT, fill=INK, width=col_w - 56)

    # Validation & metrics
    rx0, ry0, rx1, ry1 = results_box
    anchor_img = fit_image(ROOT / "reports" / "room361_pipeline" / "figures" / "room361_co2_flow_occupancy_timeseries.png", col_w - 56, 720)
    paste(canvas, anchor_img, (rx0 + 28, ry0 + 96))
    draw.text((rx0 + 28, ry0 + 835), "Figure 3. Held-out Room 361 anchor-window evaluation.", font=CAPTION_FONT, fill=MUTED)

    card_w = (col_w - 74) // 2
    card_h = 146
    metrics = [
        ("r = 0.978", "CO2 ↔ estimated occupancy"),
        ("r = 0.748", "VOC ↔ estimated occupancy"),
        ("P90 13, P99 30", "Room 354 blended estimate"),
        ("150 → 85 (-43%)", "Intervals above 30 people"),
        ("MAE 7.3, RMSE 12.5", "Physics baseline on 3 anchors"),
        ("MAE 8.5, RMSE 12.6", "Gradient boosting residual"),
    ]
    start_y = ry0 + 900
    for idx, (value, label) in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        bx0 = rx0 + 28 + col * (card_w + 18)
        by0 = start_y + row * (card_h + 18)
        metric_card(draw, (bx0, by0, bx0 + card_w, by0 + card_h), value, label)

    # Conclusions
    cx0, cy0, cx1, cy1 = conclusion_box
    draw_bullets(
        draw,
        [
            "CO2 and VOC are the strongest passive occupancy indicators in these classroom datasets.",
            "Measured airflow improves occupancy estimates by replacing unrealistic fixed-ventilation assumptions.",
            "Hybrid physics + ML methods are promising, but sparse labels currently limit supervised gains.",
            "The most defensible near-term target is coarse occupancy state estimation rather than exact people counts.",
            "Next steps: collect more manual counts, align vibration data, and add outdoor-air fraction or damper-position data when available.",
        ],
        (cx0 + 28, cy0 + 96),
        width=col_w - 56,
    )

    # References + acknowledgements
    fx0, fy0, fx1, fy1 = refs_box
    draw.text((fx0 + 28, fy0 + 96), "References", font=BODY_BOLD, fill=PURPLE_DARK)
    refs = [
        "ASHRAE 62.1. Ventilation for Acceptable Indoor Air Quality.",
        "Fisk, W. J., et al. CO2-based occupancy estimation in commercial buildings.",
        "Tennessee Tech AIEB FPB and IAQ point-history exports, 2026.",
        "Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12:2825-2830.",
    ]
    draw_bullets(draw, refs, (fx0 + 28, fy0 + 144), width=col_w - 56, font_obj=BODY_FONT)
    draw.text((fx0 + 28, fy0 + 530), "Acknowledgements", font=BODY_BOLD, fill=PURPLE_DARK)
    acks = [
        "Domain experts: Chandler Norman, Norman Walker, Elisabeth Humphrey, and Dr. Steven Anton.",
        "Tennessee Tech University and the AIEB building data sources.",
        "CSC 4260 project support team.",
    ]
    draw_bullets(draw, acks, (fx0 + 28, fy0 + 578), width=col_w - 56, font_obj=BODY_FONT)

    png_path = OUT_DIR / "aieb_occupancy_poster_canva.png"
    jpg_path = OUT_DIR / "aieb_occupancy_poster_canva.jpg"
    canvas.save(png_path)
    canvas.save(jpg_path, quality=95)
    print(f"Saved {png_path}")
    print(f"Saved {jpg_path}")


if __name__ == "__main__":
    main()

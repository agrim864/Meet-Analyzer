from __future__ import annotations

import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

BG = "#f7f8fb"
INK = "#18202f"
MUTED = "#5f6b7a"
LINE = "#d9dee8"
BLUE = "#2563eb"
GREEN = "#159957"
ORANGE = "#f59e0b"
RED = "#ef4444"
PURPLE = "#7c3aed"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, outline: str = LINE, radius: int = 18, width: int = 1) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, size: int, fill: str = INK, bold: bool = False, width: int | None = None, line_gap: int = 7) -> int:
    fnt = font(size, bold)
    lines: list[str] = []
    if width:
        avg = max(1, int(size * 0.56))
        for para in str(value).splitlines() or [""]:
            lines.extend(textwrap.wrap(para, max(1, width // avg)) or [""])
    else:
        lines = str(value).splitlines()
    y = xy[1]
    for line in lines:
        draw.text((xy[0], y), line, fill=fill, font=fnt)
        y += size + line_gap
    return y


def build_architecture() -> None:
    img = Image.new("RGB", (1800, 900), BG)
    draw = ImageDraw.Draw(img)
    text(draw, (70, 55), "Meet Analyzer Pipeline", 48, bold=True)
    text(draw, (70, 118), "From screen-share video or live frames to evidence-backed meeting intelligence", 24, MUTED)

    stages = [
        ("Video Input", "Local MP4 or public Bunny/CDN URL", BLUE),
        ("Transcript", "Deepgram prerecorded STT", GREEN),
        ("Visual Sampling", "Mode routing, cursor tracking, change detection", ORANGE),
        ("Screen Understanding", "yolo-ui macro segments + OmniParser crops", PURPLE),
        ("Report Builder", "Topic units, cursor flow, Mistral/Gemini/Qwen writer", RED),
    ]
    x = 70
    y = 260
    w = 290
    h = 180
    gap = 55
    for idx, (title, body, color) in enumerate(stages):
        rounded(draw, (x, y, x + w, y + h), "white", "#cad2e0", radius=22, width=2)
        draw.rounded_rectangle((x + 24, y + 24, x + 86, y + 86), radius=16, fill=color)
        text(draw, (x + 105, y + 27), f"{idx + 1}", 34, "white", bold=True)
        text(draw, (x + 24, y + 105), title, 25, INK, bold=True)
        text(draw, (x + 24, y + 142), body, 18, MUTED, width=w - 48, line_gap=3)
        if idx < len(stages) - 1:
            ax = x + w + 12
            ay = y + h // 2
            draw.line((ax, ay, ax + gap - 24, ay), fill="#8b97a8", width=5)
            draw.polygon([(ax + gap - 24, ay - 11), (ax + gap - 24, ay + 11), (ax + gap - 4, ay)], fill="#8b97a8")
        x += w + gap

    rounded(draw, (150, 600, 1650, 775), "white", "#cad2e0", radius=24, width=2)
    text(draw, (190, 635), "Production contract", 28, bold=True)
    bullets = [
        "Every screen area is assigned to a macro segment; low-confidence regions are preserved before falling back to main content.",
        "OmniParser runs full-frame on the first frame and crop-refreshes changed areas while keeping cached elements elsewhere.",
        "Reports include screen layout, visible UI, transcript evidence, and cursor-driven user flow.",
    ]
    y = 678
    for bullet in bullets:
        draw.ellipse((195, y + 8, 207, y + 20), fill=BLUE)
        y = text(draw, (220, y), bullet, 20, MUTED, width=1370, line_gap=4) + 2

    img.save(ASSETS / "architecture_overview.png", quality=95)


def build_metrics() -> None:
    metrics = json.loads((ASSETS / "validation_metrics.json").read_text(encoding="utf-8"))
    img = Image.new("RGB", (1800, 1050), BG)
    draw = ImageDraw.Draw(img)
    text(draw, (70, 55), "Validation Metrics", 48, bold=True)
    sample_times = ", ".join(f"{int(t)}" if float(t).is_integer() else f"{t:g}" for t in metrics.get("sample_times_sec", []))
    text(draw, (70, 118), f"Bunny sample, {metrics['frames_sampled']} varied keyframes, raw frame vs generated segment overlays", 24, MUTED)

    cards = [
        ("Frames", str(metrics["frames_sampled"]), f"sampled at {sample_times}s", BLUE),
        ("Coverage", f"{metrics['mean_grid_coverage_pct']:.0f}%", "mean grid coverage across frames", GREEN),
        ("High Conf", str(metrics["high_conf_region_count"]), "regions at or above 0.75", ORANGE),
        ("Low Conf", str(metrics["low_conf_region_count"]), "regions retained for coverage", PURPLE),
    ]
    x = 70
    for title, value, body, color in cards:
        rounded(draw, (x, 210, x + 385, 390), "white", "#cad2e0", radius=24, width=2)
        text(draw, (x + 30, 238), title, 24, MUTED, bold=True)
        text(draw, (x + 30, 278), value, 58, color, bold=True)
        text(draw, (x + 30, 348), body, 18, MUTED, width=320)
        x += 430

    rounded(draw, (70, 460, 1730, 945), "white", "#cad2e0", radius=24, width=2)
    text(draw, (105, 495), "Class reliability from sampled frames", 30, bold=True)
    palette = [BLUE, GREEN, ORANGE, PURPLE, RED]
    class_rows = [
        (name, data, palette[index % len(palette)])
        for index, (name, data) in enumerate(metrics.get("classes", {}).items())
    ]
    y = 560
    frame_count = max(1, int(metrics["frames_sampled"]))
    for name, data, color in class_rows:
        detected = int(data["detected_in_frames"])
        score = float(data["score"])
        text(draw, (110, y), name, 23, INK, bold=True)
        draw.rounded_rectangle((330, y + 6, 930, y + 34), radius=14, fill="#e8edf6")
        draw.rounded_rectangle((330, y + 6, 330 + int(600 * detected / frame_count), y + 34), radius=14, fill=color)
        text(draw, (960, y - 2), f"{detected}/{frame_count} frames", 22, INK, bold=True)
        text(draw, (1130, y - 2), f"example score {score:.2f}", 22, MUTED)
        text(draw, (1410, y - 2), data["source"], 22, MUTED)
        y += 72

    text(
        draw,
        (105, 850),
        "Current Bunny validation favors precision over decorative segmentation: unsupported sidebars are suppressed, while coverage falls back to main_content.",
        20,
        MUTED,
        width=1450,
    )
    img.save(ASSETS / "metrics_dashboard.png", quality=95)


def build_micro_metrics() -> None:
    metrics_path = ASSETS / "micro_validation_metrics.json"
    if not metrics_path.exists():
        return
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    img = Image.new("RGB", (1800, 1100), BG)
    draw = ImageDraw.Draw(img)
    text(draw, (70, 55), "Micro-Element Validation", 48, bold=True)
    text(draw, (70, 118), "OmniParser element extraction on the same Bunny keyframes used for macro validation.", 24, MUTED)

    cards = [
        ("Frames", str(metrics["frames_sampled"]), "same varied Bunny keyframes", BLUE),
        ("Elements", str(metrics["total_elements"]), "detected UI boxes", GREEN),
        ("Mean / Frame", f"{metrics['mean_elements_per_frame']:.1f}", "micro boxes per frame", ORANGE),
        ("With Text", str(metrics["text_or_caption_elements"]), "OCR or caption content attached", PURPLE),
    ]
    x = 70
    for title, value, body, color in cards:
        rounded(draw, (x, 210, x + 385, 390), "white", "#cad2e0", radius=24, width=2)
        text(draw, (x + 30, 238), title, 24, MUTED, bold=True)
        text(draw, (x + 30, 278), value, 58, color, bold=True)
        text(draw, (x + 30, 348), body, 18, MUTED, width=320)
        x += 430

    rounded(draw, (70, 460, 865, 985), "white", "#cad2e0", radius=24, width=2)
    text(draw, (105, 495), "Element type distribution", 30, bold=True)
    y = 560
    type_counts = list((metrics.get("type_counts") or {}).items())[:8]
    max_count = max([int(value) for _, value in type_counts] or [1])
    palette = [BLUE, GREEN, ORANGE, PURPLE, RED, "#0f766e", "#4b5563", "#be123c"]
    for index, (name, value) in enumerate(type_counts):
        value = int(value)
        color = palette[index % len(palette)]
        text(draw, (110, y), str(name), 22, INK, bold=True)
        draw.rounded_rectangle((305, y + 7, 730, y + 33), radius=13, fill="#e8edf6")
        draw.rounded_rectangle((305, y + 7, 305 + int(425 * value / max_count), y + 33), radius=13, fill=color)
        text(draw, (750, y - 2), str(value), 22, MUTED)
        y += 56

    rounded(draw, (935, 460, 1730, 985), "white", "#cad2e0", radius=24, width=2)
    text(draw, (970, 495), "Per-frame parser load", 30, bold=True)
    y = 560
    frame_metrics = metrics.get("frame_metrics", [])
    max_elements = max([int(item.get("element_count", 0)) for item in frame_metrics] or [1])
    for item in frame_metrics:
        count = int(item.get("element_count", 0))
        time_sec = item.get("time_sec", 0)
        text(draw, (975, y), f"{time_sec:g}s", 22, INK, bold=True)
        draw.rounded_rectangle((1095, y + 7, 1525, y + 33), radius=13, fill="#e8edf6")
        draw.rounded_rectangle((1095, y + 7, 1095 + int(430 * count / max_elements), y + 33), radius=13, fill=BLUE)
        text(draw, (1545, y - 2), f"{count} boxes", 22, MUTED)
        y += 56
    text(draw, (970, 890), "Micro regions are linked back to parent macro regions during live and batch state assembly.", 21, MUTED, width=680)
    img.save(ASSETS / "micro_metrics_dashboard.png", quality=95)


def build_class_gallery() -> None:
    metrics = json.loads((ASSETS / "validation_metrics.json").read_text(encoding="utf-8"))
    items = []
    for name, data in metrics.get("classes", {}).items():
        proof_name = Path(str(data.get("proof_image", ""))).name
        if proof_name and (ASSETS / proof_name).exists():
            items.append((name.replace("_", " ").title(), proof_name))
    thumbs = []
    for label, name in items:
        im = Image.open(ASSETS / name).convert("RGB")
        im.thumbnail((780, 360), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (820, 430), "white")
        d = ImageDraw.Draw(canvas)
        rounded(d, (0, 0, 819, 429), "white", "#cad2e0", radius=20, width=2)
        text(d, (24, 20), label, 26, INK, bold=True)
        canvas.paste(im, (20, 62))
        thumbs.append(canvas)

    img = Image.new("RGB", (1800, 1480), BG)
    draw = ImageDraw.Draw(img)
    text(draw, (70, 55), "Photo vs Output Gallery", 48, bold=True)
    text(draw, (70, 118), "For each class, left side is the input crop and right side is the generated segment output.", 24, MUTED)
    positions = [(70, 210), (910, 210), (70, 680), (910, 680), (490, 1150)]
    for thumb, pos in zip(thumbs, positions):
        img.paste(thumb, pos)
    img.save(ASSETS / "class_gallery.png", quality=95)


def build_report_preview() -> None:
    report = (ROOT / "batch_mode" / "outputs" / "mini_interval_report" / "report.md").read_text(encoding="utf-8")
    cleaned = (
        report.replace("\u00e2\u20ac\u0153", '"')
        .replace("\u00e2\u20ac\u009d", '"')
        .replace("\u00e2\u20ac\u2122", "'")
        .replace("\u00e2\u20ac\u201c", "-")
        .replace("\u00e2\u20ac\u2018", "-")
        .replace("\u00e2\u20ac\u00af", " ")
    )
    lines = [line for line in cleaned.splitlines() if line.strip()]
    excerpt = "\n".join(lines[:28])
    img = Image.new("RGB", (1800, 1180), BG)
    draw = ImageDraw.Draw(img)
    text(draw, (70, 55), "Generated Report Preview", 48, bold=True)
    text(draw, (70, 118), "Short interval report generated from sampled visual events and transcript evidence.", 24, MUTED)
    rounded(draw, (90, 210, 1710, 1060), "white", "#cad2e0", radius=26, width=2)
    y = 250
    for line in excerpt.splitlines():
        if line.startswith("# "):
            y = text(draw, (135, y), line[2:], 32, INK, bold=True, width=1500, line_gap=8) + 4
        elif line.startswith("## "):
            y = text(draw, (135, y + 12), line[3:], 26, BLUE, bold=True, width=1500, line_gap=7) + 2
        elif line.startswith("- "):
            draw.ellipse((142, y + 10, 154, y + 22), fill=BLUE)
            y = text(draw, (170, y), line[2:], 20, MUTED, width=1450, line_gap=5)
        else:
            y = text(draw, (135, y), line.replace("**", ""), 20, MUTED, width=1510, line_gap=5) + 2
        if y > 1010:
            text(draw, (135, 1015), "...", 24, MUTED, bold=True)
            break
    img.save(ASSETS / "report_preview.png", quality=95)


def main() -> int:
    build_architecture()
    build_metrics()
    build_micro_metrics()
    build_class_gallery()
    build_report_preview()
    print("README assets rebuilt in docs/assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

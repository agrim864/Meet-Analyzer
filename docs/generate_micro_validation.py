from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from real_time_mode.omniparser_adapter import OmniParserAdapter  # noqa: E402


COLORS = {
    "button": (67, 160, 71),
    "checkbox": (0, 137, 123),
    "icon": (25, 118, 210),
    "input": (251, 140, 0),
    "link": (123, 31, 162),
    "menu": (216, 27, 96),
    "tab": (0, 121, 107),
    "text": (97, 97, 97),
    "toolbar": (93, 64, 55),
    "ui_element": (84, 110, 122),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate OmniParser micro-element README validation assets.")
    parser.add_argument("--frames", required=True, help="Comma-separated raw frame image paths.")
    parser.add_argument("--output-dir", default="docs/assets", help="Output directory for images and metrics.")
    parser.add_argument("--repo-dir", default="real_time_mode/vendor/OmniParser", help="Local OmniParser repository directory.")
    parser.add_argument("--weights-dir", default="real_time_mode/vendor/OmniParser/weights", help="Local OmniParser weights directory.")
    parser.add_argument("--imgsz", type=int, default=512, help="OmniParser detector image size.")
    parser.add_argument("--max-boxes", type=int, default=60, help="Maximum boxes to draw per frame.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = [Path(item.strip()) for item in args.frames.split(",") if item.strip()]
    adapter = OmniParserAdapter(
        cache_dir=args.weights_dir,
        repo_dir=args.repo_dir,
        imgsz=args.imgsz,
    )
    records = []
    for path in frame_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        parsed = adapter.parse_regions(frame, [(0, 0, frame.shape[1], frame.shape[0])])
        elements = sorted(parsed.elements, key=lambda item: (item.bbox[1], item.bbox[0]))
        overlay = _draw_micro_overlay(frame, elements[: args.max_boxes])
        stem = path.name.replace("_raw", "").rsplit(".", 1)[0]
        overlay_path = output_dir / f"{stem}_micro.jpg"
        cv2.imwrite(str(overlay_path), overlay)
        records.append(
            {
                "time_sec": _time_from_stem(path.stem),
                "raw": str(path),
                "overlay": str(overlay_path),
                "provider": parsed.provider,
                "confidence": parsed.confidence,
                "elements": [
                    {
                        "type": item.type,
                        "bbox": list(item.bbox),
                        "text": item.text,
                        "caption": item.caption,
                        "interactive": item.interactive,
                        "confidence": item.confidence,
                        "source": item.source,
                    }
                    for item in elements
                ],
                "overlays": [
                    {
                        "type": item.type,
                        "bbox": list(item.bbox),
                        "title": item.title,
                        "confidence": item.confidence,
                        "source": item.source,
                    }
                    for item in parsed.overlays
                ],
            }
        )
    if not records:
        raise RuntimeError("No frames were parsed.")
    metrics = _metrics(records)
    _write_contact_sheet(records, output_dir)
    (output_dir / "micro_segments.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output_dir / "micro_validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "frames": len(records), "metrics": metrics}, indent=2))
    return 0


def _draw_micro_overlay(frame: np.ndarray, elements: list) -> np.ndarray:
    overlay = frame.copy()
    for idx, item in enumerate(elements, start=1):
        x1, y1, x2, y2 = [int(v) for v in item.bbox]
        color = COLORS.get(item.type, (90, 90, 90))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"u{idx}:{item.type}"
        label_w = min(frame.shape[1] - x1, max(54, min(160, 9 * len(label))))
        cv2.rectangle(overlay, (x1, max(0, y1 - 22)), (x1 + label_w, y1), color, -1)
        cv2.putText(overlay, label, (x1 + 3, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def _metrics(records: list[dict]) -> dict:
    type_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    frame_metrics = []
    for record in records:
        elements = record["elements"]
        type_counts.update(str(item.get("type", "unknown")) for item in elements)
        source_counts.update(str(item.get("source", "unknown")) for item in elements)
        interactive = sum(1 for item in elements if item.get("interactive"))
        with_text = sum(1 for item in elements if str(item.get("text") or item.get("caption") or "").strip())
        frame_metrics.append(
            {
                "time_sec": record["time_sec"],
                "provider": record["provider"],
                "element_count": len(elements),
                "interactive_count": interactive,
                "text_or_caption_count": with_text,
                "overlay_count": len(record["overlays"]),
                "top_types": dict(Counter(str(item.get("type", "unknown")) for item in elements).most_common(6)),
            }
        )
    total_elements = sum(item["element_count"] for item in frame_metrics)
    total_interactive = sum(item["interactive_count"] for item in frame_metrics)
    total_text = sum(item["text_or_caption_count"] for item in frame_metrics)
    return {
        "frames_sampled": len(records),
        "sample_times_sec": [record["time_sec"] for record in records],
        "total_elements": total_elements,
        "mean_elements_per_frame": round(total_elements / max(1, len(records)), 2),
        "interactive_elements": total_interactive,
        "text_or_caption_elements": total_text,
        "type_counts": dict(type_counts.most_common()),
        "source_counts": dict(source_counts.most_common()),
        "frame_metrics": frame_metrics,
    }


def _write_contact_sheet(records: list[dict], output_dir: Path) -> None:
    rows = []
    for record in records:
        raw = cv2.imread(record["raw"])
        overlay = cv2.imread(record["overlay"])
        raw = _add_title(_fit(raw, 420), f"Input frame @ {record['time_sec']:g}s")
        overlay = _add_title(_fit(overlay, 420), "Generated micro elements")
        raw, overlay = _equal_height(raw, overlay)
        rows.append(np.hstack([raw, overlay]))
    max_width = max(row.shape[1] for row in rows)
    rows = [cv2.copyMakeBorder(row, 0, 0, 0, max_width - row.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for row in rows]
    cv2.imwrite(str(output_dir / "micro_validation_contact_sheet.jpg"), np.vstack(rows))


def _time_from_stem(stem: str) -> float:
    digits = "".join(ch for ch in stem if ch.isdigit() or ch == ".")
    try:
        return float(digits)
    except ValueError:
        return 0.0


def _fit(image: np.ndarray, width: int) -> np.ndarray:
    height, current_width = image.shape[:2]
    scale = width / max(1, current_width)
    return cv2.resize(image, (width, int(round(height * scale))), interpolation=cv2.INTER_AREA)


def _add_title(image: np.ndarray, title: str) -> np.ndarray:
    _, width = image.shape[:2]
    bar = np.full((38, width, 3), 245, dtype=np.uint8)
    cv2.putText(bar, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    return np.vstack([bar, image])


def _equal_height(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_height = max(left.shape[0], right.shape[0])
    left = cv2.copyMakeBorder(left, 0, max_height - left.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    right = cv2.copyMakeBorder(right, 0, max_height - right.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return left, right


if __name__ == "__main__":
    raise SystemExit(main())

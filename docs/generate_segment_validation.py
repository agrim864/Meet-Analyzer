from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from batch_mode.macro_layout import MacroLayoutDetector  # noqa: E402


COLORS = {
    "header": (80, 180, 255),
    "left_sidebar": (120, 220, 120),
    "main_content": (255, 170, 80),
    "right_sidebar": (220, 120, 220),
    "footer": (120, 120, 255),
}

CLASS_EXAMPLES = {
    "header": 0,
    "left_sidebar": 0,
    "main_content": 0,
    "right_sidebar": 0,
    "footer": 0,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate raw-vs-segment proof images for README validation.")
    parser.add_argument("--source", required=True, help="Local MP4 path or public video URL.")
    parser.add_argument("--output-dir", default="docs/assets", help="Directory for proof images and metrics.")
    parser.add_argument("--times", default="2,15,58,100,145,205", help="Comma-separated sample times in seconds.")
    parser.add_argument("--weights", default="real_time_mode/vendor/uitag/yolo-ui.pt", help="Path or HF repo for yolo-ui weights.")
    parser.add_argument("--device", default="cpu", help="Detector device, for example cpu or cuda:0.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    times = [float(item.strip()) for item in args.times.split(",") if item.strip()]
    detector = MacroLayoutDetector(args.weights, device=args.device, tile_size=640, tile_step=512, max_det=128)
    records = _sample_and_detect(args.source, output_dir, times, detector)
    if not records:
        raise RuntimeError("No frames were sampled from the input video.")
    _write_contact_sheet(records, output_dir)
    metrics = _write_class_proofs(records, output_dir)
    (output_dir / "validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "frames": len(records), "metrics": metrics}, indent=2))
    return 0


def _sample_and_detect(source: str, output_dir: Path, times: list[float], detector: MacroLayoutDetector) -> list[dict]:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")
    records: list[dict] = []
    for time_sec in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        groups = detector.detect(frame)
        overlay = frame.copy()
        for group in groups:
            x1, y1, x2, y2 = [int(v) for v in group.bbox]
            color = COLORS.get(group.label, (200, 200, 200))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
            label = f"{group.label} {group.score:.2f} {group.source}"
            cv2.rectangle(overlay, (x1, max(0, y1 - 24)), (min(frame.shape[1], x1 + 380), y1), color, -1)
            cv2.putText(overlay, label, (x1 + 4, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        stem = f"frame_{int(round(time_sec)):04d}"
        raw_path = output_dir / f"{stem}_raw.jpg"
        overlay_path = output_dir / f"{stem}_segments.jpg"
        cv2.imwrite(str(raw_path), frame)
        cv2.imwrite(str(overlay_path), overlay)
        records.append(
            {
                "time_sec": time_sec,
                "raw": str(raw_path),
                "overlay": str(overlay_path),
                "groups": [
                    {
                        "label": group.label,
                        "bbox": list(group.bbox),
                        "score": float(group.score),
                        "source": group.source,
                    }
                    for group in groups
                ],
            }
        )
    cap.release()
    (output_dir / "segments.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


def _write_contact_sheet(records: list[dict], output_dir: Path) -> None:
    rows = []
    for record in records:
        raw = cv2.imread(record["raw"])
        overlay = cv2.imread(record["overlay"])
        raw = _add_title(_fit(raw, 420), f"Input frame @ {record['time_sec']:g}s")
        overlay = _add_title(_fit(overlay, 420), "Generated macro segments")
        raw, overlay = _equal_height(raw, overlay)
        rows.append(np.hstack([raw, overlay]))
    max_width = max(row.shape[1] for row in rows)
    rows = [cv2.copyMakeBorder(row, 0, 0, 0, max_width - row.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for row in rows]
    cv2.imwrite(str(output_dir / "segment_validation_contact_sheet.jpg"), np.vstack(rows))


def _write_class_proofs(records: list[dict], output_dir: Path) -> dict:
    first_by_class: dict[str, dict] = {}
    for record in records:
        for group in record["groups"]:
            first_by_class.setdefault(group["label"], {"record": record, "group": group})

    metrics = {
        "frames_sampled": len(records),
        "sample_times_sec": [record["time_sec"] for record in records],
        "classes": {},
        "frame_metrics": [],
    }
    for record in records:
        raw = cv2.imread(record["raw"])
        height, width = raw.shape[:2]
        metrics["frame_metrics"].append(
            {
                "time_sec": record["time_sec"],
                "labels": [group["label"] for group in record["groups"]],
                "coverage_grid_pct": _coverage_grid_pct(record["groups"], width, height),
                "high_conf_regions": sum(1 for group in record["groups"] if group["score"] >= 0.75),
                "low_conf_regions": sum(1 for group in record["groups"] if 0 < group["score"] < 0.75),
            }
        )

    for label in CLASS_EXAMPLES:
        if label not in first_by_class:
            continue
        item = first_by_class[label]
        record = item["record"]
        group = item["group"]
        proof_path = output_dir / f"proof_{label}.jpg"
        _write_side_by_side_crop(record, group, proof_path)
        metrics["classes"][label] = {
            "example_time_sec": record["time_sec"],
            "score": round(float(group["score"]), 4),
            "source": group["source"],
            "bbox": group["bbox"],
            "proof_image": str(proof_path).replace("\\", "/"),
            "detected_in_frames": sum(1 for record_item in records if any(group_item["label"] == label for group_item in record_item["groups"])),
        }

    frame_metrics = metrics["frame_metrics"]
    metrics["mean_grid_coverage_pct"] = round(sum(item["coverage_grid_pct"] for item in frame_metrics) / max(1, len(frame_metrics)), 2)
    metrics["high_conf_region_count"] = sum(item["high_conf_regions"] for item in frame_metrics)
    metrics["low_conf_region_count"] = sum(item["low_conf_regions"] for item in frame_metrics)
    return metrics


def _write_side_by_side_crop(record: dict, group: dict, proof_path: Path) -> None:
    raw = cv2.imread(record["raw"])
    overlay = cv2.imread(record["overlay"])
    height, width = raw.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in group["bbox"]]
    pad = 28
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(width, x2 + pad), min(height, y2 + pad)
    crop_raw = _add_title(_fit(raw[cy1:cy2, cx1:cx2], 420), f"Input crop: {group['label']}")
    crop_overlay = _add_title(_fit(overlay[cy1:cy2, cx1:cx2], 420), f"Output: {group['label']} score={group['score']:.2f}")
    crop_raw, crop_overlay = _equal_height(crop_raw, crop_overlay)
    cv2.imwrite(str(proof_path), np.hstack([crop_raw, crop_overlay]))


def _coverage_grid_pct(groups: list[dict], width: int, height: int) -> float:
    total = 0
    hits = 0
    for y in np.linspace(0, height - 1, 16):
        for x in np.linspace(0, width - 1, 24):
            total += 1
            if any(group["bbox"][0] <= x <= group["bbox"][2] and group["bbox"][1] <= y <= group["bbox"][3] for group in groups):
                hits += 1
    return round(100.0 * hits / max(1, total), 2)


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

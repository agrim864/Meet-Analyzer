from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from batch_mode.macro_layout import (
    MacroLayoutDetector,
    UIDetection,
    build_layout_groups,
    estimate_coarse_regions,
    refine_regions_from_boundary_evidence,
)
from batch_mode.state import BBox, clamp_bbox


COLORS = {
    "header": (80, 180, 255),
    "left_sidebar": (120, 220, 120),
    "main_content": (255, 170, 80),
    "right_sidebar": (220, 120, 220),
    "footer": (120, 120, 255),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Trial-only macro boundary refinement using background-color seams. "
            "This does not change the main pipeline."
        )
    )
    parser.add_argument("--input-dir", default="test-sidebar")
    parser.add_argument("--output-dir", default="test-output2")
    parser.add_argument("--weights", default="real_time_mode/vendor/uitag/yolo-ui.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = list(_iter_images(input_dir))
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        raise RuntimeError(f"No images found in {input_dir}")

    detector = MacroLayoutDetector(args.weights, device=args.device, tile_size=640, tile_step=512, max_det=128)
    detector.warmup()

    label_counts: Dict[str, int] = {}
    sidebar_images = 0
    for index, path in enumerate(paths, start=1):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        detections = detector._run_tiled_detection(frame)  # trial script: intentionally diagnostic.
        height, width = frame.shape[:2]
        regions, confidences = estimate_coarse_regions(detections, width, height)
        regions = refine_regions_from_boundary_evidence(frame, detections, regions, confidences)
        regions = refine_regions_from_background_color(frame, detections, regions, confidences)
        groups = build_layout_groups(regions, confidences, width, height)
        if any(group.label in {"left_sidebar", "right_sidebar"} for group in groups):
            sidebar_images += 1
        for group in groups:
            label_counts[group.label] = label_counts.get(group.label, 0) + 1
        overlay = draw_groups(frame, groups)
        cv2.imwrite(str(output_dir / f"{path.stem}_annotated.jpg"), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if index % 20 == 0 or index == len(paths):
            print(f"processed={index}/{len(paths)}")

    non_images = [p.name for p in output_dir.iterdir() if p.is_file() and p.suffix.lower() not in {".jpg", ".jpeg", ".png"}]
    print(
        {
            "input_images": len(paths),
            "output_dir": str(output_dir),
            "annotated_outputs": len(list(output_dir.glob("*.jpg"))),
            "sidebar_images": sidebar_images,
            "non_image_files": non_images,
            "label_counts": dict(sorted(label_counts.items())),
        }
    )
    return 0


def refine_regions_from_background_color(
    frame_bgr: np.ndarray,
    detections: Sequence[UIDetection],
    regions: Dict[str, Optional[BBox]],
    confidences: Dict[str, float],
) -> Dict[str, Optional[BBox]]:
    height, width = frame_bgr.shape[:2]
    refined = dict(regions)

    header_bottom = _header_bottom(refined, confidences)
    footer_top = int(refined["footer"][1]) if refined.get("footer") else height
    if footer_top <= header_bottom:
        header_bottom, footer_top = 0, height

    # Jurisdiction priority:
    # header owns top, left_sidebar owns left, right_sidebar owns right,
    # footer owns bottom, main_content gets the remaining area.
    left = refined.get("left_sidebar")
    if left:
        current = int(left[2])
        found = _best_color_vertical_boundary(
            frame_bgr,
            detections,
            current=current,
            min_x=max(0, current - max(80, int(width * 0.14))),
            max_x=min(width, current + max(80, int(width * 0.14))),
            y1=header_bottom,
            y2=footer_top,
            side="left",
        )
        if found is not None:
            refined["left_sidebar"] = clamp_bbox((0, header_bottom, found, footer_top), width, height)

    right = refined.get("right_sidebar")
    if right:
        current = int(right[0])
        found = _best_color_vertical_boundary(
            frame_bgr,
            detections,
            current=current,
            min_x=max(0, current - max(80, int(width * 0.14))),
            max_x=min(width, current + max(80, int(width * 0.14))),
            y1=header_bottom,
            y2=footer_top,
            side="right",
        )
        if found is not None:
            refined["right_sidebar"] = clamp_bbox((found, header_bottom, width, footer_top), width, height)

    header = refined.get("header")
    if header and float(confidences.get("header", 0.0) or 0.0) > 0:
        current = int(header[3])
        found = _best_color_horizontal_boundary(
            frame_bgr,
            detections,
            current=current,
            min_y=max(0, current - max(42, int(height * 0.035))),
            max_y=min(height, current + max(42, int(height * 0.035))),
            x1=0,
            x2=width,
        )
        if found is not None:
            refined["header"] = clamp_bbox((0, 0, width, found), width, height)
            header_bottom = found

    footer = refined.get("footer")
    if footer:
        current = int(footer[1])
        found = _best_color_horizontal_boundary(
            frame_bgr,
            detections,
            current=current,
            min_y=max(0, current - max(42, int(height * 0.035))),
            max_y=min(height, current + max(42, int(height * 0.035))),
            x1=0,
            x2=width,
        )
        if found is not None:
            refined["footer"] = clamp_bbox((0, found, width, height), width, height)
            footer_top = found

    main_x1 = refined["left_sidebar"][2] if refined.get("left_sidebar") else 0
    main_x2 = refined["right_sidebar"][0] if refined.get("right_sidebar") else width
    if main_x2 <= main_x1:
        main_x1, main_x2 = 0, width
    if footer_top <= header_bottom:
        header_bottom, footer_top = 0, height
    refined["main_content"] = clamp_bbox((main_x1, header_bottom, main_x2, footer_top), width, height)
    return refined


def _best_color_vertical_boundary(
    frame_bgr: np.ndarray,
    detections: Sequence[UIDetection],
    *,
    current: int,
    min_x: int,
    max_x: int,
    y1: int,
    y2: int,
    side: str,
) -> Optional[int]:
    height, width = frame_bgr.shape[:2]
    min_x = max(1, min(width - 2, int(min_x)))
    max_x = max(1, min(width - 2, int(max_x)))
    if max_x <= min_x or y2 <= y1:
        return None
    window = max(10, min(36, int(width * 0.018)))
    step = max(2, int((max_x - min_x) / 120))
    best_x = None
    best_strength = 0.0
    for x in range(min_x, max_x + 1, step):
        strength = _vertical_color_boundary_strength(frame_bgr, x, y1, y2, window)
        strength -= _vertical_cut_penalty(detections, x, y1, y2, width) * 0.55
        strength -= abs(x - current) * 0.0008
        if strength > best_strength:
            best_strength = strength
            best_x = x
    threshold = 0.18 if side in {"left", "right"} else 0.22
    return int(best_x) if best_x is not None and best_strength >= threshold else None


def _best_color_horizontal_boundary(
    frame_bgr: np.ndarray,
    detections: Sequence[UIDetection],
    *,
    current: int,
    min_y: int,
    max_y: int,
    x1: int,
    x2: int,
) -> Optional[int]:
    height, width = frame_bgr.shape[:2]
    min_y = max(1, min(height - 2, int(min_y)))
    max_y = max(1, min(height - 2, int(max_y)))
    if max_y <= min_y or x2 <= x1:
        return None
    window = max(8, min(32, int(height * 0.012)))
    step = max(2, int((max_y - min_y) / 100))
    best_y = None
    best_strength = 0.0
    for y in range(min_y, max_y + 1, step):
        strength = _horizontal_color_boundary_strength(frame_bgr, y, x1, x2, window)
        strength -= _horizontal_cut_penalty(detections, y, x1, x2, height) * 0.55
        strength -= abs(y - current) * 0.0008
        if strength > best_strength:
            best_strength = strength
            best_y = y
    return int(best_y) if best_y is not None and best_strength >= 0.18 else None


def _vertical_color_boundary_strength(frame_bgr: np.ndarray, x: int, y1: int, y2: int, window: int) -> float:
    height, width = frame_bgr.shape[:2]
    left = frame_bgr[max(0, y1) : min(height, y2), max(0, x - window) : max(0, x - 2)]
    right = frame_bgr[max(0, y1) : min(height, y2), min(width, x + 2) : min(width, x + window)]
    return _section_change_strength(left, right)


def _horizontal_color_boundary_strength(frame_bgr: np.ndarray, y: int, x1: int, x2: int, window: int) -> float:
    height, width = frame_bgr.shape[:2]
    top = frame_bgr[max(0, y - window) : max(0, y - 2), max(0, x1) : min(width, x2)]
    bottom = frame_bgr[min(height, y + 2) : min(height, y + window), max(0, x1) : min(width, x2)]
    return _section_change_strength(top, bottom)


def _section_change_strength(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or getattr(a, "size", 0) == 0 or getattr(b, "size", 0) == 0:
        return 0.0
    a_lab = cv2.cvtColor(a, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    b_lab = cv2.cvtColor(b, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    a_mean = np.mean(a_lab, axis=0)
    b_mean = np.mean(b_lab, axis=0)
    color_delta = float(np.linalg.norm(a_mean - b_mean) / 120.0)
    a_uniform = 1.0 - min(1.0, float(np.mean(np.std(a_lab, axis=0))) / 42.0)
    b_uniform = 1.0 - min(1.0, float(np.mean(np.std(b_lab, axis=0))) / 42.0)
    uniformity = max(0.0, (a_uniform + b_uniform) / 2.0)
    return color_delta * (0.35 + 0.65 * uniformity)


def _vertical_cut_penalty(
    detections: Sequence[UIDetection],
    x: int,
    y1: int,
    y2: int,
    width: int,
) -> float:
    span = max(1, y2 - y1)
    penalty = 0.0
    for det in detections:
        bx1, by1, bx2, by2 = [int(v) for v in det.box]
        if bx2 - bx1 > int(width * 0.75):
            continue
        if bx1 <= x <= bx2 and by2 > y1 and by1 < y2:
            penalty += (max(0, min(by2, y2) - max(by1, y1)) / span) * (0.4 + float(det.confidence))
    return penalty


def _horizontal_cut_penalty(
    detections: Sequence[UIDetection],
    y: int,
    x1: int,
    x2: int,
    height: int,
) -> float:
    span = max(1, x2 - x1)
    penalty = 0.0
    for det in detections:
        bx1, by1, bx2, by2 = [int(v) for v in det.box]
        if by2 - by1 > int(height * 0.75):
            continue
        if by1 <= y <= by2 and bx2 > x1 and bx1 < x2:
            penalty += (max(0, min(bx2, x2) - max(bx1, x1)) / span) * (0.4 + float(det.confidence))
    return penalty


def _header_bottom(regions: Dict[str, Optional[BBox]], confidences: Dict[str, float]) -> int:
    header = regions.get("header")
    if header and float(confidences.get("header", 0.0) or 0.0) > 0:
        return int(header[3])
    return 0


def draw_groups(frame: np.ndarray, groups: Sequence[object]) -> np.ndarray:
    overlay = frame.copy()
    height, width = frame.shape[:2]
    thickness = max(2, round(min(width, height) / 360))
    font_scale = max(0.45, min(width, height) / 1100)
    for group in groups:
        label_name = str(getattr(group, "label", "region"))
        x1, y1, x2, y2 = [int(v) for v in getattr(group, "bbox")]
        color = COLORS.get(label_name, (210, 210, 210))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
        label = f"{label_name} {float(getattr(group, 'score', 0.0)):.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        label_y1 = max(0, y1 - th - baseline - 8)
        label_y2 = min(height, label_y1 + th + baseline + 8)
        label_x2 = min(width, x1 + tw + 10)
        cv2.rectangle(overlay, (x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.putText(
            overlay,
            label,
            (x1 + 5, label_y2 - baseline - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
    return overlay


def _iter_images(path: Path) -> Iterable[Path]:
    for suffix in ("*.png", "*.jpg", "*.jpeg"):
        yield from sorted(path.glob(suffix))


if __name__ == "__main__":
    raise SystemExit(main())

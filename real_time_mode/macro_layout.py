from __future__ import annotations

from dataclasses import dataclass
import site
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from .state import BBox, LayoutGroup, bbox_iou, clamp_bbox


UI_MODEL_FILENAME = "yolo-ui.pt"
DEFAULT_UI_MODEL_REPO = "swaylenhayes/uitag-yolo11s-ui-detect-v1"
UI_REGION_CONFIDENCE_THRESHOLD = 0.75


@dataclass
class UIDetection:
    box: BBox
    class_name: str
    confidence: float


@dataclass
class MacroLayoutDetector:
    model_ref: str
    model_filename: str = UI_MODEL_FILENAME
    device: Optional[str] = None
    conf: float = 0.20
    iou: float = 0.45
    max_det: int = 256
    tile_size: int = 640
    tile_step: int = 512

    def __post_init__(self) -> None:
        self.device = self.device or ("cuda:0" if _cuda_available() else "cpu")
        self._model: Optional[YOLO] = None
        self._weights_path: Optional[str] = None

    def warmup(self) -> None:
        self._ensure_model()

    def detect(self, frame_bgr: Any) -> List[LayoutGroup]:
        frame_h, frame_w = frame_bgr.shape[:2]
        try:
            detections = self._run_tiled_detection(frame_bgr)
            regions, region_confidences = estimate_coarse_regions(detections, frame_w, frame_h)
            regions = refine_regions_from_boundary_evidence(frame_bgr, detections, regions, region_confidences)
            groups = build_layout_groups(regions, region_confidences, frame_w, frame_h)
            if groups:
                return groups
        except Exception:
            pass
        return [_fallback_content_group(frame_w, frame_h)]

    def _ensure_model(self) -> YOLO:
        if self._model is None:
            weights_path = self._resolve_weights_path()
            self._weights_path = weights_path
            self._model = YOLO(weights_path)
        return self._model

    def _resolve_weights_path(self) -> str:
        candidate = Path(self.model_ref)
        if candidate.exists():
            return str(candidate)
        bundled = _resolve_bundled_uitag_weights()
        if bundled is not None:
            return bundled
        repo_id = self.model_ref or DEFAULT_UI_MODEL_REPO
        return hf_hub_download(repo_id=repo_id, filename=self.model_filename)

    def _run_tiled_detection(self, frame_bgr: np.ndarray) -> List[UIDetection]:
        model = self._ensure_model()
        frame_rgb = frame_bgr[:, :, ::-1]
        height, width = frame_rgb.shape[:2]
        all_detections: List[UIDetection] = []
        device_arg: Any = 0 if str(self.device).startswith("cuda") else "cpu"
        for left, top in iter_tiles(width, height, self.tile_size, self.tile_step):
            tile = frame_rgb[top : min(top + self.tile_size, height), left : min(left + self.tile_size, width)]
            result = model.predict(
                source=tile,
                imgsz=self.tile_size,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                device=device_arg,
                verbose=False,
            )[0]
            if result.boxes is None:
                continue
            names = result.names or {}
            boxes = result.boxes.xyxy.cpu().numpy().tolist()
            scores = result.boxes.conf.cpu().numpy().tolist()
            classes = result.boxes.cls.cpu().numpy().tolist()
            for idx, raw_box in enumerate(boxes):
                x1, y1, x2, y2 = raw_box
                full_box = clamp_bbox((x1 + left, y1 + top, x2 + left, y2 + top), width, height)
                if (full_box[2] - full_box[0]) <= 4 or (full_box[3] - full_box[1]) <= 4:
                    continue
                class_name = str(names.get(int(classes[idx]), "Unknown"))
                all_detections.append(
                    UIDetection(
                        box=full_box,
                        class_name=class_name,
                        confidence=float(scores[idx]) if idx < len(scores) else 0.0,
                    )
                )
        return class_aware_nms(all_detections, iou_threshold=self.iou)


def iter_tiles(width: int, height: int, tile_size: int, tile_step: int) -> Iterable[Tuple[int, int]]:
    x_positions = _axis_positions(width, tile_size, tile_step)
    y_positions = _axis_positions(height, tile_size, tile_step)
    for top in y_positions:
        for left in x_positions:
            yield left, top


def _axis_positions(length: int, tile_size: int, tile_step: int) -> List[int]:
    if length <= tile_size:
        return [0]
    positions = list(range(0, max(1, length - tile_size + 1), tile_step))
    last = max(0, length - tile_size)
    if not positions or positions[-1] != last:
        positions.append(last)
    return positions


def class_aware_nms(detections: Sequence[UIDetection], iou_threshold: float = 0.45) -> List[UIDetection]:
    ordered = sorted(detections, key=lambda item: item.confidence, reverse=True)
    kept: List[UIDetection] = []
    for candidate in ordered:
        overlap = any(
            existing.class_name == candidate.class_name and bbox_iou(existing.box, candidate.box) > iou_threshold
            for existing in kept
        )
        if not overlap:
            kept.append(candidate)
    return kept


def estimate_coarse_regions(
    detections: Sequence[UIDetection], width: int, height: int
) -> Tuple[Dict[str, Optional[BBox]], Dict[str, float]]:
    scale_height = _effective_layout_height(width, height)
    mobile_like = width <= 800
    top_limit = min(height, int(scale_height * 0.22))
    bottom_limit = max(0, height - int(scale_height * 0.18))
    left_limit = int(width * 0.35)
    right_limit = int(width * 0.65)

    header_candidates: List[BBox] = []
    footer_candidates: List[BBox] = []
    left_sidebar_candidates: List[BBox] = []
    right_sidebar_candidates: List[BBox] = []
    header_scores: List[float] = []
    footer_scores: List[float] = []
    left_sidebar_scores: List[float] = []
    right_sidebar_scores: List[float] = []
    footer_labels: List[str] = []
    left_sidebar_labels: List[str] = []
    right_sidebar_labels: List[str] = []

    for det in detections:
        x1, y1, x2, y2 = det.box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        box_h = y2 - y1
        klass = normalize_ui_class(det.class_name)

        if klass == "sidebar":
            if cx < width / 2.0:
                left_sidebar_candidates.append(det.box)
                left_sidebar_scores.append(float(det.confidence))
                left_sidebar_labels.append(klass)
            else:
                right_sidebar_candidates.append(det.box)
                right_sidebar_scores.append(float(det.confidence))
                right_sidebar_labels.append(klass)

        if klass in {"navigation", "menu"} and cx < left_limit and box_h > 20 and _is_side_rail_candidate(det.box, width, side="left"):
            left_sidebar_candidates.append(det.box)
            left_sidebar_scores.append(float(det.confidence))
            left_sidebar_labels.append(klass)
        if klass in {"navigation", "menu"} and cx > right_limit and box_h > 20 and _is_side_rail_candidate(det.box, width, side="right"):
            right_sidebar_candidates.append(det.box)
            right_sidebar_scores.append(float(det.confidence))
            right_sidebar_labels.append(klass)

        if cy < top_limit and klass in {"navigation", "menu", "input_elements", "button"}:
            header_candidates.append(det.box)
            header_scores.append(float(det.confidence))

        if cy > bottom_limit and klass in {"information_display", "menu", "navigation", "others"}:
            footer_candidates.append(det.box)
            footer_scores.append(float(det.confidence))
            footer_labels.append(klass)

    regions: Dict[str, Optional[BBox]] = {
        "header": None,
        "left_sidebar": None,
        "right_sidebar": None,
        "main_content": None,
        "footer": None,
    }
    region_confidences: Dict[str, float] = {
        "header": 0.0,
        "left_sidebar": 0.0,
        "right_sidebar": 0.0,
        "main_content": 0.0,
        "footer": 0.0,
    }

    header_box = union_boxes(header_candidates)
    footer_box = _accept_footer_box(
        union_boxes(footer_candidates),
        footer_candidates,
        footer_scores,
        footer_labels,
        width=width,
        height=height,
    )
    left_sidebar_box = _accept_sidebar_box(
        union_boxes(left_sidebar_candidates),
        left_sidebar_candidates,
        left_sidebar_scores,
        left_sidebar_labels,
        width=width,
        height=height,
        scale_height=scale_height,
        header_limit=top_limit,
        side="left",
        mobile_like=mobile_like,
    )
    right_sidebar_box = _accept_sidebar_box(
        union_boxes(right_sidebar_candidates),
        right_sidebar_candidates,
        right_sidebar_scores,
        right_sidebar_labels,
        width=width,
        height=height,
        scale_height=scale_height,
        header_limit=top_limit,
        side="right",
        mobile_like=mobile_like,
    )

    if header_box:
        header_height = min(max(header_box[3] + 12, int(scale_height * 0.08)), int(scale_height * 0.25), height)
        regions["header"] = (0, 0, width, header_height)
        region_confidences["header"] = _region_confidence(
            header_scores,
            header_candidates,
            width=width,
            height=height,
            axis="x",
        )
    else:
        regions["header"] = (0, 0, width, min(height, int(scale_height * 0.10)))

    if footer_box:
        footer_y1 = max(footer_box[1] - 12, int(height * 0.75))
        regions["footer"] = (0, footer_y1, width, height)
        region_confidences["footer"] = _region_confidence(
            footer_scores,
            footer_candidates,
            width=width,
            height=height,
            axis="x",
        )

    header_bottom = regions["header"][3] if regions["header"] and region_confidences["header"] > 0 else 0
    footer_top = regions["footer"][1] if regions["footer"] else height

    if left_sidebar_box:
        left_edge = min(left_sidebar_box[2] + 15, int(width * 0.40))
        regions["left_sidebar"] = (0, header_bottom, left_edge, footer_top)
        region_confidences["left_sidebar"] = _region_confidence(
            left_sidebar_scores,
            left_sidebar_candidates,
            width=width,
            height=height,
            axis="y",
        )

    if right_sidebar_box:
        right_start = max(right_sidebar_box[0] - 15, int(width * 0.60))
        regions["right_sidebar"] = (right_start, header_bottom, width, footer_top)
        region_confidences["right_sidebar"] = _region_confidence(
            right_sidebar_scores,
            right_sidebar_candidates,
            width=width,
            height=height,
            axis="y",
        )

    main_x1 = regions["left_sidebar"][2] if regions["left_sidebar"] else 0
    main_x2 = regions["right_sidebar"][0] if regions["right_sidebar"] else width
    main_y1 = header_bottom
    main_y2 = footer_top
    if main_x2 <= main_x1:
        main_x1, main_x2 = 0, width
    if main_y2 <= main_y1:
        main_y1, main_y2 = 0, height
    regions["main_content"] = clamp_bbox((main_x1, main_y1, main_x2, main_y2), width, height)
    region_confidences["main_content"] = max(
        region_confidences["header"],
        region_confidences["left_sidebar"],
        region_confidences["right_sidebar"],
        region_confidences["footer"],
    )
    return regions, region_confidences


def build_layout_groups(
    regions: Dict[str, Optional[BBox]],
    region_confidences: Dict[str, float],
    frame_w: int,
    frame_h: int,
) -> List[LayoutGroup]:
    ordered_labels = ["header", "left_sidebar", "main_content", "right_sidebar", "footer"]
    z_order = {label: index for index, label in enumerate(ordered_labels)}
    accepted: Dict[str, Tuple[BBox, float, str]] = {}

    for label in ("header", "left_sidebar", "right_sidebar", "footer"):
        box = regions.get(label)
        confidence = float(region_confidences.get(label, 0.0) or 0.0)
        if not box or confidence <= 0.0:
            continue
        bbox = clamp_bbox(box, frame_w, frame_h)
        if (bbox[2] - bbox[0]) <= 0 or (bbox[3] - bbox[1]) <= 0:
            continue
        source = "uitag-yolo" if confidence > UI_REGION_CONFIDENCE_THRESHOLD else "uitag-yolo-lowconf"
        accepted[label] = (bbox, confidence, source)

    header_bottom = accepted["header"][0][3] if "header" in accepted else 0
    footer_top = accepted["footer"][0][1] if "footer" in accepted else frame_h
    main_x1 = accepted["left_sidebar"][0][2] if "left_sidebar" in accepted else 0
    main_x2 = accepted["right_sidebar"][0][0] if "right_sidebar" in accepted else frame_w
    if main_x2 <= main_x1:
        main_x1, main_x2 = 0, frame_w
    if footer_top <= header_bottom:
        header_bottom, footer_top = 0, frame_h

    main_bbox = clamp_bbox((main_x1, header_bottom, main_x2, footer_top), frame_w, frame_h)
    main_confidence = max(
        float(region_confidences.get("main_content", 0.0) or 0.0),
        *(score for _, score, _ in accepted.values()),
        0.0,
    )
    accepted["main_content"] = (main_bbox, main_confidence, "uitag-yolo-coverage")

    groups: List[LayoutGroup] = []
    for label in ordered_labels:
        item = accepted.get(label)
        if item is None:
            continue
        bbox, confidence, source = item
        groups.append(
            LayoutGroup(
                id=f"group_{label}",
                label=label,
                bbox=bbox,
                axis="xy",
                score=confidence,
                source=source,
                z_index=z_order[label],
            )
        )
    return groups or [_fallback_content_group(frame_w, frame_h)]


def refine_regions_from_boundary_evidence(
    frame_bgr: np.ndarray,
    detections: Sequence[UIDetection],
    regions: Dict[str, Optional[BBox]],
    region_confidences: Dict[str, float],
) -> Dict[str, Optional[BBox]]:
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return dict(regions)
    frame_h, frame_w = frame_bgr.shape[:2]
    refined = dict(regions)
    scale_height = _effective_layout_height(frame_w, frame_h)
    search_x = max(48, min(int(frame_w * 0.12), 240))
    search_y = max(32, min(int(scale_height * 0.12), 160))

    header_box = refined.get("header")
    footer_box = refined.get("footer")
    header_bottom = int(header_box[3]) if header_box and float(region_confidences.get("header", 0.0) or 0.0) > 0 else 0
    footer_top = int(footer_box[1]) if footer_box else frame_h
    if footer_top <= header_bottom:
        header_bottom, footer_top = 0, frame_h

    left_box = refined.get("left_sidebar")
    if left_box:
        current = int(left_box[2])
        min_x = max(int(frame_w * 0.05), current - search_x)
        max_x = min(int(frame_w * 0.45), current + search_x)
        new_x = _best_vertical_boundary(frame_bgr, detections, current, min_x, max_x, header_bottom, footer_top)
        refined["left_sidebar"] = clamp_bbox((0, header_bottom, new_x, footer_top), frame_w, frame_h)

    right_box = refined.get("right_sidebar")
    if right_box:
        current = int(right_box[0])
        min_x = max(int(frame_w * 0.55), current - search_x)
        max_x = min(int(frame_w * 0.95), current + search_x)
        new_x = _best_vertical_boundary(frame_bgr, detections, current, min_x, max_x, header_bottom, footer_top)
        refined["right_sidebar"] = clamp_bbox((new_x, header_bottom, frame_w, footer_top), frame_w, frame_h)

    header_box = refined.get("header")
    if header_box and float(region_confidences.get("header", 0.0) or 0.0) > 0:
        current = int(header_box[3])
        min_y = max(0, current - search_y)
        max_y = min(frame_h, int(scale_height * 0.35), current + search_y)
        new_y = _best_horizontal_boundary(frame_bgr, detections, current, min_y, max_y, 0, frame_w)
        refined["header"] = clamp_bbox((0, 0, frame_w, new_y), frame_w, frame_h)
        header_bottom = new_y

    footer_box = refined.get("footer")
    if footer_box:
        current = int(footer_box[1])
        min_y = max(0, current - search_y)
        max_y = min(frame_h, current + search_y)
        new_y = _best_horizontal_boundary(frame_bgr, detections, current, min_y, max_y, 0, frame_w)
        refined["footer"] = clamp_bbox((0, new_y, frame_w, frame_h), frame_w, frame_h)
        footer_top = new_y

    main_x1 = refined["left_sidebar"][2] if refined.get("left_sidebar") else 0
    main_x2 = refined["right_sidebar"][0] if refined.get("right_sidebar") else frame_w
    main_y1 = header_bottom
    main_y2 = footer_top
    if main_x2 <= main_x1:
        main_x1, main_x2 = 0, frame_w
    if main_y2 <= main_y1:
        main_y1, main_y2 = 0, frame_h
    refined["main_content"] = clamp_bbox((main_x1, main_y1, main_x2, main_y2), frame_w, frame_h)
    return refined


def _best_vertical_boundary(
    frame_bgr: np.ndarray,
    detections: Sequence[UIDetection],
    current: int,
    min_x: int,
    max_x: int,
    y1: int,
    y2: int,
) -> int:
    if max_x <= min_x or y2 <= y1:
        return current
    step = max(2, int((max_x - min_x) / 90))
    best_x = int(current)
    best_score = float("inf")
    for x in range(int(min_x), int(max_x) + 1, step):
        score = _visual_vertical_seam_score(frame_bgr, x, y1, y2)
        score += _vertical_detection_cut_penalty(detections, x, y1, y2, frame_bgr.shape[1])
        score += abs(x - current) * 0.0015
        if score < best_score:
            best_score = score
            best_x = x
    return int(best_x)


def _best_horizontal_boundary(
    frame_bgr: np.ndarray,
    detections: Sequence[UIDetection],
    current: int,
    min_y: int,
    max_y: int,
    x1: int,
    x2: int,
) -> int:
    if max_y <= min_y or x2 <= x1:
        return current
    step = max(2, int((max_y - min_y) / 90))
    best_y = int(current)
    best_score = float("inf")
    for y in range(int(min_y), int(max_y) + 1, step):
        score = _visual_horizontal_seam_score(frame_bgr, y, x1, x2)
        score += _horizontal_detection_cut_penalty(detections, y, x1, x2, frame_bgr.shape[0])
        score += abs(y - current) * 0.0015
        if score < best_score:
            best_score = score
            best_y = y
    return int(best_y)


def _visual_vertical_seam_score(frame_bgr: np.ndarray, x: int, y1: int, y2: int) -> float:
    frame_h, frame_w = frame_bgr.shape[:2]
    x1 = max(0, int(x) - 2)
    x2 = min(frame_w, int(x) + 3)
    y1 = max(0, min(frame_h, int(y1)))
    y2 = max(0, min(frame_h, int(y2)))
    strip = frame_bgr[y1:y2, x1:x2]
    return _strip_texture_score(strip)


def _visual_horizontal_seam_score(frame_bgr: np.ndarray, y: int, x1: int, x2: int) -> float:
    frame_h, frame_w = frame_bgr.shape[:2]
    y1 = max(0, int(y) - 2)
    y2 = min(frame_h, int(y) + 3)
    x1 = max(0, min(frame_w, int(x1)))
    x2 = max(0, min(frame_w, int(x2)))
    strip = frame_bgr[y1:y2, x1:x2]
    return _strip_texture_score(strip)


def _strip_texture_score(strip: np.ndarray) -> float:
    if strip is None or getattr(strip, "size", 0) == 0:
        return 1.0
    gray = strip.astype(np.float32).mean(axis=2) if strip.ndim == 3 else strip.astype(np.float32)
    texture = float(np.std(gray) / 64.0)
    dark_or_colored = float(np.mean((gray < 245).astype(np.float32))) * 0.35
    return texture + dark_or_colored


def _vertical_detection_cut_penalty(
    detections: Sequence[UIDetection],
    x: int,
    y1: int,
    y2: int,
    frame_w: int,
) -> float:
    span = max(1, y2 - y1)
    penalty = 0.0
    for det in detections:
        bx1, by1, bx2, by2 = [int(v) for v in det.box]
        if bx2 - bx1 > int(frame_w * 0.72):
            continue
        if bx1 <= x <= bx2 and by2 > y1 and by1 < y2:
            overlap = max(0, min(by2, y2) - max(by1, y1)) / span
            center_bias = 1.0 - min(1.0, abs(x - ((bx1 + bx2) / 2.0)) / max(1.0, (bx2 - bx1) / 2.0))
            penalty += (0.45 + float(det.confidence)) * overlap * (0.5 + center_bias)
    return penalty


def _horizontal_detection_cut_penalty(
    detections: Sequence[UIDetection],
    y: int,
    x1: int,
    x2: int,
    frame_h: int,
) -> float:
    span = max(1, x2 - x1)
    penalty = 0.0
    for det in detections:
        bx1, by1, bx2, by2 = [int(v) for v in det.box]
        if by2 - by1 > int(frame_h * 0.72):
            continue
        if by1 <= y <= by2 and bx2 > x1 and bx1 < x2:
            overlap = max(0, min(bx2, x2) - max(bx1, x1)) / span
            center_bias = 1.0 - min(1.0, abs(y - ((by1 + by2) / 2.0)) / max(1.0, (by2 - by1) / 2.0))
            penalty += (0.45 + float(det.confidence)) * overlap * (0.5 + center_bias)
    return penalty


def normalize_ui_class(raw_class_name: str) -> str:
    return raw_class_name.strip().lower().replace(" ", "_")


def union_boxes(boxes: Sequence[Sequence[int]]) -> Optional[BBox]:
    valid = [tuple(int(v) for v in box[:4]) for box in boxes if len(box) >= 4]
    if not valid:
        return None
    return (
        min(item[0] for item in valid),
        min(item[1] for item in valid),
        max(item[2] for item in valid),
        max(item[3] for item in valid),
    )


def _fallback_content_group(frame_w: int, frame_h: int) -> LayoutGroup:
    return LayoutGroup(
        id="group_main_content",
        label="main_content",
        bbox=(0, 0, int(frame_w), int(frame_h)),
        axis="xy",
        score=0.0,
        source="fallback",
    )


def _region_confidence(
    scores: Sequence[float],
    boxes: Sequence[Sequence[int]],
    *,
    width: int,
    height: int,
    axis: str,
) -> float:
    if not scores:
        return 0.0
    base = max(float(score) for score in scores)
    count_bonus = min(0.12, max(0, len(scores) - 1) * 0.025)
    span_bonus = 0.0
    union = union_boxes(boxes)
    if union is not None:
        x1, y1, x2, y2 = union
        if axis == "y":
            span = max(0.0, (y2 - y1) / max(1, height))
        else:
            span = max(0.0, (x2 - x1) / max(1, width))
        span_bonus = min(0.10, span * 0.12)
    return float(min(0.99, base + count_bonus + span_bonus))


def _accept_sidebar_box(
    union: Optional[BBox],
    candidates: Sequence[Sequence[int]],
    scores: Sequence[float],
    labels: Sequence[str],
    *,
    width: int,
    height: int,
    scale_height: int,
    header_limit: int,
    side: str,
    mobile_like: bool = False,
) -> Optional[BBox]:
    if union is None:
        return None
    x1, y1, x2, y2 = union
    span_w = max(0, x2 - x1)
    span_h = max(0, y2 - y1)
    if span_w < int(width * 0.06):
        return None
    if span_h < int(scale_height * 0.18):
        return None
    max_sidebar_width = 0.45 if mobile_like else 0.36
    if span_w > int(width * max_sidebar_width):
        return None
    if side == "left":
        if x1 > int(width * 0.05) or x2 > int(width * 0.40):
            return None
    else:
        if x2 < int(width * 0.95) or x1 < int(width * 0.60):
            return None

    below_header = sum(1 for box in candidates if len(box) >= 4 and int(box[3]) >= header_limit)
    vertical_bands = _vertical_band_count(candidates, scale_height)
    direct_sidebar = "sidebar" in labels and below_header >= 1 and span_h >= int(scale_height * 0.20)
    distributed_clues = len(candidates) >= 4 and vertical_bands >= 3 and span_h >= int(scale_height * 0.20)
    strong_single = any(score >= 0.84 for score in scores) and span_h >= int(scale_height * 0.30)
    if mobile_like and not direct_sidebar:
        return None
    if direct_sidebar or distributed_clues or strong_single:
        return union
    return None


def _accept_footer_box(
    union: Optional[BBox],
    candidates: Sequence[Sequence[int]],
    scores: Sequence[float],
    labels: Sequence[str],
    *,
    width: int,
    height: int,
) -> Optional[BBox]:
    if union is None:
        return None
    x1, y1, x2, y2 = union
    span_w = max(0, x2 - x1)
    span_h = max(0, y2 - y1)
    if y1 < int(height * 0.88):
        return None
    if span_w < int(width * 0.30):
        return None
    if span_h < int(height * 0.04):
        return None
    info_heavy = labels.count("information_display") >= 1
    enough_clues = len(candidates) >= 2 or any(score >= 0.84 for score in scores) or info_heavy
    return union if enough_clues else None


def _is_side_rail_candidate(box: Sequence[int], width: int, *, side: str) -> bool:
    if len(box) < 4:
        return False
    x1, _, x2, _ = [int(v) for v in box[:4]]
    box_w = x2 - x1
    if box_w > max(80, int(width * 0.34)):
        return False
    if side == "left":
        return x1 <= int(width * 0.12) and x2 <= int(width * 0.40)
    return x2 >= int(width * 0.88) and x1 >= int(width * 0.60)


def _vertical_band_count(boxes: Sequence[Sequence[int]], height: int) -> int:
    band_h = max(1, int(height * 0.10))
    bands = set()
    for box in boxes:
        if len(box) < 4:
            continue
        cy = (int(box[1]) + int(box[3])) / 2.0
        bands.add(int(cy / band_h))
    return len(bands)


def _effective_layout_height(width: int, height: int) -> int:
    """Estimate the viewport height for thresholding tall full-page captures."""
    if width <= 0 or height <= 0:
        return max(1, height)
    if width <= 800:
        return max(1, min(height, max(480, int(width * 1.25))))
    return max(1, min(height, max(720, int(width * 0.75))))


def _resolve_bundled_uitag_weights() -> Optional[str]:
    module_dir = Path(__file__).resolve().parent
    project_root = module_dir.parent
    local_candidates = [
        module_dir / "vendor" / "uitag" / UI_MODEL_FILENAME,
        project_root / "real_time_mode" / "vendor" / "uitag" / UI_MODEL_FILENAME,
    ]
    for candidate in local_candidates:
        if candidate.exists():
            return str(candidate)

    search_roots: List[Path] = []
    try:
        search_roots.extend(Path(path) for path in site.getsitepackages())
    except Exception:
        pass
    try:
        search_roots.append(Path(site.getusersitepackages()))
    except Exception:
        pass
    for root in search_roots:
        candidate = root / "uitag" / "models" / UI_MODEL_FILENAME
        if candidate.exists():
            return str(candidate)
    return None


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False

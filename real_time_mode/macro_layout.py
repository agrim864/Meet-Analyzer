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
    top_limit = int(height * 0.22)
    bottom_limit = int(height * 0.82)
    left_limit = int(width * 0.35)
    right_limit = int(width * 0.65)

    header_candidates: List[BBox] = []
    footer_candidates: List[BBox] = []
    left_sidebar_candidates: List[BBox] = []
    header_scores: List[float] = []
    footer_scores: List[float] = []
    left_sidebar_scores: List[float] = []

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

        if klass in {"navigation", "menu"} and cx < left_limit and box_h > 20:
            left_sidebar_candidates.append(det.box)
            left_sidebar_scores.append(float(det.confidence))

        if cy < top_limit and klass in {"navigation", "menu", "input_elements", "button"}:
            header_candidates.append(det.box)
            header_scores.append(float(det.confidence))

        if cy > bottom_limit and klass in {"information_display", "menu", "navigation", "others"}:
            footer_candidates.append(det.box)
            footer_scores.append(float(det.confidence))

    regions: Dict[str, Optional[BBox]] = {
        "header": None,
        "left_sidebar": None,
        "main_content": None,
        "footer": None,
    }
    region_confidences: Dict[str, float] = {
        "header": 0.0,
        "left_sidebar": 0.0,
        "main_content": 0.0,
        "footer": 0.0,
    }

    header_box = union_boxes(header_candidates)
    footer_box = union_boxes(footer_candidates)
    left_sidebar_box = union_boxes(left_sidebar_candidates)

    if header_box:
        header_height = min(max(header_box[3] + 12, int(height * 0.08)), int(height * 0.25))
        regions["header"] = (0, 0, width, header_height)
        region_confidences["header"] = _region_confidence(
            header_scores,
            header_candidates,
            width=width,
            height=height,
            axis="x",
        )
    else:
        regions["header"] = (0, 0, width, int(height * 0.10))

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

    header_bottom = regions["header"][3] if regions["header"] else 0
    footer_top = regions["footer"][1] if regions["footer"] else height

    if left_sidebar_box:
        left_edge = min(left_sidebar_box[2] + 15, int(width * 0.45))
        regions["left_sidebar"] = (0, header_bottom, left_edge, footer_top)
        region_confidences["left_sidebar"] = _region_confidence(
            left_sidebar_scores,
            left_sidebar_candidates,
            width=width,
            height=height,
            axis="y",
        )

    main_x1 = regions["left_sidebar"][2] if regions["left_sidebar"] else 0
    main_x2 = width
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
        region_confidences["footer"],
    )
    return regions, region_confidences


def build_layout_groups(
    regions: Dict[str, Optional[BBox]],
    region_confidences: Dict[str, float],
    frame_w: int,
    frame_h: int,
) -> List[LayoutGroup]:
    groups: List[LayoutGroup] = []
    ordered_labels = ["header", "left_sidebar", "main_content", "footer"]
    z_order = {label: index for index, label in enumerate(ordered_labels)}
    accepted: Dict[str, Tuple[BBox, float]] = {}
    for label in ("header", "left_sidebar", "footer"):
        box = regions.get(label)
        confidence = float(region_confidences.get(label, 0.0) or 0.0)
        if not box:
            continue
        bbox = clamp_bbox(box, frame_w, frame_h)
        if (bbox[2] - bbox[0]) <= 0 or (bbox[3] - bbox[1]) <= 0:
            continue
        if confidence > UI_REGION_CONFIDENCE_THRESHOLD or _should_use_region_for_coverage(label, bbox, frame_w, frame_h):
            accepted[label] = (bbox, confidence)

    header_bottom = accepted.get("header", ((0, 0, 0, 0), 0.0))[0][3] if "header" in accepted else 0
    footer_top = accepted.get("footer", ((0, 0, frame_w, frame_h), 0.0))[0][1] if "footer" in accepted else frame_h
    main_x1 = accepted.get("left_sidebar", ((0, 0, 0, 0), 0.0))[0][2] if "left_sidebar" in accepted else 0
    main_x2 = frame_w
    if main_x2 <= main_x1:
        main_x1, main_x2 = 0, frame_w
    if footer_top <= header_bottom:
        header_bottom, footer_top = 0, frame_h

    main_bbox = clamp_bbox((main_x1, header_bottom, main_x2, footer_top), frame_w, frame_h)
    main_confidence = max(
        float(region_confidences.get("main_content", 0.0) or 0.0),
        *(score for _, score in accepted.values()),
        0.0,
    )
    accepted["main_content"] = (main_bbox, main_confidence)

    for label in ordered_labels:
        item = accepted.get(label)
        if item is None:
            continue
        bbox, confidence = item
        groups.append(
            LayoutGroup(
                id=f"group_{label}",
                label=label,
                bbox=bbox,
                axis="xy",
                score=confidence,
                source="uitag-yolo",
                z_index=z_order[label],
            )
        )
    return groups or [_fallback_content_group(frame_w, frame_h)]


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


def _should_use_region_for_coverage(label: str, bbox: BBox, frame_w: int, frame_h: int) -> bool:
    x1, y1, x2, y2 = bbox
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    if label == "header":
        return x1 == 0 and x2 == frame_w and height >= int(frame_h * 0.06)
    if label == "footer":
        return x1 == 0 and x2 == frame_w and height >= int(frame_h * 0.06)
    if label == "left_sidebar":
        return x1 == 0 and width >= int(frame_w * 0.10) and height >= int(frame_h * 0.22)
    return False


def _resolve_bundled_uitag_weights() -> Optional[str]:
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

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from .ocr_adapter import OCRResult, OCRTextExtractor

logger = logging.getLogger(__name__)


@dataclass
class SlideLayoutBlock:
    type: str
    text: str
    bbox: List[int]
    order: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "text": self.text,
            "bbox": list(self.bbox),
            "order": self.order,
        }


@dataclass
class SlideAnalysis:
    provider: str
    layout: List[SlideLayoutBlock] = field(default_factory=list)
    full_text: str = ""
    title: str = ""
    fingerprint: str = ""
    confidence: float = 0.0
    reading_order: List[str] = field(default_factory=list)


class SuryaSlideAnalyzer:
    """Use Surya layout/OCR when available, otherwise fall back to OCR-driven slide layout extraction."""

    def __init__(self, ocr_extractor: Optional[OCRTextExtractor] = None) -> None:
        self.ocr = ocr_extractor or OCRTextExtractor()
        self._surya_available = _detect_surya()
        self._foundation_predictor = None
        self._layout_predictor = None
        self._recognition_predictor = None
        self._detection_predictor = None

    def warmup(self) -> None:
        self.ocr.warmup()
        if self._surya_available:
            self._ensure_predictors()

    def analyze(self, frame_bgr: np.ndarray) -> SlideAnalysis:
        if self._surya_available:
            try:
                return self._analyze_with_surya(frame_bgr)
            except Exception:
                logger.exception("Surya slide analysis failed; falling back to OCR layout extraction.")
        ocr = self.ocr.extract_full(frame_bgr, frame_type="slide")
        return build_slide_from_ocr_result(
            ocr,
            width=frame_bgr.shape[1],
            height=frame_bgr.shape[0],
            provider="ocr_fallback",
        )

    def _ensure_predictors(self) -> None:
        if self._layout_predictor is not None and self._recognition_predictor is not None and self._detection_predictor is not None:
            return
        from surya.detection import DetectionPredictor
        from surya.foundation import FoundationPredictor
        from surya.layout import LayoutPredictor
        from surya.recognition import RecognitionPredictor

        device = _preferred_device()
        self._foundation_predictor = FoundationPredictor(device=device)
        self._layout_predictor = LayoutPredictor(self._foundation_predictor)
        self._recognition_predictor = RecognitionPredictor(self._foundation_predictor)
        self._detection_predictor = DetectionPredictor(device=device)

    def _analyze_with_surya(self, frame_bgr: np.ndarray) -> SlideAnalysis:
        self._ensure_predictors()
        image = Image.fromarray(frame_bgr[:, :, ::-1])
        layout_result = self._layout_predictor([image])[0]
        recognition_result = self._recognition_predictor(
            [image],
            det_predictor=self._detection_predictor,
            sort_lines=True,
            return_words=False,
            math_mode=False,
        )[0]
        lines = _surya_lines_to_ocr(recognition_result)
        ocr = _normalize_surya_ocr(lines)
        analysis = build_slide_from_ocr_result(
            ocr,
            width=frame_bgr.shape[1],
            height=frame_bgr.shape[0],
            provider="surya",
        )
        analysis.layout = _merge_layout_with_text(layout_result, lines, analysis.layout)
        if not analysis.title:
            for block in analysis.layout:
                if block.type == "title" and block.text.strip():
                    analysis.title = block.text.strip()
                    break
        digest_source = {
            "title": analysis.title,
            "text": analysis.full_text[:4000],
            "layout": [(block.type, tuple(block.bbox), block.text[:160]) for block in analysis.layout[:96]],
        }
        analysis.fingerprint = hashlib.sha1(repr(digest_source).encode("utf-8")).hexdigest()
        analysis.reading_order = [block.text for block in analysis.layout if block.text.strip()]
        if analysis.confidence <= 0.0:
            analysis.confidence = ocr.confidence
        return analysis


def build_slide_from_ocr_result(ocr: OCRResult, *, width: int, height: int, provider: str) -> SlideAnalysis:
    layout: List[SlideLayoutBlock] = []
    ordered = sorted(ocr.lines, key=lambda item: (item["box"][1], item["box"][0]))
    for idx, line in enumerate(ordered, start=1):
        box = list(line["box"])
        text = str(line["text"])
        region_type = _infer_slide_block_type(text=text, box=box, width=width, height=height, order=idx)
        layout.append(SlideLayoutBlock(type=region_type, text=text, bbox=box, order=idx))
    title = ""
    for block in layout:
        if block.type == "title":
            title = block.text
            break
    if not title and layout:
        title = layout[0].text
    reading_order = [block.text for block in layout]
    digest_source = {
        "title": title,
        "text": ocr.text[:4000],
        "layout": [(block.type, tuple(block.bbox)) for block in layout[:64]],
    }
    fingerprint = hashlib.sha1(repr(digest_source).encode("utf-8")).hexdigest()
    return SlideAnalysis(
        provider=provider,
        layout=layout,
        full_text=ocr.text,
        title=title,
        fingerprint=fingerprint,
        confidence=ocr.confidence,
        reading_order=reading_order,
    )


def _infer_slide_block_type(*, text: str, box: Sequence[int], width: int, height: int, order: int) -> str:
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    compact = text.strip()
    lower = compact.lower()
    if order == 1 and y1 <= int(height * 0.22) and bw >= int(width * 0.35):
        return "title"
    if compact.startswith(("-", "*", "•")) or lower[:2].isdigit():
        return "list"
    if "|" in compact and len(compact.split("|")) >= 3:
        return "table"
    if bw >= int(width * 0.55) and bh >= int(height * 0.18):
        return "figure"
    return "text"


def _detect_surya() -> bool:
    try:
        __import__("surya")
        return True
    except Exception:
        return False


def _preferred_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _surya_lines_to_ocr(recognition_result: Any) -> List[Dict[str, Any]]:
    collected: List[Dict[str, Any]] = []
    for line in getattr(recognition_result, "text_lines", []) or []:
        text = str(getattr(line, "text", "") or "").strip()
        polygon = getattr(line, "polygon", None)
        if not text or not polygon:
            continue
        box = _polygon_to_bbox(polygon)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        confidence = float(getattr(line, "confidence", 0.0) or 0.0)
        collected.append({"text": text, "box": box, "confidence": confidence})
    return collected


def _normalize_surya_ocr(lines: Sequence[Dict[str, Any]]) -> OCRResult:
    normalized: List[Dict[str, Any]] = []
    confidences: List[float] = []
    for item in lines:
        text = str(item.get("text", "") or "").strip()
        box = [int(v) for v in item.get("box", [0, 0, 0, 0])[:4]]
        if not text or len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
            continue
        confidence = float(item.get("confidence", 0.0) or 0.0)
        normalized.append({"text": text, "box": box, "confidence": confidence, "frame_type": "slide"})
        confidences.append(confidence)
    merged_text = "\n".join(item["text"] for item in normalized)
    avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.0
    return OCRResult(text=merged_text, lines=normalized, provider="surya", confidence=avg_conf)


def _merge_layout_with_text(layout_result: Any, ocr_lines: Sequence[Dict[str, Any]], fallback_layout: Sequence[SlideLayoutBlock]) -> List[SlideLayoutBlock]:
    layout_boxes = list(getattr(layout_result, "bboxes", []) or [])
    if not layout_boxes:
        return list(fallback_layout)
    merged: List[SlideLayoutBlock] = []
    for idx, box in enumerate(layout_boxes, start=1):
        bbox = _polygon_to_bbox(getattr(box, "polygon", None))
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        raw_label = str(getattr(box, "label", "") or "text")
        label = _normalize_layout_label(raw_label)
        line_texts = []
        for line in ocr_lines:
            line_bbox = line.get("box", [0, 0, 0, 0])
            if _intersection_ratio(bbox, line_bbox) >= 0.2:
                line_texts.append(str(line.get("text", "")).strip())
        merged.append(
            SlideLayoutBlock(
                type=label,
                text=" ".join(part for part in line_texts if part).strip(),
                bbox=bbox,
                order=idx,
            )
        )
    if not merged:
        return list(fallback_layout)
    return merged


def _normalize_layout_label(label: str) -> str:
    lower = label.strip().lower()
    mapping = {
        "section_header": "title",
        "title": "title",
        "text": "text",
        "list": "list",
        "table": "table",
        "figure": "figure",
        "picture": "figure",
        "image": "figure",
    }
    return mapping.get(lower, lower or "text")


def _polygon_to_bbox(polygon: Any) -> List[int]:
    if polygon is None:
        return [0, 0, 0, 0]
    xs: List[float] = []
    ys: List[float] = []
    for point in polygon:
        try:
            x = float(point[0])
            y = float(point[1])
        except Exception:
            continue
        xs.append(x)
        ys.append(y)
    if not xs or not ys:
        return [0, 0, 0, 0]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def _intersection_ratio(a: Sequence[int], b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = [int(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [int(v) for v in b[:4]]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    base = max(1.0, float((bx2 - bx1) * (by2 - by1)))
    return inter / base

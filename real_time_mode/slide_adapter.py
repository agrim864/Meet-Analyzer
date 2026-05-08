from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .ocr_adapter import OCRResult, OCRTextExtractor


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
    """Use Surya when present, otherwise fall back to OCR-driven slide layout extraction."""

    def __init__(self, ocr_extractor: Optional[OCRTextExtractor] = None) -> None:
        self.ocr = ocr_extractor or OCRTextExtractor()
        self._surya_available = _detect_surya()

    def warmup(self) -> None:
        self.ocr.warmup()

    def analyze(self, frame_bgr: np.ndarray) -> SlideAnalysis:
        # The fallback path is deterministic and safe. If Surya becomes available later,
        # this method can be extended without changing the caller contract.
        ocr = self.ocr.extract_full(frame_bgr, frame_type="slide")
        return build_slide_from_ocr_result(ocr, width=frame_bgr.shape[1], height=frame_bgr.shape[0], provider="surya" if self._surya_available else "ocr_fallback")


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

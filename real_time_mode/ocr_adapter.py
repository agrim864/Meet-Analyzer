from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from .smart_keyframes_and_classify import (
    _choose_ocr_mode,
    _init_ocr_model,
    _resolve_ocr_backend_for_mode,
    run_ocr_full,
)


@dataclass
class OCRResult:
    text: str
    lines: List[Dict[str, Any]] = field(default_factory=list)
    provider: str = "ocr"
    confidence: float = 0.0


class OCRTextExtractor:
    def __init__(self, requested_mode: Optional[str] = None, ocr_lang: str = "en") -> None:
        self.requested_mode = requested_mode
        self.ocr_lang = str(ocr_lang or "en").strip() or "en"
        self.backend = ""
        self.use_gpu = False
        self._model: Any = None

    def warmup(self) -> None:
        self._ensure_model()

    def extract_full(self, frame_bgr: np.ndarray, *, frame_type: str) -> OCRResult:
        model = self._ensure_model()
        lines = run_ocr_full(model, self.backend, frame_bgr)
        return _normalize_ocr_lines(lines, provider=f"ocr:{self.backend}", frame_type=frame_type)

    def extract_regions(self, frame_bgr: np.ndarray, regions: Sequence[Sequence[int]], *, frame_type: str) -> OCRResult:
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            return OCRResult(text="", lines=[], provider=f"ocr:{self.backend or 'none'}", confidence=0.0)
        model = self._ensure_model()
        collected: List[Dict[str, Any]] = []
        for region in regions:
            x1, y1, x2, y2 = [int(v) for v in region[:4]]
            crop = frame_bgr[y1:y2, x1:x2]
            if crop is None or getattr(crop, "size", 0) == 0:
                continue
            crop_lines = run_ocr_full(model, self.backend, crop)
            for line in crop_lines:
                adjusted = dict(line)
                box = adjusted.get("box") or [0, 0, 0, 0]
                adjusted["box"] = [int(box[0]) + x1, int(box[1]) + y1, int(box[2]) + x1, int(box[3]) + y1]
                collected.append(adjusted)
        return _normalize_ocr_lines(collected, provider=f"ocr:{self.backend}", frame_type=frame_type)

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        mode = _choose_ocr_mode(self.requested_mode)
        backend, use_gpu = _resolve_ocr_backend_for_mode(mode)
        self.backend = backend
        self.use_gpu = bool(use_gpu)
        self._model = _init_ocr_model(backend=backend, ocr_lang=self.ocr_lang, use_gpu=self.use_gpu)
        return self._model


def _normalize_ocr_lines(lines: Sequence[Dict[str, Any]], *, provider: str, frame_type: str) -> OCRResult:
    normalized: List[Dict[str, Any]] = []
    confidences: List[float] = []
    for raw in lines or []:
        text = str(raw.get("text", "") or "").strip()
        box = [int(round(v)) for v in (raw.get("box") or [0, 0, 0, 0])[:4]]
        if not text or len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
            continue
        conf = float(raw.get("conf", raw.get("confidence", 0.0)) or 0.0)
        normalized.append(
            {
                "text": text,
                "box": box,
                "confidence": conf,
                "frame_type": frame_type,
            }
        )
        confidences.append(conf)
    merged_text = "\n".join(item["text"] for item in normalized)
    confidence = sum(confidences) / max(1, len(confidences))
    return OCRResult(text=merged_text, lines=normalized, provider=provider, confidence=confidence)

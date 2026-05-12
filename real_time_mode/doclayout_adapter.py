from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from huggingface_hub import hf_hub_download

from .state import BBox, UIElement

logger = logging.getLogger(__name__)


DEFAULT_DOCLAYOUT_REPO_ID = "juliozhao/DocLayout-YOLO-DocStructBench"
DEFAULT_DOCLAYOUT_FILENAME = "doclayout_yolo_docstructbench_imgsz1024.pt"


@dataclass
class DocLayoutPayload:
    elements: List[UIElement]
    provider: str = "doclayout_yolo"
    confidence: float = 0.0


class DocLayoutYOLOAdapter:
    def __init__(
        self,
        *,
        model_ref: str = "",
        repo_id: str = DEFAULT_DOCLAYOUT_REPO_ID,
        filename: str = DEFAULT_DOCLAYOUT_FILENAME,
        conf: float = 0.20,
        iou: float = 0.45,
        imgsz: int = 1024,
        device: str = "auto",
        enabled: bool = True,
    ) -> None:
        self.model_ref = str(model_ref or "").strip()
        self.repo_id = str(repo_id or DEFAULT_DOCLAYOUT_REPO_ID).strip()
        self.filename = str(filename or DEFAULT_DOCLAYOUT_FILENAME).strip()
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.device = str(device or "auto").strip() or "auto"
        self.enabled = bool(enabled)
        self._model: Any = None
        self._weights_path: Optional[str] = None

    def warmup(self) -> None:
        if not self.enabled:
            return
        self._ensure_model()

    def detect_full(self, frame_bgr: np.ndarray, *, frame_id: int = 0, timestamp_sec: float = 0.0) -> DocLayoutPayload:
        height, width = frame_bgr.shape[:2]
        return self.detect_regions(frame_bgr, [(0, 0, width, height)], frame_id=frame_id, timestamp_sec=timestamp_sec)

    def detect_regions(
        self,
        frame_bgr: np.ndarray,
        regions: Sequence[BBox],
        *,
        frame_id: int = 0,
        timestamp_sec: float = 0.0,
    ) -> DocLayoutPayload:
        if not self.enabled:
            return DocLayoutPayload(elements=[], provider="doclayout_disabled", confidence=0.0)
        model = self._ensure_model()
        collected: List[UIElement] = []
        confidences: List[float] = []
        frame_h, frame_w = frame_bgr.shape[:2]
        for region_index, region in enumerate(regions or [(0, 0, frame_w, frame_h)], start=1):
            x1, y1, x2, y2 = [int(v) for v in region[:4]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w, x2), min(frame_h, y2)
            crop = frame_bgr[y1:y2, x1:x2]
            if crop is None or getattr(crop, "size", 0) == 0:
                continue
            try:
                results = model.predict(
                    crop,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    iou=self.iou,
                    device=self._device_arg(),
                    verbose=False,
                )
            except TypeError:
                results = model.predict(crop, imgsz=self.imgsz, conf=self.conf, device=self._device_arg())
            for result in results or []:
                names = getattr(result, "names", {}) or {}
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue
                xyxy = boxes.xyxy.cpu().numpy().tolist()
                scores = boxes.conf.cpu().numpy().tolist()
                classes = boxes.cls.cpu().numpy().tolist()
                for idx, raw_box in enumerate(xyxy):
                    bx1, by1, bx2, by2 = [int(round(v)) for v in raw_box[:4]]
                    if bx2 <= bx1 or by2 <= by1:
                        continue
                    score = float(scores[idx]) if idx < len(scores) else 0.0
                    class_id = int(classes[idx]) if idx < len(classes) else -1
                    raw_label = str(names.get(class_id, f"class_{class_id}"))
                    normalized_type = _normalize_doclayout_type(raw_label)
                    if normalized_type in {"abandon", "background", "ignored"}:
                        continue
                    collected.append(
                        UIElement(
                            id=f"doclayout_{frame_id:05d}_{region_index:02d}_{len(collected) + 1:04d}",
                            type=normalized_type,
                            bbox=(bx1 + x1, by1 + y1, bx2 + x1, by2 + y1),
                            interactive=False,
                            confidence=score,
                            source="doclayout_yolo",
                            style={"doclayout_raw_type": raw_label},
                            last_seen_frame=frame_id,
                            timestamp=float(timestamp_sec),
                        )
                    )
                    confidences.append(score)
        return DocLayoutPayload(
            elements=_dedupe_layout_elements(collected),
            provider="doclayout_yolo",
            confidence=sum(confidences) / max(1, len(confidences)),
        )

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from doclayout_yolo import YOLOv10
        except Exception as exc:
            raise RuntimeError("doclayout-yolo is not installed. Run `pip install doclayout-yolo`.") from exc
        weights_path = self._resolve_weights_path()
        self._weights_path = weights_path
        try:
            self._model = YOLOv10(weights_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load DocLayout-YOLO weights from {weights_path}: {exc}") from exc
        return self._model

    def _resolve_weights_path(self) -> str:
        if self.model_ref:
            candidate = Path(self.model_ref)
            if candidate.exists():
                return str(candidate)
        return hf_hub_download(repo_id=self.repo_id, filename=self.filename)

    def _device_arg(self) -> str:
        if self.device == "auto":
            try:
                import torch

                return "cuda:0" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        return self.device


def _normalize_doclayout_type(raw_label: str) -> str:
    label = str(raw_label or "layout_block").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "plain_text": "text_block",
        "text": "text_block",
        "title": "title",
        "figure": "figure",
        "figure_caption": "caption",
        "table": "table",
        "table_caption": "caption",
        "table_footnote": "footnote",
        "isolate_formula": "formula",
        "formula_caption": "caption",
        "list": "list",
    }
    return aliases.get(label, label or "layout_block")


def _dedupe_layout_elements(elements: Sequence[UIElement]) -> List[UIElement]:
    ordered = sorted(elements, key=lambda item: (-item.confidence, item.bbox[1], item.bbox[0]))
    kept: List[UIElement] = []
    for element in ordered:
        duplicate = False
        ex1, ey1, ex2, ey2 = element.bbox
        earea = max(1, (ex2 - ex1) * (ey2 - ey1))
        for existing in kept:
            ix1 = max(ex1, existing.bbox[0])
            iy1 = max(ey1, existing.bbox[1])
            ix2 = min(ex2, existing.bbox[2])
            iy2 = min(ey2, existing.bbox[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if element.type == existing.type and inter / earea >= 0.82:
                duplicate = True
                break
        if not duplicate:
            kept.append(element)
    kept.sort(key=lambda item: (item.bbox[1], item.bbox[0], item.id))
    return kept

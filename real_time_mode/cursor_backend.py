from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


class CursorMotionAnalysisBackend:
    def __init__(
        self,
        model_path: str,
        *,
        model_url: str = "",
        device: Optional[str] = None,
        conf: float = 0.20,
        iou: float = 0.25,
    ) -> None:
        path = Path(model_path).resolve()
        self.model_path = str(path)
        self.model_url = str(model_url or "").strip()
        self.device = device or ("cuda:0" if _cuda_available() else "cpu")
        self.conf = float(conf)
        self.iou = float(iou)
        self.low_conf = min(0.05, self.conf)
        self.model = self._load_model()
        self.template_threshold = 0.43
        self.local_template_threshold = 0.38
        self.trail_template_threshold = 0.40
        self._cursor_templates = self._build_cursor_templates()
        self.enable_template_with_model = os.getenv(
            "REALTIME_CURSOR_ENABLE_TEMPLATE_WITH_MODEL",
            "1",
        ).strip().lower() in {"1", "true", "yes"}

    def warmup(self) -> None:
        return

    def analyze(self, *, frame_bgr: Any, prev_frame_bgr: Any = None, prev_cursor: Any = None) -> Dict[str, Any]:
        if frame_bgr is None:
            raise RuntimeError("Cursor backend received an empty frame.")
        yolo_result = self._detect_with_yolo(frame_bgr)
        best_box = yolo_result.get("bbox")
        best_score = float(yolo_result.get("confidence", 0.0) or 0.0)
        method = str(yolo_result.get("method", "none") or "none")
        if best_box is None and (self.model is None or self.enable_template_with_model):
            template_result = self._detect_with_template(
                frame_bgr,
                prev_frame_bgr=prev_frame_bgr,
                prev_cursor=prev_cursor,
            )
            best_box = template_result.get("bbox")
            best_score = float(template_result.get("confidence", 0.0) or 0.0)
            method = str(template_result.get("method", "none") or "none")
        if best_box is None:
            return {
                "bbox": None,
                "confidence": 0.0,
                "motion_type": "unknown",
                "hotspot": None,
                "type": "unknown",
                "method": method,
            }
        best_box = [int(round(v)) for v in best_box[:4]]
        motion_type = "stationary"
        if prev_cursor is not None and getattr(prev_cursor, "bbox", None):
            px1, py1, px2, py2 = prev_cursor.bbox
            cx = (best_box[0] + best_box[2]) / 2.0
            cy = (best_box[1] + best_box[3]) / 2.0
            pcx = (px1 + px2) / 2.0
            pcy = (py1 + py2) / 2.0
            dx = cx - pcx
            dy = cy - pcy
            if abs(dx) + abs(dy) < 6:
                motion_type = "stationary"
            elif abs(dx) > abs(dy) * 1.5:
                motion_type = "horizontal"
            elif abs(dy) > abs(dx) * 1.5:
                motion_type = "vertical"
            else:
                motion_type = "diagonal"
        hotspot = (best_box[0] + 2, best_box[1] + 2)
        return {
            "bbox": best_box,
            "confidence": best_score,
            "motion_type": motion_type,
            "hotspot": hotspot,
            "type": "pointer",
            "method": method,
        }

    def _detect_with_yolo(self, frame_bgr: Any) -> Dict[str, Any]:
        if self.model is None:
            return {"bbox": None, "confidence": 0.0, "method": "none"}
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        result = self.model.predict(
            source=pil_image,
            conf=self.conf,
            iou=self.iou,
            device=0 if self.device.startswith("cuda") else "cpu",
            verbose=False,
        )[0]
        boxes = result.boxes.xyxy.cpu().numpy().tolist() if result.boxes is not None else []
        scores = result.boxes.conf.cpu().numpy().tolist() if result.boxes is not None else []
        if not boxes and self.low_conf < self.conf:
            result = self.model.predict(
                source=pil_image,
                conf=self.low_conf,
                iou=self.iou,
                device=0 if self.device.startswith("cuda") else "cpu",
                verbose=False,
            )[0]
            boxes = result.boxes.xyxy.cpu().numpy().tolist() if result.boxes is not None else []
            scores = result.boxes.conf.cpu().numpy().tolist() if result.boxes is not None else []
            for box, score in zip(boxes, scores):
                rounded = [int(round(v)) for v in box[:4]]
                if float(score) >= self.conf:
                    return {"bbox": box[:4], "confidence": float(score), "method": "yolo"}
                if self._looks_like_cursor_patch(frame_bgr, rounded, threshold=self.low_conf):
                    return {"bbox": box[:4], "confidence": float(score), "method": "yolo_low_conf"}
        if not boxes:
            return {"bbox": None, "confidence": 0.0, "method": "yolo"}
        return {
            "bbox": boxes[0][:4],
            "confidence": float(scores[0]) if scores else 0.0,
            "method": "yolo",
        }

    def _detect_with_template(self, frame_bgr: Any, *, prev_frame_bgr: Any = None, prev_cursor: Any = None) -> Dict[str, Any]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        edge_map = cv2.Canny(gray, 90, 180)
        search_windows: List[Tuple[np.ndarray, int, int, float]] = []
        if prev_cursor is not None and getattr(prev_cursor, "bbox", None):
            x1, y1, x2, y2 = [int(v) for v in prev_cursor.bbox]
            pad = 80
            sx1 = max(0, x1 - pad)
            sy1 = max(0, y1 - pad)
            sx2 = min(edge_map.shape[1], x2 + pad)
            sy2 = min(edge_map.shape[0], y2 + pad)
            if sx2 - sx1 >= 24 and sy2 - sy1 >= 24:
                search_windows.append((edge_map[sy1:sy2, sx1:sx2], sx1, sy1, self.local_template_threshold))
        elif prev_cursor is not None and getattr(prev_cursor, "trail", None):
            px, py = [int(v) for v in prev_cursor.trail[-1]]
            pad = 96
            sx1 = max(0, px - pad)
            sy1 = max(0, py - pad)
            sx2 = min(edge_map.shape[1], px + pad)
            sy2 = min(edge_map.shape[0], py + pad)
            if sx2 - sx1 >= 24 and sy2 - sy1 >= 24:
                search_windows.append((edge_map[sy1:sy2, sx1:sx2], sx1, sy1, self.trail_template_threshold))
        if search_windows and prev_frame_bgr is not None:
            for region in self._motion_search_windows(frame_bgr=frame_bgr, prev_frame_bgr=prev_frame_bgr):
                sx1, sy1, sx2, sy2 = region
                if sx2 - sx1 >= 20 and sy2 - sy1 >= 20:
                    search_windows.append((edge_map[sy1:sy2, sx1:sx2], sx1, sy1, self.local_template_threshold))
        if not search_windows:
            return {"bbox": None, "confidence": 0.0, "method": "template"}

        best_score = 0.0
        best_bbox: Optional[List[int]] = None
        best_threshold = self.template_threshold
        for search_region, offset_x, offset_y, threshold in search_windows:
            for template in self._cursor_templates:
                template_edges = template["edges"]
                th, tw = template_edges.shape[:2]
                if search_region.shape[0] < th or search_region.shape[1] < tw:
                    continue
                match = cv2.matchTemplate(search_region, template_edges, cv2.TM_CCOEFF_NORMED)
                _, score, _, location = cv2.minMaxLoc(match)
                if score > best_score:
                    x = int(location[0]) + offset_x
                    y = int(location[1]) + offset_y
                    bbox = [x, y, x + tw, y + th]
                    if not self._looks_like_cursor_patch(frame_bgr, bbox, threshold=threshold):
                        continue
                    best_score = float(score)
                    best_bbox = bbox
                    best_threshold = threshold

        if best_bbox is None or best_score < best_threshold:
            return {"bbox": None, "confidence": 0.0, "method": "template"}
        return {"bbox": best_bbox, "confidence": best_score, "method": "template"}

    def _looks_like_cursor_patch(self, frame_bgr: Any, bbox: List[int], *, threshold: float) -> bool:
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        h, w = frame_bgr.shape[:2]
        if x2 <= x1 or y2 <= y1:
            return False
        x1 = max(0, min(w - 1, x1))
        x2 = max(x1 + 1, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(y1 + 1, min(h, y2))
        patch = frame_bgr[y1:y2, x1:x2]
        if patch.size == 0:
            return False
        ph, pw = patch.shape[:2]
        area = ph * pw
        if area < 60 or area > 1600:
            return False
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        contrast = float(gray.max()) - float(gray.min())
        if contrast < 35:
            return False
        dark_ratio = float(np.mean(gray < 90))
        light_ratio = float(np.mean(gray > 180))
        edge_ratio = float(np.mean(cv2.Canny(gray, 70, 160) > 0))
        if edge_ratio < 0.03 or edge_ratio > 0.55:
            return False
        return dark_ratio > 0.01 or light_ratio > 0.05

    def _motion_search_windows(self, *, frame_bgr: Any, prev_frame_bgr: Any) -> List[Tuple[int, int, int, int]]:
        try:
            current_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            previous_gray = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            return []
        if current_gray.shape != previous_gray.shape:
            return []
        diff = cv2.absdiff(current_gray, previous_gray)
        _, thresh = cv2.threshold(diff, 24, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), dtype=np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        windows: List[Tuple[int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < 20 or area > 12000:
                continue
            pad = 32
            sx1 = max(0, x - pad)
            sy1 = max(0, y - pad)
            sx2 = min(current_gray.shape[1], x + w + pad)
            sy2 = min(current_gray.shape[0], y + h + pad)
            windows.append((sx1, sy1, sx2, sy2))
        windows.sort(key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
        return windows[:8]

    def _build_cursor_templates(self) -> List[Dict[str, np.ndarray]]:
        base = np.zeros((28, 20), dtype=np.uint8)
        polygon = np.array(
            [
                [1, 1],
                [1, 20],
                [6, 15],
                [9, 26],
                [12, 25],
                [9, 14],
                [16, 14],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(base, [polygon], 255)
        cv2.polylines(base, [polygon], isClosed=True, color=255, thickness=1)
        templates: List[Dict[str, np.ndarray]] = []
        for scale in (0.85, 1.0, 1.15, 1.3, 1.5):
            resized = cv2.resize(base, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            edges = cv2.Canny(resized, 30, 120)
            templates.append({"edges": edges})
        flipped = cv2.flip(base, 1)
        for scale in (1.0, 1.3):
            resized = cv2.resize(flipped, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            edges = cv2.Canny(resized, 30, 120)
            templates.append({"edges": edges})
        return templates

    def _load_model(self) -> Optional[YOLO]:
        if self.model_url:
            try:
                return YOLO(self.model_url)
            except Exception:
                pass
        path = Path(self.model_path).resolve()
        if path.exists():
            try:
                return YOLO(str(path))
            except Exception:
                pass
        return None


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False

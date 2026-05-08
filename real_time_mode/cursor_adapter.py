from __future__ import annotations

from typing import Any, List, Optional

from .state import CursorState, UIElement, bbox_center, bbox_contains_point, bbox_intersects


class CursorMotionAnalyzer:
    def __init__(self, backend: Optional[Any] = None, max_trail: int = 12) -> None:
        if backend is None:
            raise RuntimeError(
                "cursor-motion-analysis backend is required for real_time_mode. "
                "Inject a real cursor backend instead of relying on a fallback detector."
            )
        self.backend = backend
        self.max_trail = int(max_trail)

    def analyze(
        self,
        *,
        frame_bgr: Any,
        prev_frame_bgr: Any = None,
        prev_cursor: Optional[CursorState] = None,
        elements: Optional[List[UIElement]] = None,
    ) -> CursorState:
        result = self.backend.analyze(
            frame_bgr=frame_bgr,
            prev_frame_bgr=prev_frame_bgr,
            prev_cursor=prev_cursor,
        )
        if not isinstance(result, dict):
            raise RuntimeError("cursor-motion-analysis backend returned an invalid result.")
        raw_box = result.get("bbox")
        box = tuple(int(round(v)) for v in raw_box[:4]) if raw_box else None
        confidence = float(result.get("confidence", 0.0) or 0.0)
        trail = list(prev_cursor.trail) if prev_cursor else []
        motion_type = str(result.get("motion_type", "unknown") or "unknown")
        hotspot_raw = result.get("hotspot")
        hotspot = tuple(int(round(v)) for v in hotspot_raw[:2]) if hotspot_raw else None
        if box is not None:
            center = tuple(int(round(v)) for v in bbox_center(box))
            trail.append(center)
            trail = trail[-self.max_trail :]
        if hotspot is None and box is not None:
            hotspot = (int(box[0]) + 2, int(box[1]) + 2)
        near_element_id = None
        if box is not None and elements:
            for element in elements:
                if hotspot is not None and bbox_contains_point(element.bbox, hotspot, padding=4):
                    near_element_id = element.id
                    break
            if near_element_id is None:
                expanded = (box[0] - 12, box[1] - 12, box[2] + 12, box[3] + 12)
                for element in elements:
                    if bbox_intersects(element.bbox, expanded):
                        near_element_id = element.id
                        break
        return CursorState(
            visible=box is not None,
            bbox=box,
            hotspot=hotspot,
            cursor_type=str(result.get("type", "unknown") or "unknown"),
            trail=trail,
            motion_type=motion_type if box is not None else "unknown",
            confidence=confidence,
            method=str(result.get("method", "fallback") or "fallback"),
            near_element_id=near_element_id,
        )

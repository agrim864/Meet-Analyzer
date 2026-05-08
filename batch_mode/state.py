from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

BBox = Tuple[int, int, int, int]


def clamp_bbox(box: Sequence[int], width: int, height: int) -> BBox:
    x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def bbox_area(box: Sequence[int]) -> int:
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_center(box: Sequence[int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_contains_point(box: Sequence[int], point: Sequence[float], padding: int = 0) -> bool:
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    px, py = [float(v) for v in point[:2]]
    return (x1 - padding) <= px <= (x2 + padding) and (y1 - padding) <= py <= (y2 + padding)


def bbox_intersects(a: Sequence[int], b: Sequence[int], padding: int = 0) -> bool:
    ax1, ay1, ax2, ay2 = [int(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [int(v) for v in b[:4]]
    return not (
        ax2 < bx1 - padding
        or bx2 < ax1 - padding
        or ay2 < by1 - padding
        or by2 < ay1 - padding
    )


def bbox_iou(a: Sequence[int], b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(1e-6, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return float(inter / union)


def translate_bbox(box: Sequence[int], dx: float, dy: float) -> BBox:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    return int(round(x1 + dx)), int(round(y1 + dy)), int(round(x2 + dx)), int(round(y2 + dy))


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


@dataclass
class TranscriptSegment:
    start_time: float
    end_time: float
    speaker: Optional[str]
    text: str
    is_final: bool = True
    source: str = "live"
    segment_id: Optional[int] = None

    def copy(self) -> "TranscriptSegment":
        return replace(self)


@dataclass
class ModeRouteDecision:
    mode: str
    confidence: float
    decision_path: str
    scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "confidence": self.confidence,
            "decision_path": self.decision_path,
            "scores": dict(self.scores),
        }


@dataclass
class UIElement:
    id: str
    type: str
    bbox: BBox
    text: str = ""
    interactive: bool = False
    visible: bool = True
    occluded: bool = False
    confidence: float = 0.0
    source: str = "omniparser"
    caption: Optional[str] = None
    group_id: Optional[str] = None
    parent_macro_region_id: Optional[str] = None
    z_index: int = 0
    style: Dict[str, Any] = field(default_factory=dict)
    version: int = 1
    last_seen_frame: int = 0
    stale: bool = False
    timestamp: Optional[float] = None

    def copy(self) -> "UIElement":
        return replace(self, style=dict(self.style))


@dataclass
class Overlay:
    id: str
    type: str
    bbox: BBox
    title: str = ""
    visible: bool = True
    confidence: float = 0.0
    source: str = "policy"
    z_index: int = 100

    def copy(self) -> "Overlay":
        return replace(self)


@dataclass
class LayoutGroup:
    id: str
    label: str
    bbox: BBox
    element_ids: List[str] = field(default_factory=list)
    axis: str = "xy"
    score: float = 0.0
    source: str = "cache"
    z_index: int = 0
    frame_id: int = 0
    timestamp: float = 0.0
    stable: bool = True

    def copy(self) -> "LayoutGroup":
        return replace(self, element_ids=list(self.element_ids))


@dataclass
class CursorState:
    visible: bool = False
    bbox: Optional[BBox] = None
    hotspot: Optional[Tuple[int, int]] = None
    cursor_type: str = "unknown"
    trail: List[Tuple[int, int]] = field(default_factory=list)
    motion_type: str = "unknown"
    confidence: float = 0.0
    method: str = "none"
    near_element_id: Optional[str] = None
    tracking_state: str = "missing"
    missing_frames: int = 0

    def copy(self) -> "CursorState":
        return replace(self, trail=list(self.trail))


@dataclass
class ChangeHypothesis:
    name: str
    confidence: float
    affected_regions: List[BBox] = field(default_factory=list)
    transform_params: Dict[str, Any] = field(default_factory=dict)
    reparse_regions: List[BBox] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StateDelta:
    types: List[str] = field(default_factory=list)
    changed_regions: List[BBox] = field(default_factory=list)
    reparsed_regions: List[BBox] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ScreenState:
    frame_id: int
    timestamp_sec: float
    viewport: Dict[str, int]
    frame_type: str
    mode: str
    route: ModeRouteDecision
    theme: str = "unknown"
    visual_change: bool = False
    visual_payload: Dict[str, Any] = field(default_factory=dict)
    elements: List[UIElement] = field(default_factory=list)
    overlays: List[Overlay] = field(default_factory=list)
    layout_groups: List[LayoutGroup] = field(default_factory=list)
    relations: List[str] = field(default_factory=list)
    cursor: CursorState = field(default_factory=CursorState)
    transcript_window: List[TranscriptSegment] = field(default_factory=list)
    utterance_ids: List[int] = field(default_factory=list)
    utterance_time_span: Optional[Tuple[float, float]] = None
    summary: str = ""
    delta: StateDelta = field(default_factory=StateDelta)
    confidence: Dict[str, Any] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)
    next_element_index: int = 1
    next_overlay_index: int = 1
    last_full_parse_frame_id: int = 0

    def clone(self) -> "ScreenState":
        return ScreenState(
            frame_id=self.frame_id,
            timestamp_sec=self.timestamp_sec,
            viewport=dict(self.viewport),
            frame_type=self.frame_type,
            mode=self.mode,
            route=replace(self.route, scores=dict(self.route.scores)),
            theme=self.theme,
            visual_change=self.visual_change,
            visual_payload=dict(self.visual_payload),
            elements=[item.copy() for item in self.elements],
            overlays=[item.copy() for item in self.overlays],
            layout_groups=[item.copy() for item in self.layout_groups],
            relations=list(self.relations),
            cursor=self.cursor.copy(),
            transcript_window=[segment.copy() for segment in self.transcript_window],
            utterance_ids=list(self.utterance_ids),
            utterance_time_span=self.utterance_time_span,
            summary=self.summary,
            delta=replace(
                self.delta,
                types=list(self.delta.types),
                changed_regions=list(self.delta.changed_regions),
                reparsed_regions=list(self.delta.reparsed_regions),
            ),
            confidence=dict(self.confidence),
            debug=dict(self.debug),
            next_element_index=self.next_element_index,
            next_overlay_index=self.next_overlay_index,
            last_full_parse_frame_id=self.last_full_parse_frame_id,
        )


@dataclass
class SceneStateRecord:
    event_id: str
    frame_id: int
    timestamp_sec: float
    frame_type: str
    mode: str
    theme: str
    visual_change: bool
    visual_payload: Dict[str, Any]
    transcript_window: List[Dict[str, Any]]
    summary: str
    cursor: Dict[str, Any]
    elements: List[Dict[str, Any]]
    overlays: List[Dict[str, Any]]
    layout_groups: List[Dict[str, Any]]
    relations: List[str]
    delta_from_previous: Dict[str, Any]
    confidence: Dict[str, Any]
    debug: Dict[str, Any]
    utterance_ids: List[int]
    utterance_time_span: Optional[Tuple[float, float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "frame_id": self.frame_id,
            "timestamp_sec": self.timestamp_sec,
            "frame_type": self.frame_type,
            "mode": self.mode,
            "theme": self.theme,
            "visual_change": self.visual_change,
            "visual_payload": dict(self.visual_payload),
            "transcript_window": [dict(item) for item in self.transcript_window],
            "summary": self.summary,
            "cursor": dict(self.cursor),
            "elements": [dict(item) for item in self.elements],
            "overlays": [dict(item) for item in self.overlays],
            "layout_groups": [dict(item) for item in self.layout_groups],
            "relations": list(self.relations),
            "delta_from_previous": dict(self.delta_from_previous),
            "confidence": dict(self.confidence),
            "debug": dict(self.debug),
            "utterance_ids": list(self.utterance_ids),
            "utterance_time_span": self.utterance_time_span,
        }

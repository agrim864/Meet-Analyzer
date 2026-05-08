from __future__ import annotations

import logging
from dataclasses import dataclass, field
from difflib import ndiff
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from .cursor_adapter import CursorMotionAnalyzer
from .export import export_scene_record
from .frame_summarizer import (
    FrameSummaryRequest,
    HeuristicFrameSummarizer,
    ResilientFrameSummarizer,
    SummaryMemoryTurn,
    _collect_micro_highlights,
    _looks_like_noise,
)
from .macro_layout import MacroLayoutDetector
from .merge import merge_reparsed_state
from .mode_router import ModeRouter
from .ocr_adapter import OCRResult, OCRTextExtractor
from .omniparser_adapter import OmniParserAdapter, ParsePayload
from .siglip_adapter import SiglipScreenClassifier
from .slide_adapter import SlideAnalysis, SuryaSlideAnalyzer, build_slide_from_ocr_result
from .state import (
    BBox,
    CursorState,
    LayoutGroup,
    ModeRouteDecision,
    SceneStateRecord,
    ScreenState,
    StateDelta,
    TranscriptSegment,
    UIElement,
    bbox_area,
    bbox_center,
    bbox_contains_point,
    bbox_intersects,
    clamp_bbox,
    union_boxes,
)
from .stt_adapter import UtteranceStore

logger = logging.getLogger(__name__)


@dataclass
class ChangeDetectionResult:
    changed: bool
    change_ratio: float
    ssim_score: float
    changed_regions: List[BBox] = field(default_factory=list)
    change_type: str = "none"


@dataclass
class CodeModeCache:
    prev_frame_masked: Optional[np.ndarray] = None
    prev_ocr_text: str = ""
    prev_ocr_lines: List[Dict[str, Any]] = field(default_factory=list)
    last_visual_output: Dict[str, Any] = field(default_factory=dict)
    last_state: Optional[ScreenState] = None
    last_change_time: Optional[float] = None
    last_processed_time: Optional[float] = None


@dataclass
class SlideModeCache:
    prev_frame_masked: Optional[np.ndarray] = None
    prev_slide_fingerprint: str = ""
    prev_surya_layout: List[Dict[str, Any]] = field(default_factory=list)
    prev_surya_text: str = ""
    last_visual_output: Dict[str, Any] = field(default_factory=dict)
    last_state: Optional[ScreenState] = None
    last_change_time: Optional[float] = None
    last_processed_time: Optional[float] = None


@dataclass
class DemoModeCache:
    prev_frame_masked: Optional[np.ndarray] = None
    prev_macro_layout: List[LayoutGroup] = field(default_factory=list)
    prev_micro_elements: List[UIElement] = field(default_factory=list)
    prev_page_summary: str = ""
    last_visual_output: Dict[str, Any] = field(default_factory=dict)
    last_state: Optional[ScreenState] = None
    last_change_time: Optional[float] = None
    last_processed_time: Optional[float] = None


@dataclass
class OtherModeCache:
    last_seen_time: Optional[float] = None
    last_transcript_only_summary: str = ""
    last_state: Optional[ScreenState] = None
    last_processed_time: Optional[float] = None


@dataclass
class TranscriptBufferState:
    finalized_segments: List[Dict[str, Any]] = field(default_factory=list)
    interim_segments: List[Dict[str, Any]] = field(default_factory=list)
    last_visual_event_time: Optional[float] = None
    last_summary_time: Optional[float] = None


@dataclass
class StateCache:
    current_mode: str = "other"
    previous_mode: str = "other"
    last_stable_mode: str = "other"
    mode_prediction_history: List[Dict[str, Any]] = field(default_factory=list)
    code: CodeModeCache = field(default_factory=CodeModeCache)
    slide: SlideModeCache = field(default_factory=SlideModeCache)
    demo: DemoModeCache = field(default_factory=DemoModeCache)
    other: OtherModeCache = field(default_factory=OtherModeCache)
    cursor_state: CursorState = field(default_factory=CursorState)
    transcript_buffer: TranscriptBufferState = field(default_factory=TranscriptBufferState)
    summary_memory: List[SummaryMemoryTurn] = field(default_factory=list)


class DemoAnalyzerPipeline:
    """Multi-mode real-time meeting screen analyzer with per-mode caches and transcript-aligned summary events."""

    def __init__(
        self,
        *,
        siglip_classifier: Optional[SiglipScreenClassifier] = None,
        stt_store: Optional[UtteranceStore] = None,
        omniparser: Optional[OmniParserAdapter] = None,
        cursor_analyzer: Optional[CursorMotionAnalyzer] = None,
        macro_layout_detector: Optional[MacroLayoutDetector] = None,
        ocr_extractor: Optional[OCRTextExtractor] = None,
        slide_analyzer: Optional[SuryaSlideAnalyzer] = None,
        mode_router: Optional[ModeRouter] = None,
        frame_summarizer: Optional[ResilientFrameSummarizer] = None,
        summary_memory_size: int = 8,
        code_interval_sec: float = 2.0,
        slide_interval_sec: float = 3.0,
        demo_interval_sec: float = 1.0,
        other_interval_sec: float = 5.0,
        code_change_tiny_ratio: float = 0.0004,
        slide_change_tiny_ratio: float = 0.015,
        demo_change_tiny_ratio: float = 0.0010,
        code_change_ssim_threshold: float = 0.9985,
        slide_change_ssim_threshold: float = 0.9920,
        demo_change_ssim_threshold: float = 0.9965,
        pixel_threshold: int = 18,
        change_padding_px: int = 10,
        macro_boundary_overlap: float = 0.05,
        macro_global_change_ratio: float = 0.35,
        transcript_window_sec: float = 8.0,
        cursor_mask_padding_px: int = 8,
        debug_black_cursor_mask: bool = False,
    ) -> None:
        self.siglip = siglip_classifier or SiglipScreenClassifier()
        self.stt_store = stt_store or UtteranceStore()
        self.omniparser = omniparser or OmniParserAdapter()
        self.cursor_analyzer = cursor_analyzer or CursorMotionAnalyzer()
        default_macro_weights = Path(__file__).resolve().parent / "vendor" / "uitag" / "yolo-ui.pt"
        self.macro_layout = macro_layout_detector or MacroLayoutDetector(str(default_macro_weights))
        self.ocr_extractor = ocr_extractor or OCRTextExtractor()
        self.slide_analyzer = slide_analyzer or SuryaSlideAnalyzer(self.ocr_extractor)
        self.router = mode_router or ModeRouter()
        self.frame_summarizer = frame_summarizer or ResilientFrameSummarizer(HeuristicFrameSummarizer())
        self.summary_memory_size = max(1, int(summary_memory_size))

        self.code_interval_sec = float(code_interval_sec)
        self.slide_interval_sec = float(slide_interval_sec)
        self.demo_interval_sec = float(demo_interval_sec)
        self.other_interval_sec = float(other_interval_sec)
        self.code_change_tiny_ratio = float(code_change_tiny_ratio)
        self.slide_change_tiny_ratio = float(slide_change_tiny_ratio)
        self.demo_change_tiny_ratio = float(demo_change_tiny_ratio)
        self.code_change_ssim_threshold = float(code_change_ssim_threshold)
        self.slide_change_ssim_threshold = float(slide_change_ssim_threshold)
        self.demo_change_ssim_threshold = float(demo_change_ssim_threshold)
        self.pixel_threshold = int(pixel_threshold)
        self.change_padding_px = int(change_padding_px)
        self.macro_boundary_overlap = float(macro_boundary_overlap)
        self.macro_global_change_ratio = float(macro_global_change_ratio)
        self.transcript_window_sec = float(transcript_window_sec)
        self.cursor_mask_padding_px = int(cursor_mask_padding_px)
        self.debug_black_cursor_mask = bool(debug_black_cursor_mask)

        self.cache = StateCache()
        self._frame_id = 0
        self._records: List[SceneStateRecord] = []
        self._last_record_by_mode: Dict[str, SceneStateRecord] = {}

    def ingest_utterance(self, *, start: float, end: float, speaker: str, text: str, source: str = "live") -> None:
        self.stt_store.add_utterance(start=start, end=end, speaker=speaker, text=text, source=source)
        self._sync_transcript_buffer()

    def ingest_utterances(self, utterances: Sequence[Dict[str, Any]]) -> None:
        self.stt_store.extend(utterances)
        self._sync_transcript_buffer()

    def ingest_frame(self, frame_bgr: Any, *, timestamp_sec: float) -> SceneStateRecord:
        started = perf_counter()
        stage_timings: Dict[str, float] = {}
        self._frame_id += 1
        frame_h, frame_w = frame_bgr.shape[:2]
        viewport = {"x": 0, "y": 0, "width": int(frame_w), "height": int(frame_h)}
        self._sync_transcript_buffer()

        previous_mode = self.cache.last_stable_mode
        route_started = perf_counter()
        route, mode_changed = self._route_mode(frame_bgr)
        stage_timings["route_ms"] = (perf_counter() - route_started) * 1000.0
        mode = route.mode
        self.cache.previous_mode = previous_mode
        self.cache.current_mode = mode
        self.cache.last_stable_mode = mode
        self.cache.mode_prediction_history = [item.to_dict() for item in self.router.state.history]

        cursor = CursorState()
        masked_frame: Optional[np.ndarray] = None
        if mode in {"code", "slide", "demo"}:
            cursor_started = perf_counter()
            prev_cursor = self.cache.cursor_state
            prev_masked = self._get_mode_cache(mode).prev_frame_masked
            if self._frame_shape_changed(prev_masked, frame_bgr):
                prev_masked = None
            cursor = self.cursor_analyzer.analyze(
                frame_bgr=frame_bgr,
                prev_frame_bgr=prev_masked,
                prev_cursor=prev_cursor,
                elements=self._candidate_elements_for_cursor(mode),
            )
            masked_frame = self._mask_cursor_for_change_detection(frame_bgr, cursor, prev_cursor)
            stage_timings["cursor_ms"] = (perf_counter() - cursor_started) * 1000.0
        else:
            stage_timings["cursor_ms"] = 0.0
        self.cache.cursor_state = cursor.copy()

        mode_started = perf_counter()
        if mode == "code":
            state = self._process_code_mode(frame_bgr, masked_frame, timestamp_sec, viewport, route, mode_changed)
        elif mode == "slide":
            state = self._process_slide_mode(frame_bgr, masked_frame, timestamp_sec, viewport, route, mode_changed)
        elif mode == "demo":
            state = self._process_demo_mode(frame_bgr, masked_frame, timestamp_sec, viewport, route, mode_changed)
        else:
            state = self._process_other_mode(timestamp_sec, viewport, route, mode_changed)
        stage_timings["mode_processor_ms"] = (perf_counter() - mode_started) * 1000.0

        summary_started = perf_counter()
        transcript_window = self._select_transcript_window(
            timestamp_sec=timestamp_sec,
            force=state.visual_change or mode_changed or state.mode == "other",
        )
        state.transcript_window = transcript_window
        state.utterance_ids = [int(seg.segment_id) for seg in transcript_window if seg.segment_id is not None]
        if transcript_window:
            state.utterance_time_span = (transcript_window[0].start_time, transcript_window[-1].end_time)
        state.summary = self._summarize_state(state, transcript_window, previous_mode=previous_mode, mode_changed=mode_changed)
        self._remember_summary(state, transcript_window)
        stage_timings["summary_ms"] = (perf_counter() - summary_started) * 1000.0

        elapsed_ms = (perf_counter() - started) * 1000.0
        stage_timings.update(dict(state.debug.get("stage_timings_ms") or {}))
        stage_timings["pipeline_total_ms"] = elapsed_ms
        state.debug = dict(state.debug)
        state.debug.update(
            {
                "mode_confidence": route.confidence,
                "cursor_detected": bool(cursor.visible),
                "change_score": float(state.debug.get("change_score", 0.0)),
                "processor_used": str(state.debug.get("processor_used", "none")),
                "rfdetr_ran": bool(state.debug.get("rfdetr_ran", False)),
                "omniparser_ran": bool(state.debug.get("omniparser_ran", False)),
                "processing_time_ms": elapsed_ms,
                "stage_timings_ms": stage_timings,
            }
        )
        state.confidence = dict(state.confidence)
        state.confidence["route"] = route.to_dict()
        if state.visual_change or mode_changed or state.mode == "other":
            self.cache.transcript_buffer.last_visual_event_time = float(timestamp_sec)
            self.cache.transcript_buffer.last_summary_time = float(timestamp_sec)

        record = export_scene_record(state)
        self._records.append(record)
        self._last_record_by_mode[mode] = record
        self._log_frame(state, transcript_window)
        return record

    def records(self) -> List[SceneStateRecord]:
        return list(self._records)

    def _process_code_mode(
        self,
        frame_bgr: np.ndarray,
        masked_frame: np.ndarray,
        timestamp_sec: float,
        viewport: Dict[str, int],
        route: ModeRouteDecision,
        mode_changed: bool,
    ) -> ScreenState:
        cache = self.cache.code
        resolution_changed = self._frame_shape_changed(cache.prev_frame_masked, masked_frame)
        if resolution_changed:
            cache.prev_frame_masked = None
            cache.prev_ocr_text = ""
            cache.prev_ocr_lines = []
            cache.last_visual_output = {}
            cache.last_state = None
            cache.last_processed_time = None
        change = ChangeDetectionResult(changed=True, change_ratio=1.0, ssim_score=0.0, changed_regions=[(0, 0, viewport["width"], viewport["height"])], change_type="initial")
        first_frame = mode_changed or cache.prev_frame_masked is None or cache.last_state is None
        interval_skip = (not first_frame) and cache.last_processed_time is not None and (timestamp_sec - cache.last_processed_time) < self.code_interval_sec
        if interval_skip:
            return self._reuse_cached_state(cache.last_state, timestamp_sec, route, self.cache.cursor_state, self._infer_theme(masked_frame), "code", "interval_skip", change)
        if not first_frame and not interval_skip:
            change = self._detect_change(
                cache.prev_frame_masked,
                masked_frame,
                tiny_ratio=self.code_change_tiny_ratio,
                ssim_keep_threshold=self.code_change_ssim_threshold,
            )
        if first_frame or change.changed:
            ocr_started = perf_counter()
            ocr = self._safe_extract_ocr(frame_bgr, "code")
            ocr_ms = (perf_counter() - ocr_started) * 1000.0
            ocr_diff = _diff_text(cache.prev_ocr_text, ocr.text)
            text_changed = bool(ocr_diff["added"] or ocr_diff["removed"])
            visual_change = first_frame or change.changed or text_changed
            changed_bbox = union_boxes(change.changed_regions) or (0, 0, viewport["width"], viewport["height"])
            elements = _elements_from_ocr_lines(ocr.lines, frame_id=self._frame_id, timestamp_sec=timestamp_sec, prefix="code")
            payload = {
                "mode": "code",
                "visual_change": visual_change,
                "changed_bbox": list(changed_bbox),
                "ocr_text": ocr.text,
                "ocr_diff": ocr_diff,
                "cursor": _cursor_payload(self.cache.cursor_state),
            }
            state = ScreenState(
                frame_id=self._frame_id,
                timestamp_sec=float(timestamp_sec),
                viewport=dict(viewport),
                frame_type="code",
                mode="code",
                route=route,
                theme=self._infer_theme(masked_frame),
                visual_change=visual_change,
                visual_payload=payload,
                elements=elements,
                overlays=[],
                layout_groups=[LayoutGroup(id="group_code_content", label="content", bbox=(0, 0, viewport["width"], viewport["height"]), source="ocr", frame_id=self._frame_id, timestamp=float(timestamp_sec))],
                cursor=self.cache.cursor_state.copy(),
                delta=StateDelta(
                    types=["initial_frame"] if first_frame else ["ocr_refresh" if visual_change else "cached_interval_skip"],
                    changed_regions=list(change.changed_regions),
                    reparsed_regions=[changed_bbox] if visual_change else [],
                    confidence=max(ocr.confidence, change.change_ratio),
                ),
                confidence={"ocr_provider": ocr.provider, "ocr_confidence": ocr.confidence},
                debug={
                    "change_score": change.change_ratio,
                    "processor_used": "ocr",
                    "rfdetr_ran": False,
                    "omniparser_ran": False,
                    "crop_bbox": list(changed_bbox),
                    "stage_timings_ms": {"ocr_ms": ocr_ms},
                },
            )
            if visual_change or first_frame:
                cache.prev_ocr_text = ocr.text
                cache.prev_ocr_lines = list(ocr.lines)
                cache.last_visual_output = dict(payload)
                cache.last_change_time = float(timestamp_sec)
            cache.last_state = state.clone()
            cache.last_processed_time = float(timestamp_sec)
            cache.prev_frame_masked = masked_frame.copy()
            return state
        return self._reuse_cached_state(cache.last_state, timestamp_sec, route, self.cache.cursor_state, self._infer_theme(masked_frame), "code", "cached_layout_reused", change)

    def _process_slide_mode(
        self,
        frame_bgr: np.ndarray,
        masked_frame: np.ndarray,
        timestamp_sec: float,
        viewport: Dict[str, int],
        route: ModeRouteDecision,
        mode_changed: bool,
    ) -> ScreenState:
        cache = self.cache.slide
        resolution_changed = self._frame_shape_changed(cache.prev_frame_masked, masked_frame)
        if resolution_changed:
            cache.prev_frame_masked = None
            cache.prev_slide_fingerprint = ""
            cache.prev_surya_layout = []
            cache.prev_surya_text = ""
            cache.last_visual_output = {}
            cache.last_state = None
            cache.last_processed_time = None
        change = ChangeDetectionResult(changed=True, change_ratio=1.0, ssim_score=0.0, changed_regions=[(0, 0, viewport["width"], viewport["height"])], change_type="initial")
        first_frame = mode_changed or cache.prev_frame_masked is None or cache.last_state is None
        interval_skip = (not first_frame) and cache.last_processed_time is not None and (timestamp_sec - cache.last_processed_time) < self.slide_interval_sec
        if interval_skip:
            return self._reuse_cached_state(cache.last_state, timestamp_sec, route, self.cache.cursor_state, self._infer_theme(masked_frame), "slide", "interval_skip", change)
        if not first_frame and not interval_skip:
            change = self._detect_change(
                cache.prev_frame_masked,
                masked_frame,
                tiny_ratio=self.slide_change_tiny_ratio,
                ssim_keep_threshold=self.slide_change_ssim_threshold,
            )
        if first_frame or change.changed:
            slide_started = perf_counter()
            slide = self._safe_analyze_slide(frame_bgr)
            slide_ms = (perf_counter() - slide_started) * 1000.0
            duplicate_slide = (not first_frame) and cache.prev_slide_fingerprint and cache.prev_slide_fingerprint == slide.fingerprint
            visual_change = first_frame or (change.changed and not duplicate_slide)
            payload = {
                "mode": "slide",
                "visual_change": visual_change,
                "slide_fingerprint": slide.fingerprint,
                "layout": [block.to_dict() for block in slide.layout],
                "full_text": slide.full_text,
                "cursor": _cursor_payload(self.cache.cursor_state),
            }
            elements = _elements_from_slide(slide, frame_id=self._frame_id, timestamp_sec=timestamp_sec)
            state = ScreenState(
                frame_id=self._frame_id,
                timestamp_sec=float(timestamp_sec),
                viewport=dict(viewport),
                frame_type="slide",
                mode="slide",
                route=route,
                theme=self._infer_theme(masked_frame),
                visual_change=visual_change,
                visual_payload=payload,
                elements=elements,
                overlays=[],
                layout_groups=[LayoutGroup(id="group_slide_content", label="content", bbox=(0, 0, viewport["width"], viewport["height"]), source=slide.provider, frame_id=self._frame_id, timestamp=float(timestamp_sec))],
                cursor=self.cache.cursor_state.copy(),
                delta=StateDelta(
                    types=["initial_frame"] if first_frame else ["surya_refresh" if visual_change else "duplicate_slide"],
                    changed_regions=list(change.changed_regions),
                    reparsed_regions=[(0, 0, viewport["width"], viewport["height"])] if visual_change else [],
                    confidence=max(slide.confidence, change.change_ratio),
                ),
                confidence={"slide_provider": slide.provider, "slide_confidence": slide.confidence},
                debug={
                    "change_score": change.change_ratio,
                    "processor_used": "surya",
                    "rfdetr_ran": False,
                    "omniparser_ran": False,
                    "duplicate_slide": duplicate_slide,
                    "stage_timings_ms": {"slide_analysis_ms": slide_ms},
                },
            )
            if visual_change or first_frame:
                cache.prev_slide_fingerprint = slide.fingerprint
                cache.prev_surya_layout = [block.to_dict() for block in slide.layout]
                cache.prev_surya_text = slide.full_text
                cache.last_visual_output = dict(payload)
                cache.last_change_time = float(timestamp_sec)
            cache.last_state = state.clone()
            cache.last_processed_time = float(timestamp_sec)
            cache.prev_frame_masked = masked_frame.copy()
            return state
        return self._reuse_cached_state(cache.last_state, timestamp_sec, route, self.cache.cursor_state, self._infer_theme(masked_frame), "slide", "cached_layout_reused", change)

    def _process_demo_mode(
        self,
        frame_bgr: np.ndarray,
        masked_frame: np.ndarray,
        timestamp_sec: float,
        viewport: Dict[str, int],
        route: ModeRouteDecision,
        mode_changed: bool,
    ) -> ScreenState:
        cache = self.cache.demo
        frame_h, frame_w = frame_bgr.shape[:2]
        resolution_changed = self._frame_shape_changed(cache.prev_frame_masked, masked_frame)
        if resolution_changed:
            cache.prev_frame_masked = None
            cache.prev_macro_layout = []
            cache.prev_micro_elements = []
            cache.prev_page_summary = ""
            cache.last_visual_output = {}
            cache.last_state = None
            cache.last_processed_time = None
        first_frame = mode_changed or cache.prev_frame_masked is None or cache.last_state is None
        interval_skip = (not first_frame) and cache.last_processed_time is not None and (timestamp_sec - cache.last_processed_time) < self.demo_interval_sec
        change = ChangeDetectionResult(changed=True, change_ratio=1.0, ssim_score=0.0, changed_regions=[(0, 0, frame_w, frame_h)], change_type="initial")
        reran_macro = False
        omniparser_ran = False
        stage_timings: Dict[str, float] = {}
        if interval_skip:
            return self._reuse_cached_state(cache.last_state, timestamp_sec, route, self.cache.cursor_state, self._infer_theme(masked_frame), "demo", "interval_skip", change)
        if not first_frame and not interval_skip:
            change = self._detect_change(
                cache.prev_frame_masked,
                masked_frame,
                tiny_ratio=self.demo_change_tiny_ratio,
                ssim_keep_threshold=self.demo_change_ssim_threshold,
            )
        if first_frame or change.changed:
            state = cache.last_state.clone() if cache.last_state is not None else ScreenState(
                frame_id=self._frame_id,
                timestamp_sec=float(timestamp_sec),
                viewport=dict(viewport),
                frame_type="demo",
                mode="demo",
                route=route,
            )
            state.frame_id = self._frame_id
            state.timestamp_sec = float(timestamp_sec)
            state.viewport = dict(viewport)
            state.mode = "demo"
            state.frame_type = "demo"
            state.route = route
            state.theme = self._infer_theme(masked_frame)
            state.cursor = self.cache.cursor_state.copy()

            macro_layout = [group.copy() for group in cache.prev_macro_layout]
            if first_frame or self._should_rerun_macro(change.changed_regions, viewport, macro_layout):
                macro_started = perf_counter()
                macro_layout = self._detect_macro_layout(frame_bgr, timestamp_sec)
                stage_timings["macro_layout_ms"] = (perf_counter() - macro_started) * 1000.0
                reran_macro = True
            else:
                stage_timings["macro_layout_ms"] = 0.0
            macro_layout = self._augment_macro_layout_from_micro(macro_layout, cache.prev_micro_elements, viewport)
            state.layout_groups = [group.copy() for group in macro_layout]

            crop_plan = self._plan_demo_crops(change.changed_regions, macro_layout, cache.prev_micro_elements, viewport, first_frame)
            reparsed_regions = [tuple(plan["crop_bbox"]) for plan in crop_plan]
            requires_full_parse = first_frame or any(bool(item.get("requires_full_parse")) for item in crop_plan)
            if requires_full_parse:
                reparsed_regions = [(0, 0, frame_w, frame_h)]
            parse_started = perf_counter()
            parsed = self._parse_micro_regions(frame_bgr, reparsed_regions)
            stage_timings["omniparser_ms"] = (perf_counter() - parse_started) * 1000.0
            omniparser_ran = True

            for element in parsed.elements:
                element.parent_macro_region_id = _parent_group_id_for_box(element.bbox, macro_layout)
                element.timestamp = float(timestamp_sec)
            merged_state = merge_reparsed_state(
                state=state,
                reparsed_elements=parsed.elements,
                reparsed_overlays=parsed.overlays,
                reparsed_regions=reparsed_regions,
                frame_id=self._frame_id,
            )
            merged_state.layout_groups = self._augment_macro_layout_from_micro(merged_state.layout_groups, merged_state.elements, viewport)
            merged_state.route = route
            merged_state.mode = "demo"
            merged_state.frame_type = "demo"
            merged_state.cursor = self.cache.cursor_state.copy()
            cursor_target = self._match_cursor_target(self.cache.cursor_state, merged_state.elements, merged_state.layout_groups)
            page_summary = self._build_demo_page_summary(merged_state.layout_groups, merged_state.elements, merged_state.overlays, cursor_target)
            visual_change = first_frame or change.changed
            visual_payload = {
                "mode": "demo",
                "visual_change": visual_change,
                "change_type": change.change_type,
                "macro_layout": [_serialize_layout_group(group) for group in merged_state.layout_groups],
                "micro_elements": [_serialize_micro_element(item) for item in merged_state.elements],
                "cursor": _cursor_payload(self.cache.cursor_state),
                "cursor_target": cursor_target,
                "updated_regions": [item.get("parent_macro_region_id") for item in crop_plan if item.get("parent_macro_region_id")],
                "unchanged_regions": [group.id for group in merged_state.layout_groups if group.id not in {item.get("parent_macro_region_id") for item in crop_plan if item.get("parent_macro_region_id")}],
                "page_summary": page_summary,
            }
            merged_state.visual_change = visual_change
            merged_state.visual_payload = visual_payload
            merged_state.summary = page_summary
            merged_state.confidence = {
                "parser_provider": parsed.provider,
                "parser_confidence": parsed.confidence,
                "ssim_score": change.ssim_score,
                "changed_pixel_ratio": change.change_ratio,
                "route": route.to_dict(),
            }
            merged_state.debug = {
                "change_score": change.change_ratio,
                "processor_used": "webui_pipeline",
                "rfdetr_ran": reran_macro,
                "omniparser_ran": omniparser_ran,
                "crop_bbox": [list(item) for item in reparsed_regions],
                "change_type": change.change_type,
                "stage_timings_ms": stage_timings,
            }
            merged_state.delta = StateDelta(
                types=["initial_frame"] if first_frame else ["macro_refresh" if reran_macro else "micro_refresh"],
                changed_regions=list(change.changed_regions),
                reparsed_regions=list(reparsed_regions),
                confidence=max(parsed.confidence, change.change_ratio),
            )
            merged_state.last_full_parse_frame_id = self._frame_id if reran_macro or first_frame else merged_state.last_full_parse_frame_id
            cache.prev_macro_layout = [group.copy() for group in merged_state.layout_groups]
            cache.prev_micro_elements = [item.copy() for item in merged_state.elements]
            cache.prev_page_summary = page_summary
            cache.last_visual_output = dict(visual_payload)
            cache.last_change_time = float(timestamp_sec)
            cache.last_state = merged_state.clone()
            cache.last_processed_time = float(timestamp_sec)
            cache.prev_frame_masked = masked_frame.copy()
            return merged_state
        return self._reuse_cached_state(cache.last_state, timestamp_sec, route, self.cache.cursor_state, self._infer_theme(masked_frame), "demo", "cached_layout_reused", change)

    def _process_other_mode(
        self,
        timestamp_sec: float,
        viewport: Dict[str, int],
        route: ModeRouteDecision,
        mode_changed: bool,
    ) -> ScreenState:
        cache = self.cache.other
        should_emit = mode_changed or cache.last_processed_time is None or (timestamp_sec - float(cache.last_processed_time)) >= self.other_interval_sec
        transcript_window = self.stt_store.window(
            event_time=float(timestamp_sec),
            previous_event_time=self.cache.transcript_buffer.last_summary_time,
            lookback_sec=self.transcript_window_sec,
        ) if should_emit else []
        summary = (
            "No meaningful presented screen content detected. Transcript is being captured."
            if not transcript_window
            else f"No meaningful presented screen content detected while the speaker discussed: {transcript_window[-1].text}"
        )
        state = ScreenState(
            frame_id=self._frame_id,
            timestamp_sec=float(timestamp_sec),
            viewport=dict(viewport),
            frame_type="other",
            mode="other",
            route=route,
            theme="unknown",
            visual_change=False,
            visual_payload={
                "mode": "other",
                "visual_change": False,
                "summary": summary,
            },
            cursor=self.cache.cursor_state.copy(),
            transcript_window=transcript_window,
            summary=summary,
            delta=StateDelta(types=["transcript_only"] if should_emit else ["other_idle"], confidence=1.0),
            confidence={"route": route.to_dict()},
            debug={
                "change_score": 0.0,
                "processor_used": "none",
                "rfdetr_ran": False,
                "omniparser_ran": False,
            },
        )
        cache.last_transcript_only_summary = summary
        cache.last_seen_time = float(timestamp_sec)
        cache.last_processed_time = float(timestamp_sec)
        cache.last_state = state.clone()
        return state

    def _route_mode(self, frame_bgr: np.ndarray) -> Tuple[ModeRouteDecision, bool]:
        try:
            raw_label, raw_scores = self.siglip.classify(frame_bgr)
        except Exception:
            raw_label, raw_scores = "other", {"other": 1.0}
        raw_label, raw_scores = self._apply_route_guards(frame_bgr, raw_label, raw_scores)
        demo_guard_active = self.cache.last_stable_mode == "demo" and bool(self.cache.demo.prev_macro_layout)
        return self.router.route(raw_label=raw_label, raw_scores=raw_scores, demo_guard_active=demo_guard_active)

    def _apply_route_guards(
        self,
        frame_bgr: np.ndarray,
        raw_label: str,
        raw_scores: Dict[str, float],
    ) -> Tuple[str, Dict[str, float]]:
        label_scores = {
            "slide": float(raw_scores.get("slide", raw_scores.get("slides", 0.0))),
            "code": float(raw_scores.get("code", 0.0)),
            "demo": float(raw_scores.get("demo", 0.0)),
            "other": float(raw_scores.get("other", raw_scores.get("none", 0.0))),
        }
        ordered = sorted(label_scores.items(), key=lambda item: item[1], reverse=True)
        top_score = float(ordered[0][1]) if ordered else 0.0
        second_score = float(ordered[1][1]) if len(ordered) > 1 else top_score
        margin = top_score - second_score

        if self._looks_like_other_frame(frame_bgr):
            return "other", {"other": 1.0}

        if top_score >= 0.15 and margin >= 0.02:
            return raw_label, raw_scores

        demo_score = float(label_scores["demo"])
        code_score = float(label_scores["code"])
        slide_score = float(label_scores["slide"])
        other_score = float(label_scores["other"])

        if raw_label == "demo" and demo_score >= max(code_score, slide_score, other_score) - 0.03:
            return "demo", {"demo": 1.0}

        if self.cache.last_stable_mode == "demo" and demo_score >= max(code_score, slide_score) - 0.04:
            return "demo", {"demo": 1.0}

        if demo_score >= 0.08 and demo_score >= max(code_score, slide_score):
            return "demo", {"demo": 1.0}

        return raw_label, raw_scores

    def _looks_like_other_frame(self, frame_bgr: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        dark_ratio = float((gray < 30).mean())
        bright_ratio = float((gray > 60).mean())
        edge_mean = float(cv2.Canny(gray, 50, 150).mean())
        return dark_ratio >= 0.92 and bright_ratio <= 0.08 and edge_mean <= 1.0

    def _detect_change(
        self,
        prev_masked: np.ndarray,
        current_masked: np.ndarray,
        *,
        tiny_ratio: float,
        ssim_keep_threshold: float,
    ) -> ChangeDetectionResult:
        prev_gray = cv2.cvtColor(prev_masked, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(current_masked, cv2.COLOR_BGR2GRAY)
        ssim_score = float(structural_similarity(prev_gray, curr_gray))
        diff = cv2.absdiff(prev_gray, curr_gray)
        _, thresholded = cv2.threshold(diff, self.pixel_threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), dtype=np.uint8)
        thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=1)
        thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel, iterations=2)
        changed_ratio = float(np.count_nonzero(thresholded)) / float(thresholded.size or 1)
        if changed_ratio < tiny_ratio and ssim_score >= ssim_keep_threshold:
            return ChangeDetectionResult(changed=False, change_ratio=changed_ratio, ssim_score=ssim_score, change_type="none")
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions: List[BBox] = []
        frame_h, frame_w = curr_gray.shape[:2]
        min_area = max(16, int(frame_h * frame_w * 0.00035))
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (w * h) < min_area:
                continue
            regions.append(clamp_bbox((x, y, x + w, y + h), frame_w, frame_h))
        if not regions:
            regions = [(0, 0, frame_w, frame_h)]
        change_type = self._classify_change_type(regions, frame_w, frame_h)
        return ChangeDetectionResult(
            changed=True,
            change_ratio=changed_ratio,
            ssim_score=ssim_score,
            changed_regions=regions,
            change_type=change_type,
        )

    @staticmethod
    def _frame_shape_changed(prev_frame: Optional[np.ndarray], current_frame: np.ndarray) -> bool:
        if prev_frame is None:
            return False
        return tuple(prev_frame.shape[:2]) != tuple(current_frame.shape[:2])

    def _mask_cursor_for_change_detection(
        self,
        frame_bgr: np.ndarray,
        cursor: CursorState,
        prev_cursor: CursorState,
    ) -> np.ndarray:
        masked = frame_bgr.copy()
        boxes: List[BBox] = []
        if cursor.visible and cursor.bbox is not None:
            boxes.append(cursor.bbox)
        if prev_cursor.visible and prev_cursor.bbox is not None:
            boxes.append(prev_cursor.bbox)
        for box in boxes:
            x1, y1, x2, y2 = clamp_bbox(
                (
                    box[0] - self.cursor_mask_padding_px,
                    box[1] - self.cursor_mask_padding_px,
                    box[2] + self.cursor_mask_padding_px,
                    box[3] + self.cursor_mask_padding_px,
                ),
                masked.shape[1],
                masked.shape[0],
            )
            roi = masked[max(0, y1 - 4) : min(masked.shape[0], y2 + 4), max(0, x1 - 4) : min(masked.shape[1], x2 + 4)]
            if roi.size == 0:
                continue
            if self.debug_black_cursor_mask:
                cv2.rectangle(masked, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
                continue
            mask = np.zeros(masked.shape[:2], dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            try:
                masked = cv2.inpaint(masked, mask, 3, cv2.INPAINT_TELEA)
            except Exception:
                fill = np.median(roi.reshape(-1, roi.shape[-1]), axis=0) if roi.ndim == 3 else np.array([0, 0, 0])
                masked[y1:y2, x1:x2] = fill
        return masked

    def _should_rerun_macro(
        self,
        changed_regions: Sequence[BBox],
        viewport: Dict[str, int],
        macro_layout: Sequence[LayoutGroup],
    ) -> bool:
        if not macro_layout:
            return True
        frame_area = max(1, int(viewport["width"]) * int(viewport["height"]))
        changed_area = sum(bbox_area(region) for region in changed_regions)
        if (changed_area / frame_area) >= self.macro_global_change_ratio:
            return True
        return False

    def _plan_demo_crops(
        self,
        changed_regions: Sequence[BBox],
        macro_layout: Sequence[LayoutGroup],
        existing_elements: Sequence[UIElement],
        viewport: Dict[str, int],
        first_frame: bool,
    ) -> List[Dict[str, Any]]:
        if first_frame:
            return [
                {
                    "crop_id": f"crop_{self._frame_id:05d}_full",
                    "crop_bbox": [0, 0, int(viewport["width"]), int(viewport["height"])],
                    "reason": "first_frame_full_parse",
                    "parent_macro_region_id": None,
                    "requires_full_parse": True,
                    "padding_used": 0,
                }
            ]
        plans: List[Dict[str, Any]] = []
        for idx, region in enumerate(changed_regions, start=1):
            expanded = self._expand_region(region, viewport["width"], viewport["height"])
            parent = _best_parent_group(expanded, macro_layout)
            reason = "small_region_change"
            crop_bbox = list(expanded)
            if parent is not None:
                parent_type = str(parent.label or "")
                if parent_type in {"header", "left_sidebar", "footer"}:
                    crop_bbox = list(parent.bbox)
                    reason = f"{parent_type}_region_change"
                elif parent_type == "content":
                    enclosing_elements = [item for item in existing_elements if bbox_intersects(item.bbox, expanded, padding=4)]
                    if enclosing_elements:
                        snapped = union_boxes([item.bbox for item in enclosing_elements])
                        if snapped is not None:
                            crop_bbox = list(self._expand_region(snapped, viewport["width"], viewport["height"]))
                            reason = "snap_to_parent_content"
            plans.append(
                {
                    "crop_id": f"crop_{self._frame_id:05d}_{idx:02d}",
                    "crop_bbox": crop_bbox,
                    "reason": reason,
                    "parent_macro_region_id": parent.id if parent is not None else None,
                    "requires_full_parse": False,
                    "padding_used": self.change_padding_px,
                }
            )
        if not plans:
            plans.append(
                {
                    "crop_id": f"crop_{self._frame_id:05d}_fallback",
                    "crop_bbox": [0, 0, int(viewport["width"]), int(viewport["height"])],
                    "reason": "fallback_full_parse",
                    "parent_macro_region_id": None,
                    "requires_full_parse": True,
                    "padding_used": 0,
                }
            )
        return plans

    def _detect_macro_layout(self, frame_masked: np.ndarray, timestamp_sec: float) -> List[LayoutGroup]:
        try:
            groups = self.macro_layout.detect(frame_masked)
        except Exception:
            frame_h, frame_w = frame_masked.shape[:2]
            groups = [LayoutGroup(id="group_main_content", label="main_content", bbox=(0, 0, frame_w, frame_h))]
        for group in groups:
            group.frame_id = self._frame_id
            group.timestamp = float(timestamp_sec)
        return groups

    def _augment_macro_layout_from_micro(
        self,
        macro_layout: Sequence[LayoutGroup],
        elements: Sequence[UIElement],
        viewport: Dict[str, int],
    ) -> List[LayoutGroup]:
        groups = [group.copy() for group in macro_layout]
        if not elements:
            return groups
        labels = {group.label for group in groups}
        inferred_left = self._infer_sidebar_group(elements, viewport, side="left")
        if inferred_left is not None and "left_sidebar" not in labels:
            groups.append(inferred_left)
            groups = self._clip_main_content_for_sidebar(groups, inferred_left, viewport)
        groups.sort(key=lambda item: (item.bbox[1], item.bbox[0], item.id))
        return groups

    def _clip_main_content_for_sidebar(
        self,
        groups: Sequence[LayoutGroup],
        sidebar: LayoutGroup,
        viewport: Dict[str, int],
    ) -> List[LayoutGroup]:
        adjusted: List[LayoutGroup] = []
        sx1, sy1, sx2, sy2 = sidebar.bbox
        frame_w = int(viewport["width"])
        for group in groups:
            item = group.copy()
            if item.id == sidebar.id or item.label not in {"main_content", "content"}:
                adjusted.append(item)
                continue
            x1, y1, x2, y2 = item.bbox
            overlap_h = max(0, min(y2, sy2) - max(y1, sy1))
            if x1 < sx2 < x2 and overlap_h >= max(1, min(y2 - y1, sy2 - sy1) * 0.45):
                clipped = (sx2, y1, x2, y2)
                if clipped[2] - clipped[0] >= max(40, int(frame_w * 0.25)):
                    item.bbox = clipped
                    item.score = max(float(item.score or 0.0), float(sidebar.score or 0.0))
                    if item.source == "fallback":
                        item.source = "micro_inferred_coverage"
            adjusted.append(item)
        return adjusted

    def _infer_sidebar_group(
        self,
        elements: Sequence[UIElement],
        viewport: Dict[str, int],
        *,
        side: str,
    ) -> Optional[LayoutGroup]:
        frame_w = int(viewport["width"])
        frame_h = int(viewport["height"])
        if side != "left":
            return None
        candidates = [
            item
            for item in elements
            if item.visible
            and not item.occluded
            and item.bbox[0] <= frame_w * 0.12
            and item.bbox[2] <= frame_w * 0.36
            and (item.bbox[2] - item.bbox[0]) <= max(80, int(frame_w * 0.30))
        ]
        label = "left_sidebar"
        if len(candidates) < 6:
            return None
        union = union_boxes([item.bbox for item in candidates])
        if union is None:
            return None
        x1, y1, x2, y2 = union
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        if x1 > frame_w * 0.08:
            return None
        if x2 > frame_w * 0.38 or width > frame_w * 0.34:
            return None
        if height < frame_h * 0.26 or width < frame_w * 0.08:
            return None
        centers_x = sorted((item.bbox[0] + item.bbox[2]) / 2.0 for item in candidates)
        centers_y = sorted((item.bbox[1] + item.bbox[3]) / 2.0 for item in candidates)
        p10 = centers_x[max(0, int(len(centers_x) * 0.10) - 1)]
        p90 = centers_x[min(len(centers_x) - 1, int(len(centers_x) * 0.90))]
        if (p90 - p10) > max(72, frame_w * 0.18):
            return None
        vertical_bands = {int(cy / max(1, frame_h * 0.12)) for cy in centers_y}
        if len(vertical_bands) < 3:
            return None
        score = min(0.72, 0.46 + min(0.12, len(candidates) * 0.01) + min(0.14, (height / max(1, frame_h)) * 0.20))
        return LayoutGroup(
            id=f"group_{label}_synthetic",
            label=label,
            bbox=(x1, y1, x2, y2),
            axis="xy",
            score=score,
            source="micro_inferred",
            frame_id=self._frame_id,
            timestamp=0.0,
            stable=False,
        )

    def _parse_micro_regions(self, frame_masked: np.ndarray, regions: Sequence[BBox]) -> ParsePayload:
        return self.omniparser.parse_regions(frame_masked, regions)

    def _safe_extract_ocr(self, frame_bgr: np.ndarray, mode: str) -> OCRResult:
        try:
            return self.ocr_extractor.extract_full(frame_bgr, frame_type=mode)
        except Exception as exc:
            logger.warning("OCR failed for mode=%s frame=%s: %s", mode, self._frame_id, exc)
            return OCRResult(text="", lines=[], provider="ocr_failed", confidence=0.0)

    def _ocr_elements_for_regions(
        self,
        *,
        ocr_result: OCRResult,
        regions: Sequence[BBox],
        frame_id: int,
        timestamp_sec: float,
    ) -> List[UIElement]:
        relevant_lines = _ocr_lines_for_regions(ocr_result.lines, regions)
        return _elements_from_ocr_lines(
            relevant_lines,
            frame_id=frame_id,
            timestamp_sec=timestamp_sec,
            prefix="demo_ocr",
        )

    def _safe_analyze_slide(self, frame_bgr: np.ndarray) -> SlideAnalysis:
        try:
            return self.slide_analyzer.analyze(frame_bgr)
        except Exception as exc:
            logger.warning("Slide analysis failed for frame=%s: %s", self._frame_id, exc)
            fallback_ocr = self._safe_extract_ocr(frame_bgr, "slide")
            return build_slide_from_ocr_result(
                fallback_ocr,
                width=frame_bgr.shape[1],
                height=frame_bgr.shape[0],
                provider="surya_failed",
            )

    def _match_cursor_target(
        self,
        cursor: CursorState,
        elements: Sequence[UIElement],
        layout_groups: Sequence[LayoutGroup],
    ) -> Dict[str, Any]:
        if not cursor.visible or cursor.hotspot is None:
            return {
                "element_id": None,
                "element_type": None,
                "element_text": None,
                "parent_macro_region_id": None,
                "parent_macro_region_type": None,
                "relation": "none",
                "confidence": 0.0,
            }
        candidates: List[Tuple[int, int, float, UIElement]] = []
        for element in elements:
            if bbox_contains_point(element.bbox, cursor.hotspot, padding=2):
                area = bbox_area(element.bbox)
                candidates.append((0 if element.interactive else 1, area, -element.confidence, element))
        if not candidates:
            parent = _best_parent_group((cursor.hotspot[0], cursor.hotspot[1], cursor.hotspot[0] + 1, cursor.hotspot[1] + 1), layout_groups)
            return {
                "element_id": None,
                "element_type": None,
                "element_text": None,
                "parent_macro_region_id": parent.id if parent is not None else None,
                "parent_macro_region_type": parent.label if parent is not None else None,
                "relation": "inside_region" if parent is not None else "none",
                "confidence": 0.25 if parent is not None else 0.0,
            }
        candidates.sort(key=lambda item: item[:3])
        selected = candidates[0][3]
        parent = _best_parent_group(selected.bbox, layout_groups)
        return {
            "element_id": selected.id,
            "element_type": selected.type,
            "element_text": selected.text or selected.caption,
            "parent_macro_region_id": parent.id if parent is not None else selected.parent_macro_region_id,
            "parent_macro_region_type": parent.label if parent is not None else None,
            "relation": "hovering_over",
            "confidence": max(0.4, selected.confidence),
        }

    def _build_demo_page_summary(
        self,
        layout_groups: Sequence[LayoutGroup],
        elements: Sequence[UIElement],
        overlays: Sequence[Any],
        cursor_target: Dict[str, Any],
    ) -> str:
        region_labels = [
            group.label.replace("_", " ")
            for group in layout_groups
            if str(group.label or "").strip()
        ]
        region_desc = ", ".join(dict.fromkeys(region_labels[:4])) if region_labels else "an application interface"
        highlight_candidates = []
        for item in elements:
            if not item.interactive and not (item.text or item.caption):
                continue
            highlight_candidates.append(
                {
                    "text": item.text,
                    "caption": item.caption,
                    "type": item.type,
                }
            )
        highlights = ", ".join(_collect_micro_highlights(highlight_candidates, limit=5))
        summary = f"The screen shows {region_desc}."
        if overlays:
            summary += f" An overlay is visible ({overlays[0].type})."
        if highlights:
            summary += f" Visible controls include {highlights}."
        if cursor_target.get("element_text"):
            summary += f" The cursor is hovering near {cursor_target['element_text']}."
        elif cursor_target.get("parent_macro_region_type"):
            summary += f" The cursor is inside the {cursor_target['parent_macro_region_type'].replace('_', ' ')}."
        return summary.strip()

    def _select_transcript_window(self, *, timestamp_sec: float, force: bool) -> List[TranscriptSegment]:
        if not force:
            return []
        segments = self.stt_store.window(
            event_time=float(timestamp_sec),
            previous_event_time=self.cache.transcript_buffer.last_visual_event_time,
            lookback_sec=self.transcript_window_sec,
        )
        return segments

    def _summarize_state(
        self,
        state: ScreenState,
        transcript_window: Sequence[TranscriptSegment],
        *,
        previous_mode: str,
        mode_changed: bool,
    ) -> str:
        transcript_text = " ".join(segment.text for segment in transcript_window if segment.text).strip()
        if state.mode == "code":
            ocr_text = str(state.visual_payload.get("ocr_text", "") or "").strip()
            ocr_diff = state.visual_payload.get("ocr_diff", {}) or {}
            diff_bits = []
            if ocr_diff.get("added"):
                diff_bits.append(f"added text: {ocr_diff['added']}")
            if ocr_diff.get("removed"):
                diff_bits.append(f"removed text: {ocr_diff['removed']}")
            visual_part = "The presenter is showing code."
            if diff_bits:
                visual_part += " The visible code changed with " + "; ".join(diff_bits) + "."
            elif ocr_text:
                visual_part += f" Visible text includes: {ocr_text[:220]}."
            heuristic = _join_summary_parts([_mode_transition_text(previous_mode, state.mode, mode_changed), visual_part, transcript_text and f"They are saying: {transcript_text}"])
            return self._llm_summary(
                state=state,
                heuristic_summary=heuristic,
                transcript_text=transcript_text,
                cursor_text=self._cursor_summary_text(state.cursor, state.visual_payload.get("cursor_target")),
                page_text=ocr_text[:400],
                action_text=self._action_summary_text(state),
            )
        if state.mode == "slide":
            layout = state.visual_payload.get("layout", []) or []
            title = ""
            if layout:
                title = str(layout[0].get("text", "") or "").strip()
            visual_part = "The presenter is showing a slide."
            if title:
                visual_part += f" The slide title appears to be '{title}'."
            if transcript_text:
                visual_part += f" They are discussing: {transcript_text}"
            heuristic = _join_summary_parts([_mode_transition_text(previous_mode, state.mode, mode_changed), visual_part])
            return self._llm_summary(
                state=state,
                heuristic_summary=heuristic,
                transcript_text=transcript_text,
                cursor_text=self._cursor_summary_text(state.cursor, state.visual_payload.get("cursor_target")),
                page_text=str(state.visual_payload.get("full_text", "") or "")[:500],
                action_text=self._action_summary_text(state),
            )
        if state.mode == "demo":
            page_summary = str(state.visual_payload.get("page_summary", "") or "").strip()
            change_type = str(state.visual_payload.get("change_type", "") or "").strip()
            visual_part = page_summary or "The presenter is demonstrating a web UI."
            if change_type and change_type != "none":
                visual_part += f" The detected UI change type is {change_type.replace('_', ' ')}."
            if transcript_text:
                visual_part += f" They are explaining: {transcript_text}"
            heuristic = _join_summary_parts([_mode_transition_text(previous_mode, state.mode, mode_changed), visual_part])
            return self._llm_summary(
                state=state,
                heuristic_summary=heuristic,
                transcript_text=transcript_text,
                cursor_text=self._cursor_summary_text(state.cursor, state.visual_payload.get("cursor_target")),
                page_text=str(state.visual_payload.get("page_summary", "") or "")[:500],
                action_text=self._action_summary_text(state),
            )
        heuristic = _join_summary_parts([
            _mode_transition_text(previous_mode, state.mode, mode_changed),
            str(state.visual_payload.get("summary", "") or "No meaningful presented screen content is visible."),
            transcript_text and f"The speaker said: {transcript_text}",
        ])
        return self._llm_summary(
            state=state,
            heuristic_summary=heuristic,
            transcript_text=transcript_text,
            cursor_text=self._cursor_summary_text(state.cursor, state.visual_payload.get("cursor_target")),
            page_text=str(state.visual_payload.get("summary", "") or "")[:300],
            action_text=self._action_summary_text(state),
        )

    def _llm_summary(
        self,
        *,
        state: ScreenState,
        heuristic_summary: str,
        transcript_text: str,
        cursor_text: str,
        page_text: str,
        action_text: str,
    ) -> str:
        request = FrameSummaryRequest(
            timestamp_sec=state.timestamp_sec,
            mode=state.mode,
            heuristic_summary=heuristic_summary,
            transcript_text=transcript_text,
            cursor_text=cursor_text,
            page_text=page_text,
            action_text=action_text,
            current_payload=dict(state.visual_payload),
            recent_memory=list(self.cache.summary_memory[-self.summary_memory_size :]),
        )
        summary = self.frame_summarizer.summarize(request)
        summary_backend = getattr(self.frame_summarizer, "primary", self.frame_summarizer)
        state.debug = dict(state.debug)
        state.debug["summary_provider"] = getattr(summary_backend, "provider_name", "heuristic")
        state.debug["summary_model"] = getattr(summary_backend, "model_name", "heuristic")
        return summary

    def _remember_summary(self, state: ScreenState, transcript_window: Sequence[TranscriptSegment]) -> None:
        transcript_text = " ".join(segment.text for segment in transcript_window if segment.text).strip()
        turn = SummaryMemoryTurn(
            timestamp_sec=state.timestamp_sec,
            mode=state.mode,
            summary=state.summary,
            transcript_text=transcript_text,
            cursor_text=self._cursor_summary_text(state.cursor, state.visual_payload.get("cursor_target")),
            page_text=self._memory_page_text(state),
            action_text=self._action_summary_text(state),
        )
        self.cache.summary_memory = [*self.cache.summary_memory[-(self.summary_memory_size - 1) :], turn]

    def _memory_page_text(self, state: ScreenState) -> str:
        if state.mode == "code":
            return str(state.visual_payload.get("ocr_text", "") or "")[:300]
        if state.mode == "slide":
            return str(state.visual_payload.get("full_text", "") or "")[:300]
        if state.mode == "demo":
            return str(state.visual_payload.get("page_summary", "") or "")[:300]
        return str(state.visual_payload.get("summary", "") or "")[:200]

    def _action_summary_text(self, state: ScreenState) -> str:
        if state.mode == "demo":
            change_type = str(state.visual_payload.get("change_type", "") or "").strip()
            cursor_target = state.visual_payload.get("cursor_target", {}) or {}
            target = str(cursor_target.get("element_text") or cursor_target.get("parent_macro_region_type") or "").strip()
            parts = []
            if change_type and change_type != "none":
                parts.append(f"ui change: {change_type.replace('_', ' ')}")
            if target:
                parts.append(f"cursor target: {target}")
            return "; ".join(parts)
        if state.mode == "code":
            diff = state.visual_payload.get("ocr_diff", {}) or {}
            parts = []
            if diff.get("added"):
                parts.append(f"added code text: {str(diff['added'])[:120]}")
            if diff.get("removed"):
                parts.append(f"removed code text: {str(diff['removed'])[:120]}")
            return "; ".join(parts)
        if state.mode == "slide":
            layout = state.visual_payload.get("layout", []) or []
            title = str(layout[0].get("text", "") or "").strip() if layout else ""
            return f"slide title: {title}" if title else ""
        return ""

    def _cursor_summary_text(self, cursor: CursorState, cursor_target: Optional[Dict[str, Any]]) -> str:
        if not cursor.visible:
            return "cursor not visible"
        target = cursor_target or {}
        target_text = str(target.get("element_text") or target.get("parent_macro_region_type") or "").strip()
        if _looks_like_noise(target_text):
            target_text = str(target.get("parent_macro_region_type") or "").replace("_", " ").strip()
        bits = [f"cursor {cursor.motion_type or 'present'}", f"type={cursor.cursor_type}"]
        if target_text:
            bits.append(f"over {target_text}")
        return ", ".join(bits)

    def _reuse_cached_state(
        self,
        last_state: Optional[ScreenState],
        timestamp_sec: float,
        route: ModeRouteDecision,
        cursor: CursorState,
        theme: str,
        mode: str,
        delta_type: str,
        change: ChangeDetectionResult,
    ) -> ScreenState:
        if last_state is None:
            return ScreenState(
                frame_id=self._frame_id,
                timestamp_sec=float(timestamp_sec),
                viewport={"x": 0, "y": 0, "width": 0, "height": 0},
                frame_type=mode,
                mode=mode,
                route=route,
                theme=theme,
                cursor=cursor.copy(),
                visual_change=False,
                delta=StateDelta(types=[delta_type], confidence=1.0 - change.change_ratio),
                confidence={"route": route.to_dict()},
                debug={"change_score": change.change_ratio, "processor_used": "none"},
            )
        state = last_state.clone()
        state.frame_id = self._frame_id
        state.timestamp_sec = float(timestamp_sec)
        state.route = route
        state.frame_type = mode
        state.mode = mode
        state.cursor = cursor.copy()
        state.theme = theme
        state.visual_change = False
        state.transcript_window = []
        state.utterance_ids = []
        state.utterance_time_span = None
        state.delta = StateDelta(types=[delta_type], changed_regions=[], reparsed_regions=[], confidence=max(0.0, 1.0 - change.change_ratio))
        state.debug = dict(state.debug)
        state.debug.update(
            {
                "change_score": change.change_ratio,
                "processor_used": "none",
                "rfdetr_ran": False,
                "omniparser_ran": False,
                "reused_cache": True,
            }
        )
        state.confidence = dict(state.confidence)
        state.confidence["route"] = route.to_dict()
        return state

    def _expand_region(self, region: BBox, width: int, height: int) -> BBox:
        x1, y1, x2, y2 = region
        return clamp_bbox(
            (
                x1 - self.change_padding_px,
                y1 - self.change_padding_px,
                x2 + self.change_padding_px,
                y2 + self.change_padding_px,
            ),
            width,
            height,
        )

    def _candidate_elements_for_cursor(self, mode: str) -> List[UIElement]:
        if mode == "demo":
            return [item.copy() for item in self.cache.demo.prev_micro_elements]
        if mode == "code" and self.cache.code.last_state is not None:
            return [item.copy() for item in self.cache.code.last_state.elements]
        if mode == "slide" and self.cache.slide.last_state is not None:
            return [item.copy() for item in self.cache.slide.last_state.elements]
        return []

    def _get_mode_cache(self, mode: str) -> Any:
        return getattr(self.cache, mode)

    def _sync_transcript_buffer(self) -> None:
        self.cache.transcript_buffer.finalized_segments = self.stt_store.snapshot()
        self.cache.transcript_buffer.interim_segments = self.stt_store.snapshot_interim()

    def _infer_theme(self, frame_masked: np.ndarray) -> str:
        gray = cv2.cvtColor(frame_masked, cv2.COLOR_BGR2GRAY)
        return "dark" if float(np.mean(gray)) < 110.0 else "light"

    def _classify_change_type(self, regions: Sequence[BBox], frame_w: int, frame_h: int) -> str:
        total_area = sum(bbox_area(region) for region in regions)
        frame_area = max(1, frame_w * frame_h)
        if (total_area / frame_area) >= 0.45:
            return "navigation_or_page_change"
        if len(regions) >= 3:
            return "large_layout_change"
        if regions:
            x1, y1, x2, y2 = union_boxes(regions) or regions[0]
            if (y2 - y1) >= frame_h * 0.45:
                return "scroll"
            if (x2 - x1) <= frame_w * 0.20 and (y2 - y1) <= frame_h * 0.20:
                return "small_region_change"
        return "unknown"

    def _log_frame(self, state: ScreenState, transcript_window: Sequence[TranscriptSegment]) -> None:
        cursor_label = state.cursor.cursor_type if state.cursor.visible else "none"
        if state.mode == "code":
            logger.info(
                "[frame=%s] mode=%s changed=%s ocr=%s transcript_segments=%s total=%.1fms",
                state.frame_id,
                state.mode,
                str(state.visual_change).lower(),
                "ran" if state.debug.get("processor_used") == "ocr" else "skipped",
                len(transcript_window),
                float(state.debug.get("processing_time_ms", 0.0)),
            )
            return
        if state.mode == "slide":
            logger.info(
                "[frame=%s] mode=%s changed=%s surya=%s duplicate_slide=%s transcript_segments=%s total=%.1fms",
                state.frame_id,
                state.mode,
                str(state.visual_change).lower(),
                "ran" if state.debug.get("processor_used") == "surya" else "skipped",
                bool(state.debug.get("duplicate_slide", False)),
                len(transcript_window),
                float(state.debug.get("processing_time_ms", 0.0)),
            )
            return
        logger.info(
            "[frame=%s] mode=%s conf=%.2f cursor=%s cursor_conf=%.2f changed=%s change_type=%s macro=%s micro=%s transcript_segments=%s total=%.1fms",
            state.frame_id,
            state.mode,
            float(state.route.confidence),
            cursor_label,
            float(state.cursor.confidence),
            str(state.visual_change).lower(),
            str(state.debug.get("change_type", state.visual_payload.get("change_type", "none"))),
                "macro_detector_ran" if state.debug.get("rfdetr_ran") else "macro_detector_skipped",
            "omniparser_crop" if state.debug.get("omniparser_ran") else "cache_reused",
            len(transcript_window),
            float(state.debug.get("processing_time_ms", 0.0)),
        )


IncrementalUIStatePipeline = DemoAnalyzerPipeline


def _bbox_contains(outer: BBox, inner: BBox, tolerance_ratio: float) -> bool:
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = inner
    pad_x = int((ox2 - ox1) * tolerance_ratio)
    pad_y = int((oy2 - oy1) * tolerance_ratio)
    return ix1 >= (ox1 - pad_x) and iy1 >= (oy1 - pad_y) and ix2 <= (ox2 + pad_x) and iy2 <= (oy2 + pad_y)


def _bbox_overlaps(a: BBox, b: BBox, tolerance_ratio: float) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    pad_x = int((ax2 - ax1) * tolerance_ratio)
    pad_y = int((ay2 - ay1) * tolerance_ratio)
    return not (
        ax2 + pad_x < bx1
        or bx2 < ax1 - pad_x
        or ay2 + pad_y < by1
        or by2 < ay1 - pad_y
    )


def _elements_from_ocr_lines(lines: Sequence[Dict[str, Any]], *, frame_id: int, timestamp_sec: float, prefix: str) -> List[UIElement]:
    elements: List[UIElement] = []
    for idx, line in enumerate(lines or [], start=1):
        box = tuple(int(v) for v in line.get("box", [0, 0, 0, 0])[:4])
        if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
            continue
        elements.append(
            UIElement(
                id=f"{prefix}_text_{idx:04d}",
                type="text",
                bbox=box,
                text=str(line.get("text", "") or "").strip(),
                interactive=False,
                confidence=float(line.get("confidence", 0.0) or 0.0),
                source=str(line.get("source", "ocr")),
                last_seen_frame=frame_id,
                timestamp=float(timestamp_sec),
            )
        )
    return elements


def _ocr_lines_for_regions(lines: Sequence[Dict[str, Any]], regions: Sequence[BBox]) -> List[Dict[str, Any]]:
    if not regions:
        return list(lines or [])
    filtered: List[Dict[str, Any]] = []
    for line in lines or []:
        box = tuple(int(v) for v in line.get("box", [0, 0, 0, 0])[:4])
        if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
            continue
        if any(bbox_intersects(box, region, padding=4) for region in regions):
            filtered.append(dict(line))
    return filtered


def _elements_from_slide(slide: SlideAnalysis, *, frame_id: int, timestamp_sec: float) -> List[UIElement]:
    elements: List[UIElement] = []
    for idx, block in enumerate(slide.layout, start=1):
        elements.append(
            UIElement(
                id=f"slide_block_{idx:04d}",
                type=block.type,
                bbox=tuple(int(v) for v in block.bbox[:4]),
                text=block.text,
                interactive=False,
                confidence=float(slide.confidence),
                source=slide.provider,
                last_seen_frame=frame_id,
                timestamp=float(timestamp_sec),
            )
        )
    return elements


def _serialize_layout_group(group: LayoutGroup) -> Dict[str, Any]:
    return {
        "id": group.id,
        "type": group.label,
        "bbox": list(group.bbox),
        "confidence": group.score,
        "source": group.source,
        "z_index": group.z_index,
        "frame_id": group.frame_id,
        "timestamp": group.timestamp,
        "stable": group.stable,
    }


def _serialize_micro_element(item: UIElement) -> Dict[str, Any]:
    return {
        "id": item.id,
        "type": item.type,
        "bbox": list(item.bbox),
        "text": item.text or None,
        "caption": item.caption,
        "clickable": bool(item.interactive),
        "confidence": item.confidence,
        "source": item.source,
        "parent_macro_region_id": item.parent_macro_region_id,
        "frame_id": item.last_seen_frame,
        "timestamp": item.timestamp,
        "stale": item.stale,
    }


def _cursor_payload(cursor: CursorState) -> Dict[str, Any]:
    return {
        "detected": bool(cursor.visible),
        "type": cursor.cursor_type,
        "bbox": list(cursor.bbox) if cursor.bbox else None,
        "hotspot": list(cursor.hotspot) if cursor.hotspot else None,
        "confidence": cursor.confidence,
        "method": cursor.method,
        "motion_type": cursor.motion_type,
        "near_element_id": cursor.near_element_id,
    }


def _best_parent_group(region: BBox, groups: Sequence[LayoutGroup]) -> Optional[LayoutGroup]:
    matches = [group for group in groups if bbox_intersects(group.bbox, region, padding=2)]
    if not matches:
        return None
    matches.sort(key=lambda item: (bbox_area(item.bbox), -item.score))
    return matches[0]


def _parent_group_id_for_box(box: BBox, groups: Sequence[LayoutGroup]) -> Optional[str]:
    parent = _best_parent_group(box, groups)
    return parent.id if parent is not None else None


def _diff_text(previous: str, current: str) -> Dict[str, str]:
    prev_lines = [line.strip() for line in str(previous or "").splitlines() if line.strip()]
    curr_lines = [line.strip() for line in str(current or "").splitlines() if line.strip()]
    added = [line[2:] for line in ndiff(prev_lines, curr_lines) if line.startswith("+ ")]
    removed = [line[2:] for line in ndiff(prev_lines, curr_lines) if line.startswith("- ")]
    return {
        "added": "\n".join(added[:20]),
        "removed": "\n".join(removed[:20]),
    }


def _mode_transition_text(previous_mode: str, current_mode: str, mode_changed: bool) -> str:
    if mode_changed and previous_mode != current_mode:
        return f"The screen switched from {previous_mode} mode to {current_mode} mode."
    return ""


def _join_summary_parts(parts: Sequence[Optional[str]]) -> str:
    return " ".join(part.strip() for part in parts if isinstance(part, str) and part.strip())

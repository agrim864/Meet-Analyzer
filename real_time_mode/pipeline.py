from __future__ import annotations

import logging
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass, field
from difflib import ndiff
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from .cursor_adapter import CursorMotionAnalyzer
from .doclayout_adapter import DocLayoutPayload, DocLayoutYOLOAdapter
from .export import export_scene_record
from .frame_summarizer import (
    FrameSummaryRequest,
    HeuristicFrameSummarizer,
    ResilientFrameSummarizer,
    SummaryMemoryTurn,
    _looks_like_noise,
)
from .macro_layout import MacroLayoutDetector
from .merge import merge_reparsed_state
from .mode_router import ModeRouter
from .ocr_adapter import OCRResult, OCRTextExtractor
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
DEFERRED_LLM_MARKER = "__DEFER_LLM_FOR_NEWER_FRAME__"


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
    deferred_summary_by_mode: Dict[str, ScreenState] = field(default_factory=dict)


class DemoAnalyzerPipeline:
    """Multi-mode real-time meeting screen analyzer with per-mode caches and transcript-aligned summary events."""

    def __init__(
        self,
        *,
        siglip_classifier: Optional[SiglipScreenClassifier] = None,
        stt_store: Optional[UtteranceStore] = None,
        omniparser: Optional[Any] = None,
        cursor_analyzer: Optional[CursorMotionAnalyzer] = None,
        macro_layout_detector: Optional[MacroLayoutDetector] = None,
        doclayout_detector: Optional[DocLayoutYOLOAdapter] = None,
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
        summary_timeout_sec: Optional[float] = None,
    ) -> None:
        self.siglip = siglip_classifier or SiglipScreenClassifier()
        self.stt_store = stt_store or UtteranceStore()
        self.omniparser = None
        self.cursor_analyzer = cursor_analyzer or CursorMotionAnalyzer()
        default_macro_weights = Path(__file__).resolve().parent / "vendor" / "uitag" / "yolo-ui.pt"
        self.macro_layout = macro_layout_detector or MacroLayoutDetector(str(default_macro_weights))
        self.doclayout = doclayout_detector or DocLayoutYOLOAdapter(enabled=False)
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
        if summary_timeout_sec is None:
            summary_timeout_sec = float(os.getenv("FRAME_SUMMARIZER_TIMEOUT_SEC", "5.0"))
        self.summary_timeout_sec = max(0.5, float(summary_timeout_sec))

        self.cache = StateCache()
        self._frame_id = 0
        self._records: List[SceneStateRecord] = []
        self._last_record_by_mode: Dict[str, SceneStateRecord] = {}
        self.summary_skip_checker: Optional[Callable[[ScreenState], bool]] = None
        self._llm_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="rt-llm")

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
        cursor_started = perf_counter()
        prev_cursor = self.cache.cursor_state
        prev_masked_for_cursor = self._get_mode_cache(previous_mode).prev_frame_masked if previous_mode in {"code", "slide", "demo"} else None
        if self._frame_shape_changed(prev_masked_for_cursor, frame_bgr):
            prev_masked_for_cursor = None
        with ThreadPoolExecutor(max_workers=2) as executor:
            route_future = executor.submit(self._route_mode, frame_bgr)
            cursor_future = executor.submit(
                self.cursor_analyzer.analyze,
                frame_bgr=frame_bgr,
                prev_frame_bgr=prev_masked_for_cursor,
                prev_cursor=prev_cursor,
                elements=self._candidate_elements_for_cursor(previous_mode),
            )
            route, mode_changed = route_future.result()
            stage_timings["route_ms"] = (perf_counter() - route_started) * 1000.0
            try:
                cursor = cursor_future.result()
            except Exception as exc:
                logger.warning("Cursor analysis failed for frame=%s: %s", self._frame_id, exc)
                cursor = CursorState()
            stage_timings["cursor_ms"] = (perf_counter() - cursor_started) * 1000.0
        mode = route.mode
        self.cache.previous_mode = previous_mode
        self.cache.current_mode = mode
        self.cache.last_stable_mode = mode
        self.cache.mode_prediction_history = [item.to_dict() for item in self.router.state.history]

        if mode in {"code", "slide", "demo"}:
            masked_frame = self._mask_cursor_for_change_detection(frame_bgr, cursor, prev_cursor)
        else:
            masked_frame = None
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
        state = self._maybe_reclassify_visual_state(state)
        mode = state.mode
        self.cache.current_mode = mode
        self.cache.last_stable_mode = mode
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
                "macro_detector_ran": bool(state.debug.get("macro_detector_ran", state.debug.get("rfdetr_ran", False))),
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
                    "ocr_ran": True,
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
            layout_started = perf_counter()
            doclayout_started = perf_counter()
            with ThreadPoolExecutor(max_workers=3) as executor:
                slide_future = executor.submit(self._safe_analyze_slide, frame_bgr)
                layout_future = executor.submit(self._detect_macro_layout, frame_bgr, timestamp_sec)
                doclayout_future = executor.submit(
                    self._safe_detect_doclayout_regions,
                    frame_bgr,
                    [(0, 0, viewport["width"], viewport["height"])],
                    "slide",
                    timestamp_sec,
                )
                slide = slide_future.result()
                slide_ms = (perf_counter() - slide_started) * 1000.0
                macro_layout = layout_future.result()
                layout_ms = (perf_counter() - layout_started) * 1000.0
                doclayout = doclayout_future.result()
                doclayout_ms = (perf_counter() - doclayout_started) * 1000.0
            duplicate_slide = (not first_frame) and cache.prev_slide_fingerprint and cache.prev_slide_fingerprint == slide.fingerprint
            visual_change = first_frame or (change.changed and not duplicate_slide)
            elements = [*_elements_from_slide(slide, frame_id=self._frame_id, timestamp_sec=timestamp_sec), *doclayout.elements]
            for element in elements:
                element.parent_macro_region_id = _parent_group_id_for_box(element.bbox, macro_layout)
            screen_structure = _build_demo_screen_structure(macro_layout, elements)
            payload = {
                "mode": "slide",
                "visual_change": visual_change,
                "slide_fingerprint": slide.fingerprint,
                "layout": [block.to_dict() for block in slide.layout],
                "macro_layout": [_serialize_layout_group(group) for group in macro_layout],
                "doclayout_elements": [_serialize_micro_element(item) for item in doclayout.elements],
                "screen_structure": screen_structure,
                "full_text": slide.full_text,
                "cursor": _cursor_payload(self.cache.cursor_state),
            }
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
                layout_groups=macro_layout or [LayoutGroup(id="group_slide_content", label="content", bbox=(0, 0, viewport["width"], viewport["height"]), source=slide.provider, frame_id=self._frame_id, timestamp=float(timestamp_sec))],
                cursor=self.cache.cursor_state.copy(),
                delta=StateDelta(
                    types=["initial_frame"] if first_frame else ["surya_refresh" if visual_change else "duplicate_slide"],
                    changed_regions=list(change.changed_regions),
                    reparsed_regions=[(0, 0, viewport["width"], viewport["height"])] if visual_change else [],
                    confidence=max(slide.confidence, change.change_ratio),
                ),
                confidence={
                    "slide_provider": slide.provider,
                    "slide_confidence": slide.confidence,
                    "doclayout_provider": doclayout.provider,
                    "doclayout_confidence": doclayout.confidence,
                },
                debug={
                    "change_score": change.change_ratio,
                    "processor_used": "surya",
                    "rfdetr_ran": False,
                    "macro_detector_ran": True,
                    "omniparser_ran": False,
                    "ocr_ran": True,
                    "duplicate_slide": duplicate_slide,
                    "doclayout_ran": bool(doclayout.elements),
                    "stage_timings_ms": {"slide_analysis_ms": slide_ms, "macro_layout_ms": layout_ms, "doclayout_ms": doclayout_ms},
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
            state.layout_groups = [group.copy() for group in macro_layout]

            crop_plan = self._plan_demo_crops(change.changed_regions, macro_layout, cache.prev_micro_elements, viewport, first_frame)
            reparsed_regions = [tuple(plan["crop_bbox"]) for plan in crop_plan]
            requires_full_parse = first_frame or any(bool(item.get("requires_full_parse")) for item in crop_plan)
            if requires_full_parse:
                reparsed_regions = [(0, 0, frame_w, frame_h)]

            ocr_started = perf_counter()
            doclayout_started = perf_counter()
            with ThreadPoolExecutor(max_workers=2) as executor:
                ocr_future = executor.submit(self._safe_extract_ocr_regions, frame_bgr, reparsed_regions, "demo")
                doclayout_future = executor.submit(
                    self._safe_detect_doclayout_regions,
                    frame_bgr,
                    reparsed_regions,
                    "demo",
                    timestamp_sec,
                )
                demo_ocr = ocr_future.result()
                stage_timings["ocr_ms"] = (perf_counter() - ocr_started) * 1000.0
                doclayout = doclayout_future.result()
                stage_timings["doclayout_ms"] = (perf_counter() - doclayout_started) * 1000.0
            reparsed_elements = _elements_from_ocr_lines(
                demo_ocr.lines,
                frame_id=self._frame_id,
                timestamp_sec=timestamp_sec,
                prefix="demo_ocr",
            )
            reparsed_elements.extend(doclayout.elements)

            for element in reparsed_elements:
                element.parent_macro_region_id = _parent_group_id_for_box(element.bbox, macro_layout)
                element.timestamp = float(timestamp_sec)
            merged_state = merge_reparsed_state(
                state=state,
                reparsed_elements=reparsed_elements,
                reparsed_overlays=[],
                reparsed_regions=reparsed_regions,
                frame_id=self._frame_id,
            )
            merged_state.route = route
            merged_state.mode = "demo"
            merged_state.frame_type = "demo"
            merged_state.cursor = self.cache.cursor_state.copy()
            cursor_target = self._match_cursor_target(self.cache.cursor_state, merged_state.elements, merged_state.layout_groups)
            screen_structure = _build_demo_screen_structure(merged_state.layout_groups, merged_state.elements)
            page_summary = self._build_demo_page_summary(
                merged_state.layout_groups,
                merged_state.elements,
                merged_state.overlays,
                cursor_target,
                screen_structure=screen_structure,
            )
            ocr_text = _reading_text_from_elements(merged_state.elements, source_prefix="ocr")
            visual_change = first_frame or change.changed
            visual_payload = {
                "mode": "demo",
                "visual_change": visual_change,
                "change_type": change.change_type,
                "macro_layout": [_serialize_layout_group(group) for group in merged_state.layout_groups],
                "micro_elements": [_serialize_micro_element(item) for item in merged_state.elements],
                "doclayout_elements": [_serialize_micro_element(item) for item in doclayout.elements],
                "ocr_text": ocr_text,
                "updated_ocr_text": demo_ocr.text,
                "ocr_lines": _ocr_lines_from_elements(merged_state.elements),
                "screen_structure": screen_structure,
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
                "layout_provider": "yolo_layout",
                "layout_confidence": _mean_layout_confidence(merged_state.layout_groups),
                "doclayout_provider": doclayout.provider,
                "doclayout_confidence": doclayout.confidence,
                "ocr_provider": demo_ocr.provider,
                "ocr_confidence": demo_ocr.confidence,
                "ssim_score": change.ssim_score,
                "changed_pixel_ratio": change.change_ratio,
                "route": route.to_dict(),
            }
            merged_state.debug = {
                "change_score": change.change_ratio,
                "processor_used": "layout_ocr_pipeline",
                "rfdetr_ran": reran_macro,
                "macro_detector_ran": reran_macro,
                "doclayout_ran": bool(doclayout.elements),
                "omniparser_ran": False,
                "ocr_ran": True,
                "crop_bbox": [list(item) for item in reparsed_regions],
                "change_type": change.change_type,
                "stage_timings_ms": stage_timings,
            }
            merged_state.delta = StateDelta(
                types=["initial_frame"] if first_frame else ["macro_refresh" if reran_macro else "micro_refresh"],
                changed_regions=list(change.changed_regions),
                reparsed_regions=list(reparsed_regions),
                confidence=max(demo_ocr.confidence, change.change_ratio),
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
        boundary_pad = max(4, int(min(int(viewport["width"]), int(viewport["height"])) * self.macro_boundary_overlap))
        for region in changed_regions:
            expanded = self._expand_region(region, int(viewport["width"]), int(viewport["height"]))
            intersecting = [group for group in macro_layout if bbox_intersects(expanded, group.bbox, padding=boundary_pad)]
            if not intersecting:
                return True
            if len({group.id for group in intersecting}) >= 2:
                return True
            parent = intersecting[0]
            px1, py1, px2, py2 = parent.bbox
            rx1, ry1, rx2, ry2 = expanded
            if (
                abs(rx1 - px1) <= boundary_pad
                or abs(rx2 - px2) <= boundary_pad
                or abs(ry1 - py1) <= boundary_pad
                or abs(ry2 - py2) <= boundary_pad
            ) and bbox_area(expanded) >= max(32, int(frame_area * 0.002)):
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
                elif parent_type in {"content", "main_content"}:
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
            and item.interactive
            and not _is_ocr_element(item)
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

    def _safe_extract_ocr(self, frame_bgr: np.ndarray, mode: str) -> OCRResult:
        try:
            return self.ocr_extractor.extract_full(frame_bgr, frame_type=mode)
        except Exception as exc:
            logger.warning("OCR failed for mode=%s frame=%s: %s", mode, self._frame_id, exc)
            return OCRResult(text="", lines=[], provider="ocr_failed", confidence=0.0)

    def _safe_extract_ocr_regions(self, frame_bgr: np.ndarray, regions: Sequence[BBox], mode: str) -> OCRResult:
        try:
            extract_regions = getattr(self.ocr_extractor, "extract_regions", None)
            if callable(extract_regions):
                return extract_regions(frame_bgr, regions, frame_type=mode)
            full = self.ocr_extractor.extract_full(frame_bgr, frame_type=mode)
            relevant_lines = _ocr_lines_for_regions(full.lines, regions)
            return OCRResult(
                text="\n".join(str(item.get("text", "")).strip() for item in relevant_lines if str(item.get("text", "")).strip()),
                lines=relevant_lines,
                provider=full.provider,
                confidence=full.confidence,
            )
        except Exception as exc:
            logger.warning("Region OCR failed for mode=%s frame=%s: %s", mode, self._frame_id, exc)
            return OCRResult(text="", lines=[], provider="ocr_failed", confidence=0.0)

    def _safe_detect_doclayout_regions(
        self,
        frame_bgr: np.ndarray,
        regions: Sequence[BBox],
        mode: str,
        timestamp_sec: float,
    ) -> DocLayoutPayload:
        try:
            payload = self.doclayout.detect_regions(
                frame_bgr,
                regions,
                frame_id=self._frame_id,
                timestamp_sec=timestamp_sec,
            )
            return payload
        except Exception as exc:
            logger.warning("DocLayout-YOLO failed for mode=%s frame=%s: %s", mode, self._frame_id, exc)
            return DocLayoutPayload(elements=[], provider=f"doclayout_failed:{type(exc).__name__}", confidence=0.0)

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
        screen_structure: Optional[Dict[str, Any]] = None,
    ) -> str:
        structure = screen_structure or _build_demo_screen_structure(layout_groups, elements)
        layout_desc = str(structure.get("layout_description") or "a single content area")
        inferred = str(structure.get("interface_type") or "application interface")
        region_details = [str(item) for item in structure.get("region_descriptions", []) if str(item).strip()]
        highlights = ", ".join(str(item) for item in structure.get("highlights", [])[:5])
        summary = f"The screen appears to be {inferred} with {layout_desc}."
        if region_details:
            summary += " " + " ".join(region_details[:3])
        if overlays:
            summary += f" An overlay is visible ({overlays[0].type})."
        if highlights:
            summary += f" Visible controls include {highlights}."
        cursor_sentence = _cursor_action_text(cursor_target, "")
        if cursor_sentence:
            summary += " " + cursor_sentence
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
        if self._should_defer_llm_for_newer_frame(state, mode_changed=mode_changed):
            return self._deferred_summary_for_newer_frame(state, transcript_text)
        previous_deferred = self._pop_deferred_frame_for_pair(state)
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
            state.visual_payload["summary_task"] = "code_detailed"
            if previous_deferred is not None:
                state.visual_payload["summary_task"] = "paired_frame_change"
                state.visual_payload["previous_frame"] = self._compact_state_for_pair(previous_deferred)
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
            state.visual_payload["summary_task"] = "slide_detailed"
            if previous_deferred is not None:
                state.visual_payload["summary_task"] = "paired_frame_change"
                state.visual_payload["previous_frame"] = self._compact_state_for_pair(previous_deferred)
            return self._llm_summary(
                state=state,
                heuristic_summary=heuristic,
                transcript_text=transcript_text,
                cursor_text=self._cursor_summary_text(state.cursor, state.visual_payload.get("cursor_target")),
                page_text=str(state.visual_payload.get("full_text", "") or "")[:500],
                action_text=self._action_summary_text(state),
            )
        if state.mode == "demo":
            return self._summarize_demo_by_regions(
                state=state,
                transcript_text=transcript_text,
                previous_mode=previous_mode,
                mode_changed=mode_changed,
                previous_deferred=previous_deferred,
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
        summary = self._run_summarizer_with_newer_poll(state, request)
        if summary == DEFERRED_LLM_MARKER:
            return self._deferred_summary_for_newer_frame(state, transcript_text)
        summary_backend = getattr(self.frame_summarizer, "primary", self.frame_summarizer)
        state.debug = dict(state.debug)
        state.debug["summary_provider"] = getattr(summary_backend, "provider_name", "heuristic")
        state.debug["summary_model"] = getattr(summary_backend, "model_name", "heuristic")
        return summary

    def _summarize_demo_by_regions(
        self,
        *,
        state: ScreenState,
        transcript_text: str,
        previous_mode: str,
        mode_changed: bool,
        previous_deferred: Optional[ScreenState] = None,
    ) -> str:
        structure = state.visual_payload.get("screen_structure") or {}
        regions = list(structure.get("regions") or [])
        segment_payloads = _demo_segment_payloads(regions, state.visual_payload.get("cursor_target") or {})
        page_text = _join_summary_parts(
            [
                str(state.visual_payload.get("page_summary", "") or ""),
                str(state.visual_payload.get("ocr_text", "") or ""),
            ]
        )[:1200]
        base_heuristic = _join_summary_parts(
            [
                _mode_transition_text(previous_mode, state.mode, mode_changed),
                str(state.visual_payload.get("page_summary", "") or "The presenter is demonstrating a web UI."),
                transcript_text and f"They are explaining: {transcript_text}",
            ]
        )
        started = perf_counter()
        ordered_results = {
            segment["key"]: {
                "label": segment["label"],
                "bbox": segment["bbox"],
                "description": _segment_fallback_description(segment),
            }
            for segment in segment_payloads
        }
        relationship = _demo_relationship_fallback(segment_payloads)
        if len(segment_payloads) >= 2:
            llm_relationship = self._summarize_payload(
                state=state,
                payload={
                    **dict(state.visual_payload),
                    "summary_task": "demo_relationship",
                    "related_segments": [
                        {
                            "key": segment["key"],
                            "label": segment["label"],
                            "description": ordered_results.get(segment["key"], {}).get("description", ""),
                        }
                        for segment in segment_payloads
                    ],
                    "previous_frame": self._compact_state_for_pair(previous_deferred) if previous_deferred is not None else {},
                },
                heuristic_summary=relationship,
                transcript_text=transcript_text,
                cursor_text=self._cursor_summary_text(state.cursor, state.visual_payload.get("cursor_target")),
                page_text=page_text,
                action_text=self._action_summary_text(state),
            )
            if llm_relationship == DEFERRED_LLM_MARKER:
                return self._deferred_summary_for_newer_frame(state, transcript_text)
            if llm_relationship and not _looks_like_prompt_placeholder(llm_relationship):
                relationship = llm_relationship
        if relationship:
            ordered_results["relationships"] = {"label": "relationships", "bbox": None, "description": relationship}
        results_md = _format_demo_results_markdown(state.frame_id, state.timestamp_sec, ordered_results)
        state.visual_payload["segment_descriptions"] = ordered_results
        state.visual_payload["results_md"] = results_md
        state.visual_payload["summary_task"] = "demo_segmented"
        state.debug = dict(state.debug)
        state.debug.setdefault("stage_timings_ms", {})
        state.debug["stage_timings_ms"]["segment_summary_ms"] = (perf_counter() - started) * 1000.0
        state.debug["segment_summary_count"] = len(segment_payloads)
        summary_backend = getattr(self.frame_summarizer, "primary", self.frame_summarizer)
        state.debug["summary_provider"] = getattr(summary_backend, "provider_name", "heuristic")
        state.debug["summary_model"] = getattr(summary_backend, "model_name", "heuristic")
        return results_md or base_heuristic

    def _summarize_payload(
        self,
        *,
        state: ScreenState,
        payload: Dict[str, Any],
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
            current_payload=payload,
            recent_memory=list(self.cache.summary_memory[-self.summary_memory_size :]),
        )
        return self._run_summarizer_with_newer_poll(state, request)

    def _run_summarizer_with_newer_poll(self, state: ScreenState, request: FrameSummaryRequest) -> str:
        future = self._llm_executor.submit(self.frame_summarizer.summarize, request)
        started = perf_counter()
        while True:
            try:
                return future.result(timeout=0.10)
            except TimeoutError:
                if self._should_defer_llm_for_newer_frame(state, mode_changed=False):
                    future.cancel()
                    state.debug = dict(state.debug)
                    state.debug["llm_policy"] = "skipped_inflight_near_duplicate_pending_newer_frame"
                    return DEFERRED_LLM_MARKER
                if (perf_counter() - started) >= self.summary_timeout_sec:
                    future.cancel()
                    state.debug = dict(state.debug)
                    state.debug["llm_policy"] = "summary_timeout_heuristic_fallback"
                    state.debug["summary_error"] = (
                        f"LLM summary exceeded {self.summary_timeout_sec:.1f}s; "
                        "returned heuristic summary so the live frame could render."
                    )
                    state.debug["summary_provider"] = "heuristic_timeout"
                    state.debug["summary_model"] = "heuristic"
                    return request.heuristic_summary

    def _maybe_reclassify_visual_state(self, state: ScreenState) -> ScreenState:
        if state.mode != "demo":
            return state
        if not self._demo_state_looks_like_code(state):
            return state
        state.mode = "code"
        state.frame_type = "code"
        state.visual_payload = dict(state.visual_payload)
        state.visual_payload["mode"] = "code"
        state.visual_payload["summary_task"] = "code_detailed"
        state.debug = dict(state.debug)
        state.debug["route_override"] = "demo_to_code_from_ocr"
        state.delta.types = [*list(state.delta.types or []), "mode_override_code"]
        return state

    @staticmethod
    def _demo_state_looks_like_code(state: ScreenState) -> bool:
        payload = state.visual_payload or {}
        text = " ".join(
            [
                str(payload.get("ocr_text") or ""),
                str(payload.get("page_summary") or ""),
                " ".join(str(item.get("text") or "") for item in payload.get("micro_elements") or []),
            ]
        ).lower()
        code_terms = [
            "import ",
            "inport ",
            "from ",
            "trom ",
            "class ",
            "def ",
            "return ",
            "except ",
            "pydantic",
            "requirements",
            "open editors",
            "terminal",
            "debug console",
            "explorer",
            "meet_agent",
            "frame_analysis",
            "utterances",
            ".py",
        ]
        score = sum(1 for term in code_terms if term in text)
        return score >= 3

    def _pop_deferred_frame_for_pair(self, state: ScreenState) -> Optional[ScreenState]:
        previous = self.cache.deferred_summary_by_mode.pop(state.mode, None)
        if previous is None:
            return None
        if self._states_too_different_for_pair(previous, state):
            self.cache.deferred_summary_by_mode[previous.mode] = previous
            return None
        state.debug = dict(state.debug)
        state.debug["llm_policy"] = "paired_with_previous_deferred_frame"
        return previous

    @staticmethod
    def _states_too_different_for_pair(previous: ScreenState, current: ScreenState) -> bool:
        if previous.mode != current.mode:
            return True
        change_type = str(current.visual_payload.get("change_type") or current.debug.get("change_type") or "")
        change_score = float(current.debug.get("change_score", 0.0) or 0.0)
        return change_type in {"navigation_or_page_change", "large_layout_change"} and change_score >= 0.05

    @staticmethod
    def _compact_state_for_pair(state: Optional[ScreenState]) -> Dict[str, Any]:
        if state is None:
            return {}
        payload = dict(state.visual_payload or {})
        return {
            "frame_id": state.frame_id,
            "timestamp_sec": round(float(state.timestamp_sec), 2),
            "mode": state.mode,
            "summary": str(state.summary or "")[:1200],
            "page_summary": str(payload.get("page_summary") or "")[:1000],
            "ocr_text": str(payload.get("ocr_text") or payload.get("full_text") or "")[:1600],
            "macro_layout": list(payload.get("macro_layout") or [])[:12],
            "screen_structure": payload.get("screen_structure") or {},
            "cursor_target": payload.get("cursor_target") or {},
        }

    def _should_defer_llm_for_newer_frame(self, state: ScreenState, *, mode_changed: bool) -> bool:
        checker = self.summary_skip_checker
        if checker is None or mode_changed or state.mode not in {"demo", "slide", "code"}:
            return False
        try:
            newer_waiting = bool(checker(state))
        except Exception:
            newer_waiting = False
        if not newer_waiting:
            return False
        change_type = str(state.visual_payload.get("change_type") or state.debug.get("change_type") or "")
        if change_type in {"navigation_or_page_change", "large_layout_change"}:
            return False
        change_score = float(state.debug.get("change_score", 0.0) or 0.0)
        threshold = 0.018 if state.mode == "demo" else 0.006
        previous_record = self._last_record_by_mode.get(state.mode)
        return previous_record is not None and change_score <= threshold

    def _deferred_summary_for_newer_frame(self, state: ScreenState, transcript_text: str) -> str:
        previous_record = self._last_record_by_mode.get(state.mode)
        previous_summary = str(previous_record.summary if previous_record is not None else "").strip()
        fallback = str(state.visual_payload.get("page_summary") or state.visual_payload.get("full_text") or state.visual_payload.get("ocr_text") or "").strip()
        summary = _join_summary_parts(
            [
                "Detailed LLM work for this near-duplicate frame was skipped because a newer frame was already waiting.",
                previous_summary and f"Previous stable description: {previous_summary[:500]}",
                fallback and f"Current detected evidence: {fallback[:500]}",
                transcript_text and f"Transcript during the skipped frame: {transcript_text[:300]}",
            ]
        )
        state.visual_payload["summary_task"] = "deferred_for_newer_frame"
        state.visual_payload["results_md"] = _format_deferred_results_markdown(state.frame_id, state.timestamp_sec, summary)
        state.debug = dict(state.debug)
        state.debug["summary_provider"] = "deferred"
        state.debug["summary_model"] = "newer_frame_waiting"
        state.debug["llm_policy"] = state.debug.get("llm_policy") or "skipped_near_duplicate_pending_newer_frame"
        deferred = state.clone()
        deferred.summary = state.visual_payload["results_md"]
        self.cache.deferred_summary_by_mode[state.mode] = deferred
        return state.visual_payload["results_md"]

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
            return _join_summary_parts(
                [
                    str(state.visual_payload.get("page_summary", "") or ""),
                    str(state.visual_payload.get("ocr_text", "") or ""),
                ]
            )[:450]
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
            "[frame=%s] mode=%s conf=%.2f cursor=%s cursor_conf=%.2f changed=%s change_type=%s macro=%s ocr=%s transcript_segments=%s total=%.1fms",
            state.frame_id,
            state.mode,
            float(state.route.confidence),
            cursor_label,
            float(state.cursor.confidence),
            str(state.visual_change).lower(),
            str(state.debug.get("change_type", state.visual_payload.get("change_type", "none"))),
            "macro_detector_ran" if state.debug.get("macro_detector_ran", state.debug.get("rfdetr_ran")) else "macro_detector_skipped",
            "ran" if state.debug.get("ocr_ran") else "cache_reused",
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


def _build_demo_screen_structure(layout_groups: Sequence[LayoutGroup], elements: Sequence[UIElement]) -> Dict[str, Any]:
    ordered_groups = sorted(layout_groups, key=lambda item: (item.bbox[1], item.bbox[0], item.z_index, item.id))
    group_by_id = {group.id: group for group in ordered_groups}
    by_region: Dict[str, List[UIElement]] = {group.id: [] for group in ordered_groups}
    for element in sorted(elements, key=_reading_order_key):
        parent = group_by_id.get(str(element.parent_macro_region_id or ""))
        if parent is None:
            parent = _best_parent_group(element.bbox, ordered_groups)
        if parent is not None:
            by_region.setdefault(parent.id, []).append(element)

    labels = [_human_region_label(group.label) for group in ordered_groups]
    unique_labels = list(dict.fromkeys(label for label in labels if label))
    layout_description = ", ".join(unique_labels[:5]) if unique_labels else "one main content area"
    highlights = _collect_element_highlights(elements, limit=6)
    interface_type = _infer_interface_type(elements, ordered_groups, by_region)
    region_descriptions: List[str] = []
    regions: List[Dict[str, Any]] = []
    flat_reading_order: List[Dict[str, Any]] = []

    for group in ordered_groups:
        region_elements = sorted(by_region.get(group.id, []), key=_reading_order_key)
        children = _build_nested_reading_nodes(region_elements)
        description = _describe_region(group, region_elements)
        if description:
            region_descriptions.append(description)
        flat_reading_order.extend(
            {
                "region_id": group.id,
                "region_type": group.label,
                "id": node.get("id"),
                "kind": node.get("kind"),
                "type": node.get("type"),
                "text": node.get("text") or node.get("ocr_text"),
                "bbox": node.get("bbox"),
            }
            for node in children
        )
        regions.append(
            {
                "id": group.id,
                "type": group.label,
                "label": _human_region_label(group.label),
                "bbox": list(group.bbox),
                "element_count": len(region_elements),
                "ocr_line_count": sum(1 for item in region_elements if _is_ocr_element(item)),
                "element_types": dict(Counter(_normal_element_type(item) for item in region_elements).most_common(6)),
                "sample_text": _collect_element_highlights(region_elements, limit=4),
                "children": children,
            }
        )

    return {
        "interface_type": interface_type,
        "layout_description": layout_description,
        "regions": regions,
        "reading_order": flat_reading_order,
        "region_descriptions": region_descriptions,
        "highlights": highlights,
    }


def _demo_segment_payloads(regions: Sequence[Dict[str, Any]], cursor_target: Dict[str, Any]) -> List[Dict[str, Any]]:
    priority = {"header": 0, "top_navbar": 0, "left_sidebar": 1, "right_sidebar": 2, "footer": 4}
    payloads: List[Dict[str, Any]] = []
    main_regions: List[Dict[str, Any]] = []
    for idx, region in enumerate(regions):
        region_type = str(region.get("type") or "").strip() or "region"
        if region_type in {"main_content", "content", "content_panel", "form_region", "table_region"}:
            main_regions.append(region)
            continue
        if region_type not in priority:
            continue
        payloads.append(_region_payload(region, key=f"{region_type}_{idx}", label=_human_region_label(region_type), cursor_target=cursor_target, order=priority[region_type]))
    for region in main_regions:
        payloads.extend(_split_main_region_payloads(region, cursor_target=cursor_target, max_segments=5))
    if not payloads:
        for idx, region in enumerate(regions[:6]):
            region_type = str(region.get("type") or "region").strip() or "region"
            payloads.append(_region_payload(region, key=f"{region_type}_{idx}", label=_human_region_label(region_type), cursor_target=cursor_target, order=idx))
    payloads.sort(key=lambda item: (item["order"], item["bbox"][1], item["bbox"][0], item["key"]))
    return payloads


def _split_main_region_payloads(region: Dict[str, Any], *, cursor_target: Dict[str, Any], max_segments: int) -> List[Dict[str, Any]]:
    children = sorted(list(region.get("children") or []), key=lambda item: (int((item.get("bbox") or [0, 0, 0, 0])[1]), int((item.get("bbox") or [0, 0, 0, 0])[0])))
    if not children:
        return [_region_payload(region, key="main_content_0", label=_human_region_label(str(region.get("type") or "main_content")), cursor_target=cursor_target, order=3)]
    y_gaps: List[Tuple[int, int]] = []
    previous_bottom = int((children[0].get("bbox") or [0, 0, 0, 0])[3])
    for idx, child in enumerate(children[1:], start=1):
        bbox = child.get("bbox") or [0, 0, 0, 0]
        gap = int(bbox[1]) - previous_bottom
        if gap > 80:
            y_gaps.append((gap, idx))
        previous_bottom = max(previous_bottom, int(bbox[3]))
    split_indices = sorted(idx for _, idx in sorted(y_gaps, reverse=True)[: max(0, max_segments - 1)])
    groups: List[List[Dict[str, Any]]] = []
    start = 0
    for idx in split_indices:
        groups.append(children[start:idx])
        start = idx
    groups.append(children[start:])
    payloads: List[Dict[str, Any]] = []
    for idx, group in enumerate(groups[:max_segments]):
        if not group:
            continue
        bbox = union_boxes([tuple(int(v) for v in (item.get("bbox") or [0, 0, 0, 0])[:4]) for item in group]) or tuple(region.get("bbox") or [0, 0, 0, 0])
        segment = dict(region)
        segment["children"] = group
        segment["bbox"] = list(bbox)
        segment["sample_text"] = [str(item.get("text") or item.get("ocr_text") or "").strip() for item in group if str(item.get("text") or item.get("ocr_text") or "").strip()][:8]
        payloads.append(
            _region_payload(
                segment,
                key=f"main_content_{idx}",
                label=f"main content subregion {idx + 1}",
                cursor_target=cursor_target,
                order=3,
            )
        )
    return payloads


def _region_payload(
    region: Dict[str, Any],
    *,
    key: str,
    label: str,
    cursor_target: Dict[str, Any],
    order: int = 0,
) -> Dict[str, Any]:
    bbox = [int(v) for v in (region.get("bbox") or [0, 0, 0, 0])[:4]]
    cursor_macro = str(cursor_target.get("parent_macro_region_id") or "")
    cursor_region_type = str(cursor_target.get("parent_macro_region_type") or "")
    cursor_in_region = cursor_macro == str(region.get("id") or "") or cursor_region_type == str(region.get("type") or "")
    return {
        "key": key,
        "label": label,
        "type": region.get("type"),
        "id": region.get("id"),
        "bbox": bbox,
        "order": order,
        "element_count": region.get("element_count", 0),
        "ocr_line_count": region.get("ocr_line_count", 0),
        "sample_text": region.get("sample_text") or [],
        "children": region.get("children") or [],
        "cursor": cursor_target if cursor_in_region else {},
    }


def _neighbor_summaries(segment: Dict[str, Any], segments: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sx1, sy1, sx2, sy2 = segment["bbox"]
    neighbors = []
    for other in segments:
        if other["key"] == segment["key"]:
            continue
        ox1, oy1, ox2, oy2 = other["bbox"]
        relation = []
        if ox2 <= sx1:
            relation.append("left")
        if ox1 >= sx2:
            relation.append("right")
        if oy2 <= sy1:
            relation.append("above")
        if oy1 >= sy2:
            relation.append("below")
        neighbors.append({"key": other["key"], "label": other["label"], "relative_position": " ".join(relation) or "overlapping/nearby"})
    return neighbors[:6]


def _segment_fallback_description(segment: Dict[str, Any]) -> str:
    text = ", ".join(str(item) for item in (segment.get("sample_text") or [])[:6] if str(item).strip())
    label = str(segment.get("label") or "segment")
    element_count = int(segment.get("element_count") or len(segment.get("children") or []) or 0)
    ocr_count = int(segment.get("ocr_line_count") or 0)
    type_counts = segment.get("element_types") or {}
    if not type_counts:
        type_counts = dict(Counter(str((item.get("type") or item.get("kind") or "element")).strip() for item in segment.get("children") or [] if str((item.get("type") or item.get("kind") or "")).strip()).most_common(4))
    type_text = ", ".join(f"{count} {kind}" for kind, count in list(type_counts.items())[:4] if kind)
    parts = [f"{label} spans region {segment.get('bbox')}."]
    if element_count or ocr_count:
        parts.append(f"It contains {element_count} detected items and {ocr_count} OCR text lines.")
    if type_text:
        parts.append(f"Detected content types include {type_text}.")
    if text:
        parts.append(f"Visible text includes {text}.")
    else:
        parts.append("Little readable text is visible in this region.")
    cursor = segment.get("cursor") or {}
    cursor_text = str(cursor.get("element_text") or cursor.get("element_type") or "").strip()
    if cursor_text:
        parts.append(f"The cursor is over {cursor_text}.")
    return " ".join(parts)


def _demo_relationship_fallback(segments: Sequence[Dict[str, Any]]) -> str:
    if not segments:
        return ""
    labels = [str(segment.get("label") or segment.get("key") or "region") for segment in segments[:6]]
    if len(labels) == 1:
        return f"The visible interface is primarily organized around the {labels[0]} region."
    return "The visible interface is organized into " + ", ".join(labels[:-1]) + f", and {labels[-1]}. These regions together describe the current screen layout and reading order."


def _looks_like_prompt_placeholder(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "").strip().lower())
    if not normalized:
        return True
    return (
        normalized.startswith("describe the ")
        or normalized.startswith("describe how the visible ui segments")
        or normalized.endswith("segment in detail.")
        or normalized.startswith("relationships describe")
    )


def _format_demo_results_markdown(frame_id: int, timestamp_sec: float, results: Dict[str, Dict[str, Any]]) -> str:
    lines = [f"# Frame {frame_id} Demo Results", "", f"Timestamp: {timestamp_sec:.2f}s", ""]
    for key, item in results.items():
        label = str(item.get("label") or key).replace("_", " ").title()
        lines.append(f"## {label}")
        bbox = item.get("bbox")
        if bbox:
            lines.append(f"Region: `{bbox}`")
        description = str(item.get("description") or "").strip()
        lines.append(description or "No detailed description was generated.")
        lines.append("")
    return "\n".join(lines).strip()


def _format_deferred_results_markdown(frame_id: int, timestamp_sec: float, summary: str) -> str:
    return "\n".join(
        [
            f"# Frame {frame_id} Deferred Results",
            "",
            f"Timestamp: {timestamp_sec:.2f}s",
            "",
            str(summary or "Detailed LLM work was deferred because a newer near-duplicate frame was pending.").strip(),
        ]
    ).strip()


def _mean_layout_confidence(layout_groups: Sequence[LayoutGroup]) -> float:
    scores = [float(group.score or 0.0) for group in layout_groups]
    return sum(scores) / max(1, len(scores))


def _build_nested_reading_nodes(elements: Sequence[UIElement]) -> List[Dict[str, Any]]:
    ocr_items = [item for item in elements if _is_ocr_element(item)]
    ui_items = [item for item in elements if not _is_ocr_element(item)]
    attached_ui: set[str] = set()
    attached_ocr: set[str] = set()
    ocr_parent_nodes: Dict[str, Dict[str, Any]] = {}
    top_nodes: List[Dict[str, Any]] = []

    for element in sorted(ui_items, key=_reading_order_key):
        match = _best_ocr_match(element, ocr_items)
        if match is None:
            continue
        if _bbox_contains(match.bbox, element.bbox, 0.08) or _intersection_ratio(element.bbox, match.bbox) >= 0.65:
            node = ocr_parent_nodes.setdefault(match.id, _element_structure_node(match, kind="ocr_text"))
            node.setdefault("children", []).append(_element_structure_node(element, kind="ui_element"))
            attached_ui.add(element.id)
            attached_ocr.add(match.id)
        elif _bbox_contains(element.bbox, match.bbox, 0.08) or _intersection_ratio(match.bbox, element.bbox) >= 0.65:
            node = _element_structure_node(element, kind="ui_element")
            node["children"] = [_element_structure_node(match, kind="ocr_text")]
            top_nodes.append(node)
            attached_ui.add(element.id)
            attached_ocr.add(match.id)
        elif _intersection_ratio(element.bbox, match.bbox) >= 0.25 or _intersection_ratio(match.bbox, element.bbox) >= 0.25:
            node = _element_structure_node(element, kind="ui_element")
            node["ocr_text"] = match.text
            node["sources"] = sorted({element.source, match.source})
            top_nodes.append(node)
            attached_ui.add(element.id)
            attached_ocr.add(match.id)

    top_nodes.extend(ocr_parent_nodes.values())
    for element in ui_items:
        if element.id not in attached_ui:
            top_nodes.append(_element_structure_node(element, kind="ui_element"))
    for element in ocr_items:
        if element.id not in attached_ocr:
            top_nodes.append(_element_structure_node(element, kind="ocr_text"))

    for node in top_nodes:
        node["children"] = sorted(node.get("children", []), key=lambda item: (item["bbox"][1], item["bbox"][0], item["id"]))
    return sorted(top_nodes, key=lambda item: (item["bbox"][1], item["bbox"][0], item["id"]))


def _best_ocr_match(element: UIElement, ocr_items: Sequence[UIElement]) -> Optional[UIElement]:
    candidates: List[Tuple[float, float, UIElement]] = []
    for ocr in ocr_items:
        if not bbox_intersects(element.bbox, ocr.bbox, padding=3):
            continue
        overlap = max(_intersection_ratio(element.bbox, ocr.bbox), _intersection_ratio(ocr.bbox, element.bbox))
        containment_bonus = 0.25 if _bbox_contains(ocr.bbox, element.bbox, 0.08) or _bbox_contains(element.bbox, ocr.bbox, 0.08) else 0.0
        candidates.append((overlap + containment_bonus, abs(bbox_area(element.bbox) - bbox_area(ocr.bbox)), ocr))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1], item[2].id))
    return candidates[0][2]


def _element_structure_node(element: UIElement, *, kind: str) -> Dict[str, Any]:
    text = re.sub(r"\s+", " ", str(element.text or element.caption or "").strip())
    return {
        "id": element.id,
        "kind": kind,
        "type": element.type,
        "bbox": list(element.bbox),
        "text": text,
        "interactive": bool(element.interactive),
        "confidence": float(element.confidence),
        "source": element.source,
        "children": [],
    }


def _infer_interface_type(
    elements: Sequence[UIElement],
    layout_groups: Sequence[LayoutGroup],
    by_region: Dict[str, List[UIElement]],
) -> str:
    all_text = " ".join((item.text or item.caption or item.type or "") for item in elements).lower()
    type_counts = Counter(_normal_element_type(item) for item in elements)
    labels = {str(group.label or "") for group in layout_groups}
    main_elements: List[UIElement] = []
    for group in layout_groups:
        if group.label in {"main_content", "content", "content_panel", "form_region"}:
            main_elements.extend(by_region.get(group.id, []))
    main_type_counts = Counter(_normal_element_type(item) for item in main_elements)
    input_count = type_counts["input"] + type_counts["textbox"] + type_counts["textfield"]
    main_input_count = main_type_counts["input"] + main_type_counts["textbox"] + main_type_counts["textfield"]

    if any(term in all_text for term in ("chat", "message", "prompt", "send", "assistant", "bot")) and input_count:
        return "a chatbot or messaging interface"
    if main_input_count >= 2 and (type_counts["button"] or "send" in all_text):
        return "a form-heavy or chatbot-style workspace"
    if type_counts["table"] or any(term in all_text for term in ("row", "column", "filter", "sort", "export")):
        return "a dashboard or data table"
    if any(term in all_text for term in ("def ", "class ", "import ", "return ", "const ", "function ", "{", "}")):
        return "a code or notebook workspace"
    if {"left_sidebar", "right_sidebar"} & labels and type_counts["button"] >= 3:
        return "a structured web application"
    return "an application interface"


def _describe_region(group: LayoutGroup, elements: Sequence[UIElement]) -> str:
    label = _human_region_label(group.label)
    if not elements:
        return f"The {label} is present but has no readable controls."
    counts = Counter(_normal_element_type(item) for item in elements)
    highlights = _collect_element_highlights(elements, limit=3)
    count_text = ", ".join(f"{count} {kind}" for kind, count in counts.most_common(3) if kind)
    text = f"The {label} contains {count_text or str(len(elements)) + ' elements'}"
    if highlights:
        text += f", including {', '.join(highlights)}"
    return text + "."


def _collect_element_highlights(elements: Sequence[UIElement], limit: int = 5) -> List[str]:
    highlights: List[str] = []
    seen: set[str] = set()
    for item in sorted(elements, key=_reading_order_key):
        text = str(item.text or item.caption or "").strip()
        if not text and item.interactive:
            text = _human_region_label(item.type)
        text = re.sub(r"\s+", " ", text).strip(" -:\n\t")
        if not text or len(text) > 80 or _looks_like_noise(text):
            continue
        normalized = text.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        highlights.append(text)
        if len(highlights) >= limit:
            break
    return highlights


def _reading_text_from_elements(elements: Sequence[UIElement], *, source_prefix: str) -> str:
    prefix = source_prefix.lower()
    lines = [
        re.sub(r"\s+", " ", str(item.text or "").strip())
        for item in sorted(elements, key=_reading_order_key)
        if str(item.source or "").lower().startswith(prefix) and str(item.text or "").strip()
    ]
    deduped = list(dict.fromkeys(line for line in lines if line and not _looks_like_noise(line)))
    return "\n".join(deduped)


def _ocr_lines_from_elements(elements: Sequence[UIElement]) -> List[Dict[str, Any]]:
    return [
        {
            "text": item.text,
            "box": list(item.bbox),
            "confidence": item.confidence,
            "source": item.source,
        }
        for item in sorted(elements, key=_reading_order_key)
        if _is_ocr_element(item) and str(item.text or "").strip()
    ]


def _normal_element_type(item: UIElement) -> str:
    raw = str(item.type or "").strip().lower().replace("-", "_")
    text = str(item.text or item.caption or "").strip().lower()
    if _is_doclayout_element(item):
        return raw or "layout_block"
    if raw in {"input", "input_box", "text_input", "textfield", "textbox"}:
        return "input"
    if raw in {"icon_button", "button"} or (item.interactive and any(term in text for term in ("submit", "save", "send", "next", "back"))):
        return "button"
    if raw in {"text", "label", "paragraph"}:
        return "ocr_text" if _is_ocr_element(item) else "text"
    if raw in {"table", "grid"}:
        return "table"
    if raw in {"select", "dropdown", "combobox"}:
        return "dropdown"
    return raw or "element"


def _human_region_label(value: str) -> str:
    text = str(value or "").replace("_", " ").strip()
    aliases = {
        "main content": "main content area",
        "left sidebar": "left sidebar",
        "right sidebar": "right side panel",
        "top navbar": "top navigation",
    }
    return aliases.get(text, text)


def _cursor_action_text(cursor_target: Dict[str, Any], change_type: str) -> str:
    if not isinstance(cursor_target, dict):
        return ""
    target = str(cursor_target.get("element_text") or cursor_target.get("element_type") or "").strip()
    region = str(cursor_target.get("parent_macro_region_type") or "").replace("_", " ").strip()
    relation = str(cursor_target.get("relation") or "").replace("_", " ").strip()
    change = str(change_type or "").replace("_", " ").strip()
    if target:
        if relation in {"hovering over", "focused on"}:
            sentence = f"The cursor is {relation} {target}"
        elif relation:
            sentence = f"The cursor is {relation} on {target}"
        else:
            sentence = f"The cursor is focused on {target}"
        if region:
            sentence += f" in the {region}"
    elif region:
        sentence = f"The cursor is inside the {region}"
    else:
        return ""
    if change and change not in {"none", "initial"}:
        sentence += f" while a {change} is detected"
    return sentence + "."


def _is_ocr_element(item: UIElement) -> bool:
    source = str(item.source or "").lower()
    return source.startswith("ocr") or str(item.id or "").startswith("demo_ocr")


def _is_doclayout_element(item: UIElement) -> bool:
    return str(item.source or "").lower().startswith("doclayout")


def _reading_order_key(item: UIElement) -> Tuple[int, int, str]:
    x1, y1, _, _ = item.bbox
    return int(y1), int(x1), str(item.id)


def _intersection_ratio(inner: BBox, outer: BBox) -> float:
    ix1 = max(int(inner[0]), int(outer[0]))
    iy1 = max(int(inner[1]), int(outer[1]))
    ix2 = min(int(inner[2]), int(outer[2]))
    iy2 = min(int(inner[3]), int(outer[3]))
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return float(inter / max(1, bbox_area(inner)))


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

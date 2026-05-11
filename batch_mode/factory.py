from __future__ import annotations

from dataclasses import dataclass

from .config import BatchModeSettings
from .cursor_adapter import CursorMotionAnalyzer
from .cursor_backend import CursorMotionAnalysisBackend
from .doclayout_adapter import DocLayoutYOLOAdapter
from .macro_layout import MacroLayoutDetector
from .mode_router import ModeRouter
from .ocr_adapter import OCRTextExtractor
from .omniparser_adapter import OmniParserAdapter
from .pipeline import DemoAnalyzerPipeline, IncrementalUIStatePipeline
from .siglip_adapter import SiglipScreenClassifier
from .slide_adapter import SuryaSlideAnalyzer
from .stt_adapter import DeepgramSTTAdapter, UtteranceStore


@dataclass
class RuntimeComponents:
    settings: BatchModeSettings
    siglip: SiglipScreenClassifier
    omniparser: OmniParserAdapter
    cursor_backend: CursorMotionAnalysisBackend
    macro_layout: MacroLayoutDetector
    doclayout: DocLayoutYOLOAdapter
    ocr: OCRTextExtractor
    slide: SuryaSlideAnalyzer
    router: ModeRouter
    warmed: bool = False

    def warmup(self) -> None:
        self.siglip.initialize()
        self.cursor_backend.warmup()
        self.macro_layout.warmup()
        self.doclayout.warmup()
        self.ocr.warmup()
        self.slide.warmup()
        self.omniparser.warmup()
        self.warmed = True


def build_runtime_components(settings: BatchModeSettings | None = None) -> RuntimeComponents:
    settings = settings or BatchModeSettings.from_env()
    settings.validate()
    cursor_backend = CursorMotionAnalysisBackend(
        settings.cursor_model_path,
        model_url=settings.cursor_model_url,
    )
    macro_layout = MacroLayoutDetector(
        model_ref=settings.macro_layout_model_ref,
        model_filename=settings.macro_layout_model_filename,
        conf=settings.macro_layout_conf,
        iou=settings.macro_layout_iou,
    )
    doclayout = DocLayoutYOLOAdapter(
        model_ref=settings.doclayout_model_ref,
        repo_id=settings.doclayout_repo_id,
        filename=settings.doclayout_filename,
        conf=settings.doclayout_conf,
        iou=settings.doclayout_iou,
        imgsz=settings.doclayout_imgsz,
        device=settings.doclayout_device,
        enabled=settings.doclayout_enabled,
    )
    ocr = OCRTextExtractor(requested_mode=settings.ocr_mode, ocr_lang=settings.ocr_lang)
    slide = SuryaSlideAnalyzer(ocr, enabled=settings.surya_enabled)
    router = ModeRouter(
        immediate_switch_confidence=settings.route_immediate_confidence,
        sticky_confidence_floor=settings.route_sticky_floor,
        demo_guard_margin=settings.route_demo_guard_margin,
        required_consecutive_frames=settings.route_required_consecutive,
    )
    omniparser = OmniParserAdapter(
        hf_token=settings.hf_api_key,
        repo_id=settings.omniparser_repo_id,
        cache_dir=settings.omniparser_weights_dir,
        bbox_threshold=settings.omniparser_bbox_threshold,
        iou_threshold=settings.omniparser_iou_threshold,
        imgsz=settings.omniparser_imgsz,
    )
    siglip = SiglipScreenClassifier()
    siglip.initialize()
    return RuntimeComponents(
        settings=settings,
        siglip=siglip,
        omniparser=omniparser,
        cursor_backend=cursor_backend,
        macro_layout=macro_layout,
        doclayout=doclayout,
        ocr=ocr,
        slide=slide,
        router=router,
    )


def build_pipeline(settings: BatchModeSettings | None = None) -> IncrementalUIStatePipeline:
    runtime = build_runtime_components(settings)
    cursor_analyzer = CursorMotionAnalyzer(backend=runtime.cursor_backend)
    stt_store = UtteranceStore()
    return DemoAnalyzerPipeline(
        siglip_classifier=runtime.siglip,
        stt_store=stt_store,
        omniparser=runtime.omniparser,
        cursor_analyzer=cursor_analyzer,
        macro_layout_detector=runtime.macro_layout,
        doclayout_detector=runtime.doclayout,
        ocr_extractor=runtime.ocr,
        slide_analyzer=runtime.slide,
        mode_router=runtime.router,
        code_interval_sec=runtime.settings.code_interval_sec,
        slide_interval_sec=runtime.settings.slide_interval_sec,
        demo_interval_sec=runtime.settings.demo_interval_sec,
        other_interval_sec=runtime.settings.other_interval_sec,
        code_change_tiny_ratio=runtime.settings.code_change_tiny_ratio,
        slide_change_tiny_ratio=runtime.settings.slide_change_tiny_ratio,
        demo_change_tiny_ratio=runtime.settings.demo_change_tiny_ratio,
        code_change_ssim_threshold=runtime.settings.code_change_ssim_threshold,
        slide_change_ssim_threshold=runtime.settings.slide_change_ssim_threshold,
        demo_change_ssim_threshold=runtime.settings.demo_change_ssim_threshold,
        pixel_threshold=runtime.settings.change_pixel_threshold,
        change_padding_px=runtime.settings.change_padding_px,
        macro_boundary_overlap=runtime.settings.macro_boundary_overlap,
        macro_global_change_ratio=runtime.settings.macro_global_change_ratio,
        transcript_window_sec=runtime.settings.transcript_window_sec,
        cursor_mask_padding_px=runtime.settings.cursor_mask_padding_px,
        debug_black_cursor_mask=runtime.settings.debug_black_cursor_mask,
    )


def build_stt_adapter(settings: BatchModeSettings | None = None) -> DeepgramSTTAdapter:
    settings = settings or BatchModeSettings.from_env()
    settings.validate()
    return DeepgramSTTAdapter(api_key=settings.deepgram_api_key, model=settings.deepgram_model)

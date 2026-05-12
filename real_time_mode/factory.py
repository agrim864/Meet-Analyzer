from __future__ import annotations

from dataclasses import dataclass

from .config import NewRealtimeSettings
from .cursor_adapter import CursorMotionAnalyzer
from .cursor_backend import CursorMotionAnalysisBackend
from .doclayout_adapter import DocLayoutYOLOAdapter
from .frame_summarizer import (
    BaseFrameSummarizer,
    CascadingFrameSummarizer,
    GeminiFrameSummarizer,
    HeuristicFrameSummarizer,
    LocalTransformersFrameSummarizer,
    OpenAICompatibleFrameSummarizer,
    ResilientFrameSummarizer,
)
from .macro_layout import MacroLayoutDetector
from .mode_router import ModeRouter
from .ocr_adapter import OCRTextExtractor
from .pipeline import DemoAnalyzerPipeline, IncrementalUIStatePipeline
from .siglip_adapter import SiglipScreenClassifier
from .slide_adapter import SuryaSlideAnalyzer
from .stt_adapter import DeepgramSTTAdapter, UtteranceStore


@dataclass
class RuntimeComponents:
    settings: NewRealtimeSettings
    siglip: SiglipScreenClassifier
    cursor_backend: CursorMotionAnalysisBackend
    macro_layout: MacroLayoutDetector
    doclayout: DocLayoutYOLOAdapter
    ocr: OCRTextExtractor
    slide: SuryaSlideAnalyzer
    router: ModeRouter
    frame_summarizer: ResilientFrameSummarizer
    warmed: bool = False

    def warmup(self) -> None:
        self.siglip.initialize()
        self.cursor_backend.warmup()
        self.macro_layout.warmup()
        self.doclayout.warmup()
        self.ocr.warmup()
        self.slide.warmup()
        self.frame_summarizer.warmup()
        self.warmed = True


def build_runtime_components(settings: NewRealtimeSettings | None = None) -> RuntimeComponents:
    settings = settings or NewRealtimeSettings.from_env()
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
    frame_summarizer = build_frame_summarizer(settings)
    siglip = SiglipScreenClassifier()
    siglip.initialize()
    return RuntimeComponents(
        settings=settings,
        siglip=siglip,
        cursor_backend=cursor_backend,
        macro_layout=macro_layout,
        doclayout=doclayout,
        ocr=ocr,
        slide=slide,
        router=router,
        frame_summarizer=frame_summarizer,
    )


def build_pipeline(settings: NewRealtimeSettings | None = None) -> IncrementalUIStatePipeline:
    runtime = build_runtime_components(settings)
    cursor_analyzer = CursorMotionAnalyzer(backend=runtime.cursor_backend)
    stt_store = UtteranceStore()
    return DemoAnalyzerPipeline(
        siglip_classifier=runtime.siglip,
        stt_store=stt_store,
        cursor_analyzer=cursor_analyzer,
        macro_layout_detector=runtime.macro_layout,
        doclayout_detector=runtime.doclayout,
        ocr_extractor=runtime.ocr,
        slide_analyzer=runtime.slide,
        mode_router=runtime.router,
        frame_summarizer=runtime.frame_summarizer,
        summary_memory_size=runtime.settings.frame_summary_memory_size,
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


def build_stt_adapter(settings: NewRealtimeSettings | None = None) -> DeepgramSTTAdapter:
    settings = settings or NewRealtimeSettings.from_env()
    settings.validate()
    return DeepgramSTTAdapter(api_key=settings.deepgram_api_key, model=settings.deepgram_model)


def build_frame_summarizer(settings: NewRealtimeSettings | None = None) -> ResilientFrameSummarizer:
    settings = settings or NewRealtimeSettings.from_env()
    if settings.frame_summarizer_enabled:
        api_fallbacks: list[BaseFrameSummarizer] = []
        if settings.groq_api_key:
            groq_models = [
                settings.groq_model,
                *[item.strip() for item in settings.groq_model_fallbacks.split(",") if item.strip()],
            ]
            for model_name in dict.fromkeys(groq_models):
                api_fallbacks.append(
                    OpenAICompatibleFrameSummarizer(
                        provider_name="groq",
                        api_key=settings.groq_api_key,
                        model_name=model_name,
                        endpoint="https://api.groq.com/openai/v1/chat/completions",
                        max_output_tokens=settings.frame_summarizer_max_new_tokens,
                        temperature=settings.frame_summarizer_temperature,
                    )
                )
        if settings.openrouter_api_key:
            openrouter_models = [
                settings.openrouter_model,
                *[item.strip() for item in settings.openrouter_model_fallbacks.split(",") if item.strip()],
            ]
            for model_name in dict.fromkeys(openrouter_models):
                api_fallbacks.append(
                    OpenAICompatibleFrameSummarizer(
                        provider_name="openrouter",
                        api_key=settings.openrouter_api_key,
                        model_name=model_name,
                        endpoint="https://openrouter.ai/api/v1/chat/completions",
                        max_output_tokens=settings.frame_summarizer_max_new_tokens,
                        temperature=settings.frame_summarizer_temperature,
                        extra_headers={
                            "HTTP-Referer": "http://localhost",
                            "X-Title": "meet-agent realtime",
                        },
                    )
                )
        local_enabled = settings.frame_summarizer_device.strip().lower() not in {"0", "false", "no", "none", "disabled", "off"}
        local_fallback: BaseFrameSummarizer = (
            LocalTransformersFrameSummarizer(
                model_name=settings.frame_summarizer_fallback_model,
                device=settings.frame_summarizer_device,
                max_new_tokens=settings.frame_summarizer_max_new_tokens,
                temperature=settings.frame_summarizer_temperature,
            )
            if local_enabled
            else HeuristicFrameSummarizer()
        )
        fallback_summarizer: BaseFrameSummarizer = CascadingFrameSummarizer([*api_fallbacks, local_fallback])
        if settings.frame_summarizer_provider == "gemini" and settings.gemini_api_key:
            primary_summarizer: BaseFrameSummarizer = GeminiFrameSummarizer(
                api_key=settings.gemini_api_key,
                model_name=settings.frame_summarizer_model,
                max_output_tokens=settings.frame_summarizer_max_new_tokens,
                temperature=settings.frame_summarizer_temperature,
            )
        elif settings.frame_summarizer_provider == "gemini":
            primary_summarizer = fallback_summarizer
            fallback_summarizer = HeuristicFrameSummarizer()
        else:
            primary_summarizer = CascadingFrameSummarizer([*api_fallbacks, local_fallback])
            fallback_summarizer = HeuristicFrameSummarizer()
        return ResilientFrameSummarizer(
            primary_summarizer,
            fallback=fallback_summarizer,
        )
    return ResilientFrameSummarizer(HeuristicFrameSummarizer(), fallback=HeuristicFrameSummarizer())

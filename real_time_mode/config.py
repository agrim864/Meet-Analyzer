from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def _load_env_chain() -> None:
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)


@dataclass
class NewRealtimeSettings:
    siglip_model_name: str
    deepgram_api_key: str
    deepgram_model: str
    hf_api_key: str
    ocr_mode: str
    ocr_lang: str
    surya_enabled: bool
    cursor_model_url: str
    omniparser_repo_dir: str
    omniparser_repo_id: str
    omniparser_weights_dir: str
    omniparser_bbox_threshold: float
    omniparser_iou_threshold: float
    omniparser_imgsz: int
    macro_layout_model_ref: str
    macro_layout_model_filename: str
    macro_layout_conf: float
    macro_layout_iou: float
    change_tiny_ratio: float
    change_pixel_threshold: int
    change_ssim_threshold: float
    change_padding_px: int
    macro_boundary_overlap: float
    macro_global_change_ratio: float
    route_immediate_confidence: float
    route_sticky_floor: float
    route_demo_guard_margin: float
    route_required_consecutive: int
    transcript_window_sec: float
    code_interval_sec: float
    slide_interval_sec: float
    demo_interval_sec: float
    other_interval_sec: float
    code_change_tiny_ratio: float
    slide_change_tiny_ratio: float
    demo_change_tiny_ratio: float
    code_change_ssim_threshold: float
    slide_change_ssim_threshold: float
    demo_change_ssim_threshold: float
    cursor_mask_padding_px: int
    debug_black_cursor_mask: bool
    cursor_model_path: str
    run_sample_fps: float
    run_max_frames: int
    frame_summarizer_enabled: bool
    frame_summarizer_provider: str
    frame_summarizer_model: str
    frame_summarizer_device: str
    frame_summarizer_max_new_tokens: int
    frame_summarizer_temperature: float
    frame_summary_memory_size: int
    gemini_api_key: str

    @classmethod
    def from_env(cls) -> "NewRealtimeSettings":
        _load_env_chain()
        base_dir = Path(__file__).resolve().parent
        vendor_repo = base_dir / "vendor" / "OmniParser"
        macro_layout_weights = base_dir / "vendor" / "uitag" / "yolo-ui.pt"
        return cls(
            siglip_model_name=os.getenv("SIGLIP_MODEL_NAME", "google/siglip-base-patch16-224").strip(),
            deepgram_api_key=os.getenv("DEEPGRAM_API_KEY", "").strip(),
            deepgram_model=os.getenv("DEEPGRAM_MODEL", "nova-3").strip(),
            hf_api_key=os.getenv("HF_API_KEY", "").strip(),
            ocr_mode=os.getenv("OCR_MODE", "gpu").strip(),
            ocr_lang=os.getenv("OCR_LANG", "en").strip(),
            surya_enabled=os.getenv("SURYA_ENABLED", "1").strip().lower() not in {"0", "false", "no"},
            cursor_model_url=os.getenv(
                "CURSOR_MODEL_URL",
                "https://huggingface.co/datasets/roboflow/cursor-detection/resolve/main/yolov8n_cursor.pt",
            ).strip(),
            omniparser_repo_dir=os.getenv("OMNIPARSER_REPO_DIR", str(vendor_repo)).strip(),
            omniparser_repo_id=os.getenv("OMNIPARSER_LOCAL_REPO_ID", "microsoft/OmniParser-v2.0").strip(),
            omniparser_weights_dir=os.getenv("OMNIPARSER_WEIGHTS_DIR", str(vendor_repo / "weights")).strip(),
            omniparser_bbox_threshold=float(os.getenv("OMNIPARSER_BBOX_THRESHOLD", os.getenv("OMNIPARSER_BOX_THRESHOLD", "0.05"))),
            omniparser_iou_threshold=float(os.getenv("OMNIPARSER_IOU_THRESHOLD", "0.10")),
            omniparser_imgsz=int(os.getenv("OMNIPARSER_IMGSZ", "512")),
            macro_layout_model_ref=os.getenv("MACRO_LAYOUT_MODEL_REF", str(macro_layout_weights)).strip(),
            macro_layout_model_filename=os.getenv("MACRO_LAYOUT_MODEL_FILENAME", "yolo-ui.pt").strip(),
            macro_layout_conf=float(os.getenv("MACRO_LAYOUT_CONF", "0.20")),
            macro_layout_iou=float(os.getenv("MACRO_LAYOUT_IOU", "0.45")),
            change_tiny_ratio=float(os.getenv("CHANGE_TINY_RATIO", "0.01")),
            change_pixel_threshold=int(os.getenv("CHANGE_PIXEL_THRESHOLD", "18")),
            change_ssim_threshold=float(os.getenv("CHANGE_SSIM_THRESHOLD", "0.995")),
            change_padding_px=int(os.getenv("CHANGE_PADDING_PX", "10")),
            macro_boundary_overlap=float(os.getenv("MACRO_BOUNDARY_OVERLAP", "0.05")),
            macro_global_change_ratio=float(os.getenv("MACRO_GLOBAL_CHANGE_RATIO", "0.35")),
            route_immediate_confidence=float(os.getenv("ROUTE_IMMEDIATE_CONFIDENCE", "0.82")),
            route_sticky_floor=float(os.getenv("ROUTE_STICKY_FLOOR", "0.48")),
            route_demo_guard_margin=float(os.getenv("ROUTE_DEMO_GUARD_MARGIN", "0.06")),
            route_required_consecutive=int(os.getenv("ROUTE_REQUIRED_CONSECUTIVE", "2")),
            transcript_window_sec=float(os.getenv("TRANSCRIPT_WINDOW_SEC", "8.0")),
            code_interval_sec=float(os.getenv("CODE_INTERVAL_SEC", "2.0")),
            slide_interval_sec=float(os.getenv("SLIDE_INTERVAL_SEC", "3.0")),
            demo_interval_sec=float(os.getenv("DEMO_INTERVAL_SEC", "1.0")),
            other_interval_sec=float(os.getenv("OTHER_INTERVAL_SEC", "5.0")),
            code_change_tiny_ratio=float(os.getenv("CODE_CHANGE_TINY_RATIO", "0.0004")),
            slide_change_tiny_ratio=float(os.getenv("SLIDE_CHANGE_TINY_RATIO", "0.015")),
            demo_change_tiny_ratio=float(os.getenv("DEMO_CHANGE_TINY_RATIO", "0.0010")),
            code_change_ssim_threshold=float(os.getenv("CODE_CHANGE_SSIM_THRESHOLD", "0.9985")),
            slide_change_ssim_threshold=float(os.getenv("SLIDE_CHANGE_SSIM_THRESHOLD", "0.9920")),
            demo_change_ssim_threshold=float(os.getenv("DEMO_CHANGE_SSIM_THRESHOLD", "0.9965")),
            cursor_mask_padding_px=int(os.getenv("CURSOR_MASK_PADDING_PX", "8")),
            debug_black_cursor_mask=os.getenv("DEBUG_BLACK_CURSOR_MASK", "0").strip().lower() in {"1", "true", "yes"},
            cursor_model_path=os.getenv("CURSOR_MODEL_PATH", str(base_dir / "cursor_class_best.pt")).strip(),
            run_sample_fps=float(os.getenv("NEW_RT_SAMPLE_FPS", "0.5")),
            run_max_frames=int(os.getenv("NEW_RT_MAX_FRAMES", "6")),
            frame_summarizer_enabled=os.getenv("FRAME_SUMMARIZER_ENABLED", "0").strip().lower() in {"1", "true", "yes"},
            frame_summarizer_provider=os.getenv("FRAME_SUMMARIZER_PROVIDER", "gemini").strip().lower() or "gemini",
            frame_summarizer_model=os.getenv("FRAME_SUMMARIZER_MODEL", "gemini-2.5-flash").strip(),
            frame_summarizer_device=os.getenv("FRAME_SUMMARIZER_DEVICE", "auto").strip(),
            frame_summarizer_max_new_tokens=int(os.getenv("FRAME_SUMMARIZER_MAX_NEW_TOKENS", "96")),
            frame_summarizer_temperature=float(os.getenv("FRAME_SUMMARIZER_TEMPERATURE", "0.1")),
            frame_summary_memory_size=int(os.getenv("FRAME_SUMMARY_MEMORY_SIZE", "8")),
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        )

    def validate(self) -> None:
        missing = []
        if not self.deepgram_api_key:
            missing.append("DEEPGRAM_API_KEY")
        if not self.hf_api_key:
            missing.append("HF_API_KEY")
        if missing:
            raise RuntimeError(f"Missing required environment variables for real_time_mode: {', '.join(missing)}")
        if not Path(self.cursor_model_path).exists() and not self.cursor_model_url:
            raise RuntimeError(f"Cursor model weights not found and no CURSOR_MODEL_URL configured: {self.cursor_model_path}")
        if not Path(self.omniparser_repo_dir).exists():
            raise RuntimeError(f"OmniParser repo directory not found: {self.omniparser_repo_dir}")
        if self.frame_summarizer_enabled and self.frame_summarizer_provider == "gemini" and not self.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required when FRAME_SUMMARIZER_PROVIDER=gemini.")

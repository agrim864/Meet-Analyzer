from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def _load_env_chain() -> None:
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent
    for candidate in (
        base_dir / ".env",
        project_root / ".env",
        project_root / "real_time_mode" / ".env",
    ):
        if candidate.exists():
            load_dotenv(candidate, override=False)


def _first_existing(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


@dataclass
class BatchModeSettings:
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
    batch_output_dir: str
    batch_temp_dir: str
    batch_download_timeout_sec: int
    embedding_model_name: str
    embedding_device: str
    embedding_similarity_threshold: float
    topic_similarity_threshold: float
    segment_min_sec: float
    segment_max_sec: float
    segment_target_sec: float
    llm_backend: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_timeout_sec: int
    llm_mock_enabled: bool
    qwen_model_name: str
    cluster_host: str
    cluster_user: str
    cluster_password: str
    cluster_account: str
    cluster_conda_env: str
    cluster_partition: str
    cluster_gres: str
    cluster_time_limit: str
    cluster_memory: str
    cluster_cpus_per_task: int
    cluster_log_dir: str
    cluster_project_dir: str
    cluster_jobs_dir: str

    @classmethod
    def from_env(cls) -> "BatchModeSettings":
        _load_env_chain()
        base_dir = Path(__file__).resolve().parent
        project_root = base_dir.parent
        vendor_repo = base_dir / "vendor" / "OmniParser"
        shared_weights = _first_existing(
            vendor_repo / "weights",
            project_root / "real_time_mode" / "vendor" / "OmniParser" / "weights",
        )
        cursor_weights = _first_existing(
            base_dir / "cursor_class_best.pt",
            project_root / "real_time_mode" / "cursor_class_best.pt",
            project_root / "cursor_class_best.pt",
        )
        macro_layout_weights = _first_existing(
            base_dir / "vendor" / "uitag" / "yolo-ui.pt",
            project_root / "real_time_mode" / "vendor" / "uitag" / "yolo-ui.pt",
        )
        return cls(
            siglip_model_name=os.getenv("SIGLIP_MODEL_NAME", "google/siglip-base-patch16-224").strip(),
            deepgram_api_key=os.getenv("DEEPGRAM_API_KEY", "").strip(),
            deepgram_model=os.getenv("DEEPGRAM_MODEL", "nova-3").strip(),
            hf_api_key=os.getenv("HF_API_KEY", "").strip(),
            ocr_mode=os.getenv("OCR_MODE", "cpu").strip(),
            ocr_lang=os.getenv("OCR_LANG", "en").strip(),
            surya_enabled=os.getenv("SURYA_ENABLED", "1").strip().lower() not in {"0", "false", "no"},
            cursor_model_url=os.getenv(
                "CURSOR_MODEL_URL",
                "",
            ).strip(),
            omniparser_repo_dir=os.getenv("OMNIPARSER_REPO_DIR", str(vendor_repo)).strip(),
            omniparser_repo_id=os.getenv("OMNIPARSER_LOCAL_REPO_ID", "microsoft/OmniParser-v2.0").strip(),
            omniparser_weights_dir=os.getenv("OMNIPARSER_WEIGHTS_DIR", str(shared_weights)).strip(),
            omniparser_bbox_threshold=float(os.getenv("OMNIPARSER_BBOX_THRESHOLD", os.getenv("OMNIPARSER_BOX_THRESHOLD", "0.05"))),
            omniparser_iou_threshold=float(os.getenv("OMNIPARSER_IOU_THRESHOLD", "0.10")),
            omniparser_imgsz=int(os.getenv("OMNIPARSER_IMGSZ", "640")),
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
            cursor_model_path=os.getenv("CURSOR_MODEL_PATH", str(cursor_weights)).strip(),
            run_sample_fps=float(os.getenv("BATCH_SAMPLE_FPS", "0.5")),
            run_max_frames=int(os.getenv("BATCH_MAX_FRAMES", "6")),
            batch_output_dir=os.getenv("BATCH_OUTPUT_DIR", str(base_dir / "outputs")).strip(),
            batch_temp_dir=os.getenv("BATCH_TEMP_DIR", str(base_dir / "tmp")).strip(),
            batch_download_timeout_sec=int(os.getenv("BATCH_DOWNLOAD_TIMEOUT_SEC", "600")),
            embedding_model_name=os.getenv("BATCH_EMBEDDING_MODEL", "BAAI/bge-m3").strip(),
            embedding_device=os.getenv("BATCH_EMBEDDING_DEVICE", "cuda").strip(),
            embedding_similarity_threshold=float(os.getenv("BATCH_EMBEDDING_SIMILARITY_THRESHOLD", "0.63")),
            topic_similarity_threshold=float(os.getenv("BATCH_TOPIC_SIMILARITY_THRESHOLD", "0.69")),
            segment_min_sec=float(os.getenv("BATCH_SEGMENT_MIN_SEC", "10.0")),
            segment_max_sec=float(os.getenv("BATCH_SEGMENT_MAX_SEC", "30.0")),
            segment_target_sec=float(os.getenv("BATCH_SEGMENT_TARGET_SEC", "18.0")),
            llm_backend=os.getenv("BATCH_LLM_BACKEND", "mock").strip().lower(),
            llm_base_url=os.getenv("BATCH_LLM_BASE_URL", "").strip(),
            llm_api_key=os.getenv("BATCH_LLM_API_KEY", "").strip(),
            llm_model=os.getenv("BATCH_LLM_MODEL", "unsloth/Mistral-Small-24B-Instruct-2501-unsloth-bnb-4bit").strip(),
            llm_timeout_sec=int(os.getenv("BATCH_LLM_TIMEOUT_SEC", "180")),
            llm_mock_enabled=os.getenv("BATCH_LLM_MOCK_ENABLED", "1").strip().lower() in {"1", "true", "yes"},
            qwen_model_name=os.getenv("BATCH_QWEN_MODEL_NAME", "Qwen/Qwen3-30B-A3B-FP8").strip(),
            cluster_host=os.getenv("BATCH_CLUSTER_HOST", "").strip(),
            cluster_user=os.getenv("BATCH_CLUSTER_USER", "").strip(),
            cluster_password=os.getenv("BATCH_CLUSTER_PASSWORD", "").strip(),
            cluster_account=os.getenv("BATCH_CLUSTER_ACCOUNT", "").strip(),
            cluster_conda_env=os.getenv("BATCH_CLUSTER_CONDA_ENV", "torch_clean").strip(),
            cluster_partition=os.getenv("BATCH_CLUSTER_PARTITION", "short").strip(),
            cluster_gres=os.getenv("BATCH_CLUSTER_GRES", "gpu:3g.40gb:1").strip(),
            cluster_time_limit=os.getenv("BATCH_CLUSTER_TIME_LIMIT", "02:00:00").strip(),
            cluster_memory=os.getenv("BATCH_CLUSTER_MEMORY", "64G").strip(),
            cluster_cpus_per_task=int(os.getenv("BATCH_CLUSTER_CPUS_PER_TASK", "8")),
            cluster_log_dir=os.getenv("BATCH_CLUSTER_LOG_DIR", "/path/to/Meet-Analyzer/batch_mode/myproject/logs").strip(),
            cluster_project_dir=os.getenv("BATCH_CLUSTER_PROJECT_DIR", "/path/to/Meet-Analyzer").strip(),
            cluster_jobs_dir=os.getenv("BATCH_CLUSTER_JOBS_DIR", str(base_dir / "cluster_jobs")).strip(),
        )

    def validate(self) -> None:
        missing = []
        if not self.deepgram_api_key:
            missing.append("DEEPGRAM_API_KEY")
        if not self.hf_api_key:
            missing.append("HF_API_KEY")
        if missing:
            raise RuntimeError(f"Missing required environment variables for batch_mode: {', '.join(missing)}")
        if not Path(self.cursor_model_path).exists() and not self.cursor_model_url:
            raise RuntimeError(f"Cursor model weights not found and no CURSOR_MODEL_URL configured: {self.cursor_model_path}")
        if not Path(self.omniparser_repo_dir).exists():
            raise RuntimeError(f"OmniParser repo directory not found: {self.omniparser_repo_dir}")
        Path(self.batch_output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.batch_temp_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cluster_jobs_dir).mkdir(parents=True, exist_ok=True)


NewRealtimeSettings = BatchModeSettings

from __future__ import annotations

import base64
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import modal
from pydantic import BaseModel, Field


APP_NAME = "real-time-mode-screen-understanding"
UI_MODEL_REPO = "swaylenhayes/uitag-yolo11s-ui-detect-v1"
UI_MODEL_FILENAME = "yolo-ui.pt"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REALTIME_ROOT = Path(__file__).resolve().parent
REMOTE_PROJECT_ROOT = "/root/project"
REMOTE_REALTIME_ROOT = f"{REMOTE_PROJECT_ROOT}/real_time_mode"
REMOTE_CACHE_ROOT = "/root/modal-cache"


image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    .apt_install("ffmpeg", "libgomp1")
    .pip_install(
        "fastapi==0.116.1",
        "pydantic==2.13.2",
        "python-dotenv==1.2.1",
        "numpy==1.26.4",
        "opencv-python-headless==4.11.0.86",
        "pillow==12.1.0",
        "ultralytics==8.4.12",
        "transformers==4.57.6",
        "huggingface_hub==0.36.2",
        "openai==2.32.0",
        "sentencepiece==0.2.0",
        "einops==0.8.0",
        "timm==1.0.19",
        "easyocr==1.7.2",
        "rapidocr-onnxruntime==1.4.4",
        "scikit-image==0.25.2",
        "deepgram-sdk==4.8.0",
        "dill==0.4.0",
    )
    .env(
        {
            "PYTHONPATH": REMOTE_PROJECT_ROOT,
            "HF_HOME": f"{REMOTE_CACHE_ROOT}/hf",
            "TRANSFORMERS_CACHE": f"{REMOTE_CACHE_ROOT}/hf",
            "OCR_MODE": "cpu",
            "FRAME_SUMMARIZER_ENABLED": "0",
            "CURSOR_MODEL_PATH": f"{REMOTE_REALTIME_ROOT}/cursor_class_best.pt",
            "MACRO_LAYOUT_MODEL_REF": f"{REMOTE_REALTIME_ROOT}/vendor/uitag/{UI_MODEL_FILENAME}",
            "MACRO_LAYOUT_MODEL_FILENAME": UI_MODEL_FILENAME,
            "OMNIPARSER_REPO_DIR": f"{REMOTE_REALTIME_ROOT}/vendor/OmniParser",
            "OMNIPARSER_WEIGHTS_DIR": f"{REMOTE_REALTIME_ROOT}/vendor/OmniParser/weights",
        }
    )
    .add_local_dir(
        str(REALTIME_ROOT),
        remote_path=REMOTE_REALTIME_ROOT,
        copy=False,
        ignore=[
            ".env",
            "__pycache__",
            "**/__pycache__",
            "live_outputs",
            "validation_runs",
            "validation_runs_gpu",
            "cursor_probe",
            "cursor_probe_after_fix2",
            "*.jsonl",
            "*.log",
        ],
    )
)

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("real-time-mode-cache", create_if_missing=True)
secret = modal.Secret.from_name("real-time-mode-secrets")
_shared_manager: PipelineSessionManager | None = None


class TranscriptChunkIn(BaseModel):
    start_time: float
    end_time: float
    speaker: Optional[str] = None
    text: str
    is_final: bool = True
    chunk_id: Optional[str] = None


class ClientChangeIn(BaseModel):
    changed: bool = False
    changed_pixel_ratio: float = 0.0
    changed_bboxes: List[List[int]] = Field(default_factory=list)
    merged_changed_bbox: Optional[List[int]] = None


class ProcessFrameRequest(BaseModel):
    session_id: str
    timestamp: float
    client_mode_hint: str = "unknown"
    frame_jpeg_base64: str
    client_change: ClientChangeIn = Field(default_factory=ClientChangeIn)
    recent_transcript: List[TranscriptChunkIn] = Field(default_factory=list)


class AudioTranscribeRequest(BaseModel):
    session_id: str
    audio_base64: str
    mime_type: str = "audio/webm"
    speaker: Optional[str] = None


class StartSessionRequest(BaseModel):
    session_id: Optional[str] = None
    title: Optional[str] = None


class StopSessionRequest(BaseModel):
    ended_at: Optional[float] = None


@dataclass
class SessionRuntime:
    session_id: str
    title: str
    pipeline: Any
    seen_transcript_keys: Set[Tuple[float, float, str]] = field(default_factory=set)
    frame_lock: Any = field(default_factory=threading.Lock)


class PipelineSessionManager:
    def __init__(self) -> None:
        self._runtime = None
        self._stt_adapter = None
        self._sessions: Dict[str, SessionRuntime] = {}
        self._runtime_lock = threading.Lock()
        self._warmup_thread: Optional[threading.Thread] = None
        self._warmup_error: Optional[str] = None

    def _ensure_runtime(self):
        with self._runtime_lock:
            if self._runtime is None:
                from real_time_mode.cursor_adapter import CursorMotionAnalyzer
                from real_time_mode.factory import build_runtime_components
                from real_time_mode.pipeline import DemoAnalyzerPipeline
                from real_time_mode.stt_adapter import UtteranceStore

                runtime = build_runtime_components()
                self._runtime = (runtime, CursorMotionAnalyzer, DemoAnalyzerPipeline, UtteranceStore)
            return self._runtime

    def _ensure_stt_adapter(self):
        if self._stt_adapter is None:
            from real_time_mode.factory import build_stt_adapter

            self._stt_adapter = build_stt_adapter()
        return self._stt_adapter

    def _warmup_worker(self) -> None:
        try:
            runtime, _, _, _ = self._ensure_runtime()
            if not runtime.warmed:
                runtime.warmup()
            self._ensure_stt_adapter().warmup()
            self._warmup_error = None
        except Exception as exc:
            self._warmup_error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._runtime_lock:
                self._warmup_thread = None

    def warmup(self) -> Dict[str, Any]:
        self._warmup_worker()
        return self.warmup_status()

    def kickoff_warmup(self) -> Dict[str, Any]:
        with self._runtime_lock:
            runtime_ready = bool(self._runtime is not None and self._runtime[0].warmed)
            deepgram_ready = bool(self._stt_adapter is not None and getattr(self._stt_adapter, "_client", None) is not None)
            if runtime_ready and deepgram_ready:
                return self.warmup_status()
            if self._warmup_thread is not None and self._warmup_thread.is_alive():
                return self.warmup_status()
            thread = threading.Thread(target=self._warmup_worker, name="modal-real-time-warmup", daemon=True)
            self._warmup_thread = thread
            thread.start()
        return self.warmup_status()

    def warmup_status(self) -> Dict[str, Any]:
        runtime_ready = bool(self._runtime is not None and self._runtime[0].warmed)
        deepgram_ready = bool(self._stt_adapter is not None and getattr(self._stt_adapter, "_client", None) is not None)
        return {
            "warmed": runtime_ready,
            "running": bool(self._warmup_thread is not None and self._warmup_thread.is_alive()),
            "error": self._warmup_error,
            "deepgram_ready": deepgram_ready,
        }

    def start_session(self, session_id: str, title: Optional[str]) -> SessionRuntime:
        runtime, CursorMotionAnalyzer, DemoAnalyzerPipeline, UtteranceStore = self._ensure_runtime()
        existing = self._sessions.get(session_id)
        if existing is not None:
            return existing
        pipeline = DemoAnalyzerPipeline(
            siglip_classifier=runtime.siglip,
            stt_store=UtteranceStore(),
            omniparser=runtime.omniparser,
            cursor_analyzer=CursorMotionAnalyzer(backend=runtime.cursor_backend),
            macro_layout_detector=runtime.macro_layout,
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
        session = SessionRuntime(
            session_id=session_id,
            title=title or f"Live session {session_id}",
            pipeline=pipeline,
        )
        self._sessions[session_id] = session
        return session

    def stop_session(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def transcribe_audio(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_runtime()
        self.start_session(str(payload.get("session_id", "")).strip(), None)
        audio_b64 = str(payload.get("audio_base64", "") or "")
        try:
            audio_bytes = base64.b64decode(audio_b64.split(",", 1)[-1])
        except Exception as exc:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail=f"Invalid audio payload: {type(exc).__name__}: {exc}") from exc
        utterances = self._ensure_stt_adapter().transcribe_bytes(
            audio_bytes,
            mimetype=str(payload.get("mime_type") or "audio/webm"),
        )
        speaker = str(payload.get("speaker") or "").strip()
        if speaker:
            for item in utterances:
                item["speaker"] = speaker
        return {
            "session_id": str(payload.get("session_id", "")).strip(),
            "chunks": [
                {
                    "chunk_id": f"{payload.get('session_id', '')}-{index}-{round(float(item.get('start', 0.0)), 2)}",
                    "start_time": float(item.get("start", 0.0)),
                    "end_time": float(item.get("end", 0.0)),
                    "speaker": str(item.get("speaker", "speaker_1") or "speaker_1"),
                    "text": str(item.get("text", "") or "").strip(),
                    "is_final": True,
                }
                for index, item in enumerate(utterances)
                if str(item.get("text", "") or "").strip()
            ],
        }

    def process_frame(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from fastapi import HTTPException
        import cv2
        import numpy as np

        runtime, _, _, _ = self._ensure_runtime()
        if not runtime.warmed:
            self.warmup()
        session_id = str(payload.get("session_id", "")).strip()
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        session = self.start_session(session_id, None)
        with session.frame_lock:
            for chunk in list(payload.get("recent_transcript") or []):
                transcript_key = (
                    float(chunk.get("start_time", 0.0)),
                    float(chunk.get("end_time", 0.0)),
                    str(chunk.get("text", "")).strip(),
                )
                if transcript_key in session.seen_transcript_keys:
                    continue
                session.pipeline.ingest_utterance(
                    start=float(chunk.get("start_time", 0.0)),
                    end=float(chunk.get("end_time", 0.0)),
                    speaker=str(chunk.get("speaker") or "unknown"),
                    text=str(chunk.get("text") or ""),
                    source="client",
                )
                session.seen_transcript_keys.add(transcript_key)

            frame_b64 = str(payload.get("frame_jpeg_base64", ""))
            try:
                binary = base64.b64decode(frame_b64.split(",", 1)[-1])
                data = np.frombuffer(binary, dtype=np.uint8)
                frame_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid frame payload: {type(exc).__name__}: {exc}") from exc
            if frame_bgr is None or frame_bgr.size == 0:
                raise HTTPException(status_code=400, detail="Decoded frame is empty.")

            timestamp = float(payload.get("timestamp", 0.0))
            record = session.pipeline.ingest_frame(frame_bgr, timestamp_sec=timestamp)
        route = dict(record.confidence.get("route", {}))
        debug = dict(record.debug)
        visual_payload = dict(record.visual_payload)
        client_change = dict(payload.get("client_change") or {})
        merged_bbox = client_change.get("merged_changed_bbox")
        if merged_bbox and "changed_bbox" not in visual_payload:
            visual_payload["changed_bbox"] = list(merged_bbox)

        return {
            "session_id": session_id,
            "timestamp": timestamp,
            "mode": record.mode,
            "mode_confidence": float(route.get("confidence", 0.0)),
            "decision_path": str(route.get("decision_path", "")),
            "visual_change": bool(record.visual_change),
            "processor_used": str(debug.get("processor_used", "none")),
            "cursor": dict(record.cursor),
            "visual_payload": visual_payload,
            "summary": record.summary,
            "debug": {
                "rfdetr_ran": bool(debug.get("rfdetr_ran", False)),
                "macro_detector_ran": bool(debug.get("rfdetr_ran", False)),
                "omniparser_ran": bool(debug.get("omniparser_ran", False)),
                "surya_ran": bool(debug.get("processor_used") == "surya"),
                "ocr_ran": bool(debug.get("processor_used") == "ocr"),
                "summary_provider": str(debug.get("summary_provider", "heuristic")),
                "summary_model": str(debug.get("summary_model", "heuristic")),
                "processing_time_ms": float(debug.get("processing_time_ms", 0.0)),
                "change_score": float(debug.get("change_score", 0.0)),
                "crop_bbox": debug.get("crop_bbox"),
                "client_change": client_change,
            },
            "raw_record": record.to_dict(),
        }


def get_shared_manager() -> PipelineSessionManager:
    global _shared_manager
    if _shared_manager is None:
        _shared_manager = PipelineSessionManager()
    return _shared_manager


@app.function(
    image=image,
    gpu="L4",
    secrets=[secret],
    volumes={REMOTE_CACHE_ROOT: cache_volume},
    timeout=1800,
    scaledown_window=300,
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware

    api = FastAPI(title="real_time_mode modal api")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    manager = PipelineSessionManager()

    @api.get("/health")
    def health() -> Dict[str, Any]:
        return {"ok": True, "runtime": manager.warmup_status()}

    @api.post("/sessions/start")
    @api.post("/api/sessions/start")
    async def start_session(payload: StartSessionRequest) -> Dict[str, Any]:
        session_id = str(payload.session_id or "").strip()
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        session = manager.start_session(session_id, payload.title)
        return {
            "session_id": session.session_id,
            "title": session.title,
            "status": "active",
            "runtime": manager.warmup_status(),
        }

    @api.post("/sessions/{session_id}/stop")
    @api.post("/api/sessions/{session_id}/stop")
    async def stop_session(session_id: str, payload: StopSessionRequest) -> Dict[str, Any]:
        removed = manager.stop_session(session_id)
        return {
            "session_id": session_id,
            "ended_at": payload.ended_at,
            "status": "stopped" if removed else "not_found",
        }

    @api.post("/warmup")
    @api.post("/api/warmup")
    async def warmup() -> Dict[str, Any]:
        return manager.warmup()

    @api.post("/warmup/start")
    @api.post("/api/warmup/start")
    async def warmup_start() -> Dict[str, Any]:
        return manager.kickoff_warmup()

    @api.get("/warmup/status")
    @api.get("/api/warmup/status")
    async def warmup_status() -> Dict[str, Any]:
        return manager.warmup_status()

    @api.post("/process-frame")
    @api.post("/api/process-frame")
    async def process_frame(payload: ProcessFrameRequest) -> Dict[str, Any]:
        try:
            return manager.process_frame(payload.model_dump())
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Modal frame processing failed: {type(exc).__name__}: {exc}") from exc

    @api.post("/transcribe")
    @api.post("/api/transcribe")
    async def transcribe(payload: AudioTranscribeRequest) -> Dict[str, Any]:
        try:
            return manager.transcribe_audio(payload.model_dump())
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Modal transcript processing failed: {type(exc).__name__}: {exc}") from exc

    return api


@app.function(
    image=image,
    gpu="L4",
    secrets=[secret],
    volumes={REMOTE_CACHE_ROOT: cache_volume},
    timeout=1800,
    scaledown_window=300,
)
@modal.fastapi_endpoint(method="POST", label="process-frame")
def process_frame_endpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
    return get_shared_manager().process_frame(payload)

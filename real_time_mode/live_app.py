from __future__ import annotations

import base64
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from real_time_mode.cursor_adapter import CursorMotionAnalyzer
from real_time_mode.factory import build_frame_summarizer, build_runtime_components, build_stt_adapter
from real_time_mode.frame_summarizer import FrameSummaryRequest
from real_time_mode.pipeline import DemoAnalyzerPipeline
from real_time_mode.stt_adapter import DeepgramSTTAdapter, UtteranceStore


def _load_env() -> None:
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)


def _modal_proxy_url() -> str:
    url = os.getenv("VITE_MODAL_PROCESS_FRAME_URL", "").strip()
    if not url:
        return ""
    if url.startswith("/"):
        return ""
    return url


def _modal_process_frame_url() -> str:
    base = os.getenv("MODAL_API_BASE_URL", "").strip().rstrip("/")
    if base:
        return f"{base}/api/process-frame"
    explicit = os.getenv("MODAL_PROCESS_FRAME_URL", "").strip()
    if explicit:
        return explicit
    return _modal_proxy_url()


def _modal_base_url() -> str:
    base = os.getenv("MODAL_API_BASE_URL", "").strip().rstrip("/")
    if base:
        return base
    frame_url = _modal_proxy_url().rstrip("/")
    if not frame_url:
        return ""
    if frame_url.endswith("/process-frame"):
        return frame_url[: -len("/process-frame")]
    if frame_url.endswith("/api/process-frame"):
        return frame_url[: -len("/api/process-frame")]
    return frame_url


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
    client_frame_id: int = 0
    client_sent_at: Optional[float] = None
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
    pipeline: Optional[DemoAnalyzerPipeline] = None
    seen_transcript_keys: Set[Tuple[float, float, str]] = field(default_factory=set)
    frame_lock: Any = field(default_factory=threading.Lock)
    frame_state_lock: Any = field(default_factory=threading.Lock)
    latest_client_frame_id: int = 0
    latest_completed_frame_id: int = 0


class PipelineSessionManager:
    def __init__(self) -> None:
        self._runtime = None
        self._stt_adapter: Optional[DeepgramSTTAdapter] = None
        self._frame_summarizer = None
        self._sessions: Dict[str, SessionRuntime] = {}
        self._runtime_lock = threading.Lock()
        self._warmup_thread: Optional[threading.Thread] = None
        self._warmup_error: Optional[str] = None
        self._remote_warmup_started_at: Optional[float] = None

    def _ensure_runtime(self, warm: bool = True):
        with self._runtime_lock:
            if self._runtime is None:
                self._runtime = build_runtime_components()
            runtime = self._runtime
        if warm and not runtime.warmed:
            runtime.warmup()
        if self._runtime is None:
            raise RuntimeError("Runtime failed to initialize.")
        return runtime

    def _ensure_stt_adapter(self) -> DeepgramSTTAdapter:
        with self._runtime_lock:
            if self._stt_adapter is None:
                self._stt_adapter = build_stt_adapter()
            return self._stt_adapter

    def _ensure_frame_summarizer(self):
        with self._runtime_lock:
            if self._frame_summarizer is None:
                self._frame_summarizer = build_frame_summarizer()
            return self._frame_summarizer

    def _warmup_worker(self) -> None:
        try:
            self._ensure_runtime(warm=True)
            self._ensure_stt_adapter().warmup()
            self._warmup_error = None
        except Exception as exc:
            self._warmup_error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._runtime_lock:
                self._warmup_thread = None

    def warmup_now(self) -> Dict[str, Any]:
        self._warmup_worker()
        return self.warmup_status()

    def kickoff_warmup(self) -> None:
        with self._runtime_lock:
            if self._runtime is not None and self._runtime.warmed:
                return
            if self._warmup_thread is not None and self._warmup_thread.is_alive():
                return
            thread = threading.Thread(target=self._warmup_worker, name="real-time-warmup", daemon=True)
            self._warmup_thread = thread
            thread.start()

    def warmup_status(self) -> Dict[str, Any]:
        with self._runtime_lock:
            warmed = bool(self._runtime and self._runtime.warmed)
            running = bool(self._warmup_thread and self._warmup_thread.is_alive())
        return {
            "warmed": warmed,
            "running": running,
            "error": self._warmup_error,
            "deepgram_ready": bool(self._stt_adapter is not None and self._stt_adapter._client is not None),
        }

    def note_remote_warmup_started(self) -> None:
        self._remote_warmup_started_at = time.time()

    def clear_remote_warmup_pending(self) -> None:
        self._remote_warmup_started_at = None

    def remote_warmup_pending(self, grace_sec: float = 180.0) -> bool:
        started = self._remote_warmup_started_at
        if started is None:
            return False
        return (time.time() - started) <= float(grace_sec)

    @staticmethod
    def remote_warmup_status_placeholder(remote_error: Optional[str] = None) -> Dict[str, Any]:
        return {
            "warmed": False,
            "running": True,
            "error": None,
            "deepgram_ready": False,
            "pending_remote_cold_start": True,
            "remote_warning": remote_error,
        }

    def _build_session_pipeline(self, runtime: Any) -> DemoAnalyzerPipeline:
        return DemoAnalyzerPipeline(
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

    def start_session(self, session_id: str, title: Optional[str]) -> SessionRuntime:
        existing = self._sessions.get(session_id)
        if existing is not None:
            return existing
        session = SessionRuntime(
            session_id=session_id,
            title=title or f"Live session {session_id}",
        )
        self._sessions[session_id] = session
        return session

    def stop_session(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def note_client_frame(self, session_id: str, client_frame_id: int) -> Tuple[bool, int]:
        session = self.start_session(session_id, None)
        frame_id = int(client_frame_id or 0)
        if frame_id <= 0:
            return True, session.latest_client_frame_id
        with session.frame_state_lock:
            if frame_id < session.latest_client_frame_id:
                return False, session.latest_client_frame_id
            session.latest_client_frame_id = frame_id
            return True, session.latest_client_frame_id

    def is_stale_client_frame(self, session: SessionRuntime, client_frame_id: int) -> Tuple[bool, int]:
        frame_id = int(client_frame_id or 0)
        if frame_id <= 0:
            return False, session.latest_client_frame_id
        with session.frame_state_lock:
            return frame_id < session.latest_client_frame_id, session.latest_client_frame_id

    def mark_client_frame_completed(self, session: SessionRuntime, client_frame_id: int) -> None:
        frame_id = int(client_frame_id or 0)
        if frame_id <= 0:
            return
        with session.frame_state_lock:
            session.latest_completed_frame_id = max(session.latest_completed_frame_id, frame_id)

    def process_frame(self, payload: ProcessFrameRequest) -> Dict[str, Any]:
        runtime = self._ensure_runtime(warm=True)
        session = self.start_session(payload.session_id, None)
        fresh, latest_frame_id = self.note_client_frame(payload.session_id, payload.client_frame_id)
        if not fresh:
            return _stale_frame_response(payload.session_id, payload, latest_frame_id)
        if session.pipeline is None:
            session.pipeline = self._build_session_pipeline(runtime)
        with session.frame_lock:
            stale, latest_frame_id = self.is_stale_client_frame(session, payload.client_frame_id)
            if stale:
                return _stale_frame_response(payload.session_id, payload, latest_frame_id)
            for chunk in payload.recent_transcript:
                transcript_key = (
                    float(chunk.start_time),
                    float(chunk.end_time),
                    str(chunk.text).strip(),
                )
                if transcript_key in session.seen_transcript_keys:
                    continue
                session.pipeline.ingest_utterance(
                    start=float(chunk.start_time),
                    end=float(chunk.end_time),
                    speaker=str(chunk.speaker or "unknown"),
                    text=str(chunk.text or ""),
                    source="client",
                )
                session.seen_transcript_keys.add(transcript_key)

            frame_bgr = _decode_base64_jpeg(payload.frame_jpeg_base64)
            record = session.pipeline.ingest_frame(frame_bgr, timestamp_sec=float(payload.timestamp))
            self.mark_client_frame_completed(session, payload.client_frame_id)
            stale, latest_frame_id = self.is_stale_client_frame(session, payload.client_frame_id)
            if stale:
                return _stale_frame_response(payload.session_id, payload, latest_frame_id)
        route = dict(record.confidence.get("route", {}))
        debug = dict(record.debug)
        stage_timings = dict(debug.get("stage_timings_ms") or {})
        visual_payload = dict(record.visual_payload)
        merged_bbox = payload.client_change.merged_changed_bbox
        if merged_bbox and "changed_bbox" not in visual_payload:
            visual_payload["changed_bbox"] = list(merged_bbox)

        return {
            "session_id": payload.session_id,
            "timestamp": float(payload.timestamp),
            "client_frame_id": int(payload.client_frame_id or 0),
            "client_sent_at": payload.client_sent_at,
            "stale": False,
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
                "stage_timings_ms": stage_timings,
                "change_score": float(debug.get("change_score", 0.0)),
                "crop_bbox": debug.get("crop_bbox"),
                "client_change": payload.client_change.model_dump(),
            },
            "raw_record": record.to_dict(),
        }

    def transcribe_audio(self, payload: AudioTranscribeRequest) -> Dict[str, Any]:
        self.start_session(payload.session_id, None)
        audio_bytes = _decode_base64_bytes(payload.audio_base64)
        utterances = self._ensure_stt_adapter().transcribe_bytes(audio_bytes, mimetype=payload.mime_type)
        if payload.speaker:
            for item in utterances:
                item["speaker"] = str(payload.speaker)
        return {
            "session_id": payload.session_id,
            "chunks": [
                {
                    "chunk_id": f"{payload.session_id}-{index}-{round(float(item.get('start', 0.0)), 2)}",
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


def _proxy_modal_json(method: str, path: str, *, json_payload: Optional[Dict[str, Any]] = None, timeout_sec: float = 120.0) -> Optional[JSONResponse]:
    base = _modal_base_url()
    if not base:
        return None
    url = f"{base}{path}"
    response = requests.request(method=method, url=url, json=json_payload, timeout=timeout_sec)
    response.raise_for_status()
    return JSONResponse(response.json())


def _proxy_modal_dict(method: str, path: str, *, json_payload: Optional[Dict[str, Any]] = None, timeout_sec: float = 120.0) -> Optional[Dict[str, Any]]:
    base = _modal_base_url()
    if not base:
        return None
    url = f"{base}{path}"
    response = requests.request(method=method, url=url, json=json_payload, timeout=timeout_sec)
    response.raise_for_status()
    return response.json()


def _decode_base64_jpeg(frame_b64: str) -> np.ndarray:
    try:
        payload = frame_b64.split(",", 1)[-1]
        binary = base64.b64decode(payload)
        data = np.frombuffer(binary, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid frame payload: {type(exc).__name__}: {exc}") from exc
    if frame is None or frame.size == 0:
        raise HTTPException(status_code=400, detail="Decoded frame is empty.")
    return frame


def _decode_base64_bytes(payload_b64: str) -> bytes:
    try:
        payload = str(payload_b64 or "").split(",", 1)[-1]
        return base64.b64decode(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid audio payload: {type(exc).__name__}: {exc}") from exc


def _encode_jpeg_data_url(frame_bgr: np.ndarray, *, quality: int = 80) -> str:
    success, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not success:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _stale_frame_response(session_id: str, payload: ProcessFrameRequest, latest_frame_id: int) -> Dict[str, Any]:
    return {
        "session_id": session_id,
        "timestamp": float(payload.timestamp),
        "client_frame_id": int(payload.client_frame_id or 0),
        "client_sent_at": payload.client_sent_at,
        "latest_client_frame_id": int(latest_frame_id or 0),
        "stale": True,
        "dropped": True,
        "mode": payload.client_mode_hint or "unknown",
        "mode_confidence": 0.0,
        "decision_path": "stale_client_frame_dropped",
        "visual_change": False,
        "processor_used": "stale_drop",
        "cursor": {},
        "visual_payload": {"mode": payload.client_mode_hint or "unknown", "visual_change": False},
        "summary": "Dropped stale frame.",
        "debug": {
            "processing_time_ms": 0.0,
            "stage_timings_ms": {},
            "client_change": payload.client_change.model_dump(),
            "stale_reason": "newer client frame superseded this response",
        },
        "raw_record": {},
    }


def _draw_labeled_box(frame: np.ndarray, box: Optional[List[int]], color: Tuple[int, int, int], label: str, thickness: int = 2) -> None:
    if not box or len(box) < 4:
        return
    x1, y1, x2, y2 = [int(round(value)) for value in box[:4]]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if not label:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    text_thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, text_thickness)
    text_y = max(y1 - 6, text_h + 6)
    cv2.rectangle(frame, (x1, text_y - text_h - baseline - 4), (x1 + text_w + 6, text_y + baseline), color, -1)
    cv2.putText(frame, label, (x1 + 3, text_y - 2), font, scale, (255, 255, 255), text_thickness, cv2.LINE_AA)


def _safe_label(text: Any, fallback: str) -> str:
    value = str(text or "").strip()
    if not value:
        return fallback
    if len(value) > 28:
        value = value[:25] + "..."
    return value


def _annotate_response_frame(frame_bgr: np.ndarray, response_json: Dict[str, Any]) -> str:
    visual_payload = dict(response_json.get("visual_payload") or {})
    annotated = frame_bgr.copy()
    macro_layout = list(visual_payload.get("macro_layout") or [])
    micro_elements = list(visual_payload.get("micro_elements") or [])
    cursor = dict(response_json.get("cursor") or {})
    max_labeled_micro = 120
    for index, item in enumerate(micro_elements, start=1):
        box = item.get("bbox")
        if not box or len(box) < 4:
            continue
        label = ""
        if index <= max_labeled_micro:
            label = f"u{index}:{_safe_label(item.get('text') or item.get('caption') or item.get('type'), f'e{index}')}"
        _draw_labeled_box(annotated, box, (0, 220, 120), label, thickness=1)
    for index, item in enumerate(macro_layout[:20], start=1):
        confidence = item.get("confidence")
        confidence_suffix = ""
        if isinstance(confidence, (int, float)):
            confidence_suffix = f" {float(confidence):.2f}"
        _draw_labeled_box(
            annotated,
            item.get("bbox"),
            (255, 140, 0),
            f"M{index}:{_safe_label(item.get('type') or item.get('label'), 'region')}{confidence_suffix}",
            thickness=4,
        )
    cursor_box = cursor.get("bbox")
    if cursor_box:
        _draw_labeled_box(
            annotated,
            cursor_box,
            (80, 120, 255),
            f"cursor:{_safe_label(cursor.get('type'), 'pointer')}",
            thickness=2,
        )
        hotspot = cursor.get("hotspot")
        if hotspot and len(hotspot) >= 2:
            hx, hy = int(round(hotspot[0])), int(round(hotspot[1]))
            cv2.circle(annotated, (hx, hy), 5, (80, 120, 255), -1)
    return _encode_jpeg_data_url(annotated, quality=78)


def _transcript_text_from_response(response_json: Dict[str, Any]) -> str:
    transcript_window = (
        ((response_json.get("raw_record") or {}).get("transcript_window"))
        or response_json.get("transcript_window")
        or []
    )
    parts = []
    for item in transcript_window:
        text = str((item or {}).get("text", "") or "").strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def _page_text_from_response(response_json: Dict[str, Any]) -> str:
    payload = dict(response_json.get("visual_payload") or {})
    mode = str(response_json.get("mode") or "")
    if mode == "code":
        return str(payload.get("ocr_text", "") or "")[:600]
    if mode == "slide":
        return str(payload.get("full_text", "") or "")[:800]
    if mode == "demo":
        highlights = []
        for item in list(payload.get("micro_elements") or [])[:20]:
            for key in ("text", "caption", "type"):
                value = str(item.get(key) or "").strip()
                if value:
                    highlights.append(value)
                    break
        prefix = str(payload.get("page_summary", "") or "").strip()
        return f"{prefix} Visible elements: {', '.join(highlights[:10])}".strip()[:900]
    return str(payload.get("summary", "") or "")[:400]


def _action_text_from_response(response_json: Dict[str, Any]) -> str:
    payload = dict(response_json.get("visual_payload") or {})
    change_type = str(payload.get("change_type", "") or "").replace("_", " ").strip()
    cursor_target = dict(payload.get("cursor_target") or {})
    cursor_micro = str(cursor_target.get("element_text") or cursor_target.get("element_type") or "").strip()
    cursor_macro = str(cursor_target.get("parent_macro_region_type") or "").replace("_", " ").strip()
    parts = []
    if change_type:
        parts.append(f"screen change: {change_type}")
    if cursor_micro:
        parts.append(f"cursor over: {cursor_micro}")
    elif cursor_macro:
        parts.append(f"cursor in: {cursor_macro}")
    return "; ".join(parts)


def _cursor_text_from_response(response_json: Dict[str, Any]) -> str:
    cursor = dict(response_json.get("cursor") or {})
    if not bool(cursor.get("detected")):
        return "cursor not visible"
    payload = dict(response_json.get("visual_payload") or {})
    target = dict(payload.get("cursor_target") or {})
    target_text = str(target.get("element_text") or target.get("parent_macro_region_type") or "").replace("_", " ").strip()
    cursor_type = str(cursor.get("type") or "pointer").strip()
    if target_text:
        return f"cursor type={cursor_type}, over {target_text}"
    return f"cursor type={cursor_type}"


def _resummarize_response(manager: PipelineSessionManager, response_json: Dict[str, Any]) -> Dict[str, Any]:
    enabled = os.getenv("LIVE_RESUMMARIZE_RESPONSES", "0").strip().lower() in {"1", "true", "yes"}
    if not enabled:
        response_json.setdefault("debug", {})
        response_json["debug"].setdefault("summary_provider", "remote")
        response_json["debug"].setdefault("summary_model", "remote")
        return response_json
    summarizer = manager._ensure_frame_summarizer()
    request = FrameSummaryRequest(
        timestamp_sec=float(response_json.get("timestamp", 0.0)),
        mode=str(response_json.get("mode", "other") or "other"),
        heuristic_summary=str(response_json.get("summary", "") or ""),
        transcript_text=_transcript_text_from_response(response_json),
        cursor_text=_cursor_text_from_response(response_json),
        page_text=_page_text_from_response(response_json),
        action_text=_action_text_from_response(response_json),
        current_payload=dict(response_json.get("visual_payload") or {}),
        recent_memory=[],
    )
    summary = summarizer.summarize(request)
    response_json["summary"] = summary
    response_json.setdefault("debug", {})
    backend = getattr(summarizer, "primary", summarizer)
    response_json["debug"]["summary_provider"] = getattr(backend, "provider_name", "heuristic")
    response_json["debug"]["summary_model"] = getattr(backend, "model_name", "heuristic")
    last_error = str(getattr(summarizer, "last_error", "") or "").strip()
    if last_error:
        response_json["debug"]["summary_error"] = last_error
    return response_json


_load_env()
app = FastAPI(title="real_time_mode live app")
manager = PipelineSessionManager()
frontend_dir = Path(__file__).resolve().parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(frontend_dir / "index.html")


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/health")
def health() -> Dict[str, Any]:
    modal_base = _modal_base_url()
    runtime = manager.warmup_status()
    if modal_base:
        try:
            modal_response = requests.get(f"{modal_base}/health", timeout=20)
            modal_response.raise_for_status()
            runtime = modal_response.json().get("runtime", runtime)
        except Exception as exc:
            runtime = dict(runtime)
            runtime["error"] = f"Modal health failed: {type(exc).__name__}: {exc}"
    return {"ok": True, "runtime": runtime}


@app.get("/api/supabase/schema")
def supabase_schema() -> Dict[str, Any]:
    schema_path = Path(__file__).resolve().parent / "supabase" / "schema.sql"
    return {
        "path": str(schema_path),
        "sql": schema_path.read_text(encoding="utf-8"),
    }


@app.get("/api/config")
def frontend_config() -> Dict[str, Any]:
    return {
        # Keep the browser on the same origin and let the backend proxy to Modal when configured.
        "modalProcessFrameUrl": "/api/process-frame",
        "modalProxyEnabled": bool(_modal_base_url()),
        "runtime": manager.warmup_status(),
        "supabaseUrl": os.getenv("VITE_SUPABASE_URL", "").strip(),
        "supabaseAnonKey": os.getenv("VITE_SUPABASE_ANON_KEY", "").strip(),
        "deepgramProxyUrl": os.getenv("VITE_DEEPGRAM_PROXY_URL", "").strip(),
        "enableDebugJson": os.getenv("VITE_ENABLE_DEBUG_JSON", "true").strip().lower() in {"1", "true", "yes"},
        "captureMaxSide": int(os.getenv("VITE_CAPTURE_MAX_SIDE", "512")),
        "frameJpegQuality": float(os.getenv("VITE_FRAME_JPEG_QUALITY", "0.46")),
        "maxInFlightFrames": int(os.getenv("VITE_MAX_IN_FLIGHT_FRAMES", "2")),
        "fullFrameChangeRatio": float(os.getenv("VITE_FULL_FRAME_CHANGE_RATIO", "0.55")),
        "frameIntervalsMs": {
            "demo": int(os.getenv("VITE_FRAME_INTERVAL_DEMO_MS", "300")),
            "code": int(os.getenv("VITE_FRAME_INTERVAL_CODE_MS", "450")),
            "slide": int(os.getenv("VITE_FRAME_INTERVAL_SLIDE_MS", "650")),
            "other": int(os.getenv("VITE_FRAME_INTERVAL_OTHER_MS", "1000")),
            "unknown": int(os.getenv("VITE_FRAME_INTERVAL_UNKNOWN_MS", "500")),
        },
    }


@app.post("/api/transcribe")
def transcribe_audio(payload: AudioTranscribeRequest) -> JSONResponse:
    try:
        return JSONResponse(manager.transcribe_audio(payload))
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
        base = _modal_base_url()
        if base:
            try:
                response = requests.post(
                    f"{base}/api/transcribe",
                    json=payload.model_dump(),
                    timeout=float(os.getenv("DEEPGRAM_PROXY_TIMEOUT_SEC", "60")),
                )
                if not response.ok:
                    raise RuntimeError(f"Modal returned {response.status_code}: {response.text}")
                return JSONResponse(response.json())
            except Exception as modal_exc:
                return JSONResponse(
                    {
                        "session_id": payload.session_id,
                        "chunks": [],
                        "transcript_error": (
                            f"Local transcription failed: {local_error}. "
                            f"Modal fallback failed: {type(modal_exc).__name__}: {modal_exc}"
                        ),
                    }
                )
        return JSONResponse(
            {
                "session_id": payload.session_id,
                "chunks": [],
                "transcript_error": f"Local transcription failed: {local_error}",
            }
        )


@app.post("/api/sessions/start")
def start_session(payload: StartSessionRequest) -> Dict[str, Any]:
    session_id = str(payload.session_id or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    session = manager.start_session(session_id, payload.title)
    runtime_status = manager.warmup_status()
    try:
        proxied_status = _proxy_modal_dict("GET", "/api/warmup/status", timeout_sec=20.0)
        if proxied_status is not None:
            runtime_status = proxied_status
    except Exception:
        pass
    return {
        "session_id": session.session_id,
        "title": session.title,
        "status": "active",
        "runtime": runtime_status,
    }


@app.post("/api/sessions/{session_id}/stop")
def stop_session(session_id: str, payload: StopSessionRequest) -> Dict[str, Any]:
    removed = manager.stop_session(session_id)
    return {
        "session_id": session_id,
        "ended_at": payload.ended_at,
        "status": "stopped" if removed else "not_found",
    }


@app.post("/api/warmup")
def warmup() -> Dict[str, Any]:
    try:
        proxied = _proxy_modal_dict("POST", "/api/warmup", timeout_sec=600.0)
        if proxied is not None:
            return proxied
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Modal warmup failed: {type(exc).__name__}: {exc}") from exc
    return manager.warmup_now()


@app.post("/api/warmup/start")
def warmup_start() -> Dict[str, Any]:
    try:
        manager.note_remote_warmup_started()
        proxied = _proxy_modal_dict("POST", "/api/warmup/start", timeout_sec=8.0)
        if proxied is not None:
            if proxied.get("error") and _modal_base_url():
                return manager.remote_warmup_status_placeholder(str(proxied.get("error")))
            if proxied.get("warmed") and proxied.get("deepgram_ready"):
                manager.clear_remote_warmup_pending()
            return proxied
    except Exception as exc:
        if _modal_base_url() and manager.remote_warmup_pending():
            return manager.remote_warmup_status_placeholder(f"{type(exc).__name__}: {exc}")
        raise HTTPException(status_code=502, detail=f"Modal warmup start failed: {type(exc).__name__}: {exc}") from exc
    manager.kickoff_warmup()
    return manager.warmup_status()


@app.get("/api/warmup/status")
def warmup_status() -> Dict[str, Any]:
    try:
        proxied = _proxy_modal_dict("GET", "/api/warmup/status", timeout_sec=5.0)
        if proxied is not None:
            if proxied.get("error") and _modal_base_url():
                return manager.remote_warmup_status_placeholder(str(proxied.get("error")))
            if proxied.get("warmed") and proxied.get("deepgram_ready"):
                manager.clear_remote_warmup_pending()
            return proxied
    except Exception as exc:
        if _modal_base_url() and manager.remote_warmup_pending():
            return manager.remote_warmup_status_placeholder(f"{type(exc).__name__}: {exc}")
        raise HTTPException(status_code=502, detail=f"Modal warmup status failed: {type(exc).__name__}: {exc}") from exc
    return manager.warmup_status()


@app.post("/api/process-frame")
def process_frame(payload: ProcessFrameRequest) -> JSONResponse:
    endpoint_started = time.perf_counter()
    fresh, latest_frame_id = manager.note_client_frame(payload.session_id, payload.client_frame_id)
    if not fresh:
        return JSONResponse(_stale_frame_response(payload.session_id, payload, latest_frame_id))
    process_url = _modal_process_frame_url()
    if process_url:
        try:
            proxy_started = time.perf_counter()
            response = requests.post(
                process_url,
                json=payload.model_dump(),
                timeout=float(os.getenv("MODAL_PROXY_TIMEOUT_SEC", "120")),
            )
            proxy_ms = (time.perf_counter() - proxy_started) * 1000.0
            if not response.ok:
                raise RuntimeError(f"Modal returned {response.status_code}: {response.text}")
            response_json = response.json()
            if response_json.get("stale") or response_json.get("dropped"):
                return JSONResponse(response_json)
            response_json = _resummarize_response(manager, response_json)
            annotation_started = time.perf_counter()
            response_json["annotated_image_data_url"] = _annotate_response_frame(
                _decode_base64_jpeg(payload.frame_jpeg_base64),
                response_json,
            )
            response_json.setdefault("debug", {})
            response_json["debug"].setdefault("stage_timings_ms", {})
            response_json["debug"]["stage_timings_ms"].update(
                {
                    "modal_proxy_ms": proxy_ms,
                    "annotation_ms": (time.perf_counter() - annotation_started) * 1000.0,
                    "endpoint_total_ms": (time.perf_counter() - endpoint_started) * 1000.0,
                }
            )
            return JSONResponse(response_json)
        except Exception as exc:
            fallback_enabled = os.getenv("ENABLE_LOCAL_MODAL_FALLBACK", "0").strip().lower() in {"1", "true", "yes"}
            if fallback_enabled:
                try:
                    manager_started = time.perf_counter()
                    local_response = manager.process_frame(payload)
                    manager_ms = (time.perf_counter() - manager_started) * 1000.0
                    if local_response.get("stale") or local_response.get("dropped"):
                        return JSONResponse(local_response)
                    local_response = _resummarize_response(manager, local_response)
                    annotation_started = time.perf_counter()
                    local_response["annotated_image_data_url"] = _annotate_response_frame(
                        _decode_base64_jpeg(payload.frame_jpeg_base64),
                        local_response,
                    )
                    local_response.setdefault("debug", {})
                    local_response["debug"].setdefault("stage_timings_ms", {})
                    local_response["debug"]["stage_timings_ms"].update(
                        {
                            "local_fallback_manager_ms": manager_ms,
                            "annotation_ms": (time.perf_counter() - annotation_started) * 1000.0,
                            "endpoint_total_ms": (time.perf_counter() - endpoint_started) * 1000.0,
                        }
                    )
                    local_response["debug"]["modal_proxy_error"] = f"{type(exc).__name__}: {exc}"
                    local_response["debug"]["modal_proxy_fallback"] = True
                    return JSONResponse(local_response)
                except Exception as local_exc:
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            f"Frame processing failed. Modal error: {type(exc).__name__}: {exc}. "
                            f"Local fallback error: {type(local_exc).__name__}: {local_exc}"
                        ),
                    ) from local_exc
            raise HTTPException(
                status_code=502,
                detail=f"Frame processing failed on Modal: {type(exc).__name__}: {exc}",
            ) from exc

    try:
        manager_started = time.perf_counter()
        response = manager.process_frame(payload)
        manager_ms = (time.perf_counter() - manager_started) * 1000.0
        if response.get("stale") or response.get("dropped"):
            return JSONResponse(response)
        response = _resummarize_response(manager, response)
        annotation_started = time.perf_counter()
        response["annotated_image_data_url"] = _annotate_response_frame(
            _decode_base64_jpeg(payload.frame_jpeg_base64),
            response,
        )
        response.setdefault("debug", {})
        response["debug"].setdefault("stage_timings_ms", {})
        response["debug"]["stage_timings_ms"].update(
            {
                "local_manager_ms": manager_ms,
                "annotation_ms": (time.perf_counter() - annotation_started) * 1000.0,
                "endpoint_total_ms": (time.perf_counter() - endpoint_started) * 1000.0,
            }
        )
        return JSONResponse(response)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Local frame processing failed: {type(exc).__name__}: {exc}") from exc

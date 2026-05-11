from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import cv2
import requests
import torch

from .config import BatchModeSettings
from .embeddings import BgeM3Embedder
from .factory import build_pipeline, build_stt_adapter
from .llm_adapters import LLMConfig, MockSectionWriter, build_frame_description_writer, build_section_writer
from .report_sections import build_report_sections as _canonical_build_report_sections
from .segmenter import TopicSegmenter
from .topic_linking import GlobalTopicLinker

logger = logging.getLogger(__name__)


@dataclass
class BatchAnalysisResult:
    source: str
    resolved_video_path: str
    transcript_segments: List[Dict[str, Any]]
    frame_events: List[Dict[str, Any]]
    topic_units: List[Dict[str, Any]]
    topic_clusters: List[Dict[str, Any]]
    sections: Dict[str, Any]
    llm_output: Dict[str, Any]
    output_path: str
    profiling: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "resolved_video_path": self.resolved_video_path,
            "transcript_segments": self.transcript_segments,
            "frame_events": self.frame_events,
            "topic_units": self.topic_units,
            "topic_clusters": self.topic_clusters,
            "sections": self.sections,
            "llm_output": self.llm_output,
            "output_path": self.output_path,
            "profiling": self.profiling,
        }


def build_report_sections(
    frame_events: List[Dict[str, Any]],
    transcript_segments: List[Dict[str, Any]],
    topic_units: List[Dict[str, Any]],
    topic_clusters: List[Dict[str, Any]],
    *,
    title: str,
) -> Dict[str, Any]:
    slides = [event for event in frame_events if event.get("mode") == "slide"]
    code = [event for event in frame_events if event.get("mode") == "code"]
    demo = [event for event in frame_events if event.get("mode") == "demo"]
    chronological = sorted(topic_units, key=lambda item: float(item.get("start", 0.0)))
    decisions = _extract_decisions_for_report(transcript_segments, frame_events)
    return {
        "report_title": title,
        "meeting_narrative_section": {
            "prewritten_markdown": _build_meeting_narrative_markdown(
                frame_events=frame_events,
                transcript_segments=transcript_segments,
                topic_units=chronological,
            ),
        },
        "slides_section": {
            "keyframes": slides[:30],
            "nearby_transcript": _nearby_transcript_for_report(slides, transcript_segments),
        },
        "code_section": {
            "keyframes": code[:30],
            "nearby_transcript": _nearby_transcript_for_report(code, transcript_segments),
        },
        "demo_section": {
            "events": demo[:60],
            "nearby_transcript": _nearby_transcript_for_report(demo, transcript_segments),
        },
        "chronological_section": {
            "segment_summaries": chronological,
        },
        "topic_section": {
            "clusters": topic_clusters,
        },
        "decisions_section": decisions,
        "transcript_section": {
            "prewritten_markdown": _build_full_transcript_markdown(transcript_segments),
            "segments": transcript_segments[:500],
            "full_text": _format_transcript_for_report(transcript_segments),
        },
    }


def _build_meeting_narrative_markdown(
    *,
    frame_events: List[Dict[str, Any]],
    transcript_segments: List[Dict[str, Any]],
    topic_units: List[Dict[str, Any]],
) -> str:
    if not frame_events and not transcript_segments:
        return "No screen events or transcript segments were available for this meeting."

    lines: List[str] = [
        "This section follows the meeting in chronological order, combining what was visible on screen with what was said.",
    ]
    if topic_units:
        for index, unit in enumerate(topic_units, start=1):
            start = float(unit.get("start", 0.0) or 0.0)
            end = float(unit.get("end", start) or start)
            mode = str(unit.get("dominant_class", "screen") or "screen")
            event_ids = {str(item) for item in unit.get("event_ids", []) if item}
            unit_events = [event for event in frame_events if str(event.get("event_id")) in event_ids]
            if not unit_events:
                unit_events = _events_in_range(frame_events, start, end)
            lines.extend(_format_narrative_unit(index, start, end, mode, unit, unit_events, transcript_segments))
        return "\n".join(lines).strip()

    # Fallback for short runs where topic segmentation did not emit units.
    for index, event in enumerate(frame_events, start=1):
        timestamp = float(event.get("timestamp_sec", 0.0) or 0.0)
        mode = str(event.get("mode", event.get("frame_type", "screen")) or "screen")
        summary = str(event.get("summary", "") or "A screen event was captured.").strip()
        transcript = _format_transcript_excerpt(_transcript_in_range(transcript_segments, timestamp - 4.0, timestamp + 4.0), 800)
        cursor = _cursor_sentence(event)
        line = f"{index}. **{_format_time(timestamp)} ({mode})**: {summary}"
        if cursor:
            line += f" {cursor}"
        if transcript:
            line += f"\n   Transcript: {transcript}"
        lines.append(line)
    return "\n".join(lines).strip()


def _format_narrative_unit(
    index: int,
    start: float,
    end: float,
    mode: str,
    unit: Dict[str, Any],
    events: List[Dict[str, Any]],
    transcript_segments: List[Dict[str, Any]],
) -> List[str]:
    header = f"{index}. **{_format_time(start)}-{_format_time(end)} ({mode})**"
    output = [header]
    unit_summary = str(unit.get("unit_summary", "") or "").strip()
    if unit_summary:
        output.append(f"   Overview: {unit_summary}")

    screen_lines = _screen_flow_lines(events)
    if screen_lines:
        output.append("   Screen flow:")
        output.extend(f"   - {line}" for line in screen_lines)

    transcript_text = str(unit.get("transcript_text", "") or "").strip()
    if not transcript_text:
        transcript_text = _format_transcript_excerpt(_transcript_in_range(transcript_segments, start, end), 1800)
    if transcript_text:
        output.append(f"   Transcript: {_truncate_report_text(transcript_text, 1800)}")
    return output


def _screen_flow_lines(events: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    seen: set[str] = set()
    max_events = int(os.getenv("BATCH_REPORT_EVENTS_PER_UNIT", "6"))
    for event in sorted(events, key=lambda item: float(item.get("timestamp_sec", 0.0) or 0.0)):
        timestamp = float(event.get("timestamp_sec", 0.0) or 0.0)
        summary = str(event.get("summary", "") or "").strip()
        visual = event.get("visual_payload") or {}
        if not summary and isinstance(visual, dict):
            summary = str(visual.get("page_summary", visual.get("full_text", visual.get("ocr_text", ""))) or "").strip()
        summary = _truncate_report_text(summary, 360)
        if not summary:
            continue
        normalized = re.sub(r"\s+", " ", summary).lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        cursor = _cursor_sentence(event)
        line = f"{_format_time(timestamp)}: {summary}"
        if cursor:
            line += f" {cursor}"
        lines.append(line)
        if len(lines) >= max_events:
            remaining = max(0, len(events) - max_events)
            if remaining:
                lines.append(f"{remaining} additional screen events in this interval were omitted from the narrative to keep the report readable.")
            break
    return lines


def _cursor_sentence(event: Dict[str, Any]) -> str:
    cursor = event.get("cursor")
    if not isinstance(cursor, dict):
        visual = event.get("visual_payload") or {}
        cursor = visual.get("cursor") if isinstance(visual, dict) else None
    if not isinstance(cursor, dict) or not cursor.get("detected"):
        return ""
    hotspot = cursor.get("hotspot")
    near = cursor.get("near_element_id")
    method = cursor.get("method")
    state = cursor.get("tracking_state")
    details = []
    if isinstance(hotspot, list) and len(hotspot) >= 2:
        details.append(f"hotspot near ({int(hotspot[0])}, {int(hotspot[1])})")
    if near:
        details.append(f"near element {near}")
    if state and state != "confirmed":
        details.append(f"{state} cursor track")
    elif method:
        details.append(f"detected by {method}")
    return "Cursor: " + ", ".join(details) + "." if details else "Cursor was visible."


def _build_full_transcript_markdown(transcript_segments: List[Dict[str, Any]]) -> str:
    transcript = _format_transcript_for_report(transcript_segments)
    if not transcript:
        return "No transcript segments were available for this meeting."
    return "Full transcript:\n\n" + transcript


def _format_transcript_for_report(transcript_segments: List[Dict[str, Any]]) -> str:
    lines = []
    for segment in transcript_segments[:500]:
        start = float(segment.get("start", segment.get("start_time", 0.0)) or 0.0)
        end = float(segment.get("end", segment.get("end_time", 0.0)) or 0.0)
        speaker = str(segment.get("speaker", "unknown") or "unknown")
        text = str(segment.get("text", "") or "").strip()
        if text:
            lines.append(f"[{_format_time(start)}-{_format_time(end)}] {speaker}: {text}")
    return "\n".join(lines)


def _format_transcript_excerpt(transcript_segments: List[Dict[str, Any]], limit: int) -> str:
    chunks = []
    for segment in transcript_segments:
        speaker = str(segment.get("speaker", "unknown") or "unknown")
        text = str(segment.get("text", "") or "").strip()
        if text:
            chunks.append(f"{speaker}: {text}")
    return _truncate_report_text(" ".join(chunks), limit)


def _transcript_in_range(transcript_segments: List[Dict[str, Any]], start: float, end: float) -> List[Dict[str, Any]]:
    result = []
    for segment in transcript_segments:
        seg_start = float(segment.get("start", segment.get("start_time", 0.0)) or 0.0)
        seg_end = float(segment.get("end", segment.get("end_time", 0.0)) or 0.0)
        if seg_end >= start and seg_start <= end:
            result.append(segment)
    return result


def _events_in_range(frame_events: List[Dict[str, Any]], start: float, end: float) -> List[Dict[str, Any]]:
    return [
        event for event in frame_events
        if start <= float(event.get("timestamp_sec", 0.0) or 0.0) <= end
    ]


def _nearby_transcript_for_report(events: List[Dict[str, Any]], transcript_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not events:
        return []
    start = min(float(item.get("timestamp_sec", 0.0)) for item in events) - 8.0
    end = max(float(item.get("timestamp_sec", 0.0)) for item in events) + 8.0
    return _transcript_in_range(transcript_segments, start, end)[:80]


def _extract_decisions_for_report(transcript_segments: List[Dict[str, Any]], frame_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    decision_terms = ("decide", "decision", "issue", "action", "todo", "follow up", "bug", "problem")
    transcript_hits = [
        dict(item) for item in transcript_segments
        if any(term in str(item.get("text", "")).lower() for term in decision_terms)
    ]
    related_events = [
        dict(item) for item in frame_events
        if any(term in str(item.get("summary", "")).lower() for term in decision_terms)
    ]
    return {
        "transcript_evidence": transcript_hits[:40],
        "related_events": related_events[:30],
    }


def _format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    remainder = seconds - (minutes * 60)
    return f"{minutes:02d}:{remainder:04.1f}"


def _truncate_report_text(text: str, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


build_report_sections = _canonical_build_report_sections


class BatchMeetingAnalyzer:
    def __init__(self, settings: BatchModeSettings | None = None) -> None:
        self.settings = settings or BatchModeSettings.from_env()
        self.settings.validate()
        self.pipeline = build_pipeline(self.settings)
        self.stt = build_stt_adapter(self.settings)
        self.embedder = BgeM3Embedder(
            model_name=self.settings.embedding_model_name,
            device=self.settings.embedding_device,
        )
        self.segmenter = TopicSegmenter(
            embedder=self.embedder,
            min_sec=self.settings.segment_min_sec,
            max_sec=self.settings.segment_max_sec,
            target_sec=self.settings.segment_target_sec,
            similarity_threshold=self.settings.embedding_similarity_threshold,
        )
        self.topic_linker = GlobalTopicLinker(
            embedder=self.embedder,
            similarity_threshold=self.settings.topic_similarity_threshold,
        )
        llm_backend = self._select_llm_backend()
        self.writer = build_section_writer(
            LLMConfig(
                backend=llm_backend,
                base_url=self.settings.llm_base_url,
                api_key=self.settings.llm_api_key,
                model=self.settings.llm_model or self.settings.qwen_model_name,
                timeout_sec=self.settings.llm_timeout_sec,
                mock_enabled=self.settings.llm_mock_enabled,
            )
        )
        self.frame_writer = build_frame_description_writer(
            LLMConfig(
                backend=llm_backend,
                base_url=self.settings.llm_base_url,
                api_key=self.settings.llm_api_key,
                model=self.settings.llm_model or self.settings.qwen_model_name,
                timeout_sec=self.settings.llm_timeout_sec,
                mock_enabled=self.settings.llm_mock_enabled,
            ),
            max_workers=self.settings.frame_description_workers,
        )
        self._profile: Dict[str, Any] = {}
        self._profile_lock = Lock()

    def analyze(
        self,
        *,
        source: str,
        transcript_segments: Optional[List[Dict[str, Any]]] = None,
        title: str = "",
        persist_artifacts: bool = True,
        writer: str = "",
        describe_frames: Optional[bool] = None,
    ) -> BatchAnalysisResult:
        self._reset_run_state()
        self._profile = {"stages": {}, "counts": {}}
        started = perf_counter()
        video_path = self._profile_call("resolve_video_source", lambda: self._resolve_video_source(source))

        if transcript_segments is None:
            with ThreadPoolExecutor(max_workers=2) as executor:
                transcript_future = executor.submit(self._profile_call, "transcription", lambda: self._transcribe(video_path))
                visual_future = executor.submit(self._profile_call, "visual_analysis", lambda: self._process_video(video_path))
                transcript = transcript_future.result()
                frame_events = visual_future.result()
            frame_events = self._profile_call(
                "transcript_alignment",
                lambda: self._attach_transcript_to_events(frame_events, transcript),
            )
        else:
            transcript = transcript_segments
            if transcript:
                self.pipeline.ingest_utterances(transcript)
            frame_events = self._profile_call("visual_analysis", lambda: self._process_video(video_path))

        should_describe_frames = self.settings.frame_descriptions_enabled if describe_frames is None else bool(describe_frames)
        if should_describe_frames:
            frame_description_writer = self._frame_writer_for_request(writer)
            frame_events = self._profile_call(
                "frame_llm_descriptions",
                lambda: frame_description_writer.enrich_events(
                    frame_events,
                    max_demo_main_segments=self.settings.demo_main_max_segments,
                ),
            )

        topic_units = self._profile_call(
            "topic_segmentation",
            lambda: [unit.to_dict() for unit in self.segmenter.build_units(frame_events, transcript)],
        )
        topic_clusters = self._profile_call(
            "topic_linking",
            lambda: [cluster.to_dict() for cluster in self.topic_linker.cluster(frame_events, topic_units)],
        )
        sections = self._profile_call(
            "section_building",
            lambda: self._build_sections(frame_events, transcript, topic_units, topic_clusters, title=title or Path(video_path).stem),
        )
        try:
            section_writer = self._writer_for_request(writer)
            llm_output = self._profile_call("llm_report_generation", lambda: section_writer.generate(sections=sections))
        except Exception as exc:
            logger.exception("LLM report generation failed; persisting analysis with fallback report.")
            with self._profile_lock:
                self._profile["llm_error"] = str(exc)
            llm_output = MockSectionWriter().generate(sections=sections)
            llm_output["backend"] = "fallback-mock"
            llm_output["error"] = str(exc)
        self._profile["total_elapsed_sec"] = round(perf_counter() - started, 3)
        self._profile["counts"] = {
            "transcript_segments": len(transcript),
            "frame_events": len(frame_events),
            "topic_units": len(topic_units),
            "topic_clusters": len(topic_clusters),
        }
        self._profile["visual"] = self._summarize_visual_profile(frame_events)
        output_path = self._profile_call(
            "persist_result",
            lambda: self._persist_result(
                source=source,
                video_path=video_path,
                transcript=transcript,
                frame_events=frame_events,
                topic_units=topic_units,
                topic_clusters=topic_clusters,
                sections=sections,
                llm_output=llm_output,
            ),
        ) if persist_artifacts else ""
        self._profile["total_elapsed_sec"] = round(perf_counter() - started, 3)
        return BatchAnalysisResult(
            source=source,
            resolved_video_path=video_path,
            transcript_segments=transcript,
            frame_events=frame_events,
            topic_units=topic_units,
            topic_clusters=topic_clusters,
            sections=sections,
            llm_output=llm_output,
            output_path=output_path,
            profiling=dict(self._profile),
        )

    def analyze_many(
        self,
        *,
        requests: List[Dict[str, Any]],
    ) -> List[BatchAnalysisResult]:
        results: List[BatchAnalysisResult] = []
        for item in requests:
            results.append(
                self.analyze(
                    source=str(item.get("source", "")),
                    transcript_segments=item.get("transcript_segments"),
                    title=str(item.get("title", "") or ""),
                    persist_artifacts=bool(item.get("persist_artifacts", True)),
                    writer=str(item.get("writer", "") or ""),
                    describe_frames=item.get("describe_frames"),
                )
            )
        return results

    def _select_llm_backend(self) -> str:
        backend = (self.settings.llm_backend or "").strip().lower()
        if self.settings.llm_base_url and backend in {"", "auto", "vllm"}:
            return "vllm"
        if backend == "mock" and self.settings.llm_mock_enabled:
            return "mock"
        if backend and backend not in {"mock", "auto"}:
            return backend
        if _supports_local_qwen():
            logger.info("Auto-selecting hf-local report writer for batch mode.")
            return "hf-local"
        return backend or "mock"

    def _writer_for_request(self, writer: str):
        writer = (writer or "").strip().lower()
        if not writer:
            return self.writer
        if writer in {"mock", "preserve"}:
            return build_section_writer(LLMConfig(backend="mock", model="mock-writer", mock_enabled=True))
        if writer == "gemini":
            return build_section_writer(
                LLMConfig(
                    backend="gemini",
                    api_key=self.settings.llm_api_key or os.getenv("GEMINI_API_KEY", ""),
                    model="gemini-2.5-flash",
                    timeout_sec=max(self.settings.llm_timeout_sec, 900),
                    mock_enabled=self.settings.llm_mock_enabled,
                )
            )
        if writer == "qwen":
            return build_section_writer(
                LLMConfig(
                    backend="hf-local",
                    model=self.settings.qwen_model_name,
                    timeout_sec=self.settings.llm_timeout_sec,
                    mock_enabled=self.settings.llm_mock_enabled,
                )
            )
        if writer == "mistral":
            return build_section_writer(
                LLMConfig(
                    backend="hf-local",
                    model="unsloth/Mistral-Small-24B-Instruct-2501-unsloth-bnb-4bit",
                    timeout_sec=self.settings.llm_timeout_sec,
                    mock_enabled=self.settings.llm_mock_enabled,
                )
            )
        raise ValueError(f"Unknown writer preset: {writer}. Use mistral, gemini, qwen, or mock.")

    def _frame_writer_for_request(self, writer: str):
        writer = (writer or "").strip().lower()
        if not writer:
            return self.frame_writer
        if writer in {"mock", "preserve"}:
            return build_frame_description_writer(
                LLMConfig(backend="mock", model="mock-frame-describer", mock_enabled=True),
                max_workers=self.settings.frame_description_workers,
            )
        if writer == "gemini":
            return build_frame_description_writer(
                LLMConfig(
                    backend="gemini",
                    api_key=self.settings.llm_api_key or os.getenv("GEMINI_API_KEY", ""),
                    model="gemini-2.5-flash",
                    timeout_sec=max(self.settings.llm_timeout_sec, 900),
                    mock_enabled=self.settings.llm_mock_enabled,
                ),
                max_workers=self.settings.frame_description_workers,
            )
        if writer == "qwen":
            return build_frame_description_writer(
                LLMConfig(
                    backend="hf-local",
                    model=self.settings.qwen_model_name,
                    timeout_sec=self.settings.llm_timeout_sec,
                    mock_enabled=self.settings.llm_mock_enabled,
                ),
                max_workers=max(1, min(2, self.settings.frame_description_workers)),
            )
        if writer == "mistral":
            return build_frame_description_writer(
                LLMConfig(
                    backend="hf-local",
                    model="unsloth/Mistral-Small-24B-Instruct-2501-unsloth-bnb-4bit",
                    timeout_sec=self.settings.llm_timeout_sec,
                    mock_enabled=self.settings.llm_mock_enabled,
                ),
                max_workers=max(1, min(2, self.settings.frame_description_workers)),
            )
        raise ValueError(f"Unknown writer preset: {writer}. Use mistral, gemini, qwen, or mock.")

    def _reset_run_state(self) -> None:
        self.pipeline.reset_session()

    def _resolve_video_source(self, source: str) -> str:
        parsed = urlparse(source)
        if parsed.scheme in {"http", "https"}:
            suffix = Path(parsed.path).suffix or ".mp4"
            temp_dir = Path(self.settings.batch_temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
            destination = temp_dir / f"batch_input_{abs(hash(source))}{suffix}"
            with requests.get(source, stream=True, timeout=self.settings.batch_download_timeout_sec) as response:
                response.raise_for_status()
                with destination.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            return str(destination)
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Video source not found: {source}")
        return str(path)

    def _transcribe(self, video_path: str) -> List[Dict[str, Any]]:
        try:
            return self.stt.transcribe_file(video_path)
        except Exception:
            logger.exception("Deepgram transcription failed; continuing without transcript.")
            return []

    def _process_video(self, video_path: str) -> List[Dict[str, Any]]:
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video source: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = (frame_count / fps) if fps > 0 and frame_count > 0 else 0.0
        timestamp = 0.0
        processed = 0
        current_frame_index = -1
        sequential_decode = fps > 0
        try:
            while True:
                if duration > 0 and timestamp > duration + 1e-6:
                    break
                if sequential_decode:
                    target_frame_index = max(0, int(round(timestamp * fps)))
                    while current_frame_index + 1 < target_frame_index:
                        if not capture.grab():
                            return [record.to_dict() for record in self.pipeline.records()]
                        current_frame_index += 1
                    ok, frame = capture.read()
                    current_frame_index += 1
                else:
                    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000.0)
                    ok, frame = capture.read()
                if not ok or frame is None:
                    break
                record = self.pipeline.ingest_frame(frame, timestamp_sec=timestamp)
                processed += 1
                next_interval = self._interval_for_mode(record.mode)
                timestamp += next_interval
        finally:
            capture.release()
        return [record.to_dict() for record in self.pipeline.records()]

    def _interval_for_mode(self, mode: str) -> float:
        min_sample_interval = 0.0
        if self.settings.run_sample_fps > 0:
            min_sample_interval = 1.0 / self.settings.run_sample_fps
        if mode == "demo":
            return max(1.0, min_sample_interval, self.settings.demo_interval_sec)
        if mode == "code":
            return max(1.0, min_sample_interval, self.settings.code_interval_sec)
        if mode == "slide":
            return max(1.0, min_sample_interval, self.settings.slide_interval_sec)
        return max(1.0, min_sample_interval, self.settings.other_interval_sec)

    def _build_sections(
        self,
        frame_events: List[Dict[str, Any]],
        transcript_segments: List[Dict[str, Any]],
        topic_units: List[Dict[str, Any]],
        topic_clusters: List[Dict[str, Any]],
        *,
        title: str,
    ) -> Dict[str, Any]:
        return build_report_sections(
            frame_events=frame_events,
            transcript_segments=transcript_segments,
            topic_units=topic_units,
            topic_clusters=topic_clusters,
            title=title,
        )

    def _format_transcript(self, transcript_segments: List[Dict[str, Any]]) -> str:
        lines = []
        for segment in transcript_segments[:500]:
            start = float(segment.get("start", segment.get("start_time", 0.0)) or 0.0)
            end = float(segment.get("end", segment.get("end_time", 0.0)) or 0.0)
            speaker = str(segment.get("speaker", "unknown") or "unknown")
            text = str(segment.get("text", "") or "").strip()
            if text:
                lines.append(f"[{start:.1f}-{end:.1f}] {speaker}: {text}")
        return "\n".join(lines)

    def _nearby_transcript(self, events: List[Dict[str, Any]], transcript_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not events:
            return []
        start = min(float(item.get("timestamp_sec", 0.0)) for item in events) - 8.0
        end = max(float(item.get("timestamp_sec", 0.0)) for item in events) + 8.0
        result = []
        for segment in transcript_segments:
            seg_start = float(segment.get("start", segment.get("start_time", 0.0)) or 0.0)
            seg_end = float(segment.get("end", segment.get("end_time", 0.0)) or 0.0)
            if seg_end >= start and seg_start <= end:
                result.append(dict(segment))
        return result[:80]

    def _extract_decisions(self, transcript_segments: List[Dict[str, Any]], frame_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        decision_terms = ("decide", "decision", "issue", "action", "todo", "follow up", "bug", "problem")
        transcript_hits = [
            dict(item) for item in transcript_segments
            if any(term in str(item.get("text", "")).lower() for term in decision_terms)
        ]
        related_events = [
            dict(item) for item in frame_events
            if any(term in str(item.get("summary", "")).lower() for term in decision_terms)
        ]
        return {
            "transcript_evidence": transcript_hits[:40],
            "related_events": related_events[:30],
        }

    def _persist_result(
        self,
        *,
        source: str,
        video_path: str,
        transcript: List[Dict[str, Any]],
        frame_events: List[Dict[str, Any]],
        topic_units: List[Dict[str, Any]],
        topic_clusters: List[Dict[str, Any]],
        sections: Dict[str, Any],
        llm_output: Dict[str, Any],
    ) -> str:
        base = Path(self.settings.batch_output_dir)
        base.mkdir(parents=True, exist_ok=True)
        slug = Path(video_path).stem.replace(" ", "_")
        out_dir = base / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        bundle = {
            "source": source,
            "video_path": video_path,
            "transcript_segments": transcript,
            "frame_events": frame_events,
            "topic_units": topic_units,
            "topic_clusters": topic_clusters,
            "sections": sections,
            "llm_output": llm_output,
        }
        output_path = out_dir / "analysis.json"
        bundle["profiling"] = dict(self._profile)
        output_path.write_text(json.dumps(bundle, ensure_ascii=True, indent=2), encoding="utf-8")
        markdown_path = out_dir / "report.md"
        markdown_path.write_text(str(llm_output.get("final_markdown", "")), encoding="utf-8")
        return str(output_path)

    def _profile_call(self, name: str, func):
        started = perf_counter()
        try:
            return func()
        finally:
            elapsed = perf_counter() - started
            with self._profile_lock:
                stages = self._profile.setdefault("stages", {})
                item = stages.setdefault(name, {"calls": 0, "elapsed_sec": 0.0})
                item["calls"] = int(item.get("calls", 0)) + 1
                item["elapsed_sec"] = round(float(item.get("elapsed_sec", 0.0)) + elapsed, 3)

    def _summarize_visual_profile(self, frame_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        by_mode: Dict[str, Dict[str, float]] = {}
        processor_counts: Dict[str, int] = {}
        for event in frame_events:
            mode = str(event.get("mode", event.get("frame_type", "other")) or "other")
            debug = event.get("debug") or {}
            elapsed_ms = float(debug.get("processing_time_ms", 0.0) or 0.0)
            item = by_mode.setdefault(mode, {"count": 0.0, "processing_time_ms": 0.0})
            item["count"] += 1.0
            item["processing_time_ms"] += elapsed_ms
            processor = str(debug.get("processor_used", "none") or "none")
            processor_counts[processor] = processor_counts.get(processor, 0) + 1
        modes = {}
        for mode, item in by_mode.items():
            count = max(1.0, item["count"])
            modes[mode] = {
                "count": int(item["count"]),
                "total_processing_time_ms": round(item["processing_time_ms"], 2),
                "avg_processing_time_ms": round(item["processing_time_ms"] / count, 2),
            }
        return {
            "by_mode": modes,
            "processor_counts": processor_counts,
        }

    def _attach_transcript_to_events(
        self,
        frame_events: List[Dict[str, Any]],
        transcript_segments: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not transcript_segments:
            return frame_events
        enriched: List[Dict[str, Any]] = []
        previous_visual_event_time: Optional[float] = None
        previous_mode = "other"
        for event in frame_events:
            item = dict(event)
            timestamp = float(item.get("timestamp_sec", 0.0) or 0.0)
            mode = str(item.get("mode", item.get("frame_type", "other")) or "other")
            force = bool(item.get("visual_change")) or mode != previous_mode or mode == "other"
            window = self._transcript_window_for_event(
                transcript_segments,
                event_time=timestamp,
                previous_event_time=previous_visual_event_time,
                force=force,
            )
            item["transcript_window"] = window
            item["utterance_ids"] = [seg.get("segment_id") for seg in window if seg.get("segment_id") is not None]
            if window:
                item["utterance_time_span"] = [
                    float(window[0].get("start", window[0].get("start_time", 0.0)) or 0.0),
                    float(window[-1].get("end", window[-1].get("end_time", 0.0)) or 0.0),
                ]
            item["summary"] = self._summarize_event_dict(item, window, previous_mode=previous_mode, mode_changed=mode != previous_mode)
            if force:
                previous_visual_event_time = timestamp
            previous_mode = mode
            enriched.append(item)
        return enriched

    def _transcript_window_for_event(
        self,
        transcript_segments: List[Dict[str, Any]],
        *,
        event_time: float,
        previous_event_time: Optional[float],
        force: bool,
    ) -> List[Dict[str, Any]]:
        if not force:
            return []
        lower = float(previous_event_time) if previous_event_time is not None else float(event_time) - self.settings.transcript_window_sec
        upper = float(event_time) + 0.01
        selected = []
        for idx, segment in enumerate(transcript_segments):
            seg_start = float(segment.get("start", segment.get("start_time", 0.0)) or 0.0)
            seg_end = float(segment.get("end", segment.get("end_time", seg_start)) or seg_start)
            if seg_end >= lower and seg_start <= upper:
                copied = dict(segment)
                copied.setdefault("segment_id", idx + 1)
                selected.append(copied)
        return selected

    def _summarize_event_dict(
        self,
        event: Dict[str, Any],
        transcript_window: List[Dict[str, Any]],
        *,
        previous_mode: str,
        mode_changed: bool,
    ) -> str:
        mode = str(event.get("mode", event.get("frame_type", "other")) or "other")
        transcript_text = " ".join(str(segment.get("text", "") or "") for segment in transcript_window).strip()
        transition = f"The screen switched from {previous_mode} mode to {mode} mode. " if mode_changed and previous_mode != mode else ""
        visual_payload = event.get("visual_payload") or {}
        if mode == "code":
            visual = "The presenter is showing code."
            diff = visual_payload.get("ocr_diff") or {}
            added = str(diff.get("added", "") or "").strip()
            removed = str(diff.get("removed", "") or "").strip()
            if added or removed:
                bits = []
                if added:
                    bits.append(f"added text: {added[:220]}")
                if removed:
                    bits.append(f"removed text: {removed[:220]}")
                visual += " The visible code changed with " + "; ".join(bits) + "."
            elif visual_payload.get("ocr_text"):
                visual += f" Visible text includes: {str(visual_payload.get('ocr_text'))[:220]}."
            return (transition + visual + (f" They are saying: {transcript_text}" if transcript_text else "")).strip()
        if mode == "slide":
            layout = visual_payload.get("layout") or []
            title = str(layout[0].get("text", "") or "").strip() if layout and isinstance(layout[0], dict) else ""
            visual = "The presenter is showing a slide."
            if title:
                visual += f" The slide title appears to be '{title}'."
            if transcript_text:
                visual += f" They are discussing: {transcript_text}"
            return (transition + visual).strip()
        if mode == "demo":
            visual = str(visual_payload.get("page_summary", "") or "The presenter is demonstrating a web UI.").strip()
            change_type = str(visual_payload.get("change_type", "") or "").strip()
            if change_type and change_type != "none":
                visual += f" The detected UI change type is {change_type.replace('_', ' ')}."
            if transcript_text:
                visual += f" They are explaining: {transcript_text}"
            return (transition + visual).strip()
        return (transition + (str(visual_payload.get("summary", "") or "No meaningful presented screen content detected.").strip()) + (f" Transcript: {transcript_text}" if transcript_text else "")).strip()


def _supports_local_qwen() -> bool:
    try:
        if not torch.cuda.is_available():
            return False
        total_bytes = int(torch.cuda.get_device_properties(0).total_memory)
        return total_bytes >= 30 * 1024**3
    except Exception:
        return False

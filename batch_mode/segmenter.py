from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from .embeddings import BgeM3Embedder, cosine_similarity


@dataclass
class TopicUnit:
    start: float
    end: float
    dominant_class: str
    transcript_text: str
    event_ids: List[str] = field(default_factory=list)
    ocr_text: str = ""
    screen_summaries: List[str] = field(default_factory=list)
    action_summaries: List[str] = field(default_factory=list)
    unit_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "dominant_class": self.dominant_class,
            "transcript_text": self.transcript_text,
            "event_ids": list(self.event_ids),
            "ocr_text": self.ocr_text,
            "screen_summaries": list(self.screen_summaries),
            "action_summaries": list(self.action_summaries),
            "unit_summary": self.unit_summary,
        }


class TopicSegmenter:
    def __init__(
        self,
        *,
        embedder: BgeM3Embedder | None = None,
        min_sec: float = 10.0,
        max_sec: float = 30.0,
        target_sec: float = 18.0,
        similarity_threshold: float = 0.63,
    ) -> None:
        self.embedder = embedder
        self.min_sec = float(min_sec)
        self.max_sec = float(max_sec)
        self.target_sec = float(target_sec)
        self.similarity_threshold = float(similarity_threshold)

    def build_units(self, records: Sequence[Dict[str, Any]], transcript_segments: Sequence[Dict[str, Any]]) -> List[TopicUnit]:
        if not records:
            return []
        texts = [self._event_text(record) for record in records]
        vectors = None
        if self.embedder is not None:
            vectors = self.embedder.encode(texts).vectors
        units: List[TopicUnit] = []
        bucket: List[Dict[str, Any]] = []
        for index, record in enumerate(records):
            bucket.append(record)
            elapsed = float(bucket[-1]["timestamp_sec"]) - float(bucket[0]["timestamp_sec"])
            should_split = False
            if len(bucket) >= 2:
                prev = bucket[-2]
                if self._hard_boundary(prev, record):
                    should_split = elapsed >= self.min_sec
            if vectors is not None and len(bucket) >= 2 and not should_split:
                prev_vec = vectors[index - 1]
                curr_vec = vectors[index]
                should_split = elapsed >= self.min_sec and cosine_similarity(prev_vec, curr_vec) < self.similarity_threshold
            if elapsed >= self.max_sec:
                should_split = True
            if elapsed >= self.target_sec and self._soft_boundary(bucket):
                should_split = True
            if should_split:
                units.append(self._finalize_unit(bucket[:-1], transcript_segments))
                bucket = [bucket[-1]]
        if bucket:
            units.append(self._finalize_unit(bucket, transcript_segments))
        return units

    def _hard_boundary(self, previous: Dict[str, Any], current: Dict[str, Any]) -> bool:
        if previous.get("mode") != current.get("mode"):
            return True
        if self._extract_slide_title(previous) != self._extract_slide_title(current):
            return True
        if self._extract_active_file(previous) != self._extract_active_file(current):
            return True
        if self._extract_demo_state(previous) != self._extract_demo_state(current):
            return True
        prev_change = str(previous.get("visual_payload", {}).get("change_type", "") or "")
        curr_change = str(current.get("visual_payload", {}).get("change_type", "") or "")
        return prev_change != curr_change and bool(curr_change)

    def _soft_boundary(self, bucket: Sequence[Dict[str, Any]]) -> bool:
        if len(bucket) < 2:
            return False
        modes = {item.get("mode") for item in bucket}
        return len(modes) > 1

    def _finalize_unit(self, records: Sequence[Dict[str, Any]], transcript_segments: Sequence[Dict[str, Any]]) -> TopicUnit:
        start = float(records[0]["timestamp_sec"])
        end = float(records[-1]["timestamp_sec"])
        relevant_transcript = [
            item for item in transcript_segments
            if float(item.get("end", item.get("end_time", 0.0)) or 0.0) >= start
            and float(item.get("start", item.get("start_time", 0.0)) or 0.0) <= end
        ]
        transcript_text = " ".join(str(item.get("text", "") or "").strip() for item in relevant_transcript if str(item.get("text", "") or "").strip())
        modes = Counter(str(item.get("mode", "other")) for item in records)
        dominant_class = modes.most_common(1)[0][0] if modes else "other"
        screen_summaries = [str(item.get("summary", "") or "").strip() for item in records if str(item.get("summary", "") or "").strip()]
        ocr_text = "\n".join(self._extract_ocr_text(item) for item in records if self._extract_ocr_text(item))
        action_summaries = [str(item.get("visual_payload", {}).get("change_type", "") or "").strip() for item in records if str(item.get("visual_payload", {}).get("change_type", "") or "").strip()]
        unit_summary = self._summarize_unit(dominant_class, transcript_text, screen_summaries, action_summaries)
        return TopicUnit(
            start=start,
            end=end,
            dominant_class=dominant_class,
            transcript_text=transcript_text,
            event_ids=[str(item.get("event_id")) for item in records],
            ocr_text=ocr_text,
            screen_summaries=screen_summaries,
            action_summaries=action_summaries,
            unit_summary=unit_summary,
        )

    def _extract_ocr_text(self, record: Dict[str, Any]) -> str:
        payload = record.get("visual_payload", {}) or {}
        return str(payload.get("ocr_text") or payload.get("full_text") or "").strip()

    def _extract_slide_title(self, record: Dict[str, Any]) -> str:
        payload = record.get("visual_payload", {}) or {}
        for block in payload.get("layout", []) or []:
            if str(block.get("type", "")).lower() == "title":
                return str(block.get("text", "") or "").strip().lower()
        return ""

    def _extract_active_file(self, record: Dict[str, Any]) -> str:
        payload = record.get("visual_payload", {}) or {}
        ocr_text = str(payload.get("ocr_text") or "").lower()
        for line in ocr_text.splitlines():
            if any(ext in line for ext in (".py", ".ts", ".tsx", ".js", ".json", ".ipynb")):
                return line.strip()
        return ""

    def _extract_demo_state(self, record: Dict[str, Any]) -> str:
        payload = record.get("visual_payload", {}) or {}
        summary = str(payload.get("page_summary", "") or "").strip().lower()
        return summary

    def _event_text(self, record: Dict[str, Any]) -> str:
        payload = record.get("visual_payload", {}) or {}
        return " ".join(
            part for part in [
                str(record.get("mode", "") or ""),
                str(record.get("summary", "") or ""),
                str(payload.get("page_summary", "") or ""),
                str(payload.get("ocr_text", "") or ""),
                " ".join(str(item.get("text", "") or "") for item in payload.get("layout", []) or []),
            ] if part
        )

    def _summarize_unit(
        self,
        dominant_class: str,
        transcript_text: str,
        screen_summaries: Sequence[str],
        action_summaries: Sequence[str],
    ) -> str:
        latest_screen = screen_summaries[-1] if screen_summaries else ""
        action_hint = ", ".join(item for item in action_summaries[:3] if item)
        parts = [f"Dominant mode: {dominant_class}."]
        if latest_screen:
            parts.append(latest_screen)
        if action_hint:
            parts.append(f"Key actions: {action_hint}.")
        if transcript_text:
            parts.append(f"Transcript focus: {transcript_text[:300]}")
        return " ".join(parts).strip()

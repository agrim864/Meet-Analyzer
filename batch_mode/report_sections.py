from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional


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
            "narrative_units": _build_narrative_units(
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


def _build_narrative_units(
    *,
    frame_events: List[Dict[str, Any]],
    transcript_segments: List[Dict[str, Any]],
    topic_units: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    for index, unit in enumerate(topic_units, start=1):
        start = float(unit.get("start", 0.0) or 0.0)
        end = float(unit.get("end", start) or start)
        event_ids = {str(item) for item in unit.get("event_ids", []) if item}
        unit_events = [event for event in frame_events if str(event.get("event_id")) in event_ids]
        if not unit_events:
            unit_events = _events_in_range(frame_events, start, end)
        transcript_text = str(unit.get("transcript_text", "") or "").strip()
        if not transcript_text:
            transcript_text = _format_transcript_excerpt(_transcript_in_range(transcript_segments, start, end), 1800)
        units.append(
            {
                "index": index,
                "time_range": f"{_format_time(start)}-{_format_time(end)}",
                "start": round(start, 1),
                "end": round(end, 1),
                "mode": str(unit.get("dominant_class", "screen") or "screen"),
                "overview": str(unit.get("unit_summary", "") or "").strip(),
                "screen_flow": _screen_flow_lines(unit_events),
                "transcript": _truncate_report_text(transcript_text, 1800),
            }
        )
    if units:
        return units

    for index, event in enumerate(frame_events, start=1):
        timestamp = float(event.get("timestamp_sec", 0.0) or 0.0)
        transcript = _format_transcript_excerpt(_transcript_in_range(transcript_segments, timestamp - 4.0, timestamp + 4.0), 800)
        units.append(
            {
                "index": index,
                "time_range": _format_time(timestamp),
                "start": round(timestamp, 1),
                "end": round(timestamp, 1),
                "mode": str(event.get("mode", event.get("frame_type", "screen")) or "screen"),
                "overview": str(event.get("summary", "") or "").strip(),
                "screen_flow": _screen_flow_lines([event]),
                "transcript": transcript,
            }
        )
    return units


def _format_narrative_unit(
    index: int,
    start: float,
    end: float,
    mode: str,
    unit: Dict[str, Any],
    events: List[Dict[str, Any]],
    transcript_segments: List[Dict[str, Any]],
) -> List[str]:
    output = [f"{index}. **{_format_time(start)}-{_format_time(end)} ({mode})**"]
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
    ordered = sorted(events, key=lambda item: float(item.get("timestamp_sec", 0.0) or 0.0))
    for index, event in enumerate(ordered):
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
        led_to = _cursor_led_to_sentence(event, ordered[index + 1] if index + 1 < len(ordered) else None)
        if led_to:
            line += f" {led_to}"
        lines.append(line)
        if len(lines) >= max_events:
            remaining = max(0, len(events) - max_events)
            if remaining:
                lines.append(f"{remaining} additional screen events in this interval were omitted from the narrative to keep the report readable.")
            break
    return lines


def _cursor_sentence(event: Dict[str, Any]) -> str:
    visual = event.get("visual_payload") or {}
    target = visual.get("cursor_target") if isinstance(visual, dict) else None
    if isinstance(target, dict):
        micro = str(target.get("element_text") or target.get("element_type") or "").strip()
        macro = str(target.get("parent_macro_region_type") or "").replace("_", " ").strip()
        relation = str(target.get("relation") or "").replace("_", " ").strip()
        if micro:
            location = f" in the {macro}" if macro else ""
            if relation in {"hovering over", "focused on"}:
                return f"Cursor: {relation} {micro}{location}."
            if relation:
                return f"Cursor: {relation} on {micro}{location}."
            return f"Cursor: focused on {micro}{location}."
        if macro:
            return f"Cursor: inside the {macro}."
    cursor = event.get("cursor")
    if not isinstance(cursor, dict):
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


def _cursor_led_to_sentence(event: Dict[str, Any], next_event: Optional[Dict[str, Any]]) -> str:
    visual = event.get("visual_payload") or {}
    target = visual.get("cursor_target") if isinstance(visual, dict) else None
    if not isinstance(target, dict):
        return ""
    target_text = str(target.get("element_text") or target.get("element_type") or target.get("parent_macro_region_type") or "").strip()
    if not target_text or next_event is None:
        return ""
    next_visual = next_event.get("visual_payload") or {}
    next_change = str(next_visual.get("change_type") or next_event.get("change_type") or "").replace("_", " ").strip()
    next_summary = str(next_event.get("summary", "") or "").strip()
    if next_change and next_change not in {"none", "initial"}:
        return f"This was followed by a {next_change} change."
    if next_summary:
        return f"This was followed by: {_truncate_report_text(next_summary, 140)}"
    return ""


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

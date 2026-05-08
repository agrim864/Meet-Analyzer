from __future__ import annotations

from typing import Dict, List

from .layout_groups import assign_layout_groups
from .state import LayoutGroup, SceneStateRecord, ScreenState, bbox_center, bbox_intersects


def build_relations(state: ScreenState) -> List[str]:
    relations: List[str] = []
    for group in state.layout_groups:
        label = str(group.label or "")
        if label in {"header", "top_navbar", "toolbar"}:
            relations.append(f"{group.id} spans the top navigation area")
        elif label in {"sidebar", "left_sidebar"}:
            relations.append(f"{group.id} anchors the left sidebar")
        elif label in {"right_panel", "right_sidebar", "notification_panel", "chat_panel"}:
            relations.append(f"{group.id} occupies the right-side panel")
        elif label == "footer":
            relations.append(f"{group.id} spans the footer area")
        elif label in {"content", "main_content", "content_panel", "form_region", "table_region"}:
            relations.append(f"{group.id} contains the primary content area")
    for overlay in state.overlays:
        for element in state.elements:
            if element.visible:
                relations.append(f"{overlay.id} is above {element.id}")
                break
    visible = [item for item in state.elements if item.visible and not item.occluded]
    visible = sorted(visible, key=lambda item: (item.bbox[1], item.bbox[0]))
    if len(visible) >= 2:
        first = visible[0]
        second = visible[1]
        first_center = bbox_center(first.bbox)
        second_center = bbox_center(second.bbox)
        if abs(first_center[0] - second_center[0]) > abs(first_center[1] - second_center[1]):
            if first_center[0] < second_center[0]:
                relations.append(f"{first.id} is left of {second.id}")
            else:
                relations.append(f"{first.id} is right of {second.id}")
        else:
            if first_center[1] < second_center[1]:
                relations.append(f"{first.id} is above {second.id}")
            else:
                relations.append(f"{first.id} is below {second.id}")
    return relations


def export_scene_record(state: ScreenState) -> SceneStateRecord:
    state.layout_groups = _resolve_layout_groups(state)
    relations = build_relations(state)
    state.relations = relations
    return SceneStateRecord(
        event_id=f"evt_{state.mode}_{state.frame_id:05d}",
        frame_id=state.frame_id,
        timestamp_sec=state.timestamp_sec,
        frame_type=state.frame_type,
        mode=state.mode,
        theme=state.theme,
        visual_change=bool(state.visual_change),
        visual_payload=dict(state.visual_payload),
        transcript_window=[
            {
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "speaker": segment.speaker,
                "text": segment.text,
                "is_final": segment.is_final,
            }
            for segment in state.transcript_window
        ],
        summary=state.summary,
        cursor={
            "detected": state.cursor.visible,
            "type": state.cursor.cursor_type,
            "bbox": list(state.cursor.bbox) if state.cursor.bbox else None,
            "hotspot": list(state.cursor.hotspot) if state.cursor.hotspot else None,
            "trail": [list(item) for item in state.cursor.trail],
            "motion_type": state.cursor.motion_type,
            "confidence": state.cursor.confidence,
            "method": state.cursor.method,
            "near_element_id": state.cursor.near_element_id,
            "tracking_state": state.cursor.tracking_state,
            "missing_frames": state.cursor.missing_frames,
        },
        elements=[
            {
                "id": item.id,
                "type": item.type,
                "text": item.text,
                "bbox": list(item.bbox),
                "interactive": item.interactive,
                "visible": item.visible,
                "occluded": item.occluded,
                "confidence": item.confidence,
                "source": item.source,
                "caption": item.caption,
                "group_id": item.group_id,
                "parent_macro_region_id": item.parent_macro_region_id,
                "z_index": item.z_index,
                "style": dict(item.style),
                "version": item.version,
                "stale": item.stale,
            }
            for item in state.elements
        ],
        overlays=[
            {
                "id": item.id,
                "type": item.type,
                "bbox": list(item.bbox),
                "title": item.title,
                "visible": item.visible,
                "confidence": item.confidence,
                "source": item.source,
                "z_index": item.z_index,
            }
            for item in state.overlays
        ],
        layout_groups=[
            {
                "id": item.id,
                "label": item.label,
                "bbox": list(item.bbox),
                "element_ids": list(item.element_ids),
                "axis": item.axis,
                "score": item.score,
                "source": item.source,
                "z_index": item.z_index,
                "frame_id": item.frame_id,
                "timestamp": item.timestamp,
                "stable": item.stable,
            }
            for item in state.layout_groups
        ],
        relations=relations,
        delta_from_previous={
            "type": list(state.delta.types),
            "changed_regions": [list(item) for item in state.delta.changed_regions],
            "reparsed_regions": [list(item) for item in state.delta.reparsed_regions],
            "confidence": state.delta.confidence,
        },
        confidence=dict(state.confidence),
        debug=dict(state.debug),
        utterance_ids=list(state.utterance_ids),
        utterance_time_span=state.utterance_time_span,
    )


def _resolve_layout_groups(state: ScreenState) -> List[LayoutGroup]:
    if state.layout_groups:
        for element in state.elements:
            matched_group = None
            for group in state.layout_groups:
                if bbox_intersects(element.bbox, group.bbox, padding=4):
                    matched_group = group
                    break
            element.group_id = matched_group.id if matched_group is not None else None
            element.parent_macro_region_id = matched_group.id if matched_group is not None else element.parent_macro_region_id
            if matched_group is not None:
                element.style["layout_area"] = matched_group.label
            else:
                element.style.pop("layout_area", None)
        for group in state.layout_groups:
            if not group.element_ids:
                group.element_ids = [item.id for item in state.elements if item.group_id == group.id]
        return state.layout_groups
    return assign_layout_groups(state)

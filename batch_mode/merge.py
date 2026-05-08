from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .state import BBox, Overlay, ScreenState, UIElement, bbox_iou, bbox_intersects


def merge_reparsed_state(
    *,
    state: ScreenState,
    reparsed_elements: Sequence[UIElement],
    reparsed_overlays: Sequence[Overlay],
    reparsed_regions: Sequence[BBox],
    frame_id: int,
) -> ScreenState:
    carried: List[UIElement] = [element.copy() for element in state.elements if not any(bbox_intersects(element.bbox, region) for region in reparsed_regions)]
    old_candidates = [element.copy() for element in state.elements if any(bbox_intersects(element.bbox, region) for region in reparsed_regions)]
    matched_old: set[str] = set()
    merged_new: List[UIElement] = []
    for new_element in reparsed_elements:
        element = new_element.copy()
        match = _best_element_match(element, old_candidates, matched_old)
        if match is not None:
            element.id = match.id
            element.version = match.version + (1 if match.text != element.text else 0)
            element.last_seen_frame = frame_id
            matched_old.add(match.id)
        else:
            element.id = f"el_{state.next_element_index:05d}"
            state.next_element_index += 1
            element.last_seen_frame = frame_id
        merged_new.append(element)
    for old_element in old_candidates:
        if old_element.id in matched_old:
            continue
        if old_element.occluded:
            carried.append(old_element.copy())
    carried.extend(merged_new)
    carried.sort(key=lambda item: (item.bbox[1], item.bbox[0], item.id))
    state.elements = carried

    carried_overlays: List[Overlay] = [overlay.copy() for overlay in state.overlays if not any(bbox_intersects(overlay.bbox, region) for region in reparsed_regions)]
    old_overlay_candidates = [overlay.copy() for overlay in state.overlays if any(bbox_intersects(overlay.bbox, region) for region in reparsed_regions)]
    for overlay in reparsed_overlays:
        new_overlay = overlay.copy()
        match = _best_overlay_match(new_overlay, old_overlay_candidates)
        if match is not None:
            new_overlay.id = match.id
        else:
            new_overlay.id = f"overlay_{state.next_overlay_index:05d}"
            state.next_overlay_index += 1
        carried_overlays.append(new_overlay)
    carried_overlays.sort(key=lambda item: (item.z_index, item.bbox[1], item.bbox[0]))
    state.overlays = carried_overlays
    return state


def _best_element_match(element: UIElement, candidates: Sequence[UIElement], matched_old: set[str]) -> Optional[UIElement]:
    best = None
    best_score = 0.0
    new_text = (element.text or "").strip().lower()
    for candidate in candidates:
        if candidate.id in matched_old:
            continue
        if candidate.type != element.type:
            continue
        iou = bbox_iou(candidate.bbox, element.bbox)
        text_score = 1.0 if new_text and new_text == (candidate.text or "").strip().lower() else 0.0
        score = iou * 0.8 + text_score * 0.2
        if score > best_score and (iou >= 0.25 or text_score >= 1.0):
            best_score = score
            best = candidate
    return best


def _best_overlay_match(overlay: Overlay, candidates: Sequence[Overlay]) -> Optional[Overlay]:
    best = None
    best_score = 0.0
    for candidate in candidates:
        iou = bbox_iou(candidate.bbox, overlay.bbox)
        if iou > best_score and iou >= 0.3:
            best_score = iou
            best = candidate
    return best

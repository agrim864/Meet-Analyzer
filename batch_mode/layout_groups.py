from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .state import BBox, LayoutGroup, ScreenState, UIElement, bbox_area, bbox_center, union_boxes


def assign_layout_groups(state: ScreenState) -> List[LayoutGroup]:
    width = max(1, int(state.viewport.get("width", 0) or 1))
    height = max(1, int(state.viewport.get("height", 0) or 1))
    visible = [item for item in state.elements if item.visible and not item.occluded]
    if not visible:
        for element in state.elements:
            element.group_id = None
            element.style.pop("layout_area", None)
        return []

    x_clusters = _cluster_by_axis(visible, axis="x", gap=max(48, int(width * 0.06)))
    y_clusters = _cluster_by_axis(visible, axis="y", gap=max(36, int(height * 0.05)))
    assignments: Dict[str, str] = {}
    produced: List[LayoutGroup] = []

    header_cluster = _pick_header_cluster(y_clusters, height)
    if header_cluster is not None:
        produced.append(_make_layout_cluster("group_header", "header", header_cluster, axis="y"))
        for item in header_cluster:
            assignments[item.id] = "group_header"

    footer_cluster = _pick_footer_cluster(y_clusters, height)
    if footer_cluster is not None:
        produced.append(_make_layout_cluster("group_footer", "footer", footer_cluster, axis="y"))
        for item in footer_cluster:
            assignments[item.id] = "group_footer"

    unassigned = [item for item in visible if item.id not in assignments]
    remaining_x_clusters = _cluster_by_axis(unassigned, axis="x", gap=max(48, int(width * 0.06)))
    left_cluster, right_cluster = _pick_side_clusters(remaining_x_clusters, width)

    if left_cluster:
        produced.append(_make_layout_cluster("group_sidebar", "sidebar", left_cluster, axis="x"))
        for item in left_cluster:
            assignments[item.id] = "group_sidebar"

    if right_cluster:
        produced.append(_make_layout_cluster("group_right_panel", "right_panel", right_cluster, axis="x"))
        for item in right_cluster:
            assignments[item.id] = "group_right_panel"

    center_elements = [item for item in visible if item.id not in assignments]
    if center_elements:
        produced.append(_make_layout_cluster("group_content", "content", center_elements, axis="xy"))
        for item in center_elements:
            assignments[item.id] = "group_content"

    for element in state.elements:
        group_id = assignments.get(element.id)
        element.group_id = group_id
        if group_id is None:
            element.style.pop("layout_area", None)
        else:
            element.style["layout_area"] = _group_label_from_id(group_id)

    return produced


def _cluster_by_axis(elements: Sequence[UIElement], *, axis: str, gap: int) -> List[List[UIElement]]:
    if not elements:
        return []
    indexed: List[Tuple[float, UIElement]] = []
    for item in elements:
        cx, cy = bbox_center(item.bbox)
        indexed.append((cx if axis == "x" else cy, item))
    indexed.sort(key=lambda pair: pair[0])
    clusters: List[List[UIElement]] = []
    current: List[UIElement] = [indexed[0][1]]
    previous_value = indexed[0][0]
    for value, item in indexed[1:]:
        if value - previous_value <= gap:
            current.append(item)
        else:
            clusters.append(current)
            current = [item]
        previous_value = value
    clusters.append(current)
    return clusters


def _pick_header_cluster(clusters: Sequence[Sequence[UIElement]], height: int) -> Optional[List[UIElement]]:
    candidates: List[Tuple[float, List[UIElement]]] = []
    for cluster in clusters:
        bbox = union_boxes([item.bbox for item in cluster])
        if bbox is None:
            continue
        _, cy = bbox_center(bbox)
        if cy <= height * 0.18:
            score = len(cluster) + ((bbox[2] - bbox[0]) / max(1, height))
            candidates.append((score, list(cluster)))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return candidates[0][1]


def _pick_footer_cluster(clusters: Sequence[Sequence[UIElement]], height: int) -> Optional[List[UIElement]]:
    candidates: List[Tuple[float, List[UIElement]]] = []
    for cluster in clusters:
        bbox = union_boxes([item.bbox for item in cluster])
        if bbox is None:
            continue
        _, cy = bbox_center(bbox)
        if cy >= height * 0.88:
            score = len(cluster) + ((bbox[2] - bbox[0]) / max(1, height))
            candidates.append((score, list(cluster)))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return candidates[0][1]


def _pick_side_clusters(clusters: Sequence[Sequence[UIElement]], width: int) -> Tuple[Optional[List[UIElement]], Optional[List[UIElement]]]:
    left: Optional[List[UIElement]] = None
    right: Optional[List[UIElement]] = None
    left_score = -1.0
    right_score = -1.0
    for cluster in clusters:
        bbox = union_boxes([item.bbox for item in cluster])
        if bbox is None:
            continue
        cx, _ = bbox_center(bbox)
        cluster_width = max(1, bbox[2] - bbox[0])
        total_area = sum(bbox_area(item.bbox) for item in cluster)
        score = len(cluster) + total_area / max(1, width * cluster_width)
        if cx <= width * 0.28 and score > left_score:
            left = list(cluster)
            left_score = score
        elif cx >= width * 0.72 and score > right_score:
            right = list(cluster)
            right_score = score
    return left, right


def _make_layout_cluster(cluster_id: str, label: str, elements: Sequence[UIElement], *, axis: str) -> LayoutGroup:
    bbox = union_boxes([item.bbox for item in elements]) or (0, 0, 0, 0)
    score = float(len(elements))
    return LayoutGroup(
        id=cluster_id,
        label=label,
        bbox=bbox,
        element_ids=[item.id for item in elements],
        axis=axis,
        score=score,
    )


def _group_label_from_id(group_id: str) -> str:
    return group_id.replace("group_", "", 1)

from __future__ import annotations

import base64
import io
import importlib
import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

from .state import BBox, Overlay, UIElement, bbox_iou


INTERACTIVE_LABELS = {
    "button",
    "icon",
    "toolbar",
    "input",
    "menu",
    "tab",
    "checkbox",
    "link",
}

OVERLAY_KEYWORDS = {"modal", "popup", "dialog", "dropdown", "tooltip"}


@dataclass
class ParsePayload:
    elements: List[UIElement]
    overlays: List[Overlay]
    provider: str
    confidence: float


class OmniParserAdapter:
    def __init__(
        self,
        *,
        client: Optional[Any] = None,
        hf_token: str = "",
        repo_id: str = "microsoft/OmniParser-v2.0",
        cache_dir: str = "real_time_mode/vendor/OmniParser/weights",
        bbox_threshold: float = 0.05,
        iou_threshold: float = 0.10,
        imgsz: int = 640,
        repo_dir: str | None = None,
    ) -> None:
        self.client = client or RepoBackedOmniParserClient(
            hf_token=hf_token,
            repo_id=repo_id,
            cache_dir=cache_dir,
            repo_dir=repo_dir or "real_time_mode/vendor/OmniParser",
            bbox_threshold=bbox_threshold,
            iou_threshold=iou_threshold,
            imgsz=imgsz,
        )

    def warmup(self) -> None:
        ensure = getattr(self.client, "warmup", None)
        if callable(ensure):
            ensure()

    def parse_full(self, frame_bgr: Any) -> ParsePayload:
        payload = self.client.infer(frame_bgr)
        return self._normalize_payload(payload)

    def parse_regions(self, frame_bgr: Any, regions: Sequence[BBox]) -> ParsePayload:
        collected_elements: List[UIElement] = []
        collected_overlays: List[Overlay] = []
        providers: List[str] = []
        confidences: List[float] = []
        for region in regions:
            x1, y1, x2, y2 = [int(v) for v in region]
            crop = frame_bgr[y1:y2, x1:x2]
            if crop is None or getattr(crop, "size", 0) == 0:
                continue
            normalized = self._parse_crop_with_optional_tiling(crop)
            for element in normalized.elements:
                element.bbox = (
                    element.bbox[0] + x1,
                    element.bbox[1] + y1,
                    element.bbox[2] + x1,
                    element.bbox[3] + y1,
                )
                collected_elements.append(element)
            for overlay in normalized.overlays:
                overlay.bbox = (
                    overlay.bbox[0] + x1,
                    overlay.bbox[1] + y1,
                    overlay.bbox[2] + x1,
                    overlay.bbox[3] + y1,
                )
                collected_overlays.append(overlay)
            providers.append(normalized.provider)
            confidences.append(normalized.confidence)
        provider = "+".join(sorted(set(providers))) if providers else "omniparser_region"
        confidence = sum(confidences) / max(1, len(confidences))
        return ParsePayload(
            elements=_dedupe_ui_elements(collected_elements),
            overlays=_dedupe_overlays(collected_overlays),
            provider=provider,
            confidence=confidence,
        )

    def _parse_crop_with_optional_tiling(self, crop: np.ndarray) -> ParsePayload:
        height, width = crop.shape[:2]
        if max(width, height) < 900:
            return self._normalize_payload(self.client.infer(crop))
        tiles = _tile_regions(width=width, height=height)
        collected_elements: List[UIElement] = []
        collected_overlays: List[Overlay] = []
        providers: List[str] = []
        confidences: List[float] = []
        for tx1, ty1, tx2, ty2 in tiles:
            sub = crop[ty1:ty2, tx1:tx2]
            if sub is None or getattr(sub, "size", 0) == 0:
                continue
            normalized = self._normalize_payload(self.client.infer(sub))
            for element in normalized.elements:
                element.bbox = (
                    element.bbox[0] + tx1,
                    element.bbox[1] + ty1,
                    element.bbox[2] + tx1,
                    element.bbox[3] + ty1,
                )
                collected_elements.append(element)
            for overlay in normalized.overlays:
                overlay.bbox = (
                    overlay.bbox[0] + tx1,
                    overlay.bbox[1] + ty1,
                    overlay.bbox[2] + tx1,
                    overlay.bbox[3] + ty1,
                )
                collected_overlays.append(overlay)
            providers.append(normalized.provider)
            confidences.append(normalized.confidence)
        return ParsePayload(
            elements=_dedupe_ui_elements(collected_elements),
            overlays=_dedupe_overlays(collected_overlays),
            provider="+".join(sorted(set(providers))) if providers else "omniparser_tiled",
            confidence=sum(confidences) / max(1, len(confidences)),
        )

    def _normalize_payload(self, payload: Dict[str, Any]) -> ParsePayload:
        raw_regions = payload.get("regions") or []
        elements: List[UIElement] = []
        overlays: List[Overlay] = []
        confidences: List[float] = []
        for item in raw_regions:
            if not isinstance(item, dict):
                continue
            raw_type = str(item.get("type", "ui_element") or "ui_element").strip().lower()
            source = str(item.get("source", "omniparser") or "omniparser")
            box = tuple(int(v) for v in (item.get("box", []) or [0, 0, 0, 0])[:4])
            if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                continue
            text = str(item.get("content", "") or "").strip()
            interactive = bool(item.get("interactive", item.get("interactivity", raw_type != "text")))
            confidence = float(item.get("confidence", 0.0) or 0.0)
            confidences.append(confidence)
            normalized_type = _normalize_type(raw_type=raw_type, text=text, interactive=interactive)
            if normalized_type in OVERLAY_KEYWORDS:
                overlays.append(
                    Overlay(
                        id="",
                        type=normalized_type,
                        bbox=box,
                        title=text,
                        confidence=confidence,
                        source=source,
                    )
                )
                continue
            elements.append(
                UIElement(
                    id="",
                    type=normalized_type,
                    bbox=box,
                    text=text,
                    interactive=interactive or normalized_type in INTERACTIVE_LABELS,
                    confidence=confidence,
                    source=source,
                    style={"omniparser_raw_type": raw_type},
                )
            )
        return ParsePayload(
            elements=elements,
            overlays=overlays,
            provider=str(payload.get("provider", "omniparser_repo")),
            confidence=sum(confidences) / max(1, len(confidences)),
        )


class RepoBackedOmniParserClient:
    def __init__(
        self,
        *,
        hf_token: str,
        repo_id: str,
        cache_dir: str,
        repo_dir: str,
        bbox_threshold: float,
        iou_threshold: float,
        imgsz: int,
    ) -> None:
        self.hf_token = str(hf_token or "").strip()
        self.repo_id = str(repo_id or "microsoft/OmniParser-v2.0").strip()
        self.cache_dir = Path(cache_dir).resolve()
        self.repo_dir = _resolve_repo_dir(Path(repo_dir))
        self.bbox_threshold = float(bbox_threshold)
        self.iou_threshold = float(iou_threshold)
        self.imgsz = int(imgsz)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._weights_ready = False
        self._initialized = False
        self._utils = None
        self._som_model = None
        self._caption_model_processor = None
        self._device = "cuda" if _cuda_available() else "cpu"

    def warmup(self) -> None:
        self._ensure_runtime()
        # Force one real inference path during warmup so missing modeling
        # dependencies surface before a live session starts.
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.infer(blank)

    def infer(self, frame_bgr: Any) -> Dict[str, Any]:
        self._ensure_runtime()
        if frame_bgr is None:
            raise RuntimeError("OmniParser received an empty frame.")
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)
        box_overlay_ratio = max(image.size) / 3200.0
        draw_bbox_config = {
            "text_scale": 0.8 * box_overlay_ratio,
            "text_thickness": max(int(2 * box_overlay_ratio), 1),
            "text_padding": max(int(3 * box_overlay_ratio), 1),
            "thickness": max(int(3 * box_overlay_ratio), 1),
        }
        ocr_result, _ = self._utils.check_ocr_box(
            image,
            display_img=False,
            output_bb_format="xyxy",
            goal_filtering=None,
            easyocr_args={"paragraph": False, "text_threshold": 0.9},
            use_paddleocr=False,
        )
        text_lines, ocr_boxes = ocr_result
        _, label_coordinates, parsed_content_list = self._utils.get_som_labeled_img(
            image,
            self._som_model,
            BOX_TRESHOLD=self.bbox_threshold,
            output_coord_in_ratio=True,
            ocr_bbox=ocr_boxes,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=self._caption_model_processor,
            ocr_text=text_lines,
            use_local_semantics=True,
            iou_threshold=self.iou_threshold,
            scale_img=False,
            imgsz=self.imgsz,
            batch_size=128,
        )
        width, height = image.size
        normalized_regions: List[Dict[str, Any]] = []
        for idx, item in enumerate(parsed_content_list or []):
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox") or []
            box = _ratio_or_absolute_to_box(bbox, width=width, height=height)
            if box is None:
                box = _label_coordinate_to_box(label_coordinates.get(str(idx)), width=width, height=height)
            if box is None:
                continue
            normalized_regions.append(
                {
                    "type": str(item.get("type", "ui_element") or "ui_element").strip().lower(),
                    "box": list(box),
                    "content": str(item.get("content", "") or "").strip(),
                    "interactive": bool(item.get("interactivity", item.get("interactive", False))),
                    "source": str(item.get("source", "omniparser_repo")),
                    "confidence": 1.0,
                }
            )
        return {
            "provider": "omniparser_repo_v2",
            "device": self._device,
            "frame_size": {"w": width, "h": height},
            "regions": normalized_regions,
        }

    def _ensure_runtime(self) -> None:
        if self._initialized:
            return
        _ensure_python_packages(["einops", "timm"])
        self._ensure_weights()
        self.repo_dir = _resolve_repo_dir(self.repo_dir)
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
        repo_utils = _load_repo_utils_module(self.repo_dir)

        self._utils = repo_utils
        detect_path = self.cache_dir / "icon_detect" / "model.pt"
        caption_path = self.cache_dir / "icon_caption"
        if not caption_path.exists():
            legacy = self.cache_dir / "icon_caption_florence"
            if legacy.exists():
                caption_path = legacy
        if not detect_path.exists() or not caption_path.exists():
            raise RuntimeError("OmniParser weights were not downloaded correctly.")
        self._som_model = repo_utils.get_yolo_model(model_path=str(detect_path))
        self._caption_model_processor = repo_utils.get_caption_model_processor(
            model_name="florence2",
            model_name_or_path=str(caption_path),
            device=self._device,
        )
        self._initialized = True

    def _ensure_weights(self) -> None:
        if self._weights_ready:
            return
        if not self.hf_token:
            if self._bundled_weights_present():
                self._weights_ready = True
                return
            raise RuntimeError("HF_API_KEY is required to download OmniParser weights.")
        needed_files = [
            "icon_detect/model.pt",
            "icon_detect/model.yaml",
            "icon_detect/train_args.yaml",
            "icon_caption/config.json",
            "icon_caption/generation_config.json",
            "icon_caption/model.safetensors",
        ]
        for path in needed_files:
            hf_hub_download(
                repo_id=self.repo_id,
                filename=path,
                repo_type="model",
                token=self.hf_token,
                local_dir=str(self.cache_dir),
                local_dir_use_symlinks=False,
            )
        self._weights_ready = True

    def _bundled_weights_present(self) -> bool:
        needed_files = [
            self.cache_dir / "icon_detect" / "model.pt",
            self.cache_dir / "icon_detect" / "model.yaml",
            self.cache_dir / "icon_detect" / "train_args.yaml",
            self.cache_dir / "icon_caption" / "config.json",
            self.cache_dir / "icon_caption" / "generation_config.json",
            self.cache_dir / "icon_caption" / "model.safetensors",
        ]
        return all(path.exists() for path in needed_files)


def _normalize_type(*, raw_type: str, text: str, interactive: bool) -> str:
    normalized = str(raw_type or "ui_element").strip().lower()
    compact = str(text or "").strip().lower()
    if normalized == "text":
        return "text"
    if any(keyword in compact for keyword in ("dialog", "modal", "popup")):
        return "modal"
    if any(keyword in compact for keyword in ("menu", "dropdown")):
        return "menu"
    if any(keyword in compact for keyword in ("tooltip", "hover")):
        return "tooltip"
    if interactive and compact:
        if any(keyword in compact for keyword in ("submit", "cancel", "save", "send", "share", "start", "stop", "continue", "next", "back")):
            return "button"
        if any(keyword in compact for keyword in ("search", "enter", "type here", "email", "password")):
            return "input"
        if any(keyword in compact for keyword in ("tab", "selected")):
            return "tab"
    return normalized


def _tile_regions(*, width: int, height: int) -> List[BBox]:
    overlap = 80
    cols = 2
    rows = 3 if height >= width else 2
    tile_w = max(1, int(np.ceil(width / cols)))
    tile_h = max(1, int(np.ceil(height / rows)))
    regions: List[BBox] = []
    for row in range(rows):
        for col in range(cols):
            x1 = max(0, col * tile_w - overlap)
            y1 = max(0, row * tile_h - overlap)
            x2 = min(width, (col + 1) * tile_w + overlap)
            y2 = min(height, (row + 1) * tile_h + overlap)
            regions.append((x1, y1, x2, y2))
    return regions


def _dedupe_ui_elements(elements: Sequence[UIElement]) -> List[UIElement]:
    kept: List[UIElement] = []
    for element in sorted(elements, key=lambda item: (-float(item.confidence), item.bbox[1], item.bbox[0])):
        duplicate = False
        for existing in kept:
            same_type = element.type == existing.type
            same_text = (element.text or "").strip().lower() == (existing.text or "").strip().lower()
            same_caption = (element.caption or "").strip().lower() == (existing.caption or "").strip().lower()
            overlap = bbox_iou(element.bbox, existing.bbox)
            if same_type and overlap >= 0.65 and (same_text or same_caption or not (element.text or existing.text)):
                duplicate = True
                break
        if not duplicate:
            kept.append(element)
    kept.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
    return kept


def _dedupe_overlays(overlays: Sequence[Overlay]) -> List[Overlay]:
    kept: List[Overlay] = []
    for overlay in sorted(overlays, key=lambda item: (-float(item.confidence), item.bbox[1], item.bbox[0])):
        duplicate = False
        for existing in kept:
            if overlay.type == existing.type and bbox_iou(overlay.bbox, existing.bbox) >= 0.6:
                duplicate = True
                break
        if not duplicate:
            kept.append(overlay)
    kept.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
    return kept


def _ratio_or_absolute_to_box(bbox: Sequence[float], *, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    if len(bbox) != 4:
        return None
    values = [float(v) for v in bbox]
    if max(values) <= 1.05:
        x1, y1, x2, y2 = values
        return (
            int(round(x1 * width)),
            int(round(y1 * height)),
            int(round(x2 * width)),
            int(round(y2 * height)),
        )
    return tuple(int(round(v)) for v in values[:4])  # type: ignore[return-value]


def _label_coordinate_to_box(bbox: Any, *, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    if not bbox or len(bbox) != 4:
        return None
    values = [float(v) for v in bbox]
    if max(values) <= 1.05:
        cx, cy, bw, bh = values
        x1 = int(round((cx - bw / 2.0) * width))
        y1 = int(round((cy - bh / 2.0) * height))
        x2 = int(round((cx + bw / 2.0) * width))
        y2 = int(round((cy + bh / 2.0) * height))
        return x1, y1, x2, y2
    return tuple(int(round(v)) for v in values[:4])  # type: ignore[return-value]


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _ensure_python_packages(packages: Sequence[str]) -> None:
    missing: List[str] = []
    for package in packages:
        try:
            importlib.import_module(package)
        except Exception:
            missing.append(str(package))
    if not missing:
        return
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    for package in missing:
        importlib.import_module(package)


def _resolve_repo_dir(repo_dir: Path) -> Path:
    default_repo = Path(__file__).resolve().parent / "vendor" / "OmniParser"
    candidates: List[Path] = [
        repo_dir,
        default_repo,
        Path("/root/project/real_time_mode/vendor/OmniParser"),
        Path("/root/real_time_mode/vendor/OmniParser"),
    ]
    seen: set[str] = set()
    unique_candidates: List[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(candidate)
    for candidate in unique_candidates:
        if (candidate / "util" / "utils.py").exists():
            return candidate.resolve()
    for candidate in unique_candidates:
        if candidate.exists():
            return candidate.resolve()
    return repo_dir.resolve()


def _load_repo_utils_module(repo_dir: Path) -> Any:
    util_dir = repo_dir / "util"
    utils_path = util_dir / "utils.py"
    if not utils_path.exists():
        raise RuntimeError(f"OmniParser utils module not found: {utils_path}")

    repo_root = str(repo_dir)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    util_init = util_dir / "__init__.py"
    if util_init.exists():
        package_spec = importlib.util.spec_from_file_location(
            "util",
            util_init,
            submodule_search_locations=[str(util_dir)],
        )
        if package_spec is None or package_spec.loader is None:
            raise RuntimeError(f"Failed to create OmniParser util package spec from {util_init}")
        package_module = importlib.util.module_from_spec(package_spec)
        sys.modules["util"] = package_module
        package_spec.loader.exec_module(package_module)

    module_spec = importlib.util.spec_from_file_location("util.utils", utils_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Failed to create OmniParser utils spec from {utils_path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules["util.utils"] = module
    module_spec.loader.exec_module(module)
    return module

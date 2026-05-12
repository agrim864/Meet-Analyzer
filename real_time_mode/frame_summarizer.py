from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

_SVGISH_RE = re.compile(r"^[MmLlHhVvCcSsQqTtAaZz0-9,.\-\s]+$")
_BAD_TOKEN_RE = re.compile(r"(?:https?://|www\.|aria-|svg|path|closepath)", re.IGNORECASE)
_GENERIC_ELEMENT_TYPES = {"icon", "button", "input", "text", "container", "panel", "label", "image"}


@dataclass
class SummaryMemoryTurn:
    timestamp_sec: float
    mode: str
    summary: str
    transcript_text: str = ""
    cursor_text: str = ""
    page_text: str = ""
    action_text: str = ""

    def to_prompt_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_sec": round(float(self.timestamp_sec), 2),
            "mode": self.mode,
            "summary": self.summary,
            "transcript_text": self.transcript_text,
            "cursor_text": self.cursor_text,
            "page_text": self.page_text,
            "action_text": self.action_text,
        }


@dataclass
class FrameSummaryRequest:
    timestamp_sec: float
    mode: str
    heuristic_summary: str
    transcript_text: str
    cursor_text: str
    page_text: str
    action_text: str
    current_payload: Dict[str, Any] = field(default_factory=dict)
    recent_memory: List[SummaryMemoryTurn] = field(default_factory=list)


class BaseFrameSummarizer:
    provider_name = "heuristic"
    model_name = "heuristic"

    def warmup(self) -> None:
        return None

    def summarize(self, request: FrameSummaryRequest) -> str:
        raise NotImplementedError


class HeuristicFrameSummarizer(BaseFrameSummarizer):
    def summarize(self, request: FrameSummaryRequest) -> str:
        payload = request.current_payload or {}
        if request.mode == "demo":
            summary = _heuristic_demo_summary(payload, request.transcript_text, request.cursor_text)
            return summary or request.heuristic_summary
        return request.heuristic_summary


class LocalTransformersFrameSummarizer(BaseFrameSummarizer):
    def __init__(
        self,
        *,
        model_name: str,
        device: str = "auto",
        max_new_tokens: int = 96,
        temperature: float = 0.1,
    ) -> None:
        self.model_name = str(model_name).strip()
        self.provider_name = "local_transformers"
        self.device = str(device or "auto").strip() or "auto"
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self._tokenizer = None
        self._model = None

    def warmup(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if self.device == "auto" else None
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        if device_map is None:
            target_device = self.device
            if target_device == "cuda" and not torch.cuda.is_available():
                target_device = "cpu"
            self._model.to(target_device)
        self._model.eval()

    def summarize(self, request: FrameSummaryRequest) -> str:
        self.warmup()
        tokenizer = self._tokenizer
        model = self._model
        if tokenizer is None or model is None:
            return request.heuristic_summary
        encoded = self._encode_request(tokenizer, request)
        model_device = next(model.parameters()).device
        encoded = {key: value.to(model_device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0.0,
                temperature=self.temperature if self.temperature > 0.0 else None,
                top_p=0.9 if self.temperature > 0.0 else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_ids = output[0][encoded["input_ids"].shape[1] :]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return self._clean_summary(text) or request.heuristic_summary

    def _encode_request(self, tokenizer, request: FrameSummaryRequest):
        prompt = self._build_prompt(request)
        messages = [
            {
                "role": "system",
                "content": (
                    "You write one short, factual summary of the current meeting screen state. "
                    "Focus on what happened, what the user likely did, and what the cursor was interacting with. "
                    "Do not output reasoning. Do not use bullets."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            return tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        except Exception:
            return tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)

    def _build_prompt(self, request: FrameSummaryRequest) -> str:
        return _build_detailed_screen_prompt(request)

    @staticmethod
    def _clean_summary(text: str) -> str:
        cleaned = str(text or "").strip()
        if "SUMMARY:" in cleaned:
            cleaned = cleaned.split("SUMMARY:", 1)[-1].strip()
        cleaned = cleaned.replace("<think>", "").replace("</think>", "").strip()
        return cleaned[:2400]


class GeminiFrameSummarizer(BaseFrameSummarizer):
    def __init__(
        self,
        *,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        max_output_tokens: int = 96,
        temperature: float = 0.1,
        timeout_sec: float = 4.0,
        max_retries: int = 0,
    ) -> None:
        self.api_key = str(api_key or "").strip()
        self.model_name = str(model_name or "gemini-2.5-flash").strip()
        self.provider_name = "gemini"
        self.max_output_tokens = int(max_output_tokens)
        self.temperature = float(temperature)
        self.timeout_sec = float(timeout_sec)
        self.max_retries = max(0, int(max_retries))
        self._warmed = False

    def warmup(self) -> None:
        if self._warmed:
            return
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")
        self._warmed = True

    def summarize(self, request: FrameSummaryRequest) -> str:
        self.warmup()
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": self._build_prompt(request),
                        }
                    ],
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
                "candidateCount": 1,
                "responseMimeType": "text/plain",
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        text = self._request_text(payload)
        cleaned = self._clean_summary(text)
        return cleaned or request.heuristic_summary

    def _request_text(self, payload: Dict[str, Any]) -> str:
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent?key={self.api_key}"
        )
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            response = requests.post(endpoint, json=payload, timeout=self.timeout_sec)
            if response.status_code != 429:
                response.raise_for_status()
                data = response.json()
                break
            last_error = requests.HTTPError(
                f"429 Client Error: Too Many Requests for url: {endpoint}",
                response=response,
            )
            if attempt < self.max_retries:
                time.sleep(1.5 * (attempt + 1))
        else:
            assert last_error is not None
            raise last_error
        candidates = data.get("candidates") or []
        if candidates and not any((candidate.get("content") or {}).get("parts") for candidate in candidates):
            return "OK"
        for candidate in candidates:
            content = candidate.get("content") or {}
            for part in content.get("parts") or []:
                text = str(part.get("text") or "").strip()
                if text:
                    return text
        return ""

    def _build_prompt(self, request: FrameSummaryRequest) -> str:
        return _build_detailed_screen_prompt(request)

    @staticmethod
    def _clean_summary(text: str) -> str:
        cleaned = str(text or "").strip()
        if "FINAL SUMMARY:" in cleaned:
            cleaned = cleaned.split("FINAL SUMMARY:", 1)[-1].strip()
        cleaned = cleaned.replace("<think>", "").replace("</think>", "").strip()
        cleaned = cleaned.replace("<answer>", "").replace("</answer>", "").strip()
        return cleaned[:2400]


class OpenAICompatibleFrameSummarizer(BaseFrameSummarizer):
    def __init__(
        self,
        *,
        provider_name: str,
        api_key: str,
        model_name: str,
        endpoint: str,
        max_output_tokens: int = 700,
        temperature: float = 0.1,
        timeout_sec: float = 8.0,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.provider_name = str(provider_name or "openai_compatible").strip()
        self.api_key = str(api_key or "").strip()
        self.model_name = str(model_name or "").strip()
        self.endpoint = str(endpoint or "").strip()
        self.max_output_tokens = int(max_output_tokens)
        self.temperature = float(temperature)
        self.timeout_sec = float(timeout_sec)
        self.extra_headers = dict(extra_headers or {})

    def warmup(self) -> None:
        if not self.api_key:
            raise RuntimeError(f"{self.provider_name} API key is not configured.")
        if not self.model_name:
            raise RuntimeError(f"{self.provider_name} model is not configured.")

    def summarize(self, request: FrameSummaryRequest) -> str:
        self.warmup()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You produce detailed, factual screen-understanding notes for a realtime meeting analyzer. "
                        "Use only supplied evidence and do not include hidden reasoning."
                    ),
                },
                {"role": "user", "content": _build_detailed_screen_prompt(request)},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }
        response = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout_sec)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        for choice in choices:
            message = choice.get("message") or {}
            text = str(message.get("content") or "").strip()
            if text:
                return self._clean_summary(text) or request.heuristic_summary
        return request.heuristic_summary

    @staticmethod
    def _clean_summary(text: str) -> str:
        cleaned = str(text or "").strip()
        for marker in ("DETAILED DESCRIPTION:", "FINAL SUMMARY:", "SUMMARY:"):
            if marker in cleaned:
                cleaned = cleaned.split(marker, 1)[-1].strip()
        cleaned = cleaned.replace("<think>", "").replace("</think>", "").strip()
        cleaned = cleaned.replace("<answer>", "").replace("</answer>", "").strip()
        return cleaned[:3600]


class CascadingFrameSummarizer(BaseFrameSummarizer):
    def __init__(self, summarizers: Sequence[BaseFrameSummarizer]) -> None:
        self.summarizers = [item for item in summarizers if item is not None]
        self.last_error = ""
        self.provider_name = "cascade"
        self.model_name = ",".join(getattr(item, "model_name", getattr(item, "provider_name", "")) for item in self.summarizers)

    def warmup(self) -> None:
        for summarizer in self.summarizers:
            try:
                summarizer.warmup()
                self.last_error = ""
                return
            except Exception as exc:
                self.last_error = f"{getattr(summarizer, 'provider_name', 'unknown')}: {type(exc).__name__}: {exc}"
        if self.summarizers:
            logger.warning("No cascade summarizer warmed successfully: %s", self.last_error)

    def summarize(self, request: FrameSummaryRequest) -> str:
        last_error = ""
        for summarizer in self.summarizers:
            try:
                text = summarizer.summarize(request)
                if text:
                    self.provider_name = getattr(summarizer, "provider_name", "unknown")
                    self.model_name = getattr(summarizer, "model_name", getattr(summarizer, "provider_name", "unknown"))
                    self.last_error = ""
                    return text
            except Exception as exc:
                last_error = f"{getattr(summarizer, 'provider_name', 'unknown')}: {type(exc).__name__}: {exc}"
                logger.warning("Frame summarizer fallback failed: %s", last_error)
        self.last_error = last_error
        return request.heuristic_summary


def _build_detailed_screen_prompt(request: FrameSummaryRequest) -> str:
    payload = request.current_payload or {}
    memory = [item.to_prompt_dict() for item in request.recent_memory[-6:]]
    macro_layout = payload.get("macro_layout") or []
    macro_types = [
        str(item.get("type") or item.get("label") or "").strip()
        for item in macro_layout
        if str(item.get("type") or item.get("label") or "").strip()
    ]
    micro_elements = payload.get("micro_elements") or []
    cursor_target = payload.get("cursor_target") or {}
    previous_frame = payload.get("previous_frame") or {}
    task = str(payload.get("summary_task") or "frame").strip()
    screen_structure = payload.get("screen_structure")
    if task in {"demo_region", "demo_relationship", "code_detailed", "slide_detailed", "paired_frame_change"}:
        screen_structure = _compact_screen_structure(screen_structure)
    normalized = {
        "time": round(float(request.timestamp_sec), 2),
        "mode": request.mode,
        "summary_task": task,
        "transcript_text": request.transcript_text,
        "cursor_text": request.cursor_text,
        "page_text": request.page_text,
        "action_text": request.action_text,
        "change_type": payload.get("change_type"),
        "changed_bbox": payload.get("changed_bbox"),
        "macro_types": macro_types,
        "macro_layout": _limit_dict_list(macro_layout, 12),
        "screen_structure": screen_structure,
        "ocr_lines": _limit_dict_list(payload.get("ocr_lines") or [], 80),
        "doclayout_elements": _limit_dict_list(payload.get("doclayout_elements") or [], 60),
        "slide_layout": _limit_dict_list(payload.get("layout") or [], 60),
        "full_text": str(payload.get("full_text") or "")[:3500],
        "code_text": str(payload.get("ocr_text") or "")[:5000],
        "ocr_diff": payload.get("ocr_diff"),
        "segment": payload.get("segment"),
        "segment_neighbors": payload.get("segment_neighbors"),
        "related_segments": payload.get("related_segments"),
        "micro_element_count": len(micro_elements),
        "top_visible_text": _collect_micro_highlights(micro_elements, limit=14),
        "cursor_target": {
            "text_or_type": cursor_target.get("element_text") or cursor_target.get("element_type"),
            "macro": cursor_target.get("parent_macro_region_type"),
            "relation": cursor_target.get("relation"),
            "confidence": cursor_target.get("confidence"),
        },
        "previous_frame": previous_frame,
        "recent_memory": memory,
        "heuristic_summary": request.heuristic_summary,
    }

    if task == "demo_region":
        instruction = (
            "Describe exactly one macro segment of a live UI. Be very detailed and concrete. "
            "Name the segment type, its approximate position, visible text in top-to-bottom/left-to-right order, spacing/alignment patterns, "
            "empty areas, dense clusters, and what the cursor is on if it falls inside this segment. "
            "Explain what this segment likely does in the interface. Do not invent text that is not present."
        )
    elif task == "demo_relationship":
        instruction = (
            "Describe the relationship between already-described UI segments. Explain how header, sidebars, footer, and main-content subregions "
            "relate spatially and functionally. Mention changed areas and cursor movement. Return concise prose, not JSON."
        )
    elif task == "paired_frame_change":
        instruction = (
            "Describe the current frame in detail and compare it to the previous deferred frame. "
            "Report what changed separately for header, left sidebar, right sidebar, footer, main content, and cursor when those regions exist. "
            "Focus mainly on the current screen, but preserve the before/after relationship clearly."
        )
    elif task == "code_detailed":
        instruction = (
            "Give a detailed code-screen description. Identify file/editor-like regions if visible, summarize the visible code text, explain changes from the OCR diff, "
            "and mention cursor position. Be precise about identifiers, functions, error messages, and panels when present."
        )
    elif task == "slide_detailed":
        instruction = (
            "Give a detailed slide-screen description. Use the slide OCR, slide layout blocks, and YOLO macro layout. Identify title, body structure, lists/tables/figures, "
            "relative placement, reading order, and cursor position. Include what changed from the previous slide if available."
        )
    else:
        instruction = (
            "Give a detailed live-screen description using macro layout, OCR, cursor, transcript, and recent memory. Prefer concrete visible text and region names."
        )

    return (
        "You are producing research-grade screen-understanding notes for a real-time meeting analyzer.\n"
        f"{instruction}\n"
        "Rules:\n"
        "- Use only the supplied state. If text is noisy, call it unreadable/noisy instead of guessing.\n"
        "- Preserve coordinates only as relative position language; do not dump raw JSON.\n"
        "- Mention cursor evidence exactly where it belongs.\n"
        "- Be detailed enough that someone who cannot see the image can reconstruct the screen.\n"
        "- Do not output chain-of-thought, markdown tables, or generic filler.\n\n"
        f"STATE:\n{json.dumps(normalized, ensure_ascii=True)}\n\n"
        "DETAILED DESCRIPTION:"
    )


def _limit_dict_list(items: Sequence[Any], limit: int) -> List[Any]:
    output: List[Any] = []
    for item in list(items or [])[: max(0, int(limit))]:
        if isinstance(item, dict):
            compact = {}
            for key in ("id", "type", "label", "bbox", "text", "caption", "confidence", "source", "children"):
                if key in item and key != "children":
                    value = item.get(key)
                    if isinstance(value, str):
                        value = value[:220]
                    compact[key] = value
            output.append(compact or item)
        else:
            output.append(item)
    return output


def _compact_screen_structure(structure: Any) -> Dict[str, Any]:
    if not isinstance(structure, dict):
        return {}
    compact_regions = []
    for region in list(structure.get("regions") or [])[:12]:
        if not isinstance(region, dict):
            continue
        compact_regions.append(
            {
                "id": region.get("id"),
                "type": region.get("type"),
                "label": region.get("label"),
                "bbox": region.get("bbox"),
                "element_count": region.get("element_count"),
                "ocr_line_count": region.get("ocr_line_count"),
                "sample_text": list(region.get("sample_text") or [])[:10],
                "element_types": region.get("element_types") or {},
            }
        )
    return {
        "interface_type": structure.get("interface_type"),
        "layout_description": structure.get("layout_description"),
        "regions": compact_regions,
        "highlights": list(structure.get("highlights") or [])[:12],
        "region_descriptions": list(structure.get("region_descriptions") or [])[:8],
    }


class ResilientFrameSummarizer(BaseFrameSummarizer):
    def __init__(
        self,
        primary: BaseFrameSummarizer,
        fallback: Optional[BaseFrameSummarizer] = None,
        *,
        failure_cooldown_sec: float = 60.0,
    ) -> None:
        self.primary = primary
        self.fallback = fallback or HeuristicFrameSummarizer()
        self.last_error: str = ""
        self.failure_cooldown_sec = float(failure_cooldown_sec)
        self._disabled_until = 0.0

    def warmup(self) -> None:
        try:
            self.primary.warmup()
            self.last_error = ""
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Frame summarizer warmup failed: %s", exc)

    def summarize(self, request: FrameSummaryRequest) -> str:
        if self._disabled_until and time.monotonic() < self._disabled_until:
            return self.fallback.summarize(request)
        try:
            text = self.primary.summarize(request)
            if text:
                self.last_error = ""
                self._disabled_until = 0.0
                return text
            if getattr(self.primary, "provider_name", "") == "gemini":
                self.last_error = "Gemini returned an empty response."
                self._disabled_until = time.monotonic() + self.failure_cooldown_sec
                return self.fallback.summarize(request)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Frame summarizer generation failed: %s", exc)
            if getattr(self.primary, "provider_name", "") == "gemini":
                self._disabled_until = time.monotonic() + self.failure_cooldown_sec
                return self.fallback.summarize(request)
        self.last_error = "Primary summarizer returned no text."
        self._disabled_until = time.monotonic() + self.failure_cooldown_sec
        return self.fallback.summarize(request)


def _looks_like_noise(text: str) -> bool:
    clean = str(text or "").strip()
    if not clean:
        return True
    if len(clean) < 2:
        return True
    if _BAD_TOKEN_RE.search(clean):
        return True
    if _SVGISH_RE.match(clean):
        return True
    alpha = sum(ch.isalpha() for ch in clean)
    if alpha == 0:
        return True
    if alpha / max(len(clean), 1) < 0.25 and len(clean) > 6:
        return True
    return False


def _collect_micro_highlights(micro_elements: Sequence[Dict[str, Any]], *, limit: int = 6) -> List[str]:
    seen: set[str] = set()
    highlights: List[str] = []
    for item in micro_elements:
        candidates = [
            str(item.get("text") or "").strip(),
            str(item.get("caption") or "").strip(),
            "" if str(item.get("type") or "").strip().lower() in _GENERIC_ELEMENT_TYPES else str(item.get("type") or "").strip(),
        ]
        for candidate in candidates:
            normalized = candidate.strip()
            key = normalized.lower()
            if not normalized or key in seen or _looks_like_noise(normalized):
                continue
            seen.add(key)
            highlights.append(normalized)
            break
        if len(highlights) >= limit:
            break
    return highlights


def _heuristic_demo_summary(payload: Dict[str, Any], transcript_text: str, cursor_text: str) -> str:
    page_summary = str(payload.get("page_summary", "") or "").strip()
    macro_types = [
        str(item.get("type") or item.get("label") or "").replace("_", " ").strip()
        for item in (payload.get("macro_layout") or [])
        if str(item.get("type") or item.get("label") or "").strip()
    ]
    highlights = _collect_micro_highlights(payload.get("micro_elements") or [], limit=6)
    cursor_target = payload.get("cursor_target") or {}
    change_type = str(payload.get("change_type", "") or "").replace("_", " ").strip()
    parts: List[str] = []
    if page_summary:
        parts.append(page_summary.rstrip(".") + ".")
        if highlights:
            parts.append(f"Visible items include {', '.join(highlights[:5])}.")
    elif macro_types or highlights:
        macro_text = ", ".join(dict.fromkeys(macro_types[:4])) if macro_types else "application interface"
        parts.append(f"The screen shows {macro_text}.")
        if highlights:
            parts.append(f"Visible items include {', '.join(highlights[:5])}.")
    if change_type and change_type not in {"none", "initial"}:
        article = "an" if change_type[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
        parts.append(f"The visible change is {article} {change_type}.")
    if transcript_text:
        parts.append(f"The speaker said: {transcript_text}")
    target_micro = str(cursor_target.get("element_text") or cursor_target.get("element_type") or "").strip()
    target_macro = str(cursor_target.get("parent_macro_region_type") or "").replace("_", " ").strip()
    if target_micro:
        parts.append(f"The cursor is over {target_micro}{f' in the {target_macro}' if target_macro else ''}.")
    elif target_macro:
        parts.append(f"The cursor is in the {target_macro}.")
    elif cursor_text and cursor_text != "cursor not visible":
        parts.append(cursor_text)
    return " ".join(part.strip() for part in parts if part and part.strip())[:700]

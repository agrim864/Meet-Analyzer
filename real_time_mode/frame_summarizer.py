from __future__ import annotations

import json
import logging
import re
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
        memory = [item.to_prompt_dict() for item in request.recent_memory[-8:]]
        payload = {
            "time": round(float(request.timestamp_sec), 2),
            "mode": request.mode,
            "transcript_text": request.transcript_text,
            "cursor_text": request.cursor_text,
            "page_text": request.page_text,
            "action_text": request.action_text,
            "visual_payload": request.current_payload,
            "recent_memory": memory,
            "heuristic_summary": request.heuristic_summary,
        }
        return (
            "You summarize a live meeting screen state.\n"
            "Write one short sentence about what happened on screen and what the user likely did.\n"
            "Use the cursor and transcript when helpful. Preserve continuity with recent memory.\n"
            "Do not mention uncertainty unless necessary. Do not output bullets or JSON.\n\n"
            f"STATE:\n{json.dumps(payload, ensure_ascii=True)}\n\n"
            "SUMMARY:"
        )

    @staticmethod
    def _clean_summary(text: str) -> str:
        cleaned = str(text or "").strip()
        if "SUMMARY:" in cleaned:
            cleaned = cleaned.split("SUMMARY:", 1)[-1].strip()
        cleaned = cleaned.replace("<think>", "").replace("</think>", "").strip()
        first_line = cleaned.splitlines()[0].strip() if cleaned else ""
        return first_line[:500]


class GeminiFrameSummarizer(BaseFrameSummarizer):
    def __init__(
        self,
        *,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        max_output_tokens: int = 96,
        temperature: float = 0.1,
        timeout_sec: float = 30.0,
    ) -> None:
        self.api_key = str(api_key or "").strip()
        self.model_name = str(model_name or "gemini-2.5-flash").strip()
        self.provider_name = "gemini"
        self.max_output_tokens = int(max_output_tokens)
        self.temperature = float(temperature)
        self.timeout_sec = float(timeout_sec)
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
        for attempt in range(3):
            response = requests.post(endpoint, json=payload, timeout=self.timeout_sec)
            if response.status_code != 429:
                response.raise_for_status()
                data = response.json()
                break
            last_error = requests.HTTPError(
                f"429 Client Error: Too Many Requests for url: {endpoint}",
                response=response,
            )
            if attempt < 2:
                import time

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
        macro_layout = request.current_payload.get("macro_layout") or []
        macro_types = [
            str(item.get("type") or item.get("label") or "").strip()
            for item in macro_layout
            if str(item.get("type") or item.get("label") or "").strip()
        ]
        micro_elements = request.current_payload.get("micro_elements") or []
        cursor_target = request.current_payload.get("cursor_target") or {}
        highlights = _collect_micro_highlights(micro_elements, limit=8)
        payload = {
            "time": round(float(request.timestamp_sec), 2),
            "mode": request.mode,
            "transcript_text": request.transcript_text,
            "cursor_text": request.cursor_text,
            "page_text": request.page_text,
            "action_text": request.action_text,
            "page_summary": request.current_payload.get("page_summary"),
            "change_type": request.current_payload.get("change_type"),
            "changed_bbox": request.current_payload.get("changed_bbox"),
            "macro_types": macro_types,
            "micro_element_count": len(micro_elements),
            "top_micro_elements": highlights,
            "cursor_target": {
                "micro": cursor_target.get("element_text") or cursor_target.get("element_type"),
                "macro": cursor_target.get("parent_macro_region_type"),
                "relation": cursor_target.get("relation"),
            },
            "heuristic_summary": request.heuristic_summary,
        }
        return (
            "You summarize the current state of a live meeting screen.\n"
            "Write 2 or 3 concise sentences.\n"
            "Sentence 1: clearly name what app, page, panel, editor, or UI content is visible and what changed.\n"
            "Sentence 2: summarize what the speaker said, if any transcript is present.\n"
            "Sentence 3: say where the cursor is, including the micro element and macro region if available.\n"
            "Use only the current state below. Ignore older memory and do not mention missing context.\n"
            "Be specific. Prefer concrete UI names, visible controls, panel names, change type, and cursor target.\n"
            "Do not say generic phrases like 'main content' unless that is all you truly know.\n"
            "Ignore gibberish labels, SVG path fragments, raw coordinates, and duplicated words.\n"
            "Do not use bullets, JSON, markdown headings, or chain-of-thought.\n\n"
            f"STATE:\n{json.dumps(payload, ensure_ascii=True)}\n\n"
            "FINAL SUMMARY:"
        )

    @staticmethod
    def _clean_summary(text: str) -> str:
        cleaned = str(text or "").strip()
        if "FINAL SUMMARY:" in cleaned:
            cleaned = cleaned.split("FINAL SUMMARY:", 1)[-1].strip()
        cleaned = cleaned.replace("<think>", "").replace("</think>", "").strip()
        cleaned = cleaned.replace("<answer>", "").replace("</answer>", "").strip()
        return cleaned[:700]


class ResilientFrameSummarizer(BaseFrameSummarizer):
    def __init__(self, primary: BaseFrameSummarizer, fallback: Optional[BaseFrameSummarizer] = None) -> None:
        self.primary = primary
        self.fallback = fallback or HeuristicFrameSummarizer()
        self.last_error: str = ""

    def warmup(self) -> None:
        try:
            self.primary.warmup()
            self.last_error = ""
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Frame summarizer warmup failed: %s", exc)

    def summarize(self, request: FrameSummaryRequest) -> str:
        try:
            text = self.primary.summarize(request)
            if text:
                self.last_error = ""
                return text
            if getattr(self.primary, "provider_name", "") == "gemini":
                self.last_error = "Gemini returned an empty response."
                return "Gemini not available."
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Frame summarizer generation failed: %s", exc)
            if getattr(self.primary, "provider_name", "") == "gemini":
                return "Gemini not available."
        self.last_error = "Primary summarizer returned no text."
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

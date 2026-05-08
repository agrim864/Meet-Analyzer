from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Sequence

import requests

logger = logging.getLogger(__name__)


class SectionWriter(Protocol):
    def generate(self, *, sections: Dict[str, Any]) -> Dict[str, Any]:
        ...


@dataclass
class LLMConfig:
    backend: str
    model: str
    base_url: str = ""
    api_key: str = ""
    timeout_sec: int = 180
    mock_enabled: bool = True


class MockSectionWriter:
    def generate(self, *, sections: Dict[str, Any]) -> Dict[str, Any]:
        ordered_keys = _ordered_section_keys(sections)
        output: Dict[str, Any] = {"backend": "mock", "model": "mock-writer"}
        for key in ordered_keys:
            payload = sections.get(key, {})
            output[key] = _prewritten_section(key, payload) or self._summarize_block(key, payload)
        output["final_markdown"] = _compose_final_markdown(output, ordered_keys)
        return output

    def _summarize_block(self, key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        snippets: List[str] = []
        if isinstance(payload, dict):
            for value in payload.values():
                if isinstance(value, str) and value.strip():
                    snippets.append(value.strip())
                elif isinstance(value, list) and value:
                    snippets.append(str(value[0])[:400])
        summary = " ".join(snippets)[:1200] if snippets else f"No material available for {key}."
        return {"summary": summary}


class OpenAICompatibleSectionWriter:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def generate(self, *, sections: Dict[str, Any]) -> Dict[str, Any]:
        from openai import OpenAI

        output: Dict[str, Any] = {"backend": self.config.backend, "model": self.config.model}
        ordered_keys = _ordered_section_keys(sections)
        def generate_one(name: str) -> tuple[str, Dict[str, str]]:
            client = OpenAI(base_url=self.config.base_url, api_key=self.config.api_key or "EMPTY")
            payload = sections.get(name, {})
            prewritten = _prewritten_section(name, payload)
            if prewritten:
                return name, prewritten
            logger.info("Generating report section with OpenAI-compatible model: %s", name)
            prompt = _section_prompt(name, payload)
            response = client.responses.create(model=self.config.model, input=prompt, timeout=self.config.timeout_sec)
            text = getattr(response, "output_text", "") or ""
            summary = _guard_generated_section(name, payload, _clean_generated_text(text, name))
            return name, {"summary": summary}
        output.update(_generate_sections_parallel(ordered_keys, generate_one))
        output["final_markdown"] = _compose_final_markdown(output, ordered_keys)
        return output


class OllamaSectionWriter:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def generate(self, *, sections: Dict[str, Any]) -> Dict[str, Any]:
        output: Dict[str, Any] = {"backend": "ollama", "model": self.config.model}
        ordered_keys = _ordered_section_keys(sections)
        def generate_one(name: str) -> tuple[str, Dict[str, str]]:
            payload = sections.get(name, {})
            prewritten = _prewritten_section(name, payload)
            if prewritten:
                return name, prewritten
            prompt = _section_prompt(name, payload)
            response = requests.post(
                f"{self.config.base_url.rstrip('/')}/api/generate",
                json={"model": self.config.model, "prompt": prompt, "stream": False},
                timeout=self.config.timeout_sec,
            )
            response.raise_for_status()
            data = response.json()
            summary = _guard_generated_section(name, payload, _clean_generated_text(str(data.get("response", "")), name))
            return name, {"summary": summary}
        output.update(_generate_sections_parallel(ordered_keys, generate_one))
        output["final_markdown"] = _compose_final_markdown(output, ordered_keys)
        return output


class VLLMSectionWriter(OpenAICompatibleSectionWriter):
    pass


class OpenRouterSectionWriter:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def generate(self, *, sections: Dict[str, Any]) -> Dict[str, Any]:
        output: Dict[str, Any] = {"backend": "openrouter", "model": self.config.model}
        ordered_keys = _ordered_section_keys(sections)

        def generate_one(name: str) -> tuple[str, Dict[str, str]]:
            payload = sections.get(name, {})
            prewritten = _prewritten_section(name, payload)
            if prewritten:
                return name, prewritten
            prompt = _section_prompt(name, payload)
            logger.info("Generating report section with OpenRouter model: %s", name)
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.api_key or os.getenv('OPENROUTER_API_KEY', '')}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": os.getenv("BATCH_OPENROUTER_REFERER", "http://localhost"),
                    "X-Title": os.getenv("BATCH_OPENROUTER_TITLE", "meet-agent batch report"),
                },
                json={
                    "model": self.config.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You write factual meeting-analysis report sections from structured screen "
                                "and transcript data. Do not invent events. Return only the section body "
                                "inside <answer>...</answer> tags."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": float(os.getenv("BATCH_LLM_TEMPERATURE", "0.2")),
                    "top_p": float(os.getenv("BATCH_LLM_TOP_P", "0.95")),
                    "max_tokens": int(os.getenv("BATCH_LLM_MAX_NEW_TOKENS", "4096")),
                },
                timeout=self.config.timeout_sec,
            )
            if not response.ok:
                try:
                    detail = response.json().get("error", {}).get("message", response.text)
                except Exception:
                    detail = response.text
                raise RuntimeError(f"OpenRouter request failed for {name} with HTTP {response.status_code}: {str(detail)[:500]}")
            data = response.json()
            text = str((data.get("choices") or [{}])[0].get("message", {}).get("content", "")).strip()
            summary = _guard_generated_section(name, payload, _clean_generated_text(text, name))
            return name, {"summary": summary}

        output.update(_generate_sections_parallel(ordered_keys, generate_one))
        output["final_markdown"] = _compose_final_markdown(output, ordered_keys)
        return output


class GeminiSectionWriter:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def generate(self, *, sections: Dict[str, Any]) -> Dict[str, Any]:
        model_name = (self.config.model or "gemini-2.5-pro").removeprefix("models/")
        output: Dict[str, Any] = {"backend": "gemini", "model": model_name}
        ordered_keys = _ordered_section_keys(sections)

        def generate_one(name: str) -> tuple[str, Dict[str, str]]:
            payload = sections.get(name, {})
            prewritten = _prewritten_section(name, payload)
            if prewritten:
                return name, prewritten
            prompt = _section_prompt(name, payload)
            logger.info("Generating report section with Gemini model: %s", name)
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent",
                params={"key": self.config.api_key or os.getenv("GEMINI_API_KEY", "")},
                json={
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": float(os.getenv("BATCH_LLM_TEMPERATURE", "0.2")),
                        "topP": float(os.getenv("BATCH_LLM_TOP_P", "0.95")),
                        "maxOutputTokens": int(os.getenv("BATCH_LLM_MAX_NEW_TOKENS", "4096")),
                    },
                    "systemInstruction": {
                        "parts": [
                            {
                                "text": (
                                    "You write factual meeting-analysis report sections from structured screen "
                                    "and transcript data. Do not invent events. Return only the section body "
                                    "inside <answer>...</answer> tags."
                                )
                            }
                        ]
                    },
                },
                timeout=self.config.timeout_sec,
            )
            response.raise_for_status()
            data = response.json()
            parts = (
                (data.get("candidates") or [{}])[0]
                .get("content", {})
                .get("parts", [])
            )
            text = "\n".join(str(part.get("text", "")) for part in parts if isinstance(part, dict)).strip()
            summary = _guard_generated_section(name, payload, _clean_generated_text(text, name))
            return name, {"summary": summary}

        output.update(_generate_sections_parallel(ordered_keys, generate_one))
        output["final_markdown"] = _compose_final_markdown(output, ordered_keys)
        return output


class HFLocalSectionWriter:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._tokenizer = None
        self._model = None

    def generate(self, *, sections: Dict[str, Any]) -> Dict[str, Any]:
        tokenizer, model = self._ensure_model()
        output: Dict[str, Any] = {"backend": "hf-local", "model": self.config.model}
        ordered_keys = _ordered_section_keys(sections)
        for name in ordered_keys:
            payload = sections.get(name, {})
            prewritten = _prewritten_section(name, payload)
            if prewritten:
                output[name] = prewritten
                logger.info("Using prewritten report section: %s", name)
                continue
            prompt = _section_prompt(name, payload)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You write factual sections for meeting analysis reports. "
                        "Do not reveal chain-of-thought. Do not output <think> blocks. "
                        "Return the final section body inside <answer>...</answer> tags only."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            try:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(
                rendered,
                return_tensors="pt",
                truncation=True,
                max_length=int(os.getenv("BATCH_LLM_MAX_INPUT_TOKENS", "8192")),
            ).to(model.device)
            generated = model.generate(
                **inputs,
                max_new_tokens=int(os.getenv("BATCH_LLM_MAX_NEW_TOKENS", "256")),
                do_sample=False,
                temperature=None,
                top_p=None,
            )
            new_tokens = generated[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            output[name] = {"summary": _guard_generated_section(name, payload, _clean_generated_text(text, name))}
            logger.info("Finished report section with hf-local model: %s", name)
        output["final_markdown"] = _compose_final_markdown(output, ordered_keys)
        return output

    def _ensure_model(self):
        if self._model is not None and self._tokenizer is not None:
            return self._tokenizer, self._model
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        common_kwargs = {"trust_remote_code": True}
        model_name_lower = self.config.model.lower()
        if "mistral-small-3.2-24b" in model_name_lower:
            common_kwargs["fix_mistral_regex"] = True
        if _huggingface_offline():
            common_kwargs["local_files_only"] = True
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model, **common_kwargs)
        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": "auto",
            **common_kwargs,
        }
        quantization = os.getenv("BATCH_HF_QUANTIZATION", "").strip().lower()
        if not quantization and "mistral-small-3.2-24b" in model_name_lower and "bnb-4bit" not in model_name_lower:
            quantization = "4bit"
        if quantization in {"4bit", "bnb-4bit", "bitsandbytes-4bit"}:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise RuntimeError(
                    "BATCH_HF_QUANTIZATION=4bit requires bitsandbytes. "
                    "Install batch_mode/requirements.txt in the cluster environment."
                ) from exc
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=os.getenv("BATCH_HF_BNB_4BIT_TYPE", "nf4"),
                bnb_4bit_compute_dtype=getattr(torch, os.getenv("BATCH_HF_BNB_COMPUTE_DTYPE", "bfloat16")),
                bnb_4bit_use_double_quant=os.getenv("BATCH_HF_BNB_DOUBLE_QUANT", "1").strip().lower() in {"1", "true", "yes"},
            )
            model_kwargs.pop("torch_dtype", None)
        elif quantization in {"8bit", "bnb-8bit", "bitsandbytes-8bit"}:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise RuntimeError(
                    "BATCH_HF_QUANTIZATION=8bit requires bitsandbytes. "
                    "Install batch_mode/requirements.txt in the cluster environment."
                ) from exc
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs.pop("torch_dtype", None)
        self._model = AutoModelForCausalLM.from_pretrained(self.config.model, **model_kwargs)
        self._model.eval()
        return self._tokenizer, self._model


def build_section_writer(config: LLMConfig) -> SectionWriter:
    backend = (config.backend or "mock").strip().lower()
    if backend in {"openai", "openai-compatible", "local-openai"} and config.base_url:
        return OpenAICompatibleSectionWriter(config)
    if backend == "ollama" and config.base_url:
        return OllamaSectionWriter(config)
    if backend == "vllm" and config.base_url:
        return VLLMSectionWriter(config)
    if backend in {"openrouter", "openrouter-chat"}:
        return OpenRouterSectionWriter(config)
    if backend in {"gemini", "google", "google-gemini"}:
        return GeminiSectionWriter(config)
    if backend in {"hf-local", "transformers-local"}:
        return HFLocalSectionWriter(config)
    return MockSectionWriter()


def _section_prompt(name: str, payload: Dict[str, Any]) -> str:
    if not _should_use_prewritten(name) and isinstance(payload, dict):
        payload = {key: value for key, value in payload.items() if key != "prewritten_markdown"}
    compact_payload = _compact_for_prompt(payload)
    structured_input = json.dumps(compact_payload, ensure_ascii=True, indent=2)
    max_chars = int(os.getenv("BATCH_LLM_MAX_PROMPT_CHARS", "24000"))
    if len(structured_input) > max_chars:
        structured_input = structured_input[:max_chars] + "\n... [truncated for prompt budget]"
    if name == "meeting_narrative_section":
        guidance = (
            "Write a detailed chronological narrative of the meeting. Describe what was on screen, "
            "what the presenter/user did, and what was said. Preserve useful timestamps. "
            "Use the narrative_units in order. Include enough detail that a reader can follow the entire meeting "
            "without watching the video. Use paragraphs and bullets where helpful; do not compress the flow into a short summary. "
            "Preserve transcript speaker attribution exactly: evaluator speech must not be described as presenter speech, "
            "and combined transcript excerpts from multiple speakers should be summarized as separate turns. "
            "Use page_summary, screen_structure, cursor_target, and cursor_action to explain the visible UI and user flow; "
            "describe what cursor focus or navigation was followed by when that evidence is present. "
            "Avoid vague change labels like small region change unless the visible screen effect is unknown. "
            "If transcript text is missing, do not create dialogue, speaker names, meeting titles, decisions, or project names. "
            "Treat noisy OCR as uncertain screen text, not as spoken content.\n"
        )
    elif name == "transcript_section":
        guidance = "Keep the transcript content intact and organized by timestamp and speaker.\n"
    else:
        guidance = "Write a concise factual section using only the provided material.\n"
    return (
        f"You are writing the {name} for a meeting analysis report.\n"
        f"{guidance}"
        "Use only facts that appear in the structured input. Never invent participants, companies, agendas, "
        "times of day, slide titles, decisions, quotes, or project names. "
        "Output only the final section body wrapped in <answer>...</answer>. "
        "Do not include analysis, hidden reasoning, or the section title outside those tags.\n"
        f"Section name: {name}\n"
        f"Structured input:\n{structured_input}"
    )


def _huggingface_offline() -> bool:
    return os.getenv("HF_HUB_OFFLINE", "").strip() in {"1", "true", "TRUE"} or os.getenv("TRANSFORMERS_OFFLINE", "").strip() in {"1", "true", "TRUE"}


def _ordered_section_keys(sections: Dict[str, Any]) -> List[str]:
    preferred = [
        "report_title",
        "meeting_narrative_section",
        "slides_section",
        "code_section",
        "demo_section",
        "chronological_section",
        "topic_section",
        "decisions_section",
        "transcript_section",
    ]
    seen = [key for key in preferred if key in sections]
    seen.extend(key for key in sections if key not in seen)
    return seen


def _prewritten_section(name: str, payload: Any) -> Dict[str, str] | None:
    if not _should_use_prewritten(name):
        return None
    if isinstance(payload, str) and payload.strip():
        return {"summary": payload.strip()}
    if isinstance(payload, dict):
        markdown = str(payload.get("prewritten_markdown", "") or "").strip()
        if markdown:
            return {"summary": markdown}
    return None


def _should_use_prewritten(name: str) -> bool:
    if name in {"report_title", "transcript_section"}:
        return True
    value = os.getenv("BATCH_LLM_USE_PREWRITTEN", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _generate_sections_parallel(ordered_keys: Sequence[str], generate_one) -> Dict[str, Dict[str, str]]:
    max_workers = max(1, int(os.getenv("BATCH_LLM_SECTION_WORKERS", "4")))
    if max_workers <= 1 or len(ordered_keys) <= 1:
        return dict(generate_one(key) for key in ordered_keys)
    output: Dict[str, Dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(ordered_keys))) as executor:
        future_to_key = {executor.submit(generate_one, key): key for key in ordered_keys}
        for future in as_completed(future_to_key):
            key, value = future.result()
            output[key] = value
    return output


def _compose_final_markdown(output: Dict[str, Any], ordered_keys: Sequence[str]) -> str:
    chunks: List[str] = []
    title = str(output.get("report_title", {}).get("summary", "")).strip()
    if title:
        chunks.append(f"# {title}")
    for key in ordered_keys:
        if key == "report_title":
            continue
        summary = str(output.get(key, {}).get("summary", "")).strip()
        if not summary:
            continue
        chunks.append(f"## {key.replace('_', ' ').title()}\n{summary}")
    return "\n\n".join(chunks).strip()


def _compact_for_prompt(value: Any, *, depth: int = 0) -> Any:
    if depth > 5:
        return _truncate_text(str(value), 500)
    if isinstance(value, str):
        return _truncate_text(value, 1600)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        limit = 80 if _looks_like_transcript_list(value) else 40
        compacted = [_compact_for_prompt(item, depth=depth + 1) for item in value[:limit]]
        if len(value) > limit:
            compacted.append({"omitted_items": len(value) - limit})
        return compacted
    if isinstance(value, dict):
        if _looks_like_event(value):
            return _compact_event(value, depth=depth)
        drop_keys = {
            "elements",
            "overlays",
            "layout_groups",
            "debug",
            "confidence",
            "delta",
            "viewport",
            "theme",
            "cursor",
            "macro_layout",
            "reparsed_regions",
            "changed_regions",
        }
        compact: Dict[str, Any] = {}
        for key, item in value.items():
            if key in drop_keys:
                continue
            compact[key] = _compact_for_prompt(item, depth=depth + 1)
        return compact
    return _truncate_text(str(value), 800)


def _compact_event(event: Dict[str, Any], *, depth: int) -> Dict[str, Any]:
    visual_payload = event.get("visual_payload") or {}
    compact: Dict[str, Any] = {
        "event_id": event.get("event_id"),
        "timestamp_sec": event.get("timestamp_sec"),
        "mode": event.get("mode") or event.get("frame_type"),
        "summary": _truncate_text(str(event.get("summary", "") or ""), 600),
    }
    if isinstance(visual_payload, dict):
        compact["visual"] = {}
        for key in (
            "mode",
            "visual_change",
            "change_type",
            "ocr_text",
            "full_text",
            "page_summary",
            "screen_structure",
            "cursor_target",
            "cursor_action",
            "slide_fingerprint",
        ):
            if key in visual_payload:
                compact["visual"][key] = _compact_for_prompt(visual_payload[key], depth=depth + 1)
        layout = visual_payload.get("layout")
        if isinstance(layout, list):
            compact["visual"]["layout"] = _compact_for_prompt(layout[:12], depth=depth + 1)
    transcript = event.get("transcript_window")
    if transcript:
        compact["transcript_window"] = _compact_for_prompt(transcript, depth=depth + 1)
    return compact


def _looks_like_event(value: Dict[str, Any]) -> bool:
    return "event_id" in value and ("timestamp_sec" in value or "frame_id" in value)


def _looks_like_transcript_list(value: Sequence[Any]) -> bool:
    return bool(value) and all(isinstance(item, dict) and "text" in item and ("start" in item or "start_time" in item) for item in value[:5])


def _truncate_text(text: str, limit: int) -> str:
    cleaned = str(text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


def _clean_generated_text(text: str, section_name: str) -> str:
    cleaned = str(text or "").strip()
    answer_match = re.search(r"<answer>(.*?)</answer>", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if answer_match:
        cleaned = answer_match.group(1).strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    if re.search(r"</think>", cleaned, flags=re.IGNORECASE):
        cleaned = re.split(r"</think>", cleaned, flags=re.IGNORECASE)[-1].strip()
    if re.search(r"<think>", cleaned, flags=re.IGNORECASE):
        cleaned = re.split(r"<think>", cleaned, flags=re.IGNORECASE)[0].strip()
    cleaned = re.sub(r"</?answer>", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^```(?:markdown|md)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.IGNORECASE)
    section_prefixes = [
        section_name,
        section_name.replace("_", " "),
        section_name.replace("_section", "").replace("_", " "),
    ]
    for prefix in section_prefixes:
        cleaned = re.sub(
            rf"^\s*\*{{0,2}}{re.escape(prefix)}\*{{0,2}}\s*:\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
    if cleaned.lower().startswith("summary:"):
        cleaned = cleaned.split(":", 1)[1].strip()
    if "<think>" in cleaned.lower():
        cleaned = re.sub(r"(?is)^.*?<think>", "", cleaned).strip()
        cleaned = re.sub(r"(?is)</think>.*$", "", cleaned).strip()
    return cleaned


def _guard_generated_section(name: str, payload: Any, cleaned: str) -> str:
    if not isinstance(payload, dict):
        return cleaned
    fallback = str(payload.get("prewritten_markdown", "") or "").strip()
    if not fallback:
        return cleaned
    lower = cleaned.lower()
    stock_hallucination_markers = (
        "q2 project review",
        "project alpha",
        "project beta",
        "project gamma",
        "alex ",
        "sarah ",
        "john ",
        "budget overruns",
        "resource constraints",
    )
    if any(marker in lower for marker in stock_hallucination_markers):
        logger.warning("Replacing likely hallucinated %s with evidence-based fallback.", name)
        return fallback
    if name == "transcript_section":
        segments = payload.get("segments")
        if (not segments) and "no transcript segments were available" not in lower:
            return fallback
    return cleaned

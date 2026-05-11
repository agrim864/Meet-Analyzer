from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Protocol, Sequence

import requests

logger = logging.getLogger(__name__)


class SectionWriter(Protocol):
    def generate(self, *, sections: Dict[str, Any]) -> Dict[str, Any]:
        ...


class FrameDescriptionWriter(Protocol):
    def enrich_events(self, frame_events: List[Dict[str, Any]], *, max_demo_main_segments: int = 6) -> List[Dict[str, Any]]:
        ...


@dataclass
class LLMConfig:
    backend: str
    model: str
    base_url: str = ""
    api_key: str = ""
    timeout_sec: int = 180
    mock_enabled: bool = True


class MockFrameDescriptionWriter:
    def enrich_events(self, frame_events: List[Dict[str, Any]], *, max_demo_main_segments: int = 6) -> List[Dict[str, Any]]:
        enriched: List[Dict[str, Any]] = []
        previous: Dict[str, Any] | None = None
        for event in frame_events:
            item = dict(event)
            visual = dict(item.get("visual_payload") or {})
            mode = str(item.get("mode", item.get("frame_type", "")) or "")
            if mode in {"code", "slide"}:
                detail = _fallback_frame_description(item, previous)
                visual["llm_detail"] = {
                    "backend": "mock",
                    "frame_description": detail,
                    "summary": detail,
                }
            elif mode == "demo":
                visual["llm_detail"] = _fallback_demo_detail(item, previous, max_demo_main_segments=max_demo_main_segments)
            item["visual_payload"] = visual
            enriched.append(item)
            previous = item
        return enriched


class BatchFrameDescriptionWriter:
    def __init__(self, config: LLMConfig, *, max_workers: int = 6) -> None:
        self.config = config
        self.max_workers = max(1, int(max_workers))
        self._tokenizer = None
        self._model = None
        self._hf_lock = Lock()

    def enrich_events(self, frame_events: List[Dict[str, Any]], *, max_demo_main_segments: int = 6) -> List[Dict[str, Any]]:
        if (self.config.backend or "mock").strip().lower() == "mock":
            return MockFrameDescriptionWriter().enrich_events(frame_events, max_demo_main_segments=max_demo_main_segments)
        indexed = [(idx, dict(event), dict(frame_events[idx - 1]) if idx > 0 else None) for idx, event in enumerate(frame_events)]
        if self.max_workers <= 1 or len(indexed) <= 1:
            results = [self._enrich_one(idx, event, previous, max_demo_main_segments) for idx, event, previous in indexed]
        else:
            results = []
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(indexed))) as executor:
                futures = [
                    executor.submit(self._enrich_one, idx, event, previous, max_demo_main_segments)
                    for idx, event, previous in indexed
                ]
                for future in as_completed(futures):
                    results.append(future.result())
        results.sort(key=lambda item: item[0])
        return [item[1] for item in results]

    def _enrich_one(
        self,
        idx: int,
        event: Dict[str, Any],
        previous: Dict[str, Any] | None,
        max_demo_main_segments: int,
    ) -> tuple[int, Dict[str, Any]]:
        item = dict(event)
        visual = dict(item.get("visual_payload") or {})
        mode = str(item.get("mode", item.get("frame_type", "")) or "")
        try:
            if mode in {"code", "slide"}:
                prompt = _code_slide_frame_prompt(item, previous)
                description = self._generate_text(prompt)
                visual["llm_detail"] = {
                    "backend": self.config.backend,
                    "model": self.config.model,
                    "frame_description": _clean_generated_text(description, f"{mode}_frame_description"),
                }
            elif mode == "demo":
                visual["llm_detail"] = self._describe_demo_event(item, previous, max_demo_main_segments)
        except Exception as exc:
            logger.exception("Frame-level LLM description failed for event=%s", item.get("event_id"))
            visual["llm_detail"] = {
                "backend": "fallback-mock",
                "error": str(exc),
                **(_fallback_demo_detail(item, previous, max_demo_main_segments=max_demo_main_segments) if mode == "demo" else {"frame_description": _fallback_frame_description(item, previous)}),
            }
        item["visual_payload"] = visual
        return idx, item

    def _describe_demo_event(self, event: Dict[str, Any], previous: Dict[str, Any] | None, max_demo_main_segments: int) -> Dict[str, Any]:
        segments = _demo_segments_from_event(event, max_demo_main_segments=max_demo_main_segments)
        descriptions: Dict[str, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=min(max(1, self.max_workers), max(1, len(segments)))) as executor:
            futures = {
                executor.submit(self._generate_text, _demo_segment_prompt(event, segment)): segment
                for segment in segments
            }
            for future in as_completed(futures):
                segment = futures[future]
                descriptions[segment["key"]] = {
                    "label": segment["label"],
                    "bbox": segment.get("bbox"),
                    "description": _clean_generated_text(future.result(), f"{segment['key']}_description"),
                }
        relationship = self._generate_text(_demo_relationship_prompt(event, list(descriptions.values())))
        page_summary = self._generate_text(_demo_page_summary_prompt(event, previous, descriptions, relationship))
        combined = {
            key: descriptions[key]["description"]
            for key in ("header", "left_sidebar", "right_sidebar", "footer")
            if key in descriptions
        }
        main_parts = [
            value["description"]
            for key, value in sorted(descriptions.items())
            if key.startswith("main_content")
        ]
        if main_parts:
            combined["main_content"] = main_parts
        return {
            "backend": self.config.backend,
            "model": self.config.model,
            "segments": descriptions,
            "relationships": _clean_generated_text(relationship, "demo_relationships"),
            "structured_output": combined,
            "page_summary": _clean_generated_text(page_summary, "demo_page_summary"),
        }

    def _generate_text(self, prompt: str) -> str:
        backend = (self.config.backend or "mock").strip().lower()
        if backend == "mock":
            return "Mock detailed description unavailable for this backend."
        if backend in {"openai", "openai-compatible", "local-openai", "vllm"} and self.config.base_url:
            from openai import OpenAI

            client = OpenAI(base_url=self.config.base_url, api_key=self.config.api_key or "EMPTY")
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": _FRAME_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=float(os.getenv("BATCH_LLM_TEMPERATURE", "0.2")),
                max_tokens=int(os.getenv("BATCH_FRAME_DESCRIPTION_MAX_NEW_TOKENS", os.getenv("BATCH_LLM_MAX_NEW_TOKENS", "3072"))),
                timeout=self.config.timeout_sec,
            )
            return str(response.choices[0].message.content or "").strip()
        if backend in {"gemini", "google", "google-gemini"}:
            model_name = (self.config.model or "gemini-2.5-flash").removeprefix("models/")
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent",
                params={"key": self.config.api_key or os.getenv("GEMINI_API_KEY", "")},
                json={
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": float(os.getenv("BATCH_LLM_TEMPERATURE", "0.2")),
                        "topP": float(os.getenv("BATCH_LLM_TOP_P", "0.95")),
                        "maxOutputTokens": int(os.getenv("BATCH_FRAME_DESCRIPTION_MAX_NEW_TOKENS", os.getenv("BATCH_LLM_MAX_NEW_TOKENS", "4096"))),
                    },
                    "systemInstruction": {"parts": [{"text": _FRAME_SYSTEM_PROMPT}]},
                },
                timeout=self.config.timeout_sec,
            )
            response.raise_for_status()
            parts = ((response.json().get("candidates") or [{}])[0].get("content", {}).get("parts", []))
            return "\n".join(str(part.get("text", "")) for part in parts if isinstance(part, dict)).strip()
        if backend in {"openrouter", "openrouter-chat"}:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.api_key or os.getenv('OPENROUTER_API_KEY', '')}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": os.getenv("BATCH_OPENROUTER_REFERER", "http://localhost"),
                    "X-Title": os.getenv("BATCH_OPENROUTER_TITLE", "meet-agent batch frame descriptions"),
                },
                json={
                    "model": self.config.model,
                    "messages": [
                        {"role": "system", "content": _FRAME_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": float(os.getenv("BATCH_LLM_TEMPERATURE", "0.2")),
                    "max_tokens": int(os.getenv("BATCH_FRAME_DESCRIPTION_MAX_NEW_TOKENS", os.getenv("BATCH_LLM_MAX_NEW_TOKENS", "4096"))),
                },
                timeout=self.config.timeout_sec,
            )
            response.raise_for_status()
            return str((response.json().get("choices") or [{}])[0].get("message", {}).get("content", "")).strip()
        if backend == "ollama" and self.config.base_url:
            response = requests.post(
                f"{self.config.base_url.rstrip('/')}/api/generate",
                json={"model": self.config.model, "prompt": _FRAME_SYSTEM_PROMPT + "\n\n" + prompt, "stream": False},
                timeout=self.config.timeout_sec,
            )
            response.raise_for_status()
            return str(response.json().get("response", "")).strip()
        if backend in {"hf-local", "transformers-local"}:
            return self._generate_hf_local(prompt)
        return _fallback_text_from_prompt(prompt)

    def _generate_hf_local(self, prompt: str) -> str:
        with self._hf_lock:
            tokenizer, model = self._ensure_hf_model()
            messages = [
                {"role": "system", "content": _FRAME_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            try:
                rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(
                rendered,
                return_tensors="pt",
                truncation=True,
                max_length=int(os.getenv("BATCH_LLM_MAX_INPUT_TOKENS", "24576")),
            ).to(model.device)
            generated = model.generate(
                **inputs,
                max_new_tokens=int(os.getenv("BATCH_FRAME_DESCRIPTION_MAX_NEW_TOKENS", os.getenv("BATCH_LLM_MAX_NEW_TOKENS", "3072"))),
                do_sample=False,
                temperature=None,
                top_p=None,
            )
            new_tokens = generated[0][inputs["input_ids"].shape[1]:]
            return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _ensure_hf_model(self):
        if self._model is not None and self._tokenizer is not None:
            return self._tokenizer, self._model
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        common_kwargs = {"trust_remote_code": True}
        if _huggingface_offline():
            common_kwargs["local_files_only"] = True
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model, **common_kwargs)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model,
            torch_dtype="auto",
            device_map="auto",
            **common_kwargs,
        )
        self._model.eval()
        return self._tokenizer, self._model


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


def build_frame_description_writer(config: LLMConfig, *, max_workers: int = 6) -> FrameDescriptionWriter:
    backend = (config.backend or "mock").strip().lower()
    if backend == "mock" and config.mock_enabled:
        return MockFrameDescriptionWriter()
    return BatchFrameDescriptionWriter(config, max_workers=max_workers)


_FRAME_SYSTEM_PROMPT = (
    "You are a meticulous screen-understanding analyst for recorded software demos, code walkthroughs, and slide presentations. "
    "Use only the supplied structured visual evidence. Be concrete about visible text, detected regions, spacing, hierarchy, cursor focus, "
    "and changes from the previous frame. Do not invent unseen UI states, project names, participants, or spoken content. "
    "Return only the requested description inside <answer>...</answer> tags."
)


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


def _code_slide_frame_prompt(event: Dict[str, Any], previous: Dict[str, Any] | None) -> str:
    mode = str(event.get("mode", event.get("frame_type", "screen")) or "screen")
    guidance = (
        "For this code frame, describe the editor or terminal layout, visible filenames/tabs if present, code structure, identifiers, comments, errors, cursor focus, "
        "and exact OCR changes. Explain what likely changed from the previous frame using only OCR diff, changed regions, and cursor evidence."
        if mode == "code"
        else "For this slide frame, describe the slide title, body blocks, lists, diagrams, tables, figures, spatial layout, visual emphasis, cursor focus, "
        "and what changed from the previous frame. Use Surya layout, DocLayout-YOLO blocks, OCR text, and transcript window when present."
    )
    payload = {
        "current_event": _compact_for_prompt(event),
        "previous_event": _compact_for_prompt(previous or {}),
    }
    return (
        f"{guidance}\n"
        "Be very detailed. Include the screen hierarchy, the important visible text, the detected layout boxes, and any uncertainty caused by noisy OCR. "
        "Return 2-5 dense paragraphs inside <answer> tags.\n"
        f"Evidence JSON:\n{json.dumps(payload, ensure_ascii=True, indent=2)}"
    )


def _demo_segment_prompt(event: Dict[str, Any], segment: Dict[str, Any]) -> str:
    payload = {
        "timestamp_sec": event.get("timestamp_sec"),
        "event_id": event.get("event_id"),
        "change_type": (event.get("visual_payload") or {}).get("change_type"),
        "cursor_target": (event.get("visual_payload") or {}).get("cursor_target"),
        "segment": segment,
    }
    return (
        "Describe this single UI segment in exhaustive but factual detail. Use the macro region label, segment bbox, child hierarchy, source types "
        "(DocLayout-YOLO, OmniParser, OCR), visible text, spacing, alignment, row/column grouping, whitespace, and cursor evidence. "
        "If the cursor target is inside this segment, state the exact element and why the target is likely active or important. "
        "Mention relative position within the page and whether children appear inside larger layout blocks. Return one detailed paragraph inside <answer> tags.\n"
        f"Evidence JSON:\n{json.dumps(_compact_for_prompt(payload), ensure_ascii=True, indent=2)}"
    )


def _demo_relationship_prompt(event: Dict[str, Any], segment_descriptions: List[Dict[str, Any]]) -> str:
    payload = {
        "timestamp_sec": event.get("timestamp_sec"),
        "segments": segment_descriptions,
        "macro_layout": (event.get("visual_payload") or {}).get("macro_layout"),
    }
    return (
        "Describe how the visible UI segments relate to each other. Focus on page composition, reading order, nested/adjacent regions, "
        "large gaps between main-content subsegments, and how header/sidebar/footer/main content work together. Return only one paragraph inside <answer> tags.\n"
        f"Evidence JSON:\n{json.dumps(_compact_for_prompt(payload), ensure_ascii=True, indent=2)}"
    )


def _demo_page_summary_prompt(
    event: Dict[str, Any],
    previous: Dict[str, Any] | None,
    descriptions: Dict[str, Dict[str, Any]],
    relationship: str,
) -> str:
    payload = {
        "current_event": _compact_for_prompt(event),
        "previous_event": _compact_for_prompt(previous or {}),
        "segment_descriptions": descriptions,
        "relationships": relationship,
    }
    return (
        "Write the bottom summary for this demo frame. Explain what is happening on the page, what changed from the previous page/frame, "
        "where the cursor is and the cursor flow, and why that cursor flow plausibly caused or followed the detected change. "
        "Use only the evidence; phrase uncertain causality as likely or possible. Return 1-2 detailed paragraphs inside <answer> tags.\n"
        f"Evidence JSON:\n{json.dumps(_compact_for_prompt(payload), ensure_ascii=True, indent=2)}"
    )


def _demo_segments_from_event(event: Dict[str, Any], *, max_demo_main_segments: int) -> List[Dict[str, Any]]:
    visual = event.get("visual_payload") or {}
    structure = visual.get("screen_structure") or {}
    cursor_target = visual.get("cursor_target") if isinstance(visual, dict) else {}
    regions = structure.get("regions") or []
    output: List[Dict[str, Any]] = []
    for region in regions:
        if not isinstance(region, dict):
            continue
        region_type = str(region.get("type") or "")
        if region_type in {"header", "left_sidebar", "right_sidebar", "footer"}:
            output.append(_demo_region_payload(region, key=region_type, label=region_type.replace("_", " "), cursor_target=cursor_target))
        elif region_type in {"main_content", "content", "content_panel"}:
            output.extend(_split_main_region(region, cursor_target=cursor_target, max_segments=max_demo_main_segments))
    if not output and regions:
        output.append(_demo_region_payload(regions[0], key="main_content_0", label="main content", cursor_target=cursor_target))
    return output


def _split_main_region(region: Dict[str, Any], *, cursor_target: Dict[str, Any], max_segments: int) -> List[Dict[str, Any]]:
    children = [item for item in (region.get("hierarchy") or region.get("children") or []) if isinstance(item, dict)]
    if not children:
        return [_demo_region_payload(region, key="main_content_0", label="main content", cursor_target=cursor_target)]
    boxes = [_node_bbox(item) for item in children]
    valid_pairs = [(child, box) for child, box in zip(children, boxes) if box is not None]
    if not valid_pairs:
        return [_demo_region_payload(region, key="main_content_0", label="main content", cursor_target=cursor_target)]
    valid_pairs.sort(key=lambda item: (item[1][1], item[1][0]))
    heights = [max(1, box[3] - box[1]) for _, box in valid_pairs]
    widths = [max(1, box[2] - box[0]) for _, box in valid_pairs]
    median_h = sorted(heights)[len(heights) // 2]
    median_w = sorted(widths)[len(widths) // 2]
    groups: List[List[Dict[str, Any]]] = [[valid_pairs[0][0]]]
    prev_box = valid_pairs[0][1]
    for child, box in valid_pairs[1:]:
        vertical_gap = max(0, box[1] - prev_box[3])
        horizontal_gap = max(0, box[0] - prev_box[2])
        large_gap = vertical_gap > max(48, median_h * 2.4) or horizontal_gap > max(96, median_w * 2.8)
        if large_gap and len(groups) < max(1, max_segments):
            groups.append([child])
        else:
            groups[-1].append(child)
        prev_box = box
    payloads: List[Dict[str, Any]] = []
    for idx, group in enumerate(groups[: max(1, max_segments)]):
        bbox = _union_node_boxes(group) or _node_bbox(region) or [0, 0, 0, 0]
        segment = dict(region)
        segment["children"] = group
        segment["hierarchy"] = group
        segment["bbox"] = bbox
        payloads.append(_demo_region_payload(segment, key=f"main_content_{idx}", label=f"main content subregion {idx + 1}", cursor_target=cursor_target))
    return payloads


def _demo_region_payload(region: Dict[str, Any], *, key: str, label: str, cursor_target: Dict[str, Any]) -> Dict[str, Any]:
    children = region.get("hierarchy") or region.get("children") or []
    return {
        "key": key,
        "label": label,
        "macro_region_id": region.get("id"),
        "macro_region_type": region.get("type"),
        "bbox": region.get("bbox"),
        "element_count": region.get("element_count"),
        "ocr_line_count": region.get("ocr_line_count"),
        "sample_text": region.get("sample_text") or [],
        "element_types": region.get("element_types") or {},
        "spacing": _spacing_summary(children),
        "hierarchy": children,
        "cursor": cursor_target if _cursor_belongs_to_region(cursor_target, region, key) else {},
    }


def _cursor_belongs_to_region(cursor_target: Dict[str, Any], region: Dict[str, Any], key: str) -> bool:
    if not isinstance(cursor_target, dict):
        return False
    macro_id = str(cursor_target.get("parent_macro_region_id") or "")
    macro_type = str(cursor_target.get("parent_macro_region_type") or "")
    return macro_id == str(region.get("id") or "") or macro_type == str(region.get("type") or "") or (key.startswith("main_content") and macro_type == "main_content")


def _spacing_summary(children: Any) -> Dict[str, Any]:
    nodes = [item for item in (children or []) if isinstance(item, dict)]
    boxes = [_node_bbox(item) for item in nodes]
    boxes = [box for box in boxes if box is not None]
    boxes.sort(key=lambda box: (box[1], box[0]))
    gaps = []
    for prev, curr in zip(boxes, boxes[1:]):
        gaps.append({"vertical_gap": max(0, curr[1] - prev[3]), "horizontal_gap": max(0, curr[0] - prev[2])})
    return {
        "child_count": len(nodes),
        "largest_vertical_gap": max((item["vertical_gap"] for item in gaps), default=0),
        "largest_horizontal_gap": max((item["horizontal_gap"] for item in gaps), default=0),
        "sample_gaps": gaps[:8],
    }


def _node_bbox(node: Dict[str, Any]) -> List[int] | None:
    raw = node.get("bbox")
    if not isinstance(raw, list) or len(raw) < 4:
        return None
    box = [int(v) for v in raw[:4]]
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _union_node_boxes(nodes: List[Dict[str, Any]]) -> List[int] | None:
    boxes = [_node_bbox(item) for item in nodes]
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None
    return [min(box[0] for box in boxes), min(box[1] for box in boxes), max(box[2] for box in boxes), max(box[3] for box in boxes)]


def _fallback_demo_detail(event: Dict[str, Any], previous: Dict[str, Any] | None, *, max_demo_main_segments: int) -> Dict[str, Any]:
    segments = _demo_segments_from_event(event, max_demo_main_segments=max_demo_main_segments)
    descriptions = {
        segment["key"]: {
            "label": segment["label"],
            "bbox": segment.get("bbox"),
            "description": _fallback_segment_description(segment),
        }
        for segment in segments
    }
    summary = _fallback_frame_description(event, previous)
    return {
        "backend": "mock",
        "model": "mock-frame-describer",
        "segments": descriptions,
        "relationships": _fallback_relationship_description(segments),
        "structured_output": {
            key: value["description"]
            for key, value in descriptions.items()
        },
        "page_summary": summary,
    }


def _fallback_frame_description(event: Dict[str, Any], previous: Dict[str, Any] | None) -> str:
    visual = event.get("visual_payload") or {}
    mode = str(event.get("mode", event.get("frame_type", "screen")) or "screen")
    base = str(visual.get("page_summary") or event.get("summary") or "").strip()
    if mode == "code":
        base = "Code frame. " + (str(visual.get("ocr_text") or "")[:700] or base)
    elif mode == "slide":
        base = "Slide frame. " + (str(visual.get("full_text") or "")[:700] or base)
    change = str(visual.get("change_type") or "").replace("_", " ")
    cursor = str(visual.get("cursor_action") or "").strip()
    previous_summary = str((previous or {}).get("summary") or "").strip()
    parts = [base or f"{mode} frame was captured."]
    if change and change != "none":
        parts.append(f"Detected change: {change}.")
    if cursor:
        parts.append(cursor)
    if previous_summary:
        parts.append(f"Previous frame context: {previous_summary[:240]}")
    return " ".join(parts).strip()


def _fallback_segment_description(segment: Dict[str, Any]) -> str:
    text = ", ".join(str(item) for item in (segment.get("sample_text") or [])[:8] if str(item).strip())
    parts = [f"{segment.get('label', 'segment')} spans {segment.get('bbox')}."]
    parts.append(f"It contains {segment.get('element_count', 0)} detected elements and {segment.get('ocr_line_count', 0)} OCR lines.")
    if text:
        parts.append(f"Visible text includes {text}.")
    cursor = segment.get("cursor") or {}
    cursor_text = str(cursor.get("element_text") or cursor.get("element_type") or "").strip()
    if cursor_text:
        parts.append(f"The cursor is on {cursor_text}.")
    return " ".join(parts)


def _fallback_relationship_description(segments: List[Dict[str, Any]]) -> str:
    labels = [str(segment.get("label") or segment.get("key")) for segment in segments[:8]]
    return "The UI is divided into " + ", ".join(labels) + "." if labels else "No segment relationships were available."


def _fallback_text_from_prompt(prompt: str) -> str:
    return "<answer>" + _truncate_text(prompt, 1200) + "</answer>"


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

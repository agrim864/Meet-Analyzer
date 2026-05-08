from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from batch_mode.config import BatchModeSettings  # noqa: E402
from batch_mode.llm_adapters import LLMConfig, build_section_writer  # noqa: E402
from batch_mode.llm_adapters import _compose_final_markdown, _ordered_section_keys  # noqa: E402
from batch_mode.report_sections import build_report_sections  # noqa: E402


def _supports_local_qwen() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        props = torch.cuda.get_device_properties(0)
        return int(getattr(props, "total_memory", 0)) >= 24 * 1024**3
    except Exception:
        return False


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    parser = argparse.ArgumentParser(description="Regenerate report.md and llm_output inside an existing analysis.json.")
    parser.add_argument("--analysis-json", required=True, help="Path to an existing batch_mode analysis.json.")
    parser.add_argument("--backend", default="", help="Override LLM backend, e.g. hf-local, mock, or preserve.")
    parser.add_argument("--writer", choices=["mistral", "gemini", "qwen", "preserve"], default="", help="Convenience preset for report writer.")
    args = parser.parse_args()

    settings = BatchModeSettings.from_env()
    backend = (args.backend or settings.llm_backend or "").strip().lower()
    model_override = ""
    if args.writer == "mistral":
        backend = "hf-local"
        model_override = "unsloth/Mistral-Small-24B-Instruct-2501-unsloth-bnb-4bit"
        os.environ.setdefault("BATCH_HF_QUANTIZATION", "")
        os.environ.setdefault("BATCH_LLM_USE_PREWRITTEN", "0")
        os.environ.setdefault("BATCH_LLM_MAX_PROMPT_CHARS", "60000")
        os.environ.setdefault("BATCH_LLM_MAX_INPUT_TOKENS", "24576")
        os.environ.setdefault("BATCH_LLM_MAX_NEW_TOKENS", "3072")
        os.environ.setdefault("BATCH_LLM_SECTION_WORKERS", "1")
    elif args.writer == "gemini":
        backend = "gemini"
        model_override = "gemini-2.5-flash"
        os.environ.setdefault("BATCH_LLM_USE_PREWRITTEN", "0")
        os.environ.setdefault("BATCH_LLM_MAX_PROMPT_CHARS", "80000")
        os.environ.setdefault("BATCH_LLM_MAX_NEW_TOKENS", "8192")
        os.environ.setdefault("BATCH_LLM_TIMEOUT_SEC", "900")
        os.environ.setdefault("BATCH_LLM_SECTION_WORKERS", "1")
    elif args.writer == "qwen":
        backend = "hf-local"
        model_override = "Qwen/Qwen3-30B-A3B-FP8"
    elif args.writer == "preserve":
        backend = "preserve"
    if backend in {"", "mock", "auto"} and _supports_local_qwen():
        backend = "hf-local"
    backend = backend or "mock"

    analysis_path = Path(args.analysis_json)
    data = json.loads(analysis_path.read_text(encoding="utf-8"))
    sections = data.get("sections") or {}
    if not sections:
        raise RuntimeError(f"No sections found in {analysis_path}")
    narrative_section = sections.get("meeting_narrative_section", {})
    if (
        "meeting_narrative_section" not in sections
        or "narrative_units" not in narrative_section
        or "prewritten_markdown" not in sections.get("transcript_section", {})
    ):
        sections = build_report_sections(
            frame_events=data.get("frame_events", []),
            transcript_segments=data.get("transcript_segments", []),
            topic_units=data.get("topic_units", []),
            topic_clusters=data.get("topic_clusters", []),
            title=str(sections.get("report_title") or analysis_path.parent.name),
        )
        data["sections"] = sections

    if backend in {"preserve", "existing", "no-llm"}:
        existing = data.get("llm_output") if isinstance(data.get("llm_output"), dict) else {}
        ordered_keys = _ordered_section_keys(sections)
        llm_output = {
            "backend": "preserve",
            "model": str(existing.get("model", "existing-sections")),
        }
        for key in ordered_keys:
            payload = sections.get(key, {})
            prewritten = ""
            if isinstance(payload, dict):
                prewritten = str(payload.get("prewritten_markdown", "") or "").strip()
            if prewritten:
                llm_output[key] = {"summary": prewritten}
            elif key in existing and isinstance(existing.get(key), dict):
                llm_output[key] = existing[key]
            elif key == "report_title":
                llm_output[key] = {"summary": str(sections.get("report_title", analysis_path.parent.name))}
            else:
                llm_output[key] = {"summary": ""}
        llm_output["final_markdown"] = _compose_final_markdown(llm_output, ordered_keys)
        data["llm_output"] = llm_output
        analysis_path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")
        report_path = analysis_path.with_name("report.md")
        report_path.write_text(str(llm_output.get("final_markdown", "")), encoding="utf-8")
        print(
            json.dumps(
                {
                    "analysis_json": str(analysis_path),
                    "report_md": str(report_path),
                    "backend": llm_output.get("backend"),
                    "model": llm_output.get("model"),
                    "final_markdown_chars": len(str(llm_output.get("final_markdown", ""))),
                },
                ensure_ascii=True,
            )
        )
        return 0

    writer = build_section_writer(
        LLMConfig(
            backend=backend,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key
            or (os.getenv("GEMINI_API_KEY", "").strip() if backend in {"gemini", "google", "google-gemini"} else "")
            or (os.getenv("OPENROUTER_API_KEY", "").strip() if backend in {"openrouter", "openrouter-chat"} else ""),
            model=model_override or settings.llm_model or settings.qwen_model_name,
            timeout_sec=settings.llm_timeout_sec,
            mock_enabled=settings.llm_mock_enabled,
        )
    )
    llm_output = writer.generate(sections=sections)
    data["llm_output"] = llm_output
    analysis_path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")
    report_path = analysis_path.with_name("report.md")
    report_path.write_text(str(llm_output.get("final_markdown", "")), encoding="utf-8")
    print(
        json.dumps(
            {
                "analysis_json": str(analysis_path),
                "report_md": str(report_path),
                "backend": llm_output.get("backend"),
                "model": llm_output.get("model"),
                "final_markdown_chars": len(str(llm_output.get("final_markdown", ""))),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

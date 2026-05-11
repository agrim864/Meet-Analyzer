from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from batch_mode.service import BatchMeetingAnalyzer  # noqa: E402


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    parser = argparse.ArgumentParser(description="Run batch meeting analysis on a local mp4 or public URL.")
    parser.add_argument("--source", help="Path or Bunny/public mp4 URL")
    parser.add_argument("--title", default="", help="Optional report title")
    parser.add_argument("--writer", default="", help="Report writer preset: mistral, gemini, qwen, or mock")
    parser.add_argument("--no-persist", action="store_true", help="Do not write artifacts to disk")
    parser.add_argument("--request-json", help="Path to a request.json payload written by the cluster submitter")
    parser.add_argument("--output-json", help="Optional path to write the result JSON")
    args = parser.parse_args()

    request_payload = {}
    if args.request_json:
        request_payload = json.loads(Path(args.request_json).read_text(encoding="utf-8"))

    source = args.source or request_payload.get("source")
    if not source:
        parser.error("--source or --request-json with a source field is required")

    transcript_segments = request_payload.get("transcript_segments")
    title = args.title or request_payload.get("title", "")
    writer = args.writer or request_payload.get("writer", "")
    persist_artifacts = (not args.no_persist) if not args.request_json else bool(request_payload.get("persist_artifacts", True)) and not args.no_persist
    describe_frames = request_payload.get("describe_frames")

    logging.info("Starting batch analysis source=%s title=%s writer=%s persist=%s", source, title, writer, persist_artifacts)
    analyzer = BatchMeetingAnalyzer()
    result = analyzer.analyze(
        source=source,
        transcript_segments=transcript_segments,
        title=title,
        persist_artifacts=persist_artifacts,
        writer=writer,
        describe_frames=describe_frames,
    )
    logging.info("Finished batch analysis output_path=%s frame_events=%s", result.output_path, len(result.frame_events))
    result_json = json.dumps(result.to_dict(), ensure_ascii=True, indent=2)
    if args.output_json:
        Path(args.output_json).write_text(result_json, encoding="utf-8")
    print(result_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

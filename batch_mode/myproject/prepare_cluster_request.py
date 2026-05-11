from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from batch_mode.config import BatchModeSettings  # noqa: E402
from batch_mode.stt_adapter import DeepgramSTTAdapter  # noqa: E402


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a batch-mode cluster request on the login/submission node. "
            "This keeps Deepgram/network transcription out of Slurm compute nodes."
        )
    )
    parser.add_argument("--source", required=True, help="Remote local mp4 path already visible to the cluster.")
    parser.add_argument("--request-json", required=True, help="Path where the GPU request payload should be written.")
    parser.add_argument("--title", default="", help="Optional report title.")
    parser.add_argument("--transcript-json", help="Optional sidecar path for transcript segments.")
    parser.add_argument("--skip-transcript", action="store_true", help="Write an empty transcript instead of calling Deepgram.")
    parser.add_argument("--require-transcript", action="store_true", help="Fail if no transcript segments are produced.")
    parser.add_argument("--no-persist-artifacts", action="store_true", help="Set persist_artifacts=false in the request.")
    parser.add_argument("--disable-frame-descriptions", action="store_true", help="Set describe_frames=false in the request.")
    args = parser.parse_args()

    transcript_segments: List[Dict[str, Any]] = []
    if not args.skip_transcript:
        settings = BatchModeSettings.from_env()
        adapter = DeepgramSTTAdapter(api_key=settings.deepgram_api_key, model=settings.deepgram_model)
        logging.info("Transcribing source before Slurm submit: %s", args.source)
        transcript_segments = adapter.transcribe_file(args.source)
        logging.info("Prepared %s transcript segments", len(transcript_segments))

    if args.require_transcript and not transcript_segments:
        raise RuntimeError("Transcript was required, but Deepgram returned no segments.")

    request_payload = {
        "source": args.source,
        "title": args.title,
        "transcript_segments": transcript_segments,
        "persist_artifacts": not args.no_persist_artifacts,
        "describe_frames": not args.disable_frame_descriptions,
    }
    request_path = Path(args.request_json)
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(json.dumps(request_payload, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.transcript_json:
        transcript_path = Path(args.transcript_json)
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text(json.dumps(transcript_segments, ensure_ascii=True, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "request_json": str(request_path),
                "transcript_segments": len(transcript_segments),
                "source": args.source,
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple
import subprocess
import tempfile

from deepgram import DeepgramClient, PrerecordedOptions

from .state import TranscriptSegment


@dataclass
class AssignedUtterances:
    utterances: List[Dict[str, Any]]
    utterance_ids: List[int]
    utterance_time_span: Optional[Tuple[float, float]]


class DeepgramSTTAdapter:
    def __init__(self, api_key: Optional[str] = None, model: str = "nova-3") -> None:
        self.api_key = (api_key or "").strip()
        self.model = str(model or "nova-3").strip() or "nova-3"
        self._client: Optional[DeepgramClient] = None

    def warmup(self) -> None:
        if not self.api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is required for STT transcription.")
        if self._client is None:
            self._client = DeepgramClient(api_key=self.api_key)

    def transcribe_file(self, video_path: str) -> List[Dict[str, Any]]:
        self.warmup()
        client = self._client
        if client is None:
            raise RuntimeError("Deepgram client failed to initialize.")
        path = Path(video_path)
        with path.open("rb") as handle:
            return self.transcribe_bytes(handle.read(), mimetype="video/mp4")

    def transcribe_bytes(self, audio_bytes: bytes, *, mimetype: str = "audio/webm") -> List[Dict[str, Any]]:
        self.warmup()
        client = self._client
        if client is None:
            raise RuntimeError("Deepgram client failed to initialize.")
        normalized_bytes, normalized_mimetype = self._prepare_audio_for_deepgram(audio_bytes, mimetype=mimetype)
        source = {"buffer": bytes(normalized_bytes), "mimetype": normalized_mimetype}
        options = PrerecordedOptions(
            model=self.model,
            smart_format=True,
            punctuate=True,
            utterances=True,
            diarize=True,
        )
        response = client.listen.rest.v("1").transcribe_file(source, options)
        result = response.to_dict() if hasattr(response, "to_dict") else dict(response)
        return self._normalize_response(result)

    def _normalize_response(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        utterances = result.get("results", {}).get("utterances", []) or []
        normalized: List[Dict[str, Any]] = []
        for idx, item in enumerate(utterances):
            text = str(item.get("transcript", "") or "").strip()
            if not text:
                continue
            normalized.append(
                {
                    "id": idx,
                    "start": float(item.get("start", 0.0)),
                    "end": float(item.get("end", 0.0)),
                    "speaker": str(item.get("speaker", "unknown") or "unknown"),
                    "text": text,
                    "source": "prerecorded",
                }
            )
        if normalized:
            return normalized
        alternative = (
            result.get("results", {})
            .get("channels", [{}])[0]
            .get("alternatives", [{}])[0]
        )
        text = str(alternative.get("transcript", "") or "").strip()
        if not text:
            return []
        words = alternative.get("words") or []
        start = float(words[0].get("start", 0.0)) if words else 0.0
        end = float(words[-1].get("end", start)) if words else max(start + 1.0, 1.0)
        return [
            {
                "id": 0,
                "start": start,
                "end": end,
                "speaker": "speaker_1",
                "text": text,
                "source": "prerecorded",
            }
        ]

    def _prepare_audio_for_deepgram(self, audio_bytes: bytes, *, mimetype: str) -> Tuple[bytes, str]:
        effective_type = str(mimetype or "audio/webm").lower()
        if effective_type in {"audio/wav", "audio/x-wav", "audio/wave"}:
            return bytes(audio_bytes), "audio/wav"
        suffix = _mime_to_suffix(effective_type)
        with tempfile.TemporaryDirectory(prefix="dg-audio-") as temp_dir:
            source_path = Path(temp_dir) / f"input{suffix}"
            output_path = Path(temp_dir) / "normalized.wav"
            source_path.write_bytes(bytes(audio_bytes))
            command = [
                "ffmpeg",
                "-y",
                "-i",
                str(source_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "wav",
                str(output_path),
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode != 0 or not output_path.exists():
                stderr = (completed.stderr or "").strip()
                raise RuntimeError(f"ffmpeg audio normalization failed: {stderr or 'unknown error'}")
            return output_path.read_bytes(), "audio/wav"


class UtteranceStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._utterances: List[Dict[str, Any]] = []
        self._interim_segments: List[Dict[str, Any]] = []
        self._next_id = 0
        self._last_claimed_end = 0.0

    def add_utterance(
        self,
        *,
        start: float,
        end: float,
        speaker: str,
        text: str,
        source: str = "live",
        utterance_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        clean = str(text or "").strip()
        if not clean:
            return None
        with self._lock:
            item = {
                "id": int(self._next_id if utterance_id is None else utterance_id),
                "start": float(start),
                "end": float(end),
                "speaker": str(speaker or "unknown").strip() or "unknown",
                "text": clean,
                "source": str(source or "live"),
            }
            if utterance_id is None:
                self._next_id += 1
            else:
                self._next_id = max(self._next_id, int(utterance_id) + 1)
            self._utterances.append(item)
            self._utterances.sort(key=lambda value: (float(value.get("start", 0.0)), int(value.get("id", 0))))
            return dict(item)

    def replace_interim(self, segments: Sequence[Dict[str, Any]]) -> None:
        with self._lock:
            self._interim_segments = [
                {
                    "id": int(item.get("id", -1)),
                    "start": float(item.get("start", 0.0)),
                    "end": float(item.get("end", 0.0)),
                    "speaker": str(item.get("speaker", "unknown") or "unknown"),
                    "text": str(item.get("text", "") or "").strip(),
                    "source": str(item.get("source", "live")),
                    "is_final": bool(item.get("is_final", False)),
                }
                for item in segments
                if str(item.get("text", "") or "").strip()
            ]

    def extend(self, utterances: Sequence[Dict[str, Any]]) -> None:
        for item in utterances:
            self.add_utterance(
                start=float(item.get("start", 0.0)),
                end=float(item.get("end", 0.0)),
                speaker=str(item.get("speaker", "unknown")),
                text=str(item.get("text", "")),
                source=str(item.get("source", "batch")),
                utterance_id=int(item["id"]) if "id" in item else None,
            )

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._utterances]

    def snapshot_interim(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._interim_segments]

    def claim_until(self, timestamp_sec: float, previous_timestamp_sec: float = 0.0) -> AssignedUtterances:
        with self._lock:
            selected = [
                dict(item)
                for item in self._utterances
                if float(item.get("end", 0.0)) > previous_timestamp_sec and float(item.get("end", 0.0)) <= timestamp_sec
            ]
            if not selected and timestamp_sec >= previous_timestamp_sec:
                trailing = [
                    dict(item)
                    for item in self._utterances
                    if float(item.get("end", 0.0)) > previous_timestamp_sec and previous_timestamp_sec == 0.0
                ]
                selected = trailing
            if selected:
                start = min(float(item.get("start", 0.0)) for item in selected)
                end = max(float(item.get("end", 0.0)) for item in selected)
                span: Optional[Tuple[float, float]] = (start, end)
            else:
                span = None
        return AssignedUtterances(
            utterances=selected,
            utterance_ids=[int(item.get("id", -1)) for item in selected],
            utterance_time_span=span,
        )

    def window(self, *, event_time: float, previous_event_time: Optional[float], lookback_sec: float) -> List[TranscriptSegment]:
        start_time = float(previous_event_time) if previous_event_time is not None else max(0.0, float(event_time) - float(lookback_sec))
        with self._lock:
            selected = [
                TranscriptSegment(
                    start_time=float(item.get("start", 0.0)),
                    end_time=float(item.get("end", 0.0)),
                    speaker=str(item.get("speaker", "unknown") or "unknown"),
                    text=str(item.get("text", "") or "").strip(),
                    is_final=True,
                    source=str(item.get("source", "live")),
                    segment_id=int(item.get("id", -1)),
                )
                for item in self._utterances
                if float(item.get("end", 0.0)) >= start_time and float(item.get("start", 0.0)) <= event_time
            ]
        return selected


def _mime_to_suffix(mimetype: str) -> str:
    if "webm" in mimetype:
        return ".webm"
    if "ogg" in mimetype:
        return ".ogg"
    if "mp4" in mimetype or "m4a" in mimetype:
        return ".m4a"
    if "mpeg" in mimetype or "mp3" in mimetype:
        return ".mp3"
    return ".bin"

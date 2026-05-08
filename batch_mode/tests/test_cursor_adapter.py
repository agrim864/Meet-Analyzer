from __future__ import annotations

import unittest

import numpy as np

from batch_mode.cursor_adapter import CursorMotionAnalyzer
from batch_mode.state import CursorState


class _MissingBackend:
    def analyze(self, **_: object) -> dict:
        return {
            "bbox": None,
            "confidence": 0.0,
            "motion_type": "unknown",
            "hotspot": None,
            "type": "unknown",
            "method": "yolo",
        }


class CursorAdapterTests(unittest.TestCase):
    def test_tracker_predicts_short_missing_gap(self) -> None:
        analyzer = CursorMotionAnalyzer(backend=_MissingBackend(), max_predicted_missing=2)
        previous = CursorState(
            visible=True,
            bbox=(100, 120, 112, 134),
            trail=[(100, 120), (106, 127)],
            confidence=0.7,
            tracking_state="confirmed",
            missing_frames=0,
        )
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = analyzer.analyze(frame_bgr=frame, prev_cursor=previous)
        self.assertTrue(result.visible)
        self.assertEqual(result.method, "tracker")
        self.assertEqual(result.tracking_state, "predicted")
        self.assertEqual(result.missing_frames, 1)
        self.assertEqual(result.bbox, (106, 127, 118, 141))

    def test_tracker_stops_after_missing_limit(self) -> None:
        analyzer = CursorMotionAnalyzer(backend=_MissingBackend(), max_predicted_missing=1)
        previous = CursorState(
            visible=True,
            bbox=(100, 120, 112, 134),
            trail=[(106, 127)],
            confidence=0.7,
            tracking_state="predicted",
            missing_frames=1,
        )
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = analyzer.analyze(frame_bgr=frame, prev_cursor=previous)
        self.assertFalse(result.visible)
        self.assertEqual(result.tracking_state, "missing")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import cv2
import numpy as np

from batch_mode.cursor_backend import CursorMotionAnalysisBackend


def _draw_cursor(frame: np.ndarray, x: int, y: int) -> None:
    polygon = np.array(
        [
            [x + 1, y + 1],
            [x + 1, y + 20],
            [x + 6, y + 15],
            [x + 9, y + 26],
            [x + 12, y + 25],
            [x + 9, y + 14],
            [x + 16, y + 14],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [polygon], (255, 255, 255))
    cv2.polylines(frame, [polygon], isClosed=True, color=(0, 0, 0), thickness=1)


class CursorBackendTests(unittest.TestCase):
    def test_template_fallback_ignores_blank_frame(self) -> None:
        backend = CursorMotionAnalysisBackend(model_path="missing_cursor_model.pt", model_url="")
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = backend.analyze(frame_bgr=frame, prev_frame_bgr=None, prev_cursor=None)
        self.assertIsNone(result["bbox"])

    def test_local_template_fallback_finds_visible_cursor_near_previous_trail(self) -> None:
        backend = CursorMotionAnalysisBackend(model_path="missing_cursor_model.pt", model_url="")
        frame = np.full((200, 200, 3), 255, dtype=np.uint8)
        _draw_cursor(frame, 80, 60)
        previous_cursor = type("PreviousCursor", (), {"bbox": None, "trail": [(88, 72)]})()
        result = backend.analyze(frame_bgr=frame, prev_frame_bgr=None, prev_cursor=previous_cursor)
        self.assertIsNotNone(result["bbox"])
        self.assertEqual(result["method"], "template")

    def test_template_fallback_tracks_cursor_with_motion(self) -> None:
        backend = CursorMotionAnalysisBackend(model_path="missing_cursor_model.pt", model_url="")
        previous = np.zeros((200, 200, 3), dtype=np.uint8)
        current = np.zeros((200, 200, 3), dtype=np.uint8)
        _draw_cursor(previous, 80, 60)
        _draw_cursor(current, 90, 72)
        previous_cursor = type("PreviousCursor", (), {"bbox": (80, 60, 100, 88), "trail": [(90, 74)]})()
        result = backend.analyze(frame_bgr=current, prev_frame_bgr=previous, prev_cursor=previous_cursor)
        self.assertIsNotNone(result["bbox"])
        self.assertEqual(result["method"], "template")
        x1, y1, x2, y2 = result["bbox"]
        self.assertTrue(70 <= x1 <= 110)
        self.assertTrue(55 <= y1 <= 90)

    def test_motion_template_requires_cursor_history(self) -> None:
        backend = CursorMotionAnalysisBackend(model_path="missing_cursor_model.pt", model_url="")
        previous = np.zeros((200, 200, 3), dtype=np.uint8)
        current = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(previous, (20, 20), (80, 80), (255, 255, 255), -1)
        cv2.rectangle(current, (30, 20), (90, 80), (255, 255, 255), -1)
        result = backend.analyze(frame_bgr=current, prev_frame_bgr=previous, prev_cursor=None)
        self.assertIsNone(result["bbox"])


if __name__ == "__main__":
    unittest.main()

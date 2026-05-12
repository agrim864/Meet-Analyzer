from __future__ import annotations

import unittest
import threading
import time
from unittest.mock import patch

import cv2
import numpy as np

from real_time_mode.mode_router import ModeRouter
from real_time_mode.ocr_adapter import OCRResult
from real_time_mode.pipeline import DemoAnalyzerPipeline
from real_time_mode.frame_summarizer import FrameSummaryRequest
from real_time_mode.slide_adapter import SlideAnalysis, SlideLayoutBlock, SuryaSlideAnalyzer
from real_time_mode.state import CursorState, LayoutGroup, UIElement
from real_time_mode.stt_adapter import UtteranceStore


class FakeSiglip:
    def classify(self, frame_bgr):
        marker = int(frame_bgr[0, 0, 0])
        if marker == 10 or marker == 11:
            return "code", {"code": 0.92, "demo": 0.04, "slide": 0.02, "other": 0.02}
        if marker == 20 or marker == 21:
            return "slides", {"slide": 0.93, "code": 0.02, "demo": 0.03, "other": 0.02}
        return "demo", {"demo": 0.91, "slide": 0.03, "code": 0.03, "other": 0.03}


class FakeCursorAnalyzer:
    def __init__(self, boxes=None):
        self.boxes = list(boxes or [])
        self.calls = 0

    def analyze(self, **kwargs):
        box = self.boxes[self.calls] if self.calls < len(self.boxes) else None
        self.calls += 1
        if box is None:
            return CursorState(visible=False, bbox=None, hotspot=None, cursor_type="unknown", method="none")
        return CursorState(
            visible=True,
            bbox=tuple(box),
            hotspot=(int(box[0]) + 2, int(box[1]) + 2),
            cursor_type="pointer",
            confidence=0.95,
            method="fake",
        )


class FakeOCR:
    def __init__(self):
        self.calls = 0
        self.region_calls = []

    def extract_full(self, frame_bgr, *, frame_type):
        self.calls += 1
        marker = int(frame_bgr[0, 0, 0])
        if marker == 10:
            text = "def mask_cursor():\n    return 1"
        elif marker == 11:
            text = "def mask_cursor():\n    return 2"
        else:
            text = "generic text"
        return OCRResult(
            text=text,
            lines=[
                {"text": line, "box": [10, 10 + (idx * 18), 220, 26 + (idx * 18)], "confidence": 0.9, "source": "fakeocr"}
                for idx, line in enumerate(text.splitlines())
            ],
            provider="fakeocr",
            confidence=0.9,
        )

    def extract_regions(self, frame_bgr, regions, *, frame_type):
        self.region_calls.append(list(regions))
        marker = int(frame_bgr[0, 0, 0])
        if marker == 30:
            lines = [
                {"text": "Search orders", "box": [24, 70, 150, 94], "confidence": 0.93, "source": "ocr:rapidocr"},
                {"text": "Save", "box": [24, 108, 88, 132], "confidence": 0.91, "source": "ocr:rapidocr"},
            ]
        else:
            lines = []
        return OCRResult(
            text="\n".join(line["text"] for line in lines),
            lines=lines,
            provider="ocr:rapidocr",
            confidence=0.92 if lines else 0.0,
        )


class FakeSlide:
    def __init__(self):
        self.calls = 0

    def analyze(self, frame_bgr):
        self.calls += 1
        marker = int(frame_bgr[0, 0, 0])
        if marker == 20:
            fingerprint = "slide-a"
            title = "Realtime Pipeline"
        else:
            fingerprint = "slide-a"
            title = "Realtime Pipeline"
        return SlideAnalysis(
            provider="fake-surya",
            layout=[
                SlideLayoutBlock(type="title", text=title, bbox=[20, 20, 260, 70], order=1),
                SlideLayoutBlock(type="text", text="Cursor masking keeps noise out", bbox=[20, 90, 420, 130], order=2),
            ],
            full_text=f"{title}\nCursor masking keeps noise out",
            title=title,
            fingerprint=fingerprint,
            confidence=0.88,
            reading_order=[title, "Cursor masking keeps noise out"],
        )


class FakeMacro:
    def __init__(self):
        self.calls = 0

    def detect(self, frame_bgr):
        self.calls += 1
        h, w = frame_bgr.shape[:2]
        return [
            LayoutGroup(id="group_header", label="header", bbox=(0, 0, w, 50), score=0.8, source="fake"),
            LayoutGroup(id="group_main", label="main_content", bbox=(0, 50, w, h), score=0.9, source="fake"),
        ]


class FakeDocLayout:
    def __init__(self):
        self.calls = 0
        self.last_regions = []

    def detect_regions(self, frame_bgr, regions, *, frame_id=0, timestamp_sec=0.0):
        from real_time_mode.doclayout_adapter import DocLayoutPayload

        self.calls += 1
        self.last_regions = list(regions)
        elements = []
        for idx, region in enumerate(regions, start=1):
            x1, y1, x2, y2 = [int(v) for v in region]
            elements.append(
                UIElement(
                    id=f"doc_{idx}",
                    type="text_block",
                    bbox=(max(0, x1 + 8), max(0, y1 + 8), min(x2, x1 + 120), min(y2, y1 + 44)),
                    text="",
                    interactive=False,
                    confidence=0.86,
                    source="doclayout_yolo",
                )
            )
        return DocLayoutPayload(elements=elements, provider="fake_doclayout", confidence=0.86)


class FakeSummarizer:
    def __init__(self):
        self.requests = []

    def warmup(self):
        return None

    def summarize(self, request: FrameSummaryRequest):
        self.requests.append(request)
        return f"{request.heuristic_summary} [memory={len(request.recent_memory)}]"


class SlowSummarizer:
    def __init__(self, delay=1.0):
        self.delay = float(delay)
        self.calls = 0
        self.entered = threading.Event()

    def warmup(self):
        return None

    def summarize(self, request: FrameSummaryRequest):
        self.calls += 1
        self.entered.set()
        time.sleep(self.delay)
        return "slow llm finished"


def frame(marker: int, *, patch=None):
    img = np.full((180, 280, 3), 240, dtype=np.uint8)
    cv2.rectangle(img, (20, 60), (260, 150), (220, 220, 220), thickness=-1)
    if patch is not None:
        x1, y1, x2, y2, value = patch
        img[y1:y2, x1:x2] = value
    img[0, 0, 0] = marker
    return img


class PipelineModesTests(unittest.TestCase):
    def make_pipeline(self, *, cursor_boxes=None):
        summarizer = FakeSummarizer()
        return DemoAnalyzerPipeline(
            siglip_classifier=FakeSiglip(),
            stt_store=UtteranceStore(),
            cursor_analyzer=FakeCursorAnalyzer(cursor_boxes),
            macro_layout_detector=FakeMacro(),
            doclayout_detector=FakeDocLayout(),
            ocr_extractor=FakeOCR(),
            slide_analyzer=FakeSlide(),
            mode_router=ModeRouter(immediate_switch_confidence=0.8, sticky_confidence_floor=0.45),
            frame_summarizer=summarizer,
            summary_memory_size=4,
            code_interval_sec=2.0,
            slide_interval_sec=3.0,
            demo_interval_sec=1.0,
            transcript_window_sec=8.0,
            debug_black_cursor_mask=False,
        ), summarizer

    def test_mode_cache_separation_and_mode_switch_baseline(self):
        pipeline, _ = self.make_pipeline()
        pipeline.ingest_frame(frame(10, patch=(40, 70, 70, 90, 0)), timestamp_sec=0.0)
        self.assertIsNotNone(pipeline.cache.code.prev_frame_masked)
        demo_record = pipeline.ingest_frame(frame(30, patch=(40, 70, 70, 90, 0)), timestamp_sec=3.0)
        self.assertTrue(demo_record.visual_change)
        self.assertEqual(demo_record.mode, "demo")
        self.assertIn("initial_frame", demo_record.delta_from_previous["type"])
        self.assertIsNotNone(pipeline.cache.code.prev_frame_masked)

    def test_cursor_only_motion_does_not_trigger_demo_change(self):
        pipeline, _ = self.make_pipeline(cursor_boxes=[(10, 10, 18, 18), (40, 10, 48, 18)])
        first = pipeline.ingest_frame(frame(30), timestamp_sec=0.0)
        second = pipeline.ingest_frame(frame(30), timestamp_sec=2.0)
        self.assertTrue(first.visual_change)
        self.assertFalse(second.visual_change)
        self.assertIn("cached", second.delta_from_previous["type"][0])

    def test_code_small_edit_triggers_ocr_diff(self):
        pipeline, _ = self.make_pipeline()
        pipeline.ingest_frame(frame(10, patch=(60, 90, 70, 100, 0)), timestamp_sec=0.0)
        second = pipeline.ingest_frame(frame(11, patch=(60, 90, 70, 100, 32)), timestamp_sec=2.5)
        self.assertEqual(second.mode, "code")
        self.assertTrue(second.visual_change)
        self.assertIn("return 2", second.summary)
        self.assertTrue(second.visual_payload["ocr_diff"]["added"])

    def test_slide_duplicate_fingerprint_skips_new_slide_summary(self):
        pipeline, _ = self.make_pipeline()
        pipeline.ingest_frame(frame(20, patch=(40, 80, 160, 140, 100)), timestamp_sec=0.0)
        second = pipeline.ingest_frame(frame(21, patch=(30, 70, 210, 150, 40)), timestamp_sec=3.5)
        self.assertEqual(second.mode, "slide")
        self.assertFalse(second.visual_change)
        self.assertIn("duplicate_slide", second.delta_from_previous["type"])

    def test_demo_small_change_uses_crop_and_skips_macro_rerun(self):
        pipeline, _ = self.make_pipeline()
        pipeline.ingest_frame(frame(30, patch=(120, 90, 150, 120, 60)), timestamp_sec=0.0)
        second = pipeline.ingest_frame(frame(30, patch=(126, 96, 136, 106, 20)), timestamp_sec=1.5)
        self.assertEqual(second.mode, "demo")
        self.assertTrue(second.visual_change)
        self.assertFalse(second.debug["rfdetr_ran"])
        self.assertFalse(second.debug["omniparser_ran"])
        self.assertTrue(second.debug["doclayout_ran"])
        self.assertTrue(second.debug["crop_bbox"])

    def test_demo_ocr_is_structured_with_doclayout_elements(self):
        pipeline, _ = self.make_pipeline()
        record = pipeline.ingest_frame(frame(30, patch=(120, 90, 150, 120, 60)), timestamp_sec=0.0)

        self.assertEqual(record.mode, "demo")
        self.assertTrue(record.debug["ocr_ran"])
        self.assertIn("Search orders", record.visual_payload["ocr_text"])
        self.assertEqual(record.confidence["ocr_provider"], "ocr:rapidocr")
        self.assertEqual(record.confidence["doclayout_provider"], "fake_doclayout")
        structure = record.visual_payload["screen_structure"]
        self.assertTrue(structure["reading_order"])
        main_region = next(item for item in structure["regions"] if item["type"] == "main_content")
        self.assertGreaterEqual(main_region["ocr_line_count"], 1)
        nested_nodes = [node for node in main_region["children"] if node["kind"] in {"ocr_text", "ui_element"}]
        self.assertTrue(nested_nodes)

    def test_cached_demo_frame_clears_heavy_stage_debug_flags(self):
        pipeline, _ = self.make_pipeline(cursor_boxes=[None, None])
        pipeline.ingest_frame(frame(30, patch=(120, 90, 150, 120, 60)), timestamp_sec=0.0)
        cached = pipeline.ingest_frame(frame(30, patch=(120, 90, 150, 120, 60)), timestamp_sec=2.0)
        self.assertEqual(cached.mode, "demo")
        self.assertFalse(cached.visual_change)
        self.assertTrue(cached.debug["reused_cache"])
        self.assertEqual(cached.debug["processor_used"], "none")
        self.assertFalse(cached.debug["rfdetr_ran"])
        self.assertFalse(cached.debug["omniparser_ran"])

    def test_demo_resolution_change_resets_change_baseline(self):
        pipeline, _ = self.make_pipeline(cursor_boxes=[None, None])
        pipeline.ingest_frame(frame(30, patch=(120, 90, 150, 120, 60)), timestamp_sec=0.0)
        larger = cv2.resize(frame(30, patch=(140, 110, 170, 140, 30)), (340, 220), interpolation=cv2.INTER_NEAREST)
        larger[0, 0, 0] = 30

        second = pipeline.ingest_frame(larger, timestamp_sec=1.5)

        self.assertEqual(second.mode, "demo")
        self.assertTrue(second.visual_change)
        self.assertIn("initial_frame", second.delta_from_previous["type"])
        self.assertEqual(pipeline.cache.demo.prev_frame_masked.shape[:2], (220, 340))

    def test_transcript_alignment_uses_since_previous_visual_event(self):
        pipeline, _ = self.make_pipeline()
        pipeline.ingest_utterance(start=0.1, end=0.8, speaker="spk1", text="explaining the first code block")
        first = pipeline.ingest_frame(frame(10, patch=(60, 90, 70, 100, 0)), timestamp_sec=1.0)
        pipeline.ingest_utterance(start=1.2, end=1.8, speaker="spk1", text="now updating the return value")
        second = pipeline.ingest_frame(frame(11, patch=(60, 90, 70, 100, 32)), timestamp_sec=3.5)
        self.assertEqual(len(first.transcript_window), 1)
        self.assertEqual(len(second.transcript_window), 1)
        self.assertIn("now updating", second.transcript_window[0]["text"])

    def test_llm_summary_receives_recent_memory(self):
        pipeline, summarizer = self.make_pipeline()
        first = pipeline.ingest_frame(frame(30, patch=(120, 90, 150, 120, 60)), timestamp_sec=0.0)
        second = pipeline.ingest_frame(frame(30, patch=(126, 96, 136, 106, 20)), timestamp_sec=1.5)
        self.assertIn("[memory=0]", first.summary)
        self.assertIn("[memory=1]", second.summary)
        self.assertTrue(any(len(request.recent_memory) == 0 for request in summarizer.requests))
        self.assertTrue(any(len(request.recent_memory) >= 1 for request in summarizer.requests))

    def test_pending_newer_frame_skips_near_duplicate_llm_work(self):
        pipeline, summarizer = self.make_pipeline()
        pipeline.ingest_frame(frame(30, patch=(120, 90, 150, 120, 60)), timestamp_sec=0.0)
        request_count = len(summarizer.requests)
        pipeline.summary_skip_checker = lambda _state: True

        second = pipeline.ingest_frame(frame(30, patch=(126, 96, 136, 106, 20)), timestamp_sec=1.5)

        self.assertEqual(len(summarizer.requests), request_count)
        self.assertEqual(second.debug["llm_policy"], "skipped_near_duplicate_pending_newer_frame")
        self.assertIn("Deferred Results", second.summary)

    def test_inflight_llm_can_defer_when_newer_near_duplicate_arrives(self):
        pipeline, _ = self.make_pipeline()
        pipeline.ingest_frame(frame(30, patch=(120, 90, 150, 120, 60)), timestamp_sec=0.0)
        slow = SlowSummarizer(delay=1.0)
        pipeline.frame_summarizer = slow
        newer_waiting = {"value": False}
        pipeline.summary_skip_checker = lambda _state: newer_waiting["value"]

        result = {}

        def run_frame():
            result["record"] = pipeline.ingest_frame(frame(30, patch=(126, 96, 136, 106, 20)), timestamp_sec=1.5)

        thread = threading.Thread(target=run_frame)
        started = time.perf_counter()
        thread.start()
        self.assertTrue(slow.entered.wait(timeout=2.0))
        newer_waiting["value"] = True
        thread.join(timeout=2.0)
        elapsed = time.perf_counter() - started

        self.assertFalse(thread.is_alive())
        self.assertLess(elapsed, 0.9)
        record = result["record"]
        self.assertIn("Deferred Results", record.summary)
        self.assertEqual(record.debug["llm_policy"], "skipped_inflight_near_duplicate_pending_newer_frame")

    def test_deferred_frame_is_paired_with_next_near_duplicate_summary(self):
        pipeline, summarizer = self.make_pipeline()
        pipeline.ingest_frame(frame(30, patch=(120, 90, 150, 120, 60)), timestamp_sec=0.0)
        pipeline.summary_skip_checker = lambda _state: True
        pipeline.ingest_frame(frame(30, patch=(126, 96, 136, 106, 20)), timestamp_sec=1.5)
        pipeline.summary_skip_checker = lambda _state: False

        pipeline.ingest_frame(frame(30, patch=(132, 102, 142, 112, 80)), timestamp_sec=3.0)

        paired_requests = [
            request
            for request in summarizer.requests
            if (request.current_payload or {}).get("previous_frame")
        ]
        self.assertTrue(paired_requests)
        self.assertIn(paired_requests[-1].current_payload["summary_task"], {"demo_region", "demo_relationship"})

    def test_demo_ocr_can_override_misrouted_code_frame(self):
        pipeline, _ = self.make_pipeline()

        class CodeLikeOCR(FakeOCR):
            def extract_regions(self, frame_bgr, regions, *, frame_type):
                return OCRResult(
                    text="OPEN EDITORS\nmeet_agent.py\nimport numpy as np\nfrom pydantic import BaseModel\ndef analyze():\n    return None",
                    lines=[
                        {"text": "OPEN EDITORS", "box": [8, 20, 120, 40], "confidence": 0.9, "source": "ocr:fake"},
                        {"text": "import numpy as np", "box": [80, 80, 220, 100], "confidence": 0.9, "source": "ocr:fake"},
                        {"text": "from pydantic import BaseModel", "box": [80, 105, 250, 125], "confidence": 0.9, "source": "ocr:fake"},
                        {"text": "def analyze():", "box": [80, 130, 200, 150], "confidence": 0.9, "source": "ocr:fake"},
                        {"text": "return None", "box": [100, 155, 200, 175], "confidence": 0.9, "source": "ocr:fake"},
                    ],
                    provider="ocr:fake",
                    confidence=0.9,
                )

        pipeline.ocr_extractor = CodeLikeOCR()
        record = pipeline.ingest_frame(frame(30, patch=(120, 90, 150, 120, 60)), timestamp_sec=0.0)

        self.assertEqual(record.mode, "code")
        self.assertEqual(record.debug["route_override"], "demo_to_code_from_ocr")

    def test_demo_page_summary_filters_svgish_micro_labels(self):
        pipeline, _ = self.make_pipeline()
        summary = pipeline._build_demo_page_summary(  # pylint: disable=protected-access
            layout_groups=[
                LayoutGroup(id="group_header", label="header", bbox=(0, 0, 280, 50)),
                LayoutGroup(id="group_main", label="main_content", bbox=(0, 50, 280, 180)),
            ],
            elements=[
                UIElement(id="a", type="icon", bbox=(10, 10, 30, 30), text="M0,0L0,5 4.5,5z", interactive=True),
                UIElement(id="b", type="button", bbox=(40, 40, 140, 80), text="Upload CSV", interactive=True),
            ],
            overlays=[],
            cursor_target={"element_text": "Upload CSV"},
        )
        self.assertIn("Upload CSV", summary)
        self.assertNotIn("M0,0L0,5", summary)

    def test_micro_sidebar_inference_rejects_wide_main_content_controls(self):
        pipeline = DemoAnalyzerPipeline.__new__(DemoAnalyzerPipeline)
        pipeline._frame_id = 1  # pylint: disable=protected-access
        groups = [LayoutGroup(id="group_main_content", label="main_content", bbox=(0, 0, 1200, 800), source="fallback")]
        elements = [
            UIElement(id=f"wide_{idx}", type="button", bbox=(30, 100 + idx * 80, 820, 130 + idx * 80), interactive=True)
            for idx in range(7)
        ]

        augmented = pipeline._augment_macro_layout_from_micro(  # pylint: disable=protected-access
            groups,
            elements,
            {"width": 1200, "height": 800},
        )

        self.assertEqual([group.label for group in augmented], ["main_content"])

    def test_micro_sidebar_inference_clips_main_content_when_sidebar_is_compact(self):
        pipeline = DemoAnalyzerPipeline.__new__(DemoAnalyzerPipeline)
        pipeline._frame_id = 1  # pylint: disable=protected-access
        groups = [LayoutGroup(id="group_main_content", label="main_content", bbox=(0, 0, 1200, 800), source="fallback")]
        elements = [
            UIElement(id=f"nav_{idx}", type="button", bbox=(20, 100 + idx * 70, 180, 130 + idx * 70), interactive=True)
            for idx in range(8)
        ]

        augmented = pipeline._augment_macro_layout_from_micro(  # pylint: disable=protected-access
            groups,
            elements,
            {"width": 1200, "height": 800},
        )
        left = next(group for group in augmented if group.label == "left_sidebar")
        main = next(group for group in augmented if group.label == "main_content")

        self.assertEqual(main.bbox[0], left.bbox[2])
        self.assertGreaterEqual(main.score, left.score)

    def test_router_ignores_siglip_debug_scores(self):
        router = ModeRouter(immediate_switch_confidence=0.8, sticky_confidence_floor=0.45)
        decision, changed = router.route(
            raw_label="demo",
            raw_scores={
                "slides": -0.05,
                "code": -0.06,
                "demo": -0.03,
                "none": -0.04,
                "_best_non_none_idx": 2.0,
                "_none_margin": 0.02,
            },
        )
        self.assertTrue(changed)
        self.assertEqual(decision.mode, "demo")
        self.assertEqual(decision.scores["demo"], 1.0)

    def test_surya_enabled_flag_controls_slide_analyzer(self):
        with patch("real_time_mode.slide_adapter._detect_surya", return_value=True):
            self.assertTrue(SuryaSlideAnalyzer(enabled=True)._surya_available)
            self.assertFalse(SuryaSlideAnalyzer(enabled=False)._surya_available)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from batch_mode.llm_adapters import MockSectionWriter, _clean_generated_text, _guard_generated_section
from batch_mode.macro_layout import UIDetection, build_layout_groups, estimate_coarse_regions
from batch_mode.omniparser_adapter import OmniParserAdapter
from batch_mode.pipeline import DemoAnalyzerPipeline, _build_demo_screen_structure
from batch_mode.segmenter import TopicSegmenter
from batch_mode.service import BatchMeetingAnalyzer, _supports_local_qwen, build_report_sections
from batch_mode.slide_adapter import SuryaSlideAnalyzer
from batch_mode.state import LayoutGroup, UIElement
from batch_mode.topic_linking import GlobalTopicLinker


class BatchModeTests(unittest.TestCase):
    def test_segmenter_builds_units(self) -> None:
        records = [
            {
                "event_id": "evt_1",
                "timestamp_sec": 0.0,
                "mode": "slide",
                "summary": "Architecture overview slide",
                "visual_payload": {"layout": [{"type": "title", "text": "Architecture"}], "full_text": "Architecture overview"},
            },
            {
                "event_id": "evt_2",
                "timestamp_sec": 12.0,
                "mode": "slide",
                "summary": "Upload pipeline slide",
                "visual_payload": {"layout": [{"type": "title", "text": "Upload Flow"}], "full_text": "CSV upload flow"},
            },
            {
                "event_id": "evt_3",
                "timestamp_sec": 28.0,
                "mode": "demo",
                "summary": "Upload button shown in dashboard",
                "visual_payload": {"page_summary": "Dashboard with Upload CSV button", "change_type": "navigation"},
            },
        ]
        transcript = [
            {"start": 1.0, "end": 5.0, "text": "Here is the architecture", "speaker": "s1"},
            {"start": 14.0, "end": 18.0, "text": "Now the upload flow", "speaker": "s1"},
            {"start": 29.0, "end": 31.0, "text": "This is the upload button", "speaker": "s1"},
        ]
        units = TopicSegmenter(min_sec=10, max_sec=30, target_sec=18).build_units(records, transcript)
        self.assertGreaterEqual(len(units), 2)
        self.assertEqual(units[0].dominant_class, "slide")

    def test_topic_linker_clusters_related_content(self) -> None:
        records = [
            {"event_id": "evt_1", "timestamp_sec": 0.0, "mode": "slide", "summary": "CSV upload flow architecture", "visual_payload": {}},
            {"event_id": "evt_2", "timestamp_sec": 20.0, "mode": "code", "summary": "upload endpoint implementation", "visual_payload": {}},
            {"event_id": "evt_3", "timestamp_sec": 40.0, "mode": "demo", "summary": "Upload CSV button in dashboard", "visual_payload": {}},
        ]
        units = [
            {
                "start": 0.0,
                "end": 20.0,
                "dominant_class": "slide",
                "transcript_text": "CSV upload flow overview",
                "event_ids": ["evt_1", "evt_2"],
                "ocr_text": "CSV upload",
                "screen_summaries": [],
                "action_summaries": [],
                "unit_summary": "CSV upload flow explained",
            },
            {
                "start": 40.0,
                "end": 55.0,
                "dominant_class": "demo",
                "transcript_text": "Upload button in dashboard",
                "event_ids": ["evt_3"],
                "ocr_text": "Upload CSV",
                "screen_summaries": [],
                "action_summaries": [],
                "unit_summary": "CSV upload in dashboard",
            },
        ]
        clusters = GlobalTopicLinker(similarity_threshold=0.2).cluster(records, units)
        self.assertGreaterEqual(len(clusters), 1)
        self.assertTrue(any("upload" in cluster.topic_title.lower() for cluster in clusters))

    def test_supports_local_qwen_requires_large_cuda_gpu(self) -> None:
        with patch("batch_mode.service.torch.cuda.is_available", return_value=True), patch(
            "batch_mode.service.torch.cuda.get_device_properties"
        ) as mock_props:
            mock_props.return_value.total_memory = 40 * 1024**3
            self.assertTrue(_supports_local_qwen())

    def test_generated_text_cleanup_strips_think_and_headers(self) -> None:
        raw = "<think>hidden reasoning</think>\ncode_section: The presenter edits the upload handler."
        cleaned = _clean_generated_text(raw, "code_section")
        self.assertEqual(cleaned, "The presenter edits the upload handler.")

    def test_generated_text_cleanup_prefers_answer_tag(self) -> None:
        raw = "<think>hidden reasoning</think>\n<answer>Final clean section.</answer>"
        cleaned = _clean_generated_text(raw, "topic_section")
        self.assertEqual(cleaned, "Final clean section.")

    def test_cluster_runner_stages_public_links_before_sbatch(self) -> None:
        try:
            from batch_mode.cluster_runner import ClusterBatchRunner
        except ModuleNotFoundError as exc:
            if exc.name == "paramiko":
                self.skipTest("paramiko is not installed in this environment")
            raise
        runner = ClusterBatchRunner.__new__(ClusterBatchRunner)
        runner.settings = SimpleNamespace(
            cluster_project_dir="/cluster/meet-agent",
            batch_download_timeout_sec=600,
        )
        commands = []

        def capture_command(_client, command: str) -> str:
            commands.append(command)
            return ""

        runner._exec_checked = capture_command
        remote_source = runner._prepare_remote_source(
            object(),
            request_id="req123",
            source="https://cdn.example.com/videos/play_480p.mp4?token=abc",
        )

        self.assertEqual(remote_source, "/cluster/meet-agent/batch_mode/cluster_jobs/req123/play_480p.mp4")
        self.assertEqual(len(commands), 1)
        self.assertIn("curl -L --fail", commands[0])
        self.assertIn("play_480p.mp4", commands[0])

    def test_batch_macro_layout_keeps_low_confidence_regions_for_coverage(self) -> None:
        detections = [
            UIDetection(box=(10, 20, 450, 80), class_name="Navigation", confidence=0.55),
            UIDetection(box=(15, 120, 250, 680), class_name="Sidebar", confidence=0.60),
        ]

        regions, confidences = estimate_coarse_regions(detections, width=1200, height=900)
        groups = build_layout_groups(regions, confidences, frame_w=1200, frame_h=900)
        labels = [group.label for group in groups]

        self.assertIn("header", labels)
        self.assertIn("left_sidebar", labels)
        self.assertIn("main_content", labels)
        self.assertTrue(any(group.source == "uitag-yolo-lowconf" for group in groups))
        sample_points = [(0, 0), (1199, 0), (0, 899), (1199, 899), (600, 450)]
        for x, y in sample_points:
            self.assertTrue(any(group.bbox[0] <= x <= group.bbox[2] and group.bbox[1] <= y <= group.bbox[3] for group in groups))

    def test_batch_macro_layout_assigns_unclaimed_screen_to_main_content(self) -> None:
        regions, confidences = estimate_coarse_regions([], width=1200, height=900)
        groups = build_layout_groups(regions, confidences, frame_w=1200, frame_h=900)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].label, "main_content")
        self.assertEqual(groups[0].bbox, (0, 0, 1200, 900))

    def test_batch_macro_layout_rejects_wide_left_edge_sidebar_false_positive(self) -> None:
        detections = [
            UIDetection(box=(24, 90 + idx * 70, 780, 120 + idx * 70), class_name="Navigation", confidence=0.78)
            for idx in range(6)
        ]

        regions, confidences = estimate_coarse_regions(detections, width=1200, height=900)
        groups = build_layout_groups(regions, confidences, frame_w=1200, frame_h=900)
        labels = [group.label for group in groups]

        self.assertNotIn("left_sidebar", labels)
        self.assertIn("main_content", labels)

    def test_demo_summary_builds_bigger_picture_from_elements(self) -> None:
        groups = [
            LayoutGroup(id="group_main_content", label="main_content", bbox=(0, 0, 1200, 900), source="test"),
        ]
        elements = [
            UIElement(id="input_1", type="input", bbox=(100, 700, 900, 760), text="Ask anything", interactive=True),
            UIElement(id="input_2", type="input", bbox=(100, 620, 900, 680), text="Message", interactive=True),
            UIElement(id="send", type="button", bbox=(920, 700, 1000, 760), text="Send", interactive=True),
            UIElement(id="demo_ocr_text_0001", type="text", bbox=(94, 694, 910, 766), text="Ask anything", source="ocr:rapidocr"),
        ]
        structure = _build_demo_screen_structure(groups, elements)
        summary = DemoAnalyzerPipeline.__new__(DemoAnalyzerPipeline)._build_demo_page_summary(
            groups,
            elements,
            [],
            {"element_text": "Send", "parent_macro_region_type": "main_content", "relation": "hovering_over"},
            screen_structure=structure,
        )

        self.assertIn("chatbot", summary.lower())
        self.assertIn("main content area contains", summary.lower())
        self.assertIn("cursor", summary.lower())
        self.assertEqual(structure["regions"][0]["ocr_line_count"], 1)
        ocr_parent = next(node for node in structure["regions"][0]["children"] if node["id"] == "demo_ocr_text_0001")
        self.assertTrue(any(child["id"] == "input_1" for child in ocr_parent["children"]))

    def test_omniparser_region_parse_rebases_bottom_screen_elements(self) -> None:
        class FakeParserClient:
            def infer(self, crop):
                height = crop.shape[0]
                return {
                    "provider": "fake",
                    "regions": [
                        {
                            "type": "button",
                            "box": [10, height - 30, 100, height - 10],
                            "content": "Bottom control",
                            "interactive": True,
                            "confidence": 0.9,
                        }
                    ],
                }

        frame = np.zeros((900, 1200, 3), dtype=np.uint8)
        payload = OmniParserAdapter(client=FakeParserClient()).parse_regions(
            frame,
            [(0, 0, 1200, 100), (0, 700, 1200, 900)],
        )

        self.assertEqual(len(payload.elements), 2)
        self.assertEqual(payload.elements[0].bbox, (10, 70, 100, 90))
        self.assertEqual(payload.elements[1].bbox, (10, 870, 100, 890))

    def test_report_sections_include_detailed_flow_and_full_transcript(self) -> None:
        events = [
            {
                "event_id": "evt_1",
                "timestamp_sec": 1.0,
                "mode": "demo",
                "summary": "The presenter opens the homework page and hovers over the Next button.",
                "cursor": {"detected": True, "hotspot": [320, 240], "method": "yolo"},
                "visual_payload": {
                    "cursor_target": {
                        "element_text": "Next",
                        "parent_macro_region_type": "main_content",
                        "relation": "hovering_over",
                    }
                },
            },
            {
                "event_id": "evt_2",
                "timestamp_sec": 2.0,
                "mode": "demo",
                "summary": "The next homework question appears.",
                "visual_payload": {"change_type": "navigation"},
            }
        ]
        transcript = [
            {"start": 1.0, "end": 3.0, "speaker": "speaker_0", "text": "Here we start the homework demo."}
        ]
        units = [
            {
                "start": 0.0,
                "end": 5.0,
                "dominant_class": "demo",
                "event_ids": ["evt_1", "evt_2"],
                "unit_summary": "The presenter introduces the homework page.",
                "transcript_text": "Here we start the homework demo.",
            }
        ]
        sections = build_report_sections(
            frame_events=events,
            transcript_segments=transcript,
            topic_units=units,
            topic_clusters=[],
            title="demo",
        )

        narrative = sections["meeting_narrative_section"]["prewritten_markdown"]
        full_transcript = sections["transcript_section"]["prewritten_markdown"]
        self.assertIn("Screen flow", narrative)
        self.assertIn("Cursor:", narrative)
        self.assertIn("followed by a navigation change", narrative)
        self.assertIn("Here we start the homework demo.", full_transcript)

    def test_mock_writer_preserves_prewritten_sections(self) -> None:
        sections = {
            "report_title": "demo",
            "meeting_narrative_section": {"prewritten_markdown": "Detailed narrative body."},
            "transcript_section": {"prewritten_markdown": "Full transcript body."},
        }
        output = MockSectionWriter().generate(sections=sections)
        self.assertIn("Detailed narrative body.", output["final_markdown"])
        self.assertIn("Full transcript body.", output["final_markdown"])

    def test_batch_interval_respects_sample_fps_for_demo(self) -> None:
        analyzer = BatchMeetingAnalyzer.__new__(BatchMeetingAnalyzer)
        analyzer.settings = SimpleNamespace(
            run_sample_fps=0.5,
            demo_interval_sec=1.0,
            code_interval_sec=2.0,
            slide_interval_sec=3.0,
            other_interval_sec=5.0,
        )

        self.assertEqual(analyzer._interval_for_mode("demo"), 2.0)
        self.assertEqual(analyzer._interval_for_mode("slide"), 3.0)

    def test_hallucinated_generated_section_falls_back_to_evidence(self) -> None:
        payload = {"prewritten_markdown": "Evidence-only timeline from OCR and transcript."}
        cleaned = "Alex opened the Q2 Project Review and discussed Project Alpha."

        guarded = _guard_generated_section("meeting_narrative_section", payload, cleaned)

        self.assertEqual(guarded, "Evidence-only timeline from OCR and transcript.")

    def test_surya_enabled_flag_controls_slide_analyzer(self) -> None:
        with patch("batch_mode.slide_adapter._detect_surya", return_value=True):
            self.assertTrue(SuryaSlideAnalyzer(enabled=True)._surya_available)
            self.assertFalse(SuryaSlideAnalyzer(enabled=False)._surya_available)


if __name__ == "__main__":
    unittest.main()

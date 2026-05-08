import unittest

from real_time_mode.macro_layout import UIDetection, build_layout_groups, estimate_coarse_regions


class MacroLayoutRulesTests(unittest.TestCase):
    def test_estimate_regions_builds_left_sidebar_header_footer(self) -> None:
        detections = [
            UIDetection(box=(10, 20, 450, 80), class_name="Navigation", confidence=0.91),
            UIDetection(box=(15, 110, 250, 760), class_name="Sidebar", confidence=0.88),
            UIDetection(box=(20, 900, 420, 980), class_name="Information_Display", confidence=0.79),
        ]

        regions, confidences = estimate_coarse_regions(detections, width=1200, height=1000)

        self.assertIsNotNone(regions["header"])
        self.assertIsNotNone(regions["left_sidebar"])
        self.assertIsNotNone(regions["footer"])
        self.assertIsNotNone(regions["main_content"])
        self.assertEqual(regions["left_sidebar"][0], 0)
        self.assertGreater(regions["main_content"][0], 0)
        self.assertGreater(confidences["header"], 0.75)
        self.assertGreater(confidences["left_sidebar"], 0.75)
        self.assertGreater(confidences["footer"], 0.75)

    def test_build_layout_groups_keeps_expected_macro_labels(self) -> None:
        regions = {
            "header": (0, 0, 1200, 100),
            "left_sidebar": (0, 100, 250, 900),
            "main_content": (250, 100, 1200, 900),
            "footer": (0, 900, 1200, 1000),
        }
        confidences = {
            "header": 0.91,
            "left_sidebar": 0.88,
            "main_content": 0.91,
            "footer": 0.79,
        }

        groups = build_layout_groups(regions, confidences, frame_w=1200, frame_h=1000)
        labels = [group.label for group in groups]

        self.assertEqual(labels, ["header", "left_sidebar", "main_content", "footer"])
        self.assertTrue(all(group.score > 0.75 for group in groups))

    def test_low_confidence_clues_do_not_count(self) -> None:
        detections = [
            UIDetection(box=(10, 20, 450, 80), class_name="Navigation", confidence=0.55),
            UIDetection(box=(15, 110, 250, 320), class_name="Sidebar", confidence=0.60),
        ]

        regions, confidences = estimate_coarse_regions(detections, width=1200, height=1000)
        groups = build_layout_groups(regions, confidences, frame_w=1200, frame_h=1000)

        labels = [group.label for group in groups]
        self.assertIn("left_sidebar", labels)
        self.assertIn("main_content", labels)
        self.assertTrue(any(group.label == "left_sidebar" and group.score <= 0.75 for group in groups))

    def test_repeated_sidebar_clues_can_build_confident_sidebar_region(self) -> None:
        detections = [
            UIDetection(box=(20, 100, 120, 130), class_name="Navigation", confidence=0.70),
            UIDetection(box=(24, 160, 122, 190), class_name="Navigation", confidence=0.72),
            UIDetection(box=(22, 220, 118, 250), class_name="Menu", confidence=0.69),
            UIDetection(box=(18, 280, 124, 310), class_name="Navigation", confidence=0.71),
        ]

        regions, confidences = estimate_coarse_regions(detections, width=1200, height=1000)
        groups = build_layout_groups(regions, confidences, frame_w=1200, frame_h=1000)
        labels = [group.label for group in groups]

        self.assertIn("left_sidebar", labels)
        left_group = next(group for group in groups if group.label == "left_sidebar")
        self.assertGreater(left_group.score, 0.75)

    def test_wide_left_edge_controls_do_not_become_sidebar(self) -> None:
        detections = [
            UIDetection(box=(24, 90 + idx * 70, 780, 120 + idx * 70), class_name="Navigation", confidence=0.78)
            for idx in range(6)
        ]

        regions, confidences = estimate_coarse_regions(detections, width=1200, height=900)
        groups = build_layout_groups(regions, confidences, frame_w=1200, frame_h=900)
        labels = [group.label for group in groups]

        self.assertNotIn("left_sidebar", labels)
        self.assertIn("main_content", labels)


if __name__ == "__main__":
    unittest.main()

import unittest

from real_time_mode.vendor.OmniParser.util.utils import remove_overlap_new


class OmniParserUtilsTest(unittest.TestCase):
    def test_remove_overlap_without_ocr_returns_dict_elements(self) -> None:
        boxes = [
            {"type": "icon", "bbox": [0.1, 0.1, 0.3, 0.3], "interactivity": True, "content": None},
            {"type": "icon", "bbox": [0.5, 0.5, 0.7, 0.7], "interactivity": True, "content": None},
        ]

        result = remove_overlap_new(boxes=boxes, iou_threshold=0.9, ocr_bbox=[])

        self.assertEqual(len(result), 2)
        self.assertTrue(all(isinstance(item, dict) for item in result))
        self.assertTrue(all("bbox" in item for item in result))
        self.assertTrue(all(item.get("type") == "icon" for item in result))


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mouse_detector.validate_annotations import load_annotation_records, render_comparison


class AnnotationRecordTests(unittest.TestCase):
    def test_load_annotation_records_groups_boxes_by_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            annotations_path = Path(tmp) / "annotations.json"
            annotations_path.write_text(
                json.dumps(
                    {
                        "images": [
                            {"id": 1, "file_name": "nested/a.jpg"},
                            {"id": 2, "file_name": "b.jpg"},
                        ],
                        "annotations": [
                            {"image_id": 1, "bbox": [10, 20, 30, 40]},
                            {"image_id": 1, "bbox": [1, 2, 3, 4]},
                            {"image_id": 2, "bbox": [5, 6, 7, 8]},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            records = load_annotation_records(annotations_path)

            self.assertEqual(len(records), 2)
            first_path, first_boxes = records[0]
            self.assertEqual(first_path, Path("nested/a.jpg"))
            self.assertEqual(first_boxes[0].tolist(), [10.0, 20.0, 40.0, 60.0])
            self.assertEqual(first_boxes[1].tolist(), [1.0, 2.0, 4.0, 6.0])


class RenderComparisonTests(unittest.TestCase):
    def test_render_comparison_draws_label_and_prediction_boxes(self):
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        labeled_boxes = [np.array([2, 2, 12, 12], dtype=np.float32)]
        predicted_box = np.array([20, 20, 30, 30], dtype=np.float32)

        rendered = render_comparison(frame, labeled_boxes, predicted_box)

        self.assertEqual(rendered[2, 2].tolist(), [255, 0, 0])
        self.assertEqual(rendered[20, 20].tolist(), [0, 255, 0])


if __name__ == "__main__":
    unittest.main()

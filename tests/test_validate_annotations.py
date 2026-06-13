import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


class FakeArray:
    def __init__(self, values):
        self._values = list(values)

    def tolist(self):
        return self._values


class AnnotationRecordTests(unittest.TestCase):
    def test_load_annotation_records_groups_boxes_by_image(self):
        original_modules = {}
        for name in ("cv2", "numpy", "torch", "mouse_detector.infer_videos", "mouse_detector.validate_annotations"):
            if name in sys.modules:
                original_modules[name] = sys.modules[name]

        try:
            sys.modules["cv2"] = types.ModuleType("cv2")

            numpy_module = types.ModuleType("numpy")
            numpy_module.float32 = "float32"
            numpy_module.ndarray = FakeArray
            numpy_module.array = lambda values, dtype=None: FakeArray(values)
            sys.modules["numpy"] = numpy_module

            torch_module = types.ModuleType("torch")
            torch_module.cuda = types.SimpleNamespace(is_available=lambda: False)
            sys.modules["torch"] = torch_module

            infer_module = types.ModuleType("mouse_detector.infer_videos")
            infer_module.load_model = lambda weights_path, device: None
            infer_module.select_box = lambda prediction, threshold: None
            sys.modules["mouse_detector.infer_videos"] = infer_module

            validate_annotations = importlib.import_module("mouse_detector.validate_annotations")

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

                records = validate_annotations.load_annotation_records(annotations_path)

            self.assertEqual(len(records), 2)
            first_path, first_boxes = records[0]
            self.assertEqual(first_path, Path("nested/a.jpg"))
            self.assertEqual(first_boxes[0].tolist(), [10, 20, 40, 60])
            self.assertEqual(first_boxes[1].tolist(), [1, 2, 4, 6])
        finally:
            for name in ("cv2", "numpy", "torch", "mouse_detector.infer_videos", "mouse_detector.validate_annotations"):
                sys.modules.pop(name, None)
            sys.modules.update(original_modules)


if __name__ == "__main__":
    unittest.main()

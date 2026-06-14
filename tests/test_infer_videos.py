import argparse
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from mouse_detector.cli_utils import list_video_files, parse_size


class ParseSizeTests(unittest.TestCase):
    def test_parse_size_accepts_x_format(self):
        self.assertEqual(parse_size("640x360"), (640, 360))

    def test_parse_size_accepts_comma_format(self):
        self.assertEqual(parse_size("320,240"), (320, 240))

    def test_parse_size_rejects_invalid(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_size("wrong")


class VideoListingTests(unittest.TestCase):
    def test_list_video_files_filters_extensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.mp4").write_bytes(b"")
            (root / "b.MOV").write_bytes(b"")
            (root / "c.txt").write_bytes(b"")

            files = list_video_files(root)

            self.assertEqual([p.name for p in files], ["a.mp4", "b.MOV"])


class CropFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_modules = {}
        for name in ("cv2", "torch", "mouse_detector.train", "mouse_detector.infer_videos"):
            if name in sys.modules:
                cls.original_modules[name] = sys.modules[name]

        cv2_module = types.ModuleType("cv2")
        sys.modules["cv2"] = cv2_module

        torch_module = types.ModuleType("torch")
        torch_module.Tensor = object
        torch_module.nn = types.SimpleNamespace(Module=object)
        torch_module.device = lambda value: value
        torch_module.cuda = types.SimpleNamespace(is_available=lambda: False)
        sys.modules["torch"] = torch_module

        train_module = types.ModuleType("mouse_detector.train")
        train_module.build_model = lambda num_classes: None
        sys.modules["mouse_detector.train"] = train_module

        sys.modules.pop("mouse_detector.infer_videos", None)
        cls.infer_videos = importlib.import_module("mouse_detector.infer_videos")

    @classmethod
    def tearDownClass(cls):
        for name in ("cv2", "torch", "mouse_detector.train", "mouse_detector.infer_videos"):
            sys.modules.pop(name, None)
        sys.modules.update(cls.original_modules)

    def test_crop_frame_around_box_keeps_box_centered(self):
        frame = np.arange(5 * 6 * 3, dtype=np.uint8).reshape(5, 6, 3)
        box = np.array([2, 1, 4, 3], dtype=np.float32)

        cropped = self.infer_videos.crop_frame_around_box(frame, box, (4, 2))

        np.testing.assert_array_equal(cropped, frame[1:3, 1:5])

    def test_crop_frame_around_box_pads_with_black_pixels(self):
        frame = np.arange(3 * 3 * 3, dtype=np.uint8).reshape(3, 3, 3)
        box = np.array([0, 0, 2, 2], dtype=np.float32)

        cropped = self.infer_videos.crop_frame_around_box(frame, box, (4, 4))

        expected = np.zeros((4, 4, 3), dtype=np.uint8)
        expected[1:4, 1:4] = frame
        np.testing.assert_array_equal(cropped, expected)

    def test_crop_frame_without_box_uses_frame_center(self):
        frame = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)

        cropped = self.infer_videos.crop_frame_around_box(frame, None, (4, 2))

        np.testing.assert_array_equal(cropped, frame[1:3, 1:5])


if __name__ == "__main__":
    unittest.main()

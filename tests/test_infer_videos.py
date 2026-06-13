import argparse
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()

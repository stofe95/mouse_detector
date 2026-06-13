from __future__ import annotations

import argparse
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv"}


def parse_size(value: str) -> tuple[int, int]:
    if "x" in value.lower():
        width_str, height_str = value.lower().split("x", 1)
    else:
        parts = value.split(",")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("Size must be WIDTHxHEIGHT or WIDTH,HEIGHT")
        width_str, height_str = parts

    try:
        width = int(width_str)
        height = int(height_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Size must contain integers") from exc

    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Size must be positive")
    return width, height


def list_video_files(folder: str | Path) -> list[Path]:
    root = Path(folder)
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)

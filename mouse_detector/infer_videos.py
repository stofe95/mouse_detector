from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from mouse_detector.cli_utils import list_video_files, parse_size
from mouse_detector.train import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mouse detection on all videos in a folder.")
    parser.add_argument("--weights", required=True, help="Trained model checkpoint (.pt)")
    parser.add_argument("--input-dir", required=True, help="Input folder containing videos")
    parser.add_argument("--output-dir", required=True, help="Output folder for cropped detected videos")
    parser.add_argument("--size", type=parse_size, required=True, help="Output size WIDTHxHEIGHT")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--codec", default="mp4v", help="FourCC codec for output videos (e.g. mp4v, XVID)")
    parser.add_argument("--smooth-window", type=int, default=15, help="Moving-average window size for center smoothing (1 = off)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def select_box(prediction: dict[str, torch.Tensor], threshold: float) -> np.ndarray | None:
    boxes = prediction["boxes"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()

    if len(scores) == 0:
        return None

    valid = np.where(scores >= threshold)[0]
    if len(valid) == 0:
        return None
    best_idx = int(valid[np.argmax(scores[valid])])
    return boxes[best_idx]


def crop_frame(frame: np.ndarray, center: tuple[float, float], output_size: tuple[int, int]) -> np.ndarray:
    output_width, output_height = output_size
    frame_height, frame_width = frame.shape[:2]
    center_x, center_y = center

    crop_x1 = math.floor(center_x - (output_width / 2.0) + 0.5)
    crop_y1 = math.floor(center_y - (output_height / 2.0) + 0.5)
    crop_x2 = crop_x1 + output_width
    crop_y2 = crop_y1 + output_height

    src_x1 = max(0, crop_x1)
    src_y1 = max(0, crop_y1)
    src_x2 = min(frame_width, crop_x2)
    src_y2 = min(frame_height, crop_y2)

    if frame.ndim == 2:
        cropped = np.zeros((output_height, output_width), dtype=frame.dtype)
    else:
        cropped = np.zeros((output_height, output_width, frame.shape[2]), dtype=frame.dtype)

    if src_x1 >= src_x2 or src_y1 >= src_y2:
        return cropped

    dst_x1 = src_x1 - crop_x1
    dst_y1 = src_y1 - crop_y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    cropped[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]
    return cropped


def crop_frame_around_box(frame: np.ndarray, box: np.ndarray | None, output_size: tuple[int, int]) -> np.ndarray:
    if box is None:
        center = (frame.shape[1] / 2.0, frame.shape[0] / 2.0)
    else:
        x1, y1, x2, y2 = box.astype(float)
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    return crop_frame(frame, center, output_size)


def load_model(weights_path: str | Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    model = build_model(num_classes=checkpoint.get("num_classes", 2))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def compute_centers(
    model: torch.nn.Module,
    video_path: Path,
    threshold: float,
    device: torch.device,
) -> np.ndarray:
    """Run inference on every frame and return an (N, 2) array of (cx, cy) centers.

    When a frame has no confident detection, the previous frame's center is
    reused so that a single missed detection does not cause a jump.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    centers: list[tuple[float, float]] = []
    prev_center: tuple[float, float] | None = None

    try:
        with torch.no_grad():
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                frame_height, frame_width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device) / 255.0
                prediction = model([tensor])[0]
                box = select_box(prediction, threshold)

                if box is not None:
                    x1, y1, x2, y2 = box.astype(float)
                    center: tuple[float, float] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                elif prev_center is not None:
                    center = prev_center
                else:
                    center = (frame_width / 2.0, frame_height / 2.0)

                prev_center = center
                centers.append(center)
    finally:
        capture.release()

    return np.array(centers, dtype=np.float64)


def smooth_centers(centers: np.ndarray, window: int) -> np.ndarray:
    """Apply a moving-average filter to an (N, 2) array of center coordinates.

    Edge values are padded by repeating the first and last centers so the
    crop does not drift toward the origin at the start or end of the video.
    """
    if window <= 1 or len(centers) == 0:
        return centers.copy()

    kernel = np.ones(window) / window
    pad_before = window // 2
    pad_after = window - 1 - pad_before

    smoothed = np.empty_like(centers)
    for i in range(centers.shape[1]):
        padded = np.pad(centers[:, i], (pad_before, pad_after), mode="edge")
        smoothed[:, i] = np.convolve(padded, kernel, mode="valid")
    return smoothed


def render_video(
    video_path: Path,
    output_path: Path,
    centers: np.ndarray,
    output_size: tuple[int, int],
    codec: str,
) -> None:
    """Write a cropped video using precomputed, smoothed center coordinates."""
    if len(codec) != 4:
        raise ValueError(f"Codec must be exactly 4 characters, got '{codec}'.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, output_size)
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Failed to open output writer: {output_path}")

    try:
        frame_idx = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_idx < len(centers):
                center = (float(centers[frame_idx, 0]), float(centers[frame_idx, 1]))
            else:
                center = (frame.shape[1] / 2.0, frame.shape[0] / 2.0)
            writer.write(crop_frame(frame, center, output_size))
            frame_idx += 1
    finally:
        capture.release()
        writer.release()


def process_video(
    model: torch.nn.Module,
    video_path: Path,
    output_path: Path,
    output_size: tuple[int, int],
    threshold: float,
    codec: str,
    device: torch.device,
    smooth_window: int = 15,
) -> None:
    if len(codec) != 4:
        raise ValueError(f"Codec must be exactly 4 characters, got '{codec}'.")
    centers = compute_centers(model, video_path, threshold, device)
    centers = smooth_centers(centers, smooth_window)

    centers_path = output_path.with_name(output_path.stem + "_centers.npy")
    np.save(centers_path, centers)

    render_video(video_path, output_path, centers, output_size, codec)


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model = load_model(args.weights, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = list_video_files(args.input_dir)
    if not videos:
        print("No video files found.")
        return

    for video_path in videos:
        output_path = output_dir / video_path.name
        process_video(model, video_path, output_path, args.size, args.threshold, args.codec, device, args.smooth_window)
        print(f"Processed {video_path.name} -> {output_path}")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
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
    parser.add_argument("--output-dir", required=True, help="Output folder for resized detected videos")
    parser.add_argument("--size", type=parse_size, required=True, help="Output size WIDTHxHEIGHT")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--codec", default="mp4v", help="FourCC codec for output videos (e.g. mp4v, XVID)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def select_box(prediction: dict[str, torch.Tensor], threshold: float) -> np.ndarray | None:
    boxes = prediction["boxes"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()

    if len(scores) == 0:
        return None

    valid = np.where(scores >= threshold)[0]
    if len(valid) == 0:
        best_idx = int(np.argmax(scores))
    else:
        best_idx = int(valid[np.argmax(scores[valid])])
    return boxes[best_idx]


def draw_box(frame: np.ndarray, box: np.ndarray | None) -> np.ndarray:
    if box is None:
        return frame
    x1, y1, x2, y2 = box.astype(int)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = max(x1 + 1, x2)
    y2 = max(y1 + 1, y2)
    output = frame.copy()
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return output


def load_model(weights_path: str | Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    model = build_model(num_classes=checkpoint.get("num_classes", 2))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def process_video(
    model: torch.nn.Module,
    video_path: Path,
    output_path: Path,
    output_size: tuple[int, int],
    threshold: float,
    codec: str,
    device: torch.device,
) -> None:
    if len(codec) != 4:
        raise ValueError(f"Codec must be exactly 4 characters, got '{codec}'.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*codec[:4])
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, output_size)
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Failed to open output writer: {output_path}")

    try:
        with torch.no_grad():
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device) / 255.0
                prediction = model([tensor])[0]
                box = select_box(prediction, threshold)
                drawn = draw_box(frame, box)
                resized = cv2.resize(drawn, output_size, interpolation=cv2.INTER_AREA)
                writer.write(resized)
    finally:
        capture.release()
        writer.release()


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
        process_video(model, video_path, output_path, args.size, args.threshold, args.codec, device)
        print(f"Processed {video_path.name} -> {output_path}")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()

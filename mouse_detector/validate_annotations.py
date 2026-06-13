from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from mouse_detector.infer_videos import load_model, select_box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run mouse detection on annotated COCO images and save comparison outputs."
    )
    parser.add_argument("--weights", required=True, help="Trained model checkpoint (.pt)")
    parser.add_argument("--images", required=True, help="Path to COCO image directory")
    parser.add_argument("--annotations", required=True, help="Path to COCO annotations JSON file")
    parser.add_argument("--output-dir", required=True, help="Output folder for annotated images")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_annotation_records(annotations_path: str | Path) -> list[tuple[Path, list[np.ndarray]]]:
    annotations_file = Path(annotations_path)
    with annotations_file.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    image_records = {img["id"]: img["file_name"] for img in coco["images"]}
    annotations_by_image: dict[int, list[np.ndarray]] = {}
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        box = np.array([x, y, x + w, y + h], dtype=np.float32)
        annotations_by_image.setdefault(ann["image_id"], []).append(box)

    return [
        (Path(image_records[image_id]), boxes)
        for image_id, boxes in annotations_by_image.items()
        if image_id in image_records
    ]


def draw_labeled_boxes(frame: np.ndarray, boxes: list[np.ndarray]) -> np.ndarray:
    output = frame.copy()
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = max(x1 + 1, x2)
        y2 = max(y1 + 1, y2)
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            output,
            "label",
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return output


def draw_predicted_box(frame: np.ndarray, box: np.ndarray | None) -> np.ndarray:
    output = frame.copy()
    if box is None:
        return output

    x1, y1, x2, y2 = box.astype(int)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = max(x1 + 1, x2)
    y2 = max(y1 + 1, y2)
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        output,
        "prediction",
        (x1, min(output.shape[0] - 6, y2 + 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return output


def render_comparison(frame: np.ndarray, labeled_boxes: list[np.ndarray], predicted_box: np.ndarray | None) -> np.ndarray:
    with_labels = draw_labeled_boxes(frame, labeled_boxes)
    return draw_predicted_box(with_labels, predicted_box)


def process_image(
    model: torch.nn.Module,
    image_path: Path,
    output_path: Path,
    labeled_boxes: list[np.ndarray],
    threshold: float,
    device: torch.device,
) -> None:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device) / 255.0

    with torch.no_grad():
        prediction = model([tensor])[0]

    predicted_box = select_box(prediction, threshold)
    rendered = render_comparison(frame, labeled_boxes, predicted_box)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), rendered):
        raise RuntimeError(f"Failed to write image: {output_path}")


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model = load_model(args.weights, device)

    images_dir = Path(args.images)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotation_records = load_annotation_records(args.annotations)
    if not annotation_records:
        print("No annotated images found.")
        return

    for relative_image_path, labeled_boxes in annotation_records:
        image_path = images_dir / relative_image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing annotated image: {image_path}")

        output_path = output_dir / relative_image_path
        process_image(model, image_path, output_path, labeled_boxes, args.threshold, device)
        print(f"Processed {relative_image_path} -> {output_path}")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()

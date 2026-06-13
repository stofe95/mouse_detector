from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch


class CocoMouseDataset(Dataset):
    """Simple COCO dataset wrapper for a single object class (mouse)."""

    def __init__(self, images_dir: str | Path, annotations_file: str | Path) -> None:
        self.images_dir = Path(images_dir)
        self.annotations_file = Path(annotations_file)

        with self.annotations_file.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        self.image_records = {img["id"]: img for img in coco["images"]}
        self.image_ids = list(self.image_records.keys())

        self.annotations_by_image: dict[int, list[dict[str, Any]]] = {}
        for ann in coco["annotations"]:
            self.annotations_by_image.setdefault(ann["image_id"], []).append(ann)

        missing_annotations = [img_id for img_id in self.image_ids if img_id not in self.annotations_by_image]
        if missing_annotations:
            preview_pairs = [
                (img_id, self.image_records[img_id]["file_name"]) for img_id in missing_annotations[:10]
            ]
            suffix = "..." if len(missing_annotations) > len(preview_pairs) else ""
            raise ValueError(
                "Each image must contain at least one annotation. "
                f"Missing annotations for {len(missing_annotations)} image IDs/files: {preview_pairs}{suffix}"
            )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_id = self.image_ids[index]
        image_record = self.image_records[image_id]
        image_path = self.images_dir / image_record["file_name"]

        image = Image.open(image_path).convert("RGB")
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        anns = self.annotations_by_image.get(image_id, [])
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(1)  # single mouse class
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
        }
        return image_tensor, target


def collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[tuple[torch.Tensor, ...], tuple[dict[str, torch.Tensor], ...]]:
    return tuple(zip(*batch))

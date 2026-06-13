from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from mouse_detector.data import CocoMouseDataset, collate_fn


def build_model(num_classes: int = 2) -> torch.nn.Module:
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a mouse detector on COCO data.")
    parser.add_argument("--images", required=True, help="Path to COCO image directory.")
    parser.add_argument("--annotations", required=True, help="Path to COCO annotations JSON file.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="checkpoints/mouse_detector.pt", help="Checkpoint output path.")
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dataset = CocoMouseDataset(args.images, args.annotations)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = build_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        avg_loss = epoch_loss / max(1, len(loader))
        print(f"Epoch {epoch + 1}/{args.epochs} loss={avg_loss:.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "num_classes": 2}, output_path)
    print(f"Saved model to {output_path}")


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()

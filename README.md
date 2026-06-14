# mouse_detector

PyTorch-based mouse detector for COCO datasets with:
- training on COCO images + bounding boxes
- inference across a folder of videos
- output cropped videos centered on the detected mouse

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python -m mouse_detector.train \
  --images /path/to/coco/images \
  --annotations /path/to/coco/annotations.json \
  --epochs 10 \
  --batch-size 4 \
  --output /path/to/checkpoints/mouse_detector.pt
```

## Inference on a folder of videos

```bash
python -m mouse_detector.infer_videos \
  --weights /path/to/checkpoints/mouse_detector.pt \
  --input-dir /path/to/input_videos \
  --output-dir /path/to/output_videos \
  --centers-dir /path/to/center_cache \
  --size 640x360 \
  --codec mp4v \
  --threshold 0.5
```

Output videos are written to `--output-dir` with the same filenames and the exact requested output size. Each frame is cropped around the center of the selected detection, and any area beyond the source frame is filled with black pixels.
Center coordinates are saved as `*_centers.npy` in `--centers-dir` (or in `--output-dir` if omitted).

To redraw videos at a different crop size without running inference again:

```bash
python -m mouse_detector.infer_videos \
  --input-dir /path/to/input_videos \
  --output-dir /path/to/redrawn_videos \
  --centers-dir /path/to/center_cache \
  --size 512x512 \
  --codec mp4v \
  --reuse-centers
```

## Validation on annotated images

```bash
python -m mouse_detector.validate_annotations \
  --weights /path/to/checkpoints/mouse_detector.pt \
  --images /path/to/coco/images \
  --annotations /path/to/coco/annotations.json \
  --output-dir /path/to/output_images \
  --threshold 0.5
```

Output images are written to `--output-dir` with the same relative filenames as the annotated COCO images and include blue labeled boxes plus the green inferred box.

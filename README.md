# mouse_detector

PyTorch-based mouse detector for COCO datasets with:
- training on COCO images + bounding boxes
- inference across a folder of videos
- output resized videos with mouse boxes drawn

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
  --size 640x360 \
  --codec mp4v \
  --threshold 0.5
```

Output videos are written to `--output-dir` with the same filenames and the exact requested output size.

# DetectKit

A computer-vision toolkit for object detection and image segmentation. DetectKit
provides training and inference pipelines for modern detectors — from anchor-free
one-stage models to transformer-based detection heads — with a focus on real-time
performance.

## Features

- **Object detection** — train and run one-stage and two-stage detectors on COCO
  and custom datasets, with bounding-box regression and non-maximum suppression.
- **Instance & semantic segmentation** — pixel-level masks via a shared backbone
  and lightweight segmentation heads.
- **Vision transformers** — optional ViT and Swin backbones for image
  classification and detection.
- **Data augmentation** — mosaic, mixup, random affine, and color-jitter pipelines
  built on OpenCV and PIL.
- **Export** — ONNX and TensorRT export for edge deployment.

## Quick start

```python
from detectkit import Detector

model = Detector.pretrained("detectkit-l")
results = model.predict("street.jpg")
results.show()  # draws bounding boxes and class labels
```

DetectKit targets computer-vision research on detection, segmentation, and image
classification.

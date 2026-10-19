# Photo Selection Assistant

> **Status:** In Progress (Active Development)
>
> **Academic Context:** Bachelor of Science (BSc) Thesis – _Photo selection assistant based on sharpness detection and composition analysis_
> **Institution:** Warsaw University of Technology | Computer Science (Artificial Intelligence)

## About The Project

The **Photo Selection Assistant** helps photographers cull their photos by automatically detecting blurred images and assessing image composition.

This repository contains the research part of the thesis: datasets preparation, model training, experiments and evaluation. Trained models are exported to ONNX and used by a separate desktop application (C# / .NET, Avalonia).

## Key Features

- **Blur Detection:** EfficientNetV2-S classifier distinguishing sharp photos (including intentional bokeh) from blurred ones, compared against classical Laplacian variance and YOLO-based ROI approaches.
- **Composition Assessment:** (In Progress) Model assessing the composition quality of a photo.

## Project Structure

```
common/              shared helpers: paths, seeding, plotting, YOLO bounding boxes
sharpness/           blur detection
├── config.py        current experiment settings and paths
├── data.py          dataset assembly and the train/val split
├── model.py         model factories and transforms
├── train.py         training
├── evaluation/      confusion matrix, thresholds, error analysis, Grad-CAM, methods comparison
├── classic/         Laplacian variance experiments
└── tools/           CUHK labeling, ROI dataset preparation, ONNX export
results/             metrics and plots (tracked in git)
artifacts/           checkpoints, ONNX models, pretrained weights (not tracked)
datasets/            datasets (not tracked)
```

## Getting Started

```bash
uv sync
```

Expected local data (not included in the repository):

- `datasets/` – CERTH Image Blur Dataset, CUHK Blur Detection Dataset, Kaggle Blur Dataset, EBB! (Everything is Better with Bokeh) dataset.
- `artifacts/pretrained/yolov8n.pt` – YOLOv8n weights.

All scripts are run as modules from the repository root, e.g.:

```bash
uv run python -m sharpness.train
uv run python -m sharpness.evaluation.plot_confusion_matrix
uv run python -m sharpness.tools.export_to_onnx
```

## Results

Blur detection, EfficientNetV2-S (384px), validation split:

| Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|
| 0.971 | 0.975 | 0.950 | 0.962 |

Plots and per-experiment metrics are in `results/sharpness/`.

## Tech Stack

- **Machine Learning & Vision:** Python, PyTorch, torchvision, OpenCV, Ultralytics YOLO, scikit-learn
- **Deployment:** ONNX

## Author

**Miłosz Andryszczuk**
_Python & AI Engineering Student_
[LinkedIn](https://linkedin.com/in/milosz-andryszczuk) | [GitHub](https://github.com/andrymil)

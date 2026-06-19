# Robotic Trash Boat — YOLOv8 Object Detection

A computer vision system for a robotic trash-collecting boat that detects floating waste (cans, plastic, bottles) in water using **YOLOv8** real-time object detection.

## Overview

This project uses a **YOLOv8 Nano** model fine-tuned on two Roboflow datasets to detect 3 classes of floating debris:

| Class ID | Name | Source Dataset |
|----------|------|----------------|
| 0 | Cans | [cans-fdboa](https://universe.roboflow.com/dataset-t7hz7/cans-fdboa/dataset/3) |
| 1 | Plastic | Combined dataset |
| 2 | Bottle | [bottle-f2u4m](https://universe.roboflow.com/sdp2/bottle-f2u4m/dataset/1) |

## Model Performance (train_fixed_labels — 10 epochs)

| Metric | Value |
|--------|-------|
| mAP50 | **97.3%** |
| mAP50-95 | **74.4%** |
| Precision | **95.5%** |
| Recall | **96.7%** |

## Setup

### Prerequisites
- Python 3.10+
- Webcam (for real-time detection)

### Installation

```bash
# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start (Interactive Menu)
```bash
python main.py
```
This opens a menu with options to:
1. **Start real-time detection** — webcam feed with YOLO inference
2. **Camera settings test** — verify webcam focus/exposure
3. **Print model info** — display loaded model classes

### Direct Detection
```bash
python camera_detection.py
```
Press **'q'** to quit the detection window.

### Google Colab (GPU Training)
See [README_COLAB.md](README_COLAB.md) for instructions on training the model with GPU acceleration in Google Colab.

## Project Structure

```
├── camera_detection.py          # Real-time webcam detection with filters
├── camera_detection_1.py        # Webcam focus/exposure test utility
├── model_detection.py           # Print model class info
├── main.py                      # Interactive CLI launcher
├── requirements.txt             # Python dependencies
├── data_colab.yaml              # Dataset config for Colab training
├── yolov8n.pt                   # Base YOLOv8 Nano weights
│
├── can_dataset/                 # Cans dataset (Roboflow)
│   ├── train/
│   └── valid/
│
├── plastic_bottle_dataset/      # Bottle dataset (Roboflow)
│   ├── train/
│   ├── valid/
│   └── test/
│
├── runs/detect/
│   ├── train/                   # Training run 1 (from base)
│   └── train2/                  # Training run 2 (fine-tuned, best)
│       └── weights/
│           └── best.pt          # ← Active model weights
│
└── idt_to_yolo_autolabel_pipeline.ipynb  # Auto-labeling pipeline
```

## Detection Features

- **Confidence threshold**: 55% — filters out weak/uncertain detections
- **Size filter**: Rejects boxes covering >30% of the frame (e.g., faces misdetected as cans)
- **Aspect ratio filter**: Rejects near-square detections (cans/bottles are elongated)
- **Frame skipping**: Processes every 3rd frame for smooth performance on CPU
- **Per-class colors**: Cans (orange), Plastic (blue), Bottles (green)

## License

Datasets are provided under **CC BY 4.0** license via Roboflow.

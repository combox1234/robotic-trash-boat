# most recent vrs - Colab Version

This folder is a Colab-ready copy of `new_vr`.

## 1) Open Google Colab and enable GPU
- Runtime -> Change runtime type -> Hardware accelerator -> GPU

## 2) Install dependencies
```python
!pip install -q ultralytics opencv-python
```

## 3) Move/copy this folder into Colab workspace
Use one of the options below.

Option A: Upload `most recent vrs` zip and extract to `/content/most_recent_vrs`
```python
!mkdir -p /content/most_recent_vrs
# After uploading the zip file (for example: most_recent_vrs.zip), run:
#!unzip -q /content/most_recent_vrs.zip -d /content/most_recent_vrs
```

Option B: Use Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
# Example copy command (edit source path to your Drive location):
#!cp -r "/content/drive/MyDrive/most recent vrs" /content/most_recent_vrs
```

## 4) Train model in Colab
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='/content/most_recent_vrs/data_colab.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    project='/content/most_recent_vrs/runs',
    name='detect_colab'
)
```

## 5) Validate
```python
from ultralytics import YOLO

best = YOLO('/content/most_recent_vrs/runs/detect_colab/weights/best.pt')
metrics = best.val(data='/content/most_recent_vrs/data_colab.yaml')
print(metrics)
```

## 6) Inference on image
```python
from ultralytics import YOLO

model = YOLO('/content/most_recent_vrs/runs/detect_colab/weights/best.pt')
res = model.predict('/content/most_recent_vrs/pb.jpeg', conf=0.25, save=True)
print('Prediction saved in:', res[0].save_dir)
```

## Notes
- Webcam scripts in this project are for local PC and are not suitable for standard Colab runtime webcam capture.
- Original folder `new_vr` is untouched.

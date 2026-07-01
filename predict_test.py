from ultralytics import YOLO
import os

model = YOLO('runs/detect/train_fixed_labels/weights/best.pt')

# Pick an image from can dataset and one from plastic bottle dataset
can_img = 'can_dataset/valid/images/beverage_cans-105_jpg.rf.7da30bc79786e7d056bef27af8a16383.jpg'
bottle_img = 'plastic_bottle_dataset/valid/images/091_jpg.rf.9a7c490e124a31ea51dbd386aa99b24b.jpg'

for img in [can_img, bottle_img]:
    print(f"\n--- Predicting on {img} ---")
    results = model(img, verbose=False)
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"Detected: {model.names[cls]} with confidence {conf:.2f}")

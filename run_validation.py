from ultralytics import YOLO

# Load the best trained model
model = YOLO('runs/detect/train2/weights/best.pt')

# Run validation on the local dataset
print("Starting validation...")
metrics = model.val(data='can_dataset/data.yaml', imgsz=640)

print("\n--- Validation Results ---")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP75: {metrics.box.map75:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")

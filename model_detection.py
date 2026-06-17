from ultralytics import YOLO

# Load your model
model = YOLO("runs/detect/train/weights/best.pt")

# Print the model classes
print(model.names)

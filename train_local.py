from ultralytics import YOLO

# Start from the pre-trained weights instead of scratch for faster fine-tuning
model = YOLO('runs/detect/train2/weights/best.pt')

print("Starting training with corrected labels...")
# Train for 10 epochs. (Usually 50+ is needed for high accuracy, but this gets us a working baseline)
results = model.train(
    data='can_dataset/data.yaml', 
    epochs=10, 
    imgsz=320,  # Smaller image size for faster CPU training
    batch=8, 
    name='train_fixed_labels',
    workers=0  # Prevents multiprocessing errors on Windows
)

print("Training complete! The new model is saved in runs/detect/train_fixed_labels/")

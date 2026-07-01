# 🧠 brain.md — Exhaustive Technical Brain Dump

## Robotic Trash Boat — YOLOv8 Object Detection Platform

> **Generated**: 2026-06-26  
> **Scope**: Every single file in the repository, line-by-line, with full mechanical detail.  
> **Project purpose**: A computer vision system for a robotic trash-collecting boat. It detects floating waste (cans, plastic, bottles) in water bodies using a fine-tuned YOLOv8 Nano model, running real-time inference through a webcam feed.

---

# Table of Contents

1. [main.py](#mainpy)
2. [camera_detection.py](#camera_detectionpy)
3. [camera_detection_1.py](#camera_detection_1py)
4. [model_detection.py](#model_detectionpy)
5. [train_local.py](#train_localpy)
6. [fix_labels.py](#fix_labelspy)
7. [predict_test.py](#predict_testpy)
8. [run_validation.py](#run_validationpy)
9. [requirements.txt](#requirementstxt)
10. [.gitignore](#gitignore)
11. [data_colab.yaml](#data_colabyaml)
12. [can_dataset/data.yaml](#can_datasetdatayaml)
13. [plastic_bottle_dataset/data.yaml](#plastic_bottle_datasetdatayaml)
14. [can_dataset/README.dataset.txt](#can_datasetreaddatasettxt)
15. [can_dataset/README.roboflow.txt](#can_datasetreadroboflow)
16. [README.md](#readmemd)
17. [README_COLAB.md](#readme_colabmd)
18. [CONTRIBUTING.md](#contributingmd)
19. [yolov8n.pt](#yolov8npt)
20. [pb.jpeg & pb_1.jpeg](#pbjpeg--pb_1jpeg)
21. [idt_to_yolo_autolabel_pipeline.ipynb](#idt_to_yolo_autolabel_pipelineipynb)
22. [can_dataset/ (directory structure)](#can_dataset-directory)
23. [plastic_bottle_dataset/ (directory structure)](#plastic_bottle_dataset-directory)
24. [runs/detect/ (training output directories)](#runsdetect-training-output-directories)
25. [runs/detect/train/args.yaml](#runsdetecttrainargs)
26. [runs/detect/train2/args.yaml](#runsdetecttrain2args)
27. [runs/detect/train_fixed_labels/args.yaml](#runsdetecttrain_fixed_labelsargs)
28. [runs/detect/train/results.csv](#runsdetecttrainresultscsv)
29. [runs/detect/train2/results.csv](#runsdetecttrain2resultscsv)
30. [runs/detect/train_fixed_labels/results.csv](#runsdetecttrain_fixed_labelsresultscsv)
31. [useless/ (directory)](#useless-directory)
32. [docs/ (directory)](#docs-directory)
33. [__pycache__/ (directory)](#__pycache__-directory)
34. [.venv/ (directory)](#venv-directory)
35. [.git/ (directory)](#git-directory)
36. [Global: System Architecture Overview](#-system-architecture-overview)
37. [Global: Request Lifecycle](#-request-lifecycle)
38. [Global: Data Layer Deep Dive](#-data-layer-deep-dive)
39. [Global: Framework Usage](#-framework-usage)
40. [Global: Design Patterns Used](#-design-patterns-used)
41. [Global: Problems Faced & How They Were Solved](#-problems-faced--how-they-were-solved)
42. [Global: Full Dependency Map](#-full-dependency-map)
43. [Global: Inter-File Dependency Graph](#-inter-file-dependency-graph)

---

# `main.py`

**Full path**: `most recent vrs/main.py`  
**Size**: 1,193 bytes, 36 lines  
**Language**: Python 3

---

## 1. File Purpose

`main.py` is the **primary entry point** and **interactive CLI launcher** for the entire application. It presents a numbered text menu to the user and dispatches execution to one of three child scripts depending on the user's selection. It acts as a thin orchestrator — it contains zero ML logic, zero CV logic, and zero data manipulation. Its sole job is user input → subprocess launch.

**What would break if removed**: The user would lose the single-command entry point (`python main.py`) and would need to know which specific script to run manually. No ML functionality would be lost.

---

## 2. Code Walkthrough (Line-by-Line)

### Lines 1–2: Imports

```python
import sys
import subprocess
```

- **`sys`** (Python stdlib): Provides access to `sys.executable` (the path to the currently-running Python interpreter) and `sys.exit()` for terminating the process with a specific exit code.
- **`subprocess`** (Python stdlib): Provides `subprocess.run()` for spawning child processes. Used here to launch the detection scripts as separate OS processes rather than importing and calling them directly. This design isolates the child script's global-scope side effects (e.g., `camera_detection.py` opens a webcam and enters an infinite loop at module level).

### Lines 4–13: `main()` function — Menu printing

```python
def main():
    print("====================================================")
    print("   Robotic Trash Boat - YOLOv8 Detection Platform   ")
    print("====================================================")
    print("1. Start Real-time Object Detection (camera_detection.py)")
    print("2. Run Camera Settings / Focus Test (camera_detection_1.py)")
    print("3. Exit")
    print("====================================================")
```

Five `print()` calls render an ASCII-bordered menu to stdout. Each option maps to a distinct script file. Option 3 is a clean exit path.

### Line 14: User input

```python
    choice = input("Select an option (1-3): ").strip()
```

`input()` blocks the process until the user presses Enter. `.strip()` removes leading/trailing whitespace from the entered string, preventing issues like `"1 "` not matching `'1'`.

### Lines 16–20: Script mapping dictionary

```python
    scripts = {
        '1': 'camera_detection.py',
        '2': 'camera_detection_1.py',
    }
```

A dictionary mapping user choice strings to script filenames. Uses string keys (not integers) because `input()` returns a string. Option `'3'` is deliberately omitted — it is handled separately as an exit condition.

### Lines 22–24: Exit path

```python
    if choice == '3':
        print("\nExiting. Have a great day!")
        sys.exit(0)
```

If the user typed `'3'`, prints a farewell message and terminates the process with exit code 0 (success). `sys.exit(0)` raises a `SystemExit` exception, which Python's runtime catches and uses to terminate cleanly.

### Lines 25–30: Script dispatch

```python
    elif choice in scripts:
        script = scripts[choice]
        print(f"\nLaunching {script}...")
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"\n{script} exited with code {result.returncode}")
```

- `choice in scripts` checks if the user's input is a valid key in the `scripts` dict.
- `sys.executable` resolves to the absolute path of the Python interpreter that is running `main.py` itself (e.g., `c:\...\\.venv\Scripts\python.exe`). This ensures the child script runs under the same virtual environment.
- `subprocess.run([sys.executable, script])` spawns a new Python process. It **blocks** until that child process terminates. The child's stdout/stderr are inherited (i.e., they print directly to the same terminal).
- After the child exits, `result.returncode` is checked. A non-zero return code indicates the child crashed or exited with an error; the code prints a diagnostic message.

### Lines 31–33: Invalid input

```python
    else:
        print("\nInvalid selection. Exiting.")
        sys.exit(1)
```

Any input that is not `'1'`, `'2'`, `'3'`, or `'4'` falls through to this branch. It prints an error and exits with code 1 (failure).

### Lines 35–36: Module guard

```python
if __name__ == "__main__":
    main()
```

Standard Python idiom. `main()` is only called when this file is executed directly (`python main.py`), not when it is imported as a module by another file. In this project, no other file imports `main.py`.

---

## 3. Functions

### `main()`
- **Name**: `main`
- **Purpose**: Display an interactive text menu and dispatch to the selected child script.
- **Parameters**: None
- **Return value**: `None` (implicitly). The function terminates via `sys.exit()` for options 4 and invalid input, or falls through after `subprocess.run()` completes for options 1-3.
- **Side effects**:
  - Writes to stdout (6 menu lines + status messages)
  - Reads from stdin (`input()`)
  - Spawns a child process via `subprocess.run()`
  - Calls `sys.exit()` for exit/invalid paths
- **Error handling**: Only checks `result.returncode != 0` from the child process; no try/except. If `subprocess.run()` itself raises an exception (e.g., file not found), the program crashes with an unhandled `FileNotFoundError`.
- **Called by**: The `if __name__ == "__main__"` guard at module level.
- **Calls**: `print()`, `input()`, `sys.exit()`, `subprocess.run()`

---

## 4. Libraries & Dependencies

| Library | Specific Import | Why Used | What It Does | Alternatives | Gotchas |
|---------|----------------|----------|--------------|-------------|---------|
| `sys` (stdlib) | Module-level | Get Python interpreter path, exit process | `sys.executable` returns the path to the running Python binary; `sys.exit()` terminates | Could use `os._exit()` but that skips cleanup; could use `exit()` builtin but it's meant for interactive use | `sys.exit()` raises `SystemExit`, so it can be caught by a `try/except Exception` if not careful |
| `subprocess` (stdlib) | Module-level | Launch child scripts as separate processes | `subprocess.run()` creates a child process, waits for it, returns a `CompletedProcess` object | Could use `os.system()` (less safe, no return code object), `exec()` (runs in same process), or `importlib` (shares process) | The child process inherits the parent's CWD, which means the relative path `'camera_detection.py'` must exist relative to where `python main.py` was invoked |

---

## 5. Data Structures

| Structure | Type | Contents | Purpose |
|-----------|------|----------|---------|
| `scripts` | `dict[str, str]` | `{'1': 'camera_detection.py', '2': 'camera_detection_1.py', '3': 'model_detection.py'}` | Maps menu option strings to script filenames |
| `choice` | `str` | User-entered string, stripped of whitespace | Holds the raw user input to be matched against `scripts` keys or `'4'` |
| `result` | `subprocess.CompletedProcess` | Contains `.returncode`, `.args`, `.stdout`, `.stderr` | Holds the outcome of the child process execution |

---

## 6. Algorithms & Logic Patterns

- **Pattern**: Simple dispatch table (dictionary lookup). O(1) average-case lookup in the `scripts` dict.
- **Flow**: Sequential — read input → check exit → check valid option → else invalid. No loops, no recursion, no retry logic. The menu is presented exactly once per execution.

---

## 7. Problems This File Solves

- Provides a **single unified entry point** so users don't need to memorize individual script names.
- Ensures the correct Python interpreter (from the virtual environment) is used for child scripts.
- Provides basic error reporting via return code checking.

---

## 8. Known / Likely Problems & Edge Cases

| Problem | Impact | Severity |
|---------|--------|----------|
| **No input loop**: If the user enters an invalid option, the program exits instead of re-prompting. | Poor UX; user must re-run `python main.py` | Low |
| **No CWD validation**: The script assumes `camera_detection.py` etc. exist in the current working directory. If `main.py` is invoked from a different directory, `subprocess.run` will fail with `FileNotFoundError`. | Crash on wrong CWD | Medium |
| **No exception handling**: No try/except around `subprocess.run()`. If a script file doesn't exist, the traceback is raw Python. | Confusing error for non-technical users | Low |
| **Blocking subprocess**: The main process is blocked while the child runs. For `camera_detection.py`, which runs an infinite loop, this is intentional, but it means the menu is unreachable until the child exits. | Intentional behavior, but no way to return to menu | Low |
| **No Windows path handling**: Uses bare filenames, which works only if CWD is correct. No `os.path.join` or `pathlib.Path` usage. | Fragile | Low |

---
---

# `camera_detection.py`

**Full path**: `most recent vrs/camera_detection.py`  
**Size**: 3,820 bytes, 96 lines  
**Language**: Python 3

---

## 1. File Purpose

This is the **primary real-time object detection script** — the core deliverable of the entire project. It:
1. Loads a fine-tuned YOLOv8 model from disk
2. Opens the default webcam
3. Enters an infinite loop reading frames
4. Runs YOLO inference on every Nth frame (frame skipping for performance)
5. Applies post-processing filters (bounding box size, aspect ratio) to reject false positives
6. Draws annotated bounding boxes with per-class colors
7. Displays the annotated frame in an OpenCV window

**What would break if removed**: The entire real-time detection capability of the system. This is the file that makes the robotic trash boat "see."

---

## 2. Code Walkthrough (Line-by-Line)

### Line 1–2: Imports

```python
import cv2
from ultralytics import YOLO
```

- **`cv2`**: OpenCV's Python binding. Used for webcam capture (`VideoCapture`), image drawing (`rectangle`, `putText`), and window display (`imshow`).
- **`YOLO`**: The high-level model class from the Ultralytics YOLOv8 library. Wraps model loading, inference, NMS, and result parsing into a single callable object.

### Lines 4–12: Configuration Constants

```python
MODEL_PATH = "runs/detect/train_fixed_labels/weights/best.pt"
CONF_THRESHOLD = 0.55
INFER_SIZE = 320
PROCESS_EVERY_N = 3
MAX_BOX_AREA_RATIO = 0.30
MIN_ASPECT_RATIO = 1.3
CAM_WIDTH = 640
CAM_HEIGHT = 480
```

| Constant | Value | Purpose |
|----------|-------|---------|
| `MODEL_PATH` | `"runs/detect/train_fixed_labels/weights/best.pt"` | Relative path to the best fine-tuned model weights. Points to the output of the final training run. |
| `CONF_THRESHOLD` | `0.55` | Minimum confidence score for YOLO to report a detection. Detections below 55% confidence are silently discarded by the model before they even reach the post-processing filters. Higher than the typical 0.25 default — tuned to reduce false positives in real-world conditions. |
| `INFER_SIZE` | `320` | The image is resized to 320×320 pixels before being fed to the neural network. Smaller than the default 640 — deliberately chosen to maximize inference speed on CPU at the expense of some detection accuracy for small/distant objects. |
| `PROCESS_EVERY_N` | `3` | YOLO inference runs only on every 3rd frame. Frames in between reuse the previous annotated frame. This is a performance optimization for CPU-bound systems — reduces YOLO calls by 66%. |
| `MAX_BOX_AREA_RATIO` | `0.30` | Post-inference filter: any bounding box covering more than 30% of the total frame area is discarded. This rejects false positives where the model detects a large region (e.g., a face, a wall) as a "can." |
| `MIN_ASPECT_RATIO` | `1.3` | Post-inference filter: the ratio of the longer side to the shorter side of the bounding box must be at least 1.3. Cans and bottles are elongated objects; near-square detections are suspicious and likely false positives. |
| `CAM_WIDTH` / `CAM_HEIGHT` | `640` / `480` | Requested webcam resolution. These are hints to the webcam driver — the actual resolution depends on hardware support. 640×480 (VGA) is universally supported. |

### Lines 14–20: Color Configuration

```python
CLASS_COLORS = {
    0: (0, 200, 255),   # Cans   → orange
    1: (255, 100, 100),  # Plastic → blue
    2: (100, 255, 100),  # Bottle  → green
}
DEFAULT_COLOR = (0, 255, 0)
```

- `CLASS_COLORS` maps class IDs to BGR color tuples (OpenCV uses BGR, not RGB).
  - Class 0 (Cans): `(0, 200, 255)` → orange in BGR (B=0, G=200, R=255)
  - Class 1 (Plastic): `(255, 100, 100)` → blue-ish in BGR
  - Class 2 (Bottle): `(100, 255, 100)` → green in BGR
- `DEFAULT_COLOR` is a fallback green used if a class ID is not in the dictionary (shouldn't happen with the current 3-class model, but defensive).

### Lines 22–25: Model Loading (global scope)

```python
print(f"Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print(f"Classes: {model.names}")
```

- `YOLO(MODEL_PATH)` loads the PyTorch model weights from disk. Under the hood, this:
  1. Reads the `.pt` file (a PyTorch checkpoint containing model architecture + weights + training metadata)
  2. Reconstructs the YOLOv8 Nano architecture
  3. Loads the trained weights into the model
  4. Moves the model to the available device (CPU in this case, as no GPU is specified)
- `model.names` is a dict like `{0: 'Cans', 1: 'Plastic', 2: 'Bottle'}` read from the checkpoint metadata.
- This code runs at **module load time** (global scope), meaning the model is loaded as soon as the file is executed or imported. This is why `main.py` uses `subprocess.run()` instead of `import`.

### Lines 27–36: Webcam Initialization (global scope)

```python
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened. Press 'q' to quit.")
```

- `cv2.VideoCapture(0)`: Opens the default webcam (device index 0). On Windows, this uses the DirectShow or Media Foundation backend. Returns a `VideoCapture` object.
- `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)`: Requests the webcam driver to set the horizontal resolution to 640 pixels. This is a *request* — the driver may ignore it.
- `cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)`: Same for vertical resolution.
- `cap.isOpened()`: Returns `True` if the `VideoCapture` object successfully connected to a camera device. Returns `False` if no camera is available, the device is in use, or drivers are missing.
- `exit()`: Builtin function that raises `SystemExit`. Terminates the process if no webcam is available.

### Lines 38–39: Frame Counter Initialization

```python
frame_count = 0
annotated_frame = None
```

- `frame_count`: Integer counter tracking total frames read from the webcam. Used for the frame-skipping logic (`frame_count % PROCESS_EVERY_N`).
- `annotated_frame`: Holds the most recently annotated frame (with bounding boxes drawn). Initialized to `None` because no frames have been processed yet. This variable is reused between inference frames to avoid a blank screen during skipped frames.

### Lines 41–90: Main Detection Loop

```python
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break
```

- `cap.read()` grabs and decodes one frame from the webcam. Returns a tuple `(success_flag, frame_array)`.
  - `ret` is `True` if a frame was successfully captured, `False` otherwise (camera disconnected, end of video file, etc.).
  - `frame` is a NumPy array of shape `(480, 640, 3)` with dtype `uint8`, in BGR color order.
- If `ret` is `False`, the loop breaks and the program proceeds to cleanup.

```python
    frame_count += 1

    if frame_count % PROCESS_EVERY_N == 0 or annotated_frame is None:
```

- `frame_count` increments every iteration.
- The condition `frame_count % PROCESS_EVERY_N == 0` is `True` every 3rd frame (frames 3, 6, 9, ...).
- The `or annotated_frame is None` clause ensures the very first frame is always processed (otherwise the display would show nothing until frame 3).

```python
        results = model(frame, conf=CONF_THRESHOLD, imgsz=INFER_SIZE, verbose=False)
```

- `model(frame, ...)` calls the YOLO model's `__call__` method, which internally:
  1. **Preprocessing**: Resizes the input frame from 640×480 to 320×320, normalizes pixel values to [0, 1], converts BGR→RGB, converts to a PyTorch tensor.
  2. **Inference**: Forward-passes the tensor through the YOLOv8 Nano neural network (a modified CSPDarknet backbone + PANet neck + decoupled detection head).
  3. **Postprocessing**: Applies Non-Maximum Suppression (NMS) with IoU threshold 0.7 (default), filters detections below `conf=0.55`.
  4. **Returns**: A list of `ultralytics.engine.results.Results` objects (one per input image; here, always length 1).
- `verbose=False` suppresses per-frame inference logs (timing, detection counts) that Ultralytics prints by default.

```python
        annotated_frame = frame.copy()
        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w
```

- `frame.copy()` creates a deep copy of the raw frame. This is necessary because the bounding box drawing functions modify the image in-place; without the copy, the original `frame` would be mutated.
- `frame.shape[:2]` extracts `(height, width)` from the NumPy array's shape tuple `(480, 640, 3)`.
- `frame_area = 480 * 640 = 307,200` — total pixel count, used as the denominator for the box area ratio filter.

```python
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
```

- `results` is a list with one `Results` object (since we passed a single image).
- `result.boxes` is a `ultralytics.engine.results.Boxes` object containing all detected bounding boxes.
- `box.xyxy[0]` is a 1D tensor of shape `(4,)` containing `[x_top_left, y_top_left, x_bottom_right, y_bottom_right]` in pixel coordinates.
- `map(int, ...)` converts each coordinate from a PyTorch tensor element to a Python `int`.
- `box.conf[0]` is a scalar tensor containing the confidence score (0.0 to 1.0).
- `box.cls[0]` is a scalar tensor containing the predicted class index (0, 1, or 2).

```python
                # ── Filter 1: Bounding box size ──
                box_w = x2 - x1
                box_h = y2 - y1
                box_area = box_w * box_h
                if box_area / frame_area > MAX_BOX_AREA_RATIO:
                    continue  # Too big — likely a false positive (e.g. face)
```

- Computes the area of the detection bounding box in pixels.
- If the box covers more than 30% of the frame (`box_area / 307200 > 0.30`), the detection is discarded.
- **Rationale**: The model was trained on close-up images of cans and bottles. In real-world webcam footage, very large detections are almost always false positives (e.g., a person's face, a wall texture, a large reflection).

```python
                # ── Filter 2: Aspect ratio ──
                aspect = max(box_w, box_h) / max(min(box_w, box_h), 1)
                if aspect < MIN_ASPECT_RATIO:
                    continue  # Too square — probably not a can/bottle
```

- Computes the aspect ratio as `longer_side / shorter_side`. This is always ≥ 1.0.
- `max(min(box_w, box_h), 1)` prevents division by zero if either dimension is 0 (edge case: degenerate bounding box).
- If the aspect ratio is less than 1.3, the box is roughly square. Cans and bottles are tall, narrow objects, so a square detection is likely a false positive.

```python
                # ── Draw detection ──
                label = f"{model.names[cls]} {conf:.0%}"
                color = CLASS_COLORS.get(cls, DEFAULT_COLOR)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
```

- `model.names[cls]` looks up the class name string (e.g., `'Cans'`, `'Plastic'`, `'Bottle'`).
- `f"{...} {conf:.0%}"` formats the confidence as a percentage without decimals (e.g., `"Cans 87%"`).
- `CLASS_COLORS.get(cls, DEFAULT_COLOR)` retrieves the class-specific color or falls back to green.
- `cv2.rectangle(...)` draws a colored rectangle on the annotated frame. Arguments: image, top-left corner, bottom-right corner, color (BGR), thickness (2 pixels).
- `cv2.putText(...)` draws the label text above the bounding box. `(x1, y1 - 10)` positions the text 10 pixels above the top-left corner. `cv2.FONT_HERSHEY_SIMPLEX` is a sans-serif font. `0.6` is the font scale factor. `2` is the line thickness.

```python
    cv2.imshow("Robotic Trash Boat - Object Detection", annotated_frame)
```

- Displays the annotated frame in a named OpenCV window. If the window doesn't exist, it is created. If it exists, its contents are updated.
- On skipped frames (when YOLO is not run), `annotated_frame` still holds the previous annotated image, so the display remains responsive.

```python
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

- `cv2.waitKey(1)` waits up to 1 millisecond for a key press. Returns the key code, or -1 if no key was pressed.
- `& 0xFF` masks the key code to 8 bits (necessary on some Linux systems where the key code is 32-bit).
- `ord('q')` is the ASCII code for 'q' (113).
- If 'q' is pressed, the loop breaks.

### Lines 92–95: Cleanup

```python
cap.release()
cv2.destroyAllWindows()
print("Detection stopped.")
```

- `cap.release()` releases the webcam device, freeing the hardware resource.
- `cv2.destroyAllWindows()` closes all OpenCV display windows.
- Prints a final status message.

---

## 3. Functions

This file has **no user-defined functions**. All logic runs at module/global scope in a procedural style. The only "function call" abstraction is the YOLO model's `__call__` method.

**Implication**: The file cannot be imported as a module without triggering the entire webcam capture loop. This is why `main.py` uses `subprocess.run()` to launch it.

---

## 4. Libraries & Dependencies

| Library | Import | Version | Why Chosen | Technical Detail | Alternatives | Gotchas |
|---------|--------|---------|-----------|-----------------|-------------|---------|
| `cv2` (OpenCV) | `import cv2` | Installed via `opencv-python` | Industry-standard for real-time computer vision; extremely fast C++ backend | `VideoCapture` wraps OS-specific camera APIs (DirectShow on Windows); `imshow` uses HighGUI for cross-platform window management | `Pillow` (no webcam support), `imageio` (slower), `PyQt5` (heavier for display) | OpenCV uses BGR color order (not RGB); `imshow` requires `waitKey()` to process window events — without it, the window freezes |
| `ultralytics` (YOLO) | `from ultralytics import YOLO` | Latest (installed via pip) | Official Ultralytics implementation of YOLOv8; includes training, inference, export, and tracking in one package | `YOLO(path)` auto-detects model type from checkpoint; `model(frame)` does preprocess + inference + NMS in one call | `torch` + manual model loading, `detectron2` (heavier), `mmdetection` (more complex API) | `model()` returns a list even for single images; `box.xyxy` is a 2D tensor even for a single box; `verbose=True` (default) prints logs to stderr for every frame |

---

## 5. Data Structures

| Structure | Type | Shape/Schema | Purpose |
|-----------|------|-------------|---------|
| `frame` | `numpy.ndarray` | `(480, 640, 3)`, dtype `uint8` | Raw webcam frame in BGR format |
| `annotated_frame` | `numpy.ndarray` | `(480, 640, 3)`, dtype `uint8` | Copy of frame with bounding boxes drawn on it |
| `results` | `list[Results]` | Length 1 | YOLO inference output containing all detections |
| `result.boxes` | `Boxes` | Variable number of rows | Contains `.xyxy`, `.conf`, `.cls` tensors for all detections |
| `box.xyxy[0]` | `torch.Tensor` | `(4,)` float32 | `[x1, y1, x2, y2]` bounding box coordinates |
| `box.conf[0]` | `torch.Tensor` | scalar float32 | Detection confidence (0 to 1) |
| `box.cls[0]` | `torch.Tensor` | scalar float32 | Predicted class index (0, 1, or 2) |
| `CLASS_COLORS` | `dict[int, tuple[int,int,int]]` | 3 entries | Maps class ID to BGR color |

---

## 6. Algorithms & Logic Patterns

### Frame Skipping (Temporal Subsampling)
- **Algorithm**: Modulo-based frame selection. Process frame if `frame_count % 3 == 0`.
- **Complexity**: O(1) per frame for the decision.
- **Trade-off**: Reduces CPU load by ~66% at the cost of detection latency (detections can be up to ~100ms stale at 30fps).

### Bounding Box Area Filter
- **Algorithm**: Brute-force per-detection check. `box_area / frame_area > threshold`.
- **Complexity**: O(d) where d = number of detections per frame (typically < 10).
- **Rationale**: Eliminates large false positives without additional model complexity.

### Aspect Ratio Filter
- **Algorithm**: `max(w, h) / min(w, h)` — orientation-invariant aspect ratio.
- **Complexity**: O(d).
- **Rationale**: Domain-specific heuristic — cans and bottles are elongated. This filter is cheap and effective.

### Non-Maximum Suppression (NMS)
- **Algorithm**: Performed inside `model()` by Ultralytics, not in this file. Uses greedy NMS with IoU threshold 0.7.
- **Complexity**: O(d² log d) — sorts by confidence, then pairwise IoU computation.

---

## 7. Problems This File Solves

- **Real-time waste detection**: Continuously processes webcam frames to identify cans, plastic, and bottles.
- **False positive reduction**: Two post-inference filters dramatically reduce misdetections.
- **CPU performance**: Frame skipping and reduced inference size (320px) make it feasible to run on a laptop CPU without a GPU.
- **Visual feedback**: Color-coded bounding boxes with confidence labels give immediate visual confirmation.

---

## 8. Known / Likely Problems & Edge Cases

| Problem | Detail | Severity |
|---------|--------|----------|
| **No GPU utilization** | Model runs on CPU. On a laptop, inference takes ~100-200ms per frame. | Medium |
| **Hard-coded model path** | `MODEL_PATH` is a relative path. If CWD changes, model loading fails. | Medium |
| **First frame special case** | `annotated_frame is None` forces processing of frame 1, but `frame_count` starts at 0 and is incremented to 1 before the check, so the first processed frame is frame 1 (1 % 3 = 1 ≠ 0), but the `or annotated_frame is None` clause handles this. | Low (handled) |
| **Camera auto-focus jitter** | Unlike `camera_detection_1.py`, this file doesn't disable auto-focus. The webcam may hunt for focus during detection, causing blur. | Medium |
| **Label text clipping** | `cv2.putText` at `(x1, y1 - 10)` can draw text above the frame boundary if `y1 < 10`. OpenCV clips the text, but it becomes unreadable. | Low |
| **No multi-camera support** | Hard-coded to device index 0. Cannot select a USB camera if multiple are connected. | Low |
| **No recording capability** | Detected frames are displayed but never saved to disk. | Low |
| **Thread safety** | OpenCV `imshow` must be called from the main thread. This code runs entirely single-threaded, so it's fine, but if future refactoring moves inference to a thread, `imshow` will crash. | Low |

---
---

# `camera_detection_1.py`

**Full path**: `most recent vrs/camera_detection_1.py`  
**Size**: 745 bytes, 28 lines  
**Language**: Python 3

---

## 1. File Purpose

This is a **webcam diagnostic/calibration utility**. It opens the webcam with manually configured focus and exposure settings, displays the raw feed without any YOLO inference, and lets the user visually verify that the camera is working and properly focused.

**What would break if removed**: Loss of the camera testing utility. The detection pipeline (`camera_detection.py`) would still work, but users could not easily diagnose webcam issues.

---

## 2. Code Walkthrough (Line-by-Line)

### Line 1: Import

```python
import cv2
```

Only OpenCV is needed — no YOLO, no model loading. This makes the script very lightweight and fast to start.

### Lines 3–9: Webcam Initialization with Manual Settings

```python
cap = cv2.VideoCapture(0)

# Try disabling auto-focus and auto-exposure (might not work for all webcams)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
cap.set(cv2.CAP_PROP_FOCUS, 50)  # Adjust focus (try different values)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Set manual exposure
cap.set(cv2.CAP_PROP_EXPOSURE, -5)  # Adjust exposure (try different values)
```

| Property | Value | Effect |
|----------|-------|--------|
| `CAP_PROP_AUTOFOCUS` | `0` | Disables automatic focus hunting. The lens stays at a fixed focal distance. |
| `CAP_PROP_FOCUS` | `50` | Sets the manual focus distance. The meaning of `50` is driver-dependent — it's typically a value between 0 and 255. Mid-range values focus at ~30-50cm, suitable for tabletop objects. |
| `CAP_PROP_AUTO_EXPOSURE` | `0.25` | Switches from automatic to manual exposure control. The value `0.25` is the OpenCV convention for "manual mode" on many drivers. |
| `CAP_PROP_EXPOSURE` | `-5` | Sets the manual exposure level. Negative values mean shorter exposure times (less light, less blur). The value is on a logarithmic scale (2^(-5) = 1/32 second). |

**Important comment in the code**: *"might not work for all webcams"* — these `cap.set()` calls are hints to the driver. Many cheap webcams ignore manual focus/exposure settings entirely.

### Lines 11–13: Webcam Check

```python
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
```

Identical pattern to `camera_detection.py`.

### Lines 15–24: Display Loop

```python
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not capture frame.")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

A minimal display loop:
- Reads frames continuously
- Displays the **raw** frame (no annotation, no inference)
- Window title is `"Camera Test"` (distinct from the detection window)
- Exits on 'q' keypress

### Lines 26–27: Cleanup

```python
cap.release()
cv2.destroyAllWindows()
```

Standard resource cleanup.

---

## 3. Functions

**No functions defined.** Entirely procedural, global-scope code.

---

## 4. Libraries & Dependencies

| Library | Import | Why | Alternative |
|---------|--------|-----|-------------|
| `cv2` (OpenCV) | `import cv2` | Webcam access and display | None suitable for this purpose |

---

## 5. Data Structures

| Structure | Type | Purpose |
|-----------|------|---------|
| `cap` | `cv2.VideoCapture` | Webcam handle |
| `frame` | `numpy.ndarray (480, 640, 3)` | Raw captured frame |
| `ret` | `bool` | Frame capture success flag |

---

## 6. Algorithms & Logic Patterns

None. This is a pass-through display script — frames are read and shown without transformation.

---

## 7. Problems This File Solves

- Allows the user to verify the webcam is functioning before running detection.
- Provides a way to visually assess focus and exposure settings.
- Serves as an isolated debugging tool — if `camera_detection.py` fails, this script can determine whether the issue is camera-related or model-related.

---

## 8. Known / Likely Problems & Edge Cases

| Problem | Detail | Severity |
|---------|--------|----------|
| **Driver-specific behavior** | The `cap.set()` calls for focus and exposure have no effect on many webcams, especially built-in laptop cameras. | Medium |
| **Magic numbers** | Focus value `50` and exposure value `-5` are hard-coded. Different webcams need different values. Should be configurable (command-line args or config file). | Low |
| **No resolution setting** | Unlike `camera_detection.py`, this script does not set `CAP_PROP_FRAME_WIDTH` / `CAP_PROP_FRAME_HEIGHT`. The webcam uses its default resolution, which may differ from the detection script. | Low |

---
---

# `model_detection.py`

**Full path**: `most recent vrs/model_detection.py`  
**Size**: 164 bytes, 8 lines  
**Language**: Python 3

---

## 1. File Purpose

A **minimal diagnostic script** that loads the trained YOLO model and prints its class names to stdout. It exists purely for verification — confirming that the model file is valid, loadable, and contains the expected class mapping.

**What would break if removed**: No production functionality lost. Diagnostic convenience only.

---

## 2. Code Walkthrough (Line-by-Line)

```python
from ultralytics import YOLO

# Load your model
model = YOLO("runs/detect/train_fixed_labels/weights/best.pt")

# Print the model classes
print(model.names)
```

- Line 1: Imports the `YOLO` class from Ultralytics.
- Line 4: Loads the model from the same path used by `camera_detection.py`. This triggers full model deserialization — reading the checkpoint, reconstructing the architecture, loading weights.
- Line 7: `model.names` is a dictionary attribute of the loaded model. It maps integer class IDs to string class names. Expected output: `{0: 'Cans', 1: 'Plastic', 2: 'Bottle'}`.

---

## 3. Functions

**No functions defined.**

---

## 4. Libraries & Dependencies

| Library | Import | Purpose |
|---------|--------|---------|
| `ultralytics` | `from ultralytics import YOLO` | Load the model and access its metadata |

---

## 5. Data Structures

| Structure | Type | Contents |
|-----------|------|----------|
| `model` | `ultralytics.YOLO` | Full YOLOv8 model object |
| `model.names` | `dict[int, str]` | `{0: 'Cans', 1: 'Plastic', 2: 'Bottle'}` |

---

## 6. Algorithms & Logic Patterns

None. This is a print statement.

---

## 7. Problems This File Solves

- Quick sanity check: "Does the model file exist and is it a valid YOLO checkpoint?"
- Verification that the class mapping in the model matches expectations (3 classes with the correct names).

---

## 8. Known / Likely Problems & Edge Cases

| Problem | Detail |
|---------|--------|
| **Slow for a print script** | Loading a full YOLO model takes 1-3 seconds (reading ~6MB of weights). All that work just to print 3 class names. A lighter approach would be to read only the checkpoint metadata. |
| **Hard-coded path** | Same fragility as `camera_detection.py`. |

---
---

# `gui.py`

**Full path**: `most recent vrs/gui.py`  
**Language**: Python 3

---

## 1. File Purpose

A graphical launcher for the Robotic Trash Boat detection platform, built using standard `tkinter`. It provides a user-friendly UI to start various detection modules as separate background processes without freezing the UI.

---

## 2. Code Walkthrough (Line-by-Line)

- **Imports**: Uses `subprocess` and `sys` to spawn independent processes, `threading` to keep the UI responsive during process execution, and `tkinter` / `ttk` for the UI components.
- **`TrashBoatGUI.__init__`**: Sets up the main window (500x400, custom background color, `clam` ttk theme for modern looks) and places the launch buttons.
- **`run_script(self, script_name)`**: The core launcher function. Uses a background thread to call `subprocess.Popen([sys.executable, script_name])`. Handles errors safely using `messagebox.showerror`.
- **`main()`**: Bootstraps the `tk.Tk()` root and starts the `mainloop`.

---

## 3. Functions

| Function | Purpose |
|----------|---------|
| `__init__` | Initializes the `tkinter` application and draws the UI. |
| `run_script` | Spawns a background thread that executes the specified Python script via `subprocess`. |
| `main` | Application entry point. |

---

## 4. Libraries & Dependencies

| Library | Import | Purpose |
|---------|--------|---------|
| `tkinter` | `import tkinter as tk` | Core GUI framework. |
| `subprocess` | `import subprocess` | Spawning child scripts. |
| `threading` | `import threading` | Prevents the GUI from locking up while a script runs. |

---
---

# `train_local.py`

**Full path**: `most recent vrs/train_local.py`  
**Size**: 644 bytes, 18 lines  
**Language**: Python 3

---

## 1. File Purpose

This script **fine-tunes (continues training) the YOLOv8 model** using the local can dataset. It starts from a previously trained checkpoint (`train2/weights/best.pt`) and trains for 10 more epochs with corrected labels. This is the script that produced the final `train_fixed_labels` model weights that are used by the detection scripts.

**What would break if removed**: The ability to re-train the model locally. The existing trained weights in `runs/detect/train_fixed_labels/` would still work, but they could not be regenerated.

---

## 2. Code Walkthrough (Line-by-Line)

### Line 1: Import

```python
from ultralytics import YOLO
```

### Lines 3–4: Model Loading

```python
# Start from the pre-trained weights instead of scratch for faster fine-tuning
model = YOLO('runs/detect/train2/weights/best.pt')
```

- Loads the model from `train2` (the second training run). This is **transfer learning** — the model already knows how to detect objects from 50 epochs of prior training. Starting from these weights means the 10 additional epochs can focus on learning the corrected labels.
- The comment explicitly documents the rationale.

### Lines 6–15: Training Invocation

```python
print("Starting training with corrected labels...")
results = model.train(
    data='can_dataset/data.yaml', 
    epochs=10, 
    imgsz=320,
    batch=8, 
    name='train_fixed_labels',
    workers=0
)
```

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `data` | `'can_dataset/data.yaml'` | Path to the dataset configuration file. This YAML defines train/val image paths, class count (3), and class names. |
| `epochs` | `10` | Number of complete passes through the training data. Low by ML standards (typically 50-300), but the model is already pre-trained. |
| `imgsz` | `320` | Input image size for training. Smaller than the default 640 to reduce memory usage and training time on CPU. |
| `batch` | `8` | Mini-batch size. 8 images are processed simultaneously per gradient update. Smaller than the default 16 because CPU training is memory-constrained. |
| `name` | `'train_fixed_labels'` | Output directory name. Results are saved to `runs/detect/train_fixed_labels/`. |
| `workers` | `0` | Number of data-loading worker processes. Set to 0 to disable multiprocessing. **Critical for Windows**: Python multiprocessing on Windows uses `spawn` instead of `fork`, and Ultralytics' data loaders can crash with `workers > 0` due to pickle errors with certain CUDA/OpenCV configurations. |

The `model.train()` method:
1. Reads the YAML to find image directories
2. Creates train and validation data loaders
3. Runs the training loop: forward pass → compute loss → backward pass → optimizer step
4. Saves checkpoints (`best.pt` and `last.pt`) at the end
5. Generates plots (loss curves, PR curves, confusion matrix)
6. Returns a results object with final metrics

### Lines 17: Completion Message

```python
print("Training complete! The new model is saved in runs/detect/train_fixed_labels/")
```

---

## 3. Functions

**No user-defined functions.** The `model.train()` method is provided by Ultralytics.

---

## 4. Libraries & Dependencies

| Library | Import | Purpose | Critical Detail |
|---------|--------|---------|----------------|
| `ultralytics` | `from ultralytics import YOLO` | Model loading and training | `model.train()` internally uses PyTorch's autograd for backpropagation, SGD/Adam optimizer, and custom YOLO loss functions (box loss + cls loss + dfl loss) |

---

## 5. Data Structures

| Structure | Type | Purpose |
|-----------|------|---------|
| `model` | `YOLO` | YOLOv8 model loaded from checkpoint |
| `results` | `ultralytics.utils.metrics.DetMetrics` | Training results containing final mAP, precision, recall |

---

## 6. Algorithms & Logic Patterns

### Transfer Learning / Fine-Tuning
- **Pattern**: Load pre-trained weights → train on new/corrected data for a small number of epochs.
- **Why**: Training from scratch (random weights) requires thousands of epochs and massive datasets. Starting from `train2/best.pt` (which was trained for 50 epochs) means the model already knows general object features. The 10 additional epochs fine-tune it for the corrected label mapping.

### Loss Functions (inside Ultralytics)
- **Box loss** (`box: 7.5`): CIoU (Complete Intersection over Union) loss for bounding box regression.
- **Classification loss** (`cls: 0.5`): Binary Cross-Entropy loss for class prediction.
- **DFL loss** (`dfl: 1.5`): Distribution Focal Loss for refined bounding box regression (YOLOv8-specific).

---

## 7. Problems This File Solves

- **Label correction**: After `fix_labels.py` remapped the plastic bottle labels from class 0 to class 2, this script retrains the model on the corrected data so it learns the proper 3-class mapping.
- **Local training**: Enables training on a personal machine (CPU) without needing Google Colab or a GPU server.

---

## 8. Known / Likely Problems & Edge Cases

| Problem | Detail | Severity |
|---------|--------|----------|
| **CPU training is very slow** | Each epoch took ~50 seconds (see `results.csv`). Total training: ~8 minutes for 10 epochs. This is acceptable only because of the small dataset and small image size. | Medium |
| **10 epochs may not be enough** | The model achieved 97.3% mAP50, which is good, but more epochs could potentially improve mAP50-95 (currently 74.4%). | Low |
| **`workers=0` slows data loading** | With 0 workers, data loading is synchronous in the main process. On a machine with multiple CPU cores, `workers=2` or `4` would improve throughput — but risks Windows-specific multiprocessing bugs. | Low |
| **Small batch size** | `batch=8` with `imgsz=320` means each gradient update sees only 8 small images. This can lead to noisy gradient estimates and slower convergence. | Low |
| **Hard-coded paths** | Both the model path and data YAML path are hard-coded strings. | Medium |

---
---

# `fix_labels.py`

**Full path**: `most recent vrs/fix_labels.py`  
**Size**: 1,067 bytes, 39 lines  
**Language**: Python 3

---

## 1. File Purpose

A **one-time data preprocessing script** that corrects the class labels in the plastic bottle dataset. The original Roboflow export labeled all bottles as class 0 (the only class in its original single-class dataset). But in the combined 3-class system, class 0 is "Cans" and class 2 is "Bottle." This script remaps all class `0` labels to class `2` in every label file across the `plastic_bottle_dataset` train/valid/test splits.

**What would break if removed**: If run before training, nothing (the fix has already been applied). If the labels need to be re-fixed after a fresh dataset download, this script would be needed.

---

## 2. Code Walkthrough (Line-by-Line)

### Lines 1–2: Imports

```python
import os
from pathlib import Path
```

- `os`: Imported but **never used** in the code. This is a dead import — likely left over from an earlier version that used `os.path`.
- `pathlib.Path`: Python's modern file path abstraction. Used here for clean path construction with the `/` operator.

### Lines 4–5: Configuration

```python
base_dir = Path("plastic_bottle_dataset")
```

Creates a `Path` object pointing to the plastic bottle dataset root.

### Line 8: Split List

```python
splits = ["train", "valid", "test"]
```

A list of the three dataset splits to process.

### Line 10: Counter

```python
files_updated = 0
```

Tracks how many label files were actually modified.

### Lines 12–36: Main Processing Loop

```python
for split in splits:
    labels_dir = base_dir / split / "labels"
    if not labels_dir.exists():
        continue
```

- Constructs the path `plastic_bottle_dataset/train/labels` (etc.) using Path's `/` operator.
- Skips non-existent directories gracefully.

```python
    for txt_file in labels_dir.glob("*.txt"):
        with open(txt_file, 'r') as f:
            lines = f.readlines()
```

- `labels_dir.glob("*.txt")` yields every `.txt` file in the labels directory.
- Reads all lines of each label file into a list of strings.

```python
        new_lines = []
        changed = False
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
```

- Iterates through each line of the label file.
- `line.strip().split()` splits the line by whitespace into a list of strings. A YOLO label line looks like: `0 0.5048 0.5324 0.2788 0.6442` — class ID followed by normalized center_x, center_y, width, height.
- Empty lines are skipped.

```python
            if parts[0] == '0':
                parts[0] = '2'
                changed = True
            new_lines.append(" ".join(parts) + "\n")
```

- **The key transformation**: If the class ID (first element) is `'0'`, change it to `'2'`.
- This remaps from the plastic bottle dataset's single-class scheme (where all bottles are class 0) to the combined 3-class scheme (where bottles are class 2).
- The line is reconstructed from the parts list and appended to `new_lines`.

```python
        if changed:
            with open(txt_file, 'w') as f:
                f.writelines(new_lines)
            files_updated += 1
```

- Only writes the file back to disk if at least one line was actually modified. This is an optimization — avoids unnecessary disk I/O for files that don't contain class 0.
- Overwrites the original file in-place.

### Line 38: Summary

```python
print(f"Successfully updated {files_updated} label files in plastic_bottle_dataset.")
```

---

## 3. Functions

**No functions defined.** Entirely procedural.

---

## 4. Libraries & Dependencies

| Library | Import | Used? | Purpose |
|---------|--------|-------|---------|
| `os` | `import os` | **NO** — dead import | Likely left over from earlier version |
| `pathlib.Path` | `from pathlib import Path` | Yes | Path construction with `/` operator; `.glob()` for file discovery; `.exists()` for directory checks |

---

## 5. Data Structures

| Structure | Type | Purpose |
|-----------|------|---------|
| `base_dir` | `Path` | Root of the plastic bottle dataset |
| `splits` | `list[str]` | `["train", "valid", "test"]` |
| `lines` | `list[str]` | All lines from one label file |
| `parts` | `list[str]` | One label line split by whitespace: `['0', '0.5048', '0.5324', '0.2788', '0.6442']` |
| `new_lines` | `list[str]` | Rebuilt lines with corrected class IDs |

---

## 6. Algorithms & Logic Patterns

### String-Based Class Remapping
- **Algorithm**: Linear scan of each line in each file, string comparison and replacement.
- **Complexity**: O(F × L) where F = number of files, L = average lines per file. In practice: ~469 files × ~1-5 lines = ~2000 operations. Runs in under 1 second.
- **Alternative**: Could use `sed` on Linux, but Python is cross-platform.

---

## 7. Problems This File Solves

The **class collision problem**: When merging two Roboflow datasets that were exported independently:
- The cans dataset has class 0 = Cans
- The bottle dataset has class 0 = Bottle (its only class)

Without this fix, the model would train with bottles labeled as cans, leading to a model that cannot distinguish between the two.

---

## 8. Known / Likely Problems & Edge Cases

| Problem | Detail | Severity |
|---------|--------|----------|
| **Non-idempotent** | Running this script twice is safe — on the second run, no files have class 0, so `files_updated` = 0. | N/A (handled) |
| **Dead `os` import** | `import os` is never used. Minor code smell. | Very low |
| **Destructive in-place edit** | Original label files are overwritten. No backup is created. If the remapping logic were wrong, the original data would be lost (but the original ZIP is in `useless/`). | Low |
| **Assumes single-class source** | Only remaps class `0` → `2`. If the bottle dataset had multiple classes, this would incorrectly remap all class-0 objects. | Low (correct for this dataset) |
| **No validation** | Doesn't verify that the bounding box coordinates are valid (0-1 range, width > 0, etc.). | Low |

---
---

# `predict_test.py`

**Full path**: `most recent vrs/predict_test.py`  
**Size**: 690 bytes, 18 lines  
**Language**: Python 3

---

## 1. File Purpose

A **test/validation script** that runs the trained model on two specific validation images — one can image and one bottle image — and prints the predicted class and confidence for each detection. Used to quickly verify that the model can correctly distinguish between cans and bottles after training.

**What would break if removed**: No production impact. This is a developer tool.

---

## 2. Code Walkthrough (Line-by-Line)

### Lines 1–2: Imports

```python
from ultralytics import YOLO
import os
```

- `YOLO` for model loading and inference.
- `os` is imported but **never used** — another dead import.

### Line 4: Model Loading

```python
model = YOLO('runs/detect/train_fixed_labels/weights/best.pt')
```

Loads the same model as `camera_detection.py`.

### Lines 6–8: Test Image Paths

```python
can_img = 'can_dataset/valid/images/beverage_cans-105_jpg.rf.7da30bc79786e7d056bef27af8a16383.jpg'
bottle_img = 'plastic_bottle_dataset/valid/images/091_jpg.rf.9a7c490e124a31ea51dbd386aa99b24b.jpg'
```

Two hard-coded paths to specific validation images. The long filenames are Roboflow's naming convention: `{original_name}.rf.{hash}.jpg`.

### Lines 10–17: Inference Loop

```python
for img in [can_img, bottle_img]:
    print(f"\n--- Predicting on {img} ---")
    results = model(img, verbose=False)
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"Detected: {model.names[cls]} with confidence {conf:.2f}")
```

- Iterates over the two test images.
- `model(img, verbose=False)` runs inference with default settings (conf=0.25, imgsz=640).
- For each detection, prints the class name and confidence score formatted to 2 decimal places.
- Expected output (if model is correct):
  ```
  --- Predicting on can_dataset/valid/images/... ---
  Detected: Cans with confidence 0.92
  
  --- Predicting on plastic_bottle_dataset/valid/images/... ---
  Detected: Bottle with confidence 0.88
  ```

---

## 3. Functions

**No functions defined.**

---

## 4. Libraries & Dependencies

| Library | Import | Used? | Purpose |
|---------|--------|-------|---------|
| `ultralytics` | `from ultralytics import YOLO` | Yes | Model and inference |
| `os` | `import os` | **NO** — dead import | Unused |

---

## 5. Data Structures

Same as `camera_detection.py`: `model`, `results`, `box.cls`, `box.conf`.

---

## 6. Algorithms & Logic Patterns

Straight-line inference — no filtering, no frame skipping. Simplest possible prediction pipeline.

---

## 7. Problems This File Solves

Quick smoke test: "After training, does the model correctly classify a can as `Cans` and a bottle as `Bottle`?"

---

## 8. Known / Likely Problems & Edge Cases

| Problem | Detail |
|---------|--------|
| **Hard-coded image paths** | If the specific images don't exist (deleted, renamed), the script crashes. |
| **No confidence threshold** | Uses the default 0.25 confidence, which is lower than `camera_detection.py`'s 0.55. May show more detections. |
| **Dead `os` import** | Same as `fix_labels.py`. |

---
---

# `run_validation.py`

**Full path**: `most recent vrs/run_validation.py`  
**Size**: 487 bytes, 16 lines  
**Language**: Python 3

---

## 1. File Purpose

Runs **formal YOLO validation** (mAP calculation) on the complete validation set. Unlike `predict_test.py` which tests individual images, this computes aggregate metrics across all validation images: mAP50-95, mAP50, mAP75, Precision, and Recall.

**What would break if removed**: Loss of the formal evaluation pipeline. The model's reported metrics (mAP 97.3%, etc.) were generated by this script or similar Ultralytics validation internally.

---

## 2. Code Walkthrough (Line-by-Line)

```python
from ultralytics import YOLO

# Load the best trained model
model = YOLO('runs/detect/train2/weights/best.pt')
```

**Note**: This loads from `train2`, not `train_fixed_labels`. This means the validation metrics computed by this script are for the **second training run** (before the label fix), not the final model. This is likely a mistake — the script should probably use `train_fixed_labels/weights/best.pt`.

```python
# Run validation on the local dataset
print("Starting validation...")
metrics = model.val(data='can_dataset/data.yaml', imgsz=640)
```

- `model.val()` runs the full YOLO validation pipeline:
  1. Loads all validation images from the paths specified in `data.yaml`
  2. Runs inference on each image
  3. Computes IoU between predicted and ground-truth boxes
  4. Calculates precision, recall, and mAP at multiple IoU thresholds
- `imgsz=640` uses the full default resolution (unlike training, which used 320).

```python
print("\n--- Validation Results ---")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP75: {metrics.box.map75:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
```

| Metric | Attribute | Meaning |
|--------|-----------|---------|
| mAP50-95 | `metrics.box.map` | Mean Average Precision averaged over IoU thresholds 0.50 to 0.95 in steps of 0.05. The strictest and most comprehensive metric. |
| mAP50 | `metrics.box.map50` | Mean Average Precision at IoU threshold 0.50. More lenient — a detection is "correct" if it overlaps the ground truth by at least 50%. |
| mAP75 | `metrics.box.map75` | mAP at IoU 0.75. Stricter than mAP50, requires 75% overlap. |
| Precision | `metrics.box.mp` | Mean precision across classes. Of all detections, what fraction are correct? |
| Recall | `metrics.box.mr` | Mean recall across classes. Of all ground-truth objects, what fraction were detected? |

---

## 3. Functions

**No functions defined.**

---

## 4. Libraries & Dependencies

| Library | Import | Purpose |
|---------|--------|---------|
| `ultralytics` | `from ultralytics import YOLO` | Model loading and validation |

---

## 5. Data Structures

| Structure | Type | Purpose |
|-----------|------|---------|
| `metrics` | `ultralytics.utils.metrics.DetMetrics` | Aggregated validation metrics |
| `metrics.box` | `Metric` object | Contains `.map`, `.map50`, `.map75`, `.mp`, `.mr` |

---

## 6. Algorithms & Logic Patterns

### Mean Average Precision (mAP) Computation (inside Ultralytics)
- **Algorithm**: For each class, sort detections by confidence → compute precision-recall curve → integrate the curve (using the all-point interpolation method) → average across classes → average across IoU thresholds.
- **Complexity**: O(D log D) per class for sorting, O(D × G) for IoU computation (D = detections, G = ground truths).

---

## 7. Problems This File Solves

Formal, quantitative evaluation of model quality — critical for comparing different training runs and deciding if label corrections improved performance.

---

## 8. Known / Likely Problems & Edge Cases

| Problem | Detail | Severity |
|---------|--------|----------|
| **Wrong model loaded** | Uses `train2/best.pt` instead of `train_fixed_labels/best.pt`. The metrics reported may not reflect the final model. | **High** |
| **imgsz mismatch** | Validates at 640px but the model was trained at 320px. This can cause minor metric differences due to different input scales. | Low |

---
---

# `requirements.txt`

**Full path**: `most recent vrs/requirements.txt`  
**Size**: 44 bytes, 5 lines

---

## 1. File Purpose

Declares the project's Python package dependencies. Used by `pip install -r requirements.txt` to install all required libraries into a virtual environment.

---

## 2. Contents (Line-by-Line)

```
ultralytics
opencv-python
torch
torchvision
```

| Package | Purpose | Why Listed |
|---------|---------|------------|
| `ultralytics` | YOLOv8 implementation. Provides the `YOLO` class for training, inference, validation, and export. | Core dependency — used by 6 of the 7 Python scripts. |
| `opencv-python` | Computer vision library providing `cv2`. Used for webcam access and image display. | Used by `camera_detection.py` and `camera_detection_1.py`. |
| `torch` | PyTorch — the deep learning framework underlying YOLOv8. Provides tensor operations, autograd, neural network layers, optimizers. | Transitive dependency of `ultralytics`, but explicitly listed for clarity. |
| `torchvision` | PyTorch's computer vision library. Provides image transforms, pre-trained model utilities, and dataset loaders. | Transitive dependency of `ultralytics`. |

**Notable omissions**:
- No version pinning (e.g., `ultralytics==8.1.0`). This means `pip install` will grab the latest version, which could introduce breaking API changes.
- No `numpy` — it's a transitive dependency of both OpenCV and PyTorch, so it gets installed automatically.
- No `pillow` — also installed transitively.
- The 5th line is blank (trailing newline).

---

## 3. Known / Likely Problems

| Problem | Detail |
|---------|--------|
| **No version pins** | A future `pip install -r requirements.txt` could install incompatible versions. |
| **Redundant entries** | `torch` and `torchvision` are auto-installed by `ultralytics`. Listing them explicitly is harmless but redundant. |
| **No CUDA specification** | `torch` without a specific index URL installs CPU-only by default. For GPU support, the user would need `pip install torch --index-url https://download.pytorch.org/whl/cu121` or similar. |

---
---

# `.gitignore`

**Full path**: `most recent vrs/.gitignore`  
**Size**: 263 bytes, 16 lines

---

## 1. File Purpose

Controls which files and directories Git excludes from version control. Prevents large binary files, temporary artifacts, and environment-specific directories from being committed.

---

## 2. Contents (Line-by-Line)

```gitignore
# Git Ignore
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
tempCodeRunnerFile.*

# Allow training runs and weights but ignore prediction outputs
runs/detect/predict*

# Large dataset archives
*.zip

# Base YOLO weights (downloadable, no need to track)
yolov8n.pt
```

| Pattern | What It Ignores | Why |
|---------|----------------|-----|
| `.venv/` | The Python virtual environment directory | Large (~200MB+), machine-specific, recreatable via `pip install -r requirements.txt` |
| `__pycache__/` | Python bytecode cache directories | Auto-generated, machine-specific |
| `*.pyc` | Individual Python bytecode files | Redundant with `__pycache__/` rule, but covers edge cases |
| `.ipynb_checkpoints/` | Jupyter notebook auto-save checkpoints | Temporary, auto-generated |
| `tempCodeRunnerFile.*` | VS Code "Code Runner" extension temporary files | Editor-specific artifacts |
| `runs/detect/predict*` | YOLO prediction output directories (`predict`, `predict2`, ..., `predict7`) | Contain output images from ad-hoc predictions; large and ephemeral |
| `*.zip` | All ZIP archives | The dataset ZIPs (`can_dataset.zip`, `plastic_bottle_dataset.zip`) are 4.5MB and 13.4MB respectively — too large for Git |
| `yolov8n.pt` | Base YOLOv8 Nano pre-trained weights | 6.5MB binary; freely downloadable from Ultralytics |

**Notable**: Training run outputs (`runs/detect/train*`) are **not** ignored. The trained model weights (`best.pt`, ~6MB each) and training artifacts (plots, CSVs) are tracked in Git.

---

## 3. Known / Likely Problems

| Problem | Detail |
|---------|--------|
| **Training weights are tracked** | `runs/detect/train*/weights/best.pt` files are ~6MB each. Git is not ideal for large binary files; these should arguably use Git LFS. |
| **Dataset images are tracked** | The `can_dataset/` and `plastic_bottle_dataset/` directories contain hundreds of images that are tracked in Git. This inflates the repository size. |

---
---

# `data_colab.yaml`

**Full path**: `most recent vrs/data_colab.yaml`  
**Size**: 379 bytes, 14 lines

---

## 1. File Purpose

YOLO dataset configuration file for **Google Colab** training. It defines the same dataset structure as `can_dataset/data.yaml` but with Colab-specific absolute paths (`/content/most_recent_vrs/...`).

---

## 2. Contents

```yaml
train:
  - /content/most_recent_vrs/can_dataset/train/images
  - /content/most_recent_vrs/plastic_bottle_dataset/train/images

val:
  - /content/most_recent_vrs/can_dataset/valid/images
  - /content/most_recent_vrs/plastic_bottle_dataset/valid/images

test:
  - /content/most_recent_vrs/plastic_bottle_dataset/test/images

nc: 3
names: ['Cans', 'Plastic', 'Bottle']
```

| Field | Type | Value | Purpose |
|-------|------|-------|---------|
| `train` | List of paths | Two directories (can + bottle training images) | Tells YOLO where to find training images. YOLO infers the label path by replacing `/images/` with `/labels/` in each path. |
| `val` | List of paths | Two directories (can + bottle validation images) | Validation images for mAP computation during training. |
| `test` | List of paths | One directory (bottle test images) | Optional test set. Only the bottle dataset has a test split. |
| `nc` | Integer | `3` | Number of classes. Must match the `names` list length. |
| `names` | List of strings | `['Cans', 'Plastic', 'Bottle']` | Class names mapped to indices 0, 1, 2. |

---

## 3. Known / Likely Problems

| Problem | Detail |
|---------|--------|
| **Hard-coded Colab path** | Only works if the dataset is extracted to exactly `/content/most_recent_vrs/`. |
| **Class 1 (Plastic) has no dedicated dataset** | The `Plastic` class appears in the class list but has no dedicated training images in either dataset. It's unclear where Plastic annotations come from — possibly from the can dataset (which also contains some plastic waste images). |

---
---

# `can_dataset/data.yaml`

**Full path**: `most recent vrs/can_dataset/data.yaml`  
**Size**: 680 bytes, 22 lines

---

## 1. File Purpose

The **primary dataset configuration file** used for local training and validation. This is the file referenced by `train_local.py` (`data='can_dataset/data.yaml'`). It defines the complete multi-dataset setup.

---

## 2. Contents

```yaml
path: c:/Users/patil/Documents/clg/sem 4/CEP/cep project/most recent vrs
train:
  - can_dataset/train/images
  - plastic_bottle_dataset/train/images

val:
  - can_dataset/valid/images
  - plastic_bottle_dataset/valid/images

test:
  - plastic_bottle_dataset/test/images

nc: 3
names: ['Cans', 'Plastic', 'Bottle']


roboflow:
  workspace: dataset-t7hz7
  project: cans-fdboa
  version: 3
  license: CC BY 4.0
  url: https://universe.roboflow.com/dataset-t7hz7/cans-fdboa/dataset/3
```

| Field | Value | Purpose |
|-------|-------|---------|
| `path` | Absolute Windows path to project root | Base directory. `train` and `val` paths are relative to this. |
| `train` | List of 2 relative paths | Combined training set from both datasets |
| `val` | List of 2 relative paths | Combined validation set |
| `test` | Bottle dataset test split | Optional test set |
| `nc` | `3` | Three classes |
| `names` | `['Cans', 'Plastic', 'Bottle']` | Class mapping: 0=Cans, 1=Plastic, 2=Bottle |
| `roboflow` | Metadata block | Provenance info — workspace, project, version, license, URL for the cans dataset |

**Critical detail**: The `path` field contains a **hard-coded absolute path** to a specific user's machine (`c:/Users/patil/...`). This file only works on this specific computer. If the repo is cloned elsewhere, training will fail with path-not-found errors.

---

## 3. Known / Likely Problems

| Problem | Detail | Severity |
|---------|--------|----------|
| **Hard-coded absolute path** | `path: c:/Users/patil/...` makes this non-portable. Other developers must edit this file. | **High** |
| **YOLO label discovery convention** | YOLO finds labels by replacing `/images/` with `/labels/` in each path. If the directory structure doesn't follow this convention, labels won't be found. | Low (structure is correct) |

---
---

# `plastic_bottle_dataset/data.yaml`

**Full path**: `most recent vrs/plastic_bottle_dataset/data.yaml`  
**Size**: 241 bytes, 13 lines

---

## 1. File Purpose

The **original** dataset configuration file that was shipped with the Roboflow bottle dataset export. This file is **not used** by any script in the project — the combined `can_dataset/data.yaml` is used instead. This file remains for reference.

---

## 2. Contents

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 1
names: ['bottle']

roboflow:
  workspace: sdp2
  project: bottle-f2u4m
  version: 1
  license: CC BY 4.0
  url: https://universe.roboflow.com/sdp2/bottle-f2u4m/dataset/1
```

| Field | Value | Note |
|-------|-------|------|
| `nc` | `1` | **Single-class** — this is why `fix_labels.py` was needed |
| `names` | `['bottle']` | Only one class, mapped to index 0 |
| Relative paths | `../train/images` | Relative to this file's parent directory |

---
---

# `can_dataset/README.dataset.txt`

**Full path**: `most recent vrs/can_dataset/README.dataset.txt`  
**Size**: 124 bytes, 7 lines

---

## Contents

```
# Cans > Metal cans
https://universe.roboflow.com/dataset-t7hz7/cans-fdboa

Provided by a Roboflow user
License: CC BY 4.0
```

A Roboflow-generated metadata file identifying the source of the cans dataset: workspace `dataset-t7hz7`, project `cans-fdboa`, CC BY 4.0 license.

---

# `can_dataset/README.roboflow.txt`

**Full path**: `most recent vrs/can_dataset/README.roboflow.txt`  
**Size**: 972 bytes, 30 lines

---

## Contents (Key Information)

- Dataset: **Cans - v3 Metal cans**
- Exported: January 19, 2023
- **223 images** total
- Annotation format: **YOLOv8**
- Preprocessing: Auto-orientation (EXIF stripping), resize to **416×416** (stretch)
- Augmentation: **None**

---
---

# `README.md`

**Full path**: `most recent vrs/README.md`  
**Size**: 3,379 bytes, 103 lines

---

## 1. File Purpose

The **primary project documentation**. Provides an overview of the project, model performance metrics, setup instructions, usage guide, project structure, and detection feature descriptions.

---

## 2. Key Information Documented

- **Project description**: Computer vision for a robotic trash-collecting boat
- **3 detection classes**: Cans (class 0), Plastic (class 1), Bottle (class 2)
- **Dataset sources**: Roboflow — `cans-fdboa` (dataset-t7hz7) and `bottle-f2u4m` (sdp2)
- **Model performance** (train_fixed_labels, 10 epochs):
  - mAP50: **97.3%**
  - mAP50-95: **74.4%**
  - Precision: **95.5%**
  - Recall: **96.7%**
- **Prerequisites**: Python 3.10+, webcam
- **Entry point**: `python main.py`
- **Detection features**: 55% confidence threshold, 30% area filter, 1.3 aspect ratio filter, 3-frame skip, per-class colors
- **License**: CC BY 4.0 (datasets)

---

## 3. Known Issues in README

| Issue | Detail |
|-------|--------|
| **Project structure outdated** | Shows `train2` as "best" model but code uses `train_fixed_labels` |
| **Missing scripts** | `fix_labels.py`, `predict_test.py`, `run_validation.py`, `train_local.py` are not listed in the project structure |

---
---

# `README_COLAB.md`

**Full path**: `most recent vrs/README_COLAB.md`  
**Size**: 1,874 bytes, 68 lines

---

## 1. File Purpose

Step-by-step instructions for training the model in **Google Colab** with GPU acceleration. Covers:
1. Enabling GPU runtime
2. Installing dependencies
3. Uploading/mounting the dataset
4. Training command
5. Validation command
6. Single-image inference

---

## 2. Key Details

- Training uses `data_colab.yaml` (Colab-specific paths)
- Trains from base `yolov8n.pt` (not fine-tuned weights)
- 50 epochs, 640px, batch 16 (larger than local training)
- Notes that webcam scripts are for local PC only

---
---

# `CONTRIBUTING.md`

**Full path**: `most recent vrs/CONTRIBUTING.md`  
**Size**: 2,649 bytes, 61 lines

---

## 1. File Purpose

A **beginner-friendly guide** for contributing via GitHub's fork-and-pull-request workflow. Contains 8 numbered steps from forking the repo through opening a PR and waiting for review.

---

## 2. Key Details

- Repository URL referenced: `https://github.com/ComBox360/robotic-trash-boat`
- Target audience: New Git/GitHub users (the language is very introductory)
- Covers: Fork → Clone → Branch → Edit → Commit → Push → PR → Review

---
---

# `yolov8n.pt`

**Full path**: `most recent vrs/yolov8n.pt`  
**Size**: 6,549,796 bytes (6.5 MB)

---

## 1. File Purpose

The **base YOLOv8 Nano pre-trained weights** from Ultralytics. This is the starting point for the first training run (`runs/detect/train`). The model was pre-trained on the COCO dataset (80 classes, 330K images) by Ultralytics and published at `https://github.com/ultralytics/assets`.

**Architecture**: YOLOv8 Nano — the smallest variant with ~3.2 million parameters. Uses a CSPDarknet53 backbone with C2f modules, PANet feature pyramid neck, and decoupled detection head.

---

## 2. What It Contains

A PyTorch checkpoint file (`.pt`) containing:
- Model architecture specification
- Pre-trained weights (float32 tensors)
- Training metadata (hyperparameters, class names from COCO)
- Optimizer state (ignored when fine-tuning)

**Ignored by Git**: Listed in `.gitignore` because it's freely downloadable and takes 6.5MB.

---
---

# `pb.jpeg` & `pb_1.jpeg`

**Full paths**: `most recent vrs/pb.jpeg` (8,076 bytes), `most recent vrs/pb_1.jpeg` (3,348 bytes)

---

## 1. File Purpose

Two **test images** of plastic bottles, likely used for ad-hoc prediction testing. `pb.jpeg` is referenced in `README_COLAB.md` as the inference example image.

---
---

# `idt_to_yolo_autolabel_pipeline.ipynb`

**Full path**: `most recent vrs/idt_to_yolo_autolabel_pipeline.ipynb`  
**Size**: 26,112 bytes

---

## 1. File Purpose

A **Jupyter notebook** implementing an auto-labeling pipeline. Based on the filename, it converts from some format ("idt") to YOLO annotation format. This would be used to semi-automatically generate bounding box annotations for training data, reducing the manual labeling effort.

*(Note: `.ipynb` files cannot be viewed by our tools, so detailed cell-by-cell analysis is not possible.)*

---
---

# `can_dataset/` (directory)

**Full path**: `most recent vrs/can_dataset/`

---

## Structure

```
can_dataset/
├── README.dataset.txt      (124 bytes)
├── README.roboflow.txt      (972 bytes)
├── data.yaml                (680 bytes)
├── train/
│   ├── images/              (132 images)
│   ├── labels/              (132 .txt files)
│   └── labels.cache         (155,223 bytes)
└── valid/
    ├── images/              (90 images)
    ├── labels/              (90 .txt files)
    └── labels.cache         (39,173 bytes)
```

### Image Details
- **Total**: 222 images (132 train + 90 valid)
- **Format**: JPEG, 416×416 pixels (resized by Roboflow during export)
- **Content**: Photos of metal beverage cans in various environments

### Label Format (YOLO)
Each `.txt` file corresponds to one image. Each line represents one bounding box:
```
<class_id> <center_x> <center_y> <width> <height>
```
All coordinates are normalized to [0, 1] relative to image dimensions.

Example (`beverage_cans-104_jpg.rf.307f...txt`):
```
0 0.30528846153846156 0.5048076923076923 0.37740384615384615 0.8858173076923077
0 0.7151442307692307 0.6153846153846154 0.39783653846153844 0.6225961538461539
```
- Two cans detected in this image (class 0 = Cans)
- Center-x, Center-y, Width, Height in normalized coordinates

### `labels.cache`
Binary cache file generated by Ultralytics during the first training run. Contains pre-parsed label data to speed up subsequent training runs. This file is a pickled Python dict mapping image paths to their label arrays.

---
---

# `plastic_bottle_dataset/` (directory)

**Full path**: `most recent vrs/plastic_bottle_dataset/`

---

## Structure

```
plastic_bottle_dataset/
├── data.yaml                (241 bytes)
├── train/
│   ├── images/              (404 images)
│   └── labels/              (404 .txt files)
├── valid/
│   ├── images/              (43 images)
│   └── labels/              (43 .txt files)
└── test/
    ├── images/              (22 images)
    └── labels/              (22 .txt files)
```

### Image Details
- **Total**: 469 images (404 train + 43 valid + 22 test)
- **Format**: JPEG, 416×416 pixels
- **Content**: Photos of plastic bottles
- **Augmentation applied**: Random Gaussian blur (0-1.25 pixels), 3 versions per source image (so 469 = ~156 originals × 3)

### Label Format
After `fix_labels.py` was run, all labels use class `2` (Bottle):
```
2 0.7067307692307693 0.6069711538461539 0.28725961538461536 0.4831730769230769
```

### Notable Difference from `can_dataset`
- Has a **test** split (22 images), whereas `can_dataset` does not.
- No `labels.cache` file — possibly because this dataset was processed differently or the cache was not generated.

---
---

# `runs/detect/` (Training Output Directories)

**Full path**: `most recent vrs/runs/detect/`

Contains 13 subdirectories representing different YOLO runs:

| Directory | Type | Description |
|-----------|------|-------------|
| `train/` | Training | First training run from base `yolov8n.pt`, 50 epochs, 640px |
| `train2/` | Training | Second run, fine-tuning from `train/best.pt`, 50 epochs, 640px, freeze=10 |
| `train_fixed_labels/` | Training | **Final model**. Fine-tuning from `train2/best.pt`, 10 epochs, 320px, corrected labels |
| `val/` | Validation | Output from `run_validation.py` (empty) |
| `val2/` | Validation | Another validation run |
| `val3/` | Validation | Another validation run |
| `predict/` through `predict7/` | Prediction | Ad-hoc prediction outputs (ignored by Git) |

---
---

# `runs/detect/train/args.yaml`

**Full path**: `most recent vrs/runs/detect/train/args.yaml`  
**Size**: 1,670 bytes, 107 lines

---

## Key Configuration (First Training Run)

| Parameter | Value | Note |
|-----------|-------|------|
| `model` | `yolov8n.pt` | Started from base COCO-pretrained weights |
| `data` | `F:/programs/python/new_vr/can_dataset/data.yaml` | **Old path** — this was run on a different machine/directory |
| `epochs` | `50` | Full training run |
| `batch` | `16` | Default batch size |
| `imgsz` | `640` | Full resolution |
| `workers` | `8` | Multiprocessing enabled (likely run on a different OS or with compatible config) |
| `freeze` | `null` | No layer freezing — all layers trained |
| `lr0` | `0.01` | Initial learning rate |
| `optimizer` | `auto` | Ultralytics auto-selects (typically SGD with momentum) |
| `mosaic` | `1.0` | Mosaic augmentation enabled (100% probability) |
| `amp` | `true` | Automatic Mixed Precision (float16) for faster training |

---
---

# `runs/detect/train2/args.yaml`

**Full path**: `most recent vrs/runs/detect/train2/args.yaml`  
**Size**: 1,693 bytes, 107 lines

---

## Key Configuration (Second Training Run)

| Parameter | Value | Note |
|-----------|-------|------|
| `model` | `runs/detect/train/weights/best.pt` | Fine-tuning from first run |
| `data` | `F:/programs/python/new_vr/can_dataset/data.yaml` | Same old path |
| `epochs` | `50` | Another 50 epochs |
| `freeze` | **`10`** | **First 10 layers frozen** — backbone features preserved, only head layers trained. This is a classic transfer learning technique. |
| Everything else | Same as `train` | Same hyperparameters |

---
---

# `runs/detect/train_fixed_labels/args.yaml`

**Full path**: `most recent vrs/runs/detect/train_fixed_labels/args.yaml`  
**Size**: 1,693 bytes, 107 lines

---

## Key Configuration (Final Training Run)

| Parameter | Value | Note |
|-----------|-------|------|
| `model` | `runs/detect/train2/weights/best.pt` | Fine-tuning from second run |
| `data` | `can_dataset/data.yaml` | **Local path** (run on the current machine) |
| `epochs` | `10` | Short fine-tuning |
| `imgsz` | `320` | Reduced for CPU speed |
| `batch` | `8` | Reduced for CPU memory |
| `workers` | `0` | Disabled for Windows compatibility |
| `freeze` | `null` | All layers unfrozen — full fine-tuning with corrected labels |

---
---

# `runs/detect/train/results.csv`

**Full path**: `most recent vrs/runs/detect/train/results.csv`  
**Size**: 6,224 bytes, 52 lines (header + 50 epochs)

---

## Training Progression (First Run — 50 Epochs)

| Epoch | mAP50 | mAP50-95 | Notes |
|-------|-------|----------|-------|
| 1 | 0.900 | 0.592 | High recall (0.994) from the start due to COCO pretraining |
| 10 | 0.819 | 0.484 | Performance dip — learning rate still climbing |
| 25 | 0.915 | 0.620 | Stabilizing |
| 50 | 0.955 | 0.709 | Final: mAP50=95.5%, mAP50-95=70.9% |

**Time anomaly**: Epoch 25 took 15,224 seconds (~4.2 hours) while other epochs took ~50 seconds. This suggests the machine was hibernated or paused during training.

---
---

# `runs/detect/train2/results.csv`

**Full path**: `most recent vrs/runs/detect/train2/results.csv`  
**Size**: 6,467 bytes, 52 lines

---

## Training Progression (Second Run — 50 Epochs, Frozen Layers)

| Epoch | mAP50 | mAP50-95 | Notes |
|-------|-------|----------|-------|
| 1 | 0.920 | 0.643 | Starts high because of pre-training |
| 24 | 0.964 | 0.728 | Peak mAP50-95 |
| 50 | 0.957 | 0.726 | Final: mAP50=95.7%, mAP50-95=72.6% |

**Marginal improvement** over run 1. Freezing the first 10 layers slightly improved mAP50-95 (70.9% → 72.6%).

---
---

# `runs/detect/train_fixed_labels/results.csv`

**Full path**: `most recent vrs/runs/detect/train_fixed_labels/results.csv`  
**Size**: 1,449 bytes, 12 lines (header + 10 epochs)

---

## Training Progression (Final Run — 10 Epochs, Corrected Labels)

| Epoch | mAP50 | mAP50-95 | Precision | Recall |
|-------|-------|----------|-----------|--------|
| 1 | 0.441 | 0.322 | 0.869 | 0.331 | Performance drop: model adjusting to corrected labels |
| 2 | 0.946 | 0.705 | 0.701 | 0.936 | Rapid recovery |
| 8 | 0.971 | 0.740 | 0.954 | 0.951 | Near-peak |
| 10 | **0.973** | **0.745** | **0.955** | **0.967** | **Final model** |

**Key insight**: Epoch 1 shows a significant performance drop (mAP50: 0.441) because the model was trained on the old (incorrect) labels and now sees corrected labels. By epoch 2, it has largely adapted, demonstrating the power of transfer learning.

**Training time**: ~50 seconds per epoch, ~8.4 minutes total.

---
---

# `useless/` (directory)

**Full path**: `most recent vrs/useless/`

---

## Contents

```
useless/
├── README.dataset.txt           (126 bytes) — Bottle dataset source info
├── README.roboflow.txt          (1,067 bytes) — Bottle dataset export details
├── camera_detection.ipynb       (4,510 bytes) — Notebook version of detection
├── camera_detection_1.ipynb     (2,886 bytes) — Notebook version of camera test
├── can_dataset.zip              (4,577,511 bytes / 4.6 MB) — Compressed can dataset
├── colab_new_vr.ipynb           (2,969 bytes) — Colab training notebook
├── imgx.png                     (136,506 bytes) — Unknown image
├── main.ipynb                   (1,348 bytes) — Notebook version of main menu
├── model_detection.ipynb        (1,682 bytes) — Notebook version of model info
├── plastic_bottle_dataset.zip   (13,413,335 bytes / 13.4 MB) — Compressed bottle dataset
└── docs_duplicates/
    ├── Community_Engagement_REPORT[1].docx     (314,479 bytes)
    └── Community_Engagement_REPORT[1][1][2].docx (314,476 bytes)
```

## Purpose

A dumping ground for files that are no longer active in the project:
- **Jupyter notebooks** (`.ipynb`): Earlier versions of the Python scripts, likely from when the project was developed in Google Colab. Superseded by the `.py` files.
- **Dataset ZIPs**: Original Roboflow exports. Kept as backups in case the extracted datasets need to be recreated.
- **`docs_duplicates/`**: Duplicate copies of the Community Engagement Report (a college course deliverable, not technical documentation).
- **`imgx.png`**: An image file of unknown purpose.

---
---

# `docs/` (directory)

**Full path**: `most recent vrs/docs/`

---

## Contents

```
docs/
├── Community_Engagement_REPORT.docx       (313,184 bytes)
└── Community_Engagement_REPORT[1][1].docx (3,630,831 bytes / 3.6 MB)
```

## Purpose

Contains the **Community Engagement Report** — a college course document (CEP = Community Engagement Project). This is a non-technical, non-code deliverable. The second file `[1][1]` is significantly larger (3.6 MB vs 313 KB), suggesting it contains embedded images or media.

---
---

# `__pycache__/` (directory)

**Full path**: `most recent vrs/__pycache__/`

---

## Contents

```
__pycache__/
├── camera_detection.cpython-313.pyc   (3,187 bytes)
├── camera_detection_1.cpython-313.pyc (1,241 bytes)
├── main.cpython-313.pyc               (1,768 bytes)
└── model_detection.cpython-313.pyc    (294 bytes)
```

Python bytecode cache files compiled by CPython 3.13. These are auto-generated when the scripts are executed. The `cpython-313` suffix indicates Python 3.13 was used.

**Ignored by Git** (listed in `.gitignore`).

---
---

# `.venv/` (directory)

**Full path**: `most recent vrs/.venv/`

The Python virtual environment containing installed packages (ultralytics, opencv-python, torch, torchvision, and all transitive dependencies). Created by `python -m venv .venv`.

**Ignored by Git** (listed in `.gitignore`).

---
---

# `.git/` (directory)

**Full path**: `most recent vrs/.git/`

Standard Git repository metadata. Contains the full revision history, refs, hooks, and configuration. The remote repository is likely `https://github.com/ComBox360/robotic-trash-boat` (based on `CONTRIBUTING.md`).

---
---
---

# 🗺️ System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           USER                                      │
│                     (runs python main.py)                            │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        main.py                                      │
│                   (Interactive CLI Menu)                             │
│                                                                     │
│  Option 1 ──► subprocess.run() ──► camera_detection.py              │
│  Option 2 ──► subprocess.run() ──► camera_detection_1.py            │
│  Option 3 ──► subprocess.run() ──► model_detection.py               │
│  Option 4 ──► sys.exit(0)                                           │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    camera_detection.py                               │
│              (Primary Real-Time Detection)                           │
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌───────────┐ │
│  │  Webcam   │───►│  Frame   │───►│ YOLOv8 Nano  │───►│  Filters  │ │
│  │ cv2.Video │    │  Buffer  │    │  Inference   │    │ Area+AR   │ │
│  │ Capture(0)│    │ (skip 2  │    │ (320×320)    │    │           │ │
│  │           │    │  of 3)   │    │              │    │           │ │
│  └──────────┘    └──────────┘    └──────────────┘    └─────┬─────┘ │
│                                                            │       │
│                                                            ▼       │
│                                              ┌──────────────────┐  │
│                                              │  cv2.imshow()    │  │
│                                              │  Display Window  │  │
│                                              └──────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                                 │
│              (Offline, run separately)                               │
│                                                                     │
│  fix_labels.py ──► Correct class IDs (0→2) in label files           │
│        │                                                            │
│        ▼                                                            │
│  train_local.py ──► model.train() ──► runs/detect/train_fixed_labels│
│        │                                     │                      │
│        ▼                                     ▼                      │
│  predict_test.py ──► Smoke test on 2 images                         │
│  run_validation.py ──► Full mAP evaluation                          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    MODEL WEIGHTS CHAIN                               │
│                                                                     │
│  yolov8n.pt (COCO pretrained)                                       │
│       │                                                             │
│       ▼  50 epochs, 640px, batch 16                                 │
│  runs/detect/train/weights/best.pt                                  │
│       │                                                             │
│       ▼  50 epochs, 640px, freeze=10                                │
│  runs/detect/train2/weights/best.pt                                 │
│       │                                                             │
│       ▼  10 epochs, 320px, corrected labels                         │
│  runs/detect/train_fixed_labels/weights/best.pt  ◄── ACTIVE MODEL  │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Architectural Properties

- **No database**: All data is in files (images, YOLO .txt labels, PyTorch .pt checkpoints).
- **No network requests at runtime**: The detection system is fully offline once the model is trained.
- **No queues or caches**: Processing is synchronous and single-threaded.
- **No API/server**: This is a desktop application, not a web service.
- **Synchronous everywhere**: `main.py` blocks on `subprocess.run()`, `camera_detection.py` blocks on `cap.read()`, training blocks on `model.train()`.

---

# 🔁 Request Lifecycle

## Lifecycle 1: Real-Time Detection (most common)

```
User runs `python main.py`
  → main() prints menu
  → User types "1", presses Enter
  → input() returns "1"
  → scripts["1"] → "camera_detection.py"
  → subprocess.run([sys.executable, "camera_detection.py"])
    │
    ├── YOLO("runs/detect/train_fixed_labels/weights/best.pt")
    │     └── PyTorch loads checkpoint, reconstructs model
    │
    ├── cv2.VideoCapture(0)
    │     └── Opens webcam via DirectShow/MSMF
    │
    └── while True:
          ├── cap.read()         → numpy array (480, 640, 3)
          ├── frame_count++
          ├── if frame_count % 3 == 0:
          │     ├── model(frame)  → preprocess → inference → NMS → Results
          │     ├── for each detection:
          │     │     ├── Extract x1,y1,x2,y2,conf,cls
          │     │     ├── Filter: box area > 30% frame → skip
          │     │     ├── Filter: aspect ratio < 1.3 → skip
          │     │     ├── cv2.rectangle() → draw box
          │     │     └── cv2.putText() → draw label
          │     └── annotated_frame = result
          ├── cv2.imshow(annotated_frame)
          └── if 'q' pressed → break → cap.release() → destroyAllWindows()
```

## Lifecycle 2: Training

```
User runs `python fix_labels.py`
  → Scans plastic_bottle_dataset/{train,valid,test}/labels/*.txt
  → Rewrites class 0 → class 2 in each file
  → Prints count of modified files

User runs `python train_local.py`
  → YOLO("runs/detect/train2/weights/best.pt")
  → model.train(data='can_dataset/data.yaml', epochs=10, ...)
    │
    ├── Reads data.yaml → finds train/val image directories
    ├── Creates DataLoaders (batch=8, workers=0)
    ├── For each epoch (1-10):
    │     ├── For each batch:
    │     │     ├── Load 8 images + labels
    │     │     ├── Preprocess (resize to 320, normalize, augment)
    │     │     ├── Forward pass through YOLOv8 Nano
    │     │     ├── Compute losses (box + cls + dfl)
    │     │     ├── Backward pass (autograd)
    │     │     └── Optimizer step (SGD)
    │     └── Validation pass → compute mAP metrics
    │
    └── Save best.pt and last.pt to runs/detect/train_fixed_labels/weights/
```

---

# 🧱 Data Layer Deep Dive

## There is no traditional database. All data storage is file-based.

### Dataset Storage Format

```
dataset_name/
├── split/          (train, valid, test)
│   ├── images/     (JPEG files, 416×416)
│   └── labels/     (TXT files, YOLO format)
└── data.yaml       (metadata and class mapping)
```

### YOLO Label Format (per line)
```
<class_id> <center_x> <center_y> <width> <height>
```
- All values except `class_id` are floats in [0, 1], normalized to image dimensions.
- `class_id` is an integer (0-indexed).

### Dataset Statistics

| Dataset | Train Images | Val Images | Test Images | Total | Classes |
|---------|-------------|------------|-------------|-------|---------|
| `can_dataset` | 132 | 90 | 0 | 222 | 1 (Cans, class 0) |
| `plastic_bottle_dataset` | 404 | 43 | 22 | 469 | 1 (Bottle, class 2 after fix) |
| **Combined** | **536** | **133** | **22** | **691** | **3** (class 1/Plastic is implicit) |

### Model Weight Storage
- Format: PyTorch checkpoint (`.pt`)
- Size: ~6.2 MB each
- Contains: model architecture + weights + training metadata + optimizer state

### No ORM, No Migrations, No Indexing
The project has zero database technology. The closest thing to "indexing" is the `labels.cache` files, which are pickled Python dictionaries that map image paths to pre-parsed NumPy arrays of labels.

---

# ⚙️ Framework Usage

## Ultralytics YOLOv8

### Configuration and Initialization
- Initialized via `YOLO(path)` constructor that auto-detects whether `path` is a model architecture YAML or a checkpoint `.pt` file.
- No global config file — all settings are passed as arguments to `model.train()`, `model.val()`, and `model()`.
- Training state is saved to `runs/detect/<name>/` directory, including `args.yaml` (full config snapshot), `results.csv` (per-epoch metrics), plots, and weights.

### Features Used
| Feature | Where Used | Purpose |
|---------|------------|---------|
| `YOLO(path)` | All scripts | Model loading |
| `model(image)` | `camera_detection.py`, `predict_test.py` | Single-image inference |
| `model.train()` | `train_local.py` | Fine-tuning |
| `model.val()` | `run_validation.py` | Formal evaluation |
| `model.names` | `camera_detection.py`, `model_detection.py`, `predict_test.py` | Class name lookup |
| `result.boxes.xyxy` | `camera_detection.py`, `predict_test.py` | Bounding box coordinates |
| `result.boxes.conf` | Same | Confidence scores |
| `result.boxes.cls` | Same | Class IDs |

### Non-Obvious Behaviors Relied On
1. **Auto label discovery**: YOLO finds label files by replacing `/images/` with `/labels/` in the image path. The directory structure must follow this convention exactly.
2. **Multi-path datasets**: The `train` and `val` fields in `data.yaml` can be lists of directories. YOLO concatenates them into a single dataset.
3. **Auto device selection**: With `device: null`, YOLO uses CUDA if available, otherwise CPU. In this project, it runs on CPU.
4. **Mosaic augmentation auto-disable**: The `close_mosaic: 10` setting disables mosaic augmentation for the last 10 epochs to stabilize training. With only 10 epochs in `train_fixed_labels`, mosaic is effectively disabled for the entire run.

## OpenCV (cv2)

### Configuration
- No explicit configuration. OpenCV is used via its functional API.
- Video backend is auto-selected (DirectShow or MSMF on Windows).

### Features Used
| Feature | Where Used | Purpose |
|---------|------------|---------|
| `VideoCapture(0)` | `camera_detection.py`, `camera_detection_1.py` | Webcam access |
| `cap.read()` | Same | Frame capture |
| `cap.set()` | Both | Camera property configuration |
| `cap.release()` | Both | Resource cleanup |
| `cv2.rectangle()` | `camera_detection.py` | Drawing bounding boxes |
| `cv2.putText()` | `camera_detection.py` | Drawing labels |
| `cv2.imshow()` | Both | Window display |
| `cv2.waitKey()` | Both | Key event polling + event loop pump |
| `cv2.destroyAllWindows()` | Both | Window cleanup |

### Non-Obvious Behavior
- `cv2.waitKey(n)` is not just for key detection — it also processes window events (resize, move, repaint). Without calling `waitKey()`, `imshow()` windows freeze and become unresponsive.

---

# 🧠 Design Patterns Used

| Pattern | Where | Why |
|---------|-------|-----|
| **Dispatch Table** | `main.py` — `scripts` dict maps input to filenames | Cleaner than if-elif chains; easy to add new options |
| **Strategy Pattern** (informal) | Configuration constants in `camera_detection.py` (`CONF_THRESHOLD`, `MAX_BOX_AREA_RATIO`, etc.) | Filtering strategies are parameterized at the top of the file, making them easy to tune without modifying logic |
| **Template Method** (via Ultralytics) | `model.train()` defines the training loop skeleton; hyperparameters customize the behavior | Standard ML training pattern |
| **Pipeline / Filter Chain** | Detection pipeline in `camera_detection.py`: raw frame → YOLO inference → area filter → aspect ratio filter → drawing | Each filter independently removes false positives |
| **Singleton** (implicit) | `model = YOLO(...)` at module level — one model instance for the lifetime of the process | Avoids repeated model loading (which takes seconds) |

---

# 🚧 Problems Faced & How They Were Solved

## Problem 1: Class ID Collision

**Issue**: The can dataset used class 0 for cans. The bottle dataset also used class 0 for bottles (since it was a single-class dataset). When merged, all objects had class 0.

**Solution**: Created `fix_labels.py` to remap class 0 → class 2 in all bottle dataset label files. Then retrained the model with corrected labels (`train_fixed_labels`).

**Trade-off**: Destructive in-place edit of label files. No backup created (though the original ZIP is preserved in `useless/`).

## Problem 2: False Positive Detections

**Issue**: The YOLO model sometimes detected non-target objects (faces, walls, reflections) as cans or bottles.

**Solution**: Two post-inference filters in `camera_detection.py`:
1. **Area filter**: Reject boxes > 30% of frame area.
2. **Aspect ratio filter**: Reject boxes with aspect ratio < 1.3 (too square).

**Trade-off**: These filters are domain-specific heuristics. They work for the "trash in water" use case but would need adjustment for other deployment scenarios.

## Problem 3: Slow CPU Inference

**Issue**: Running YOLO on every webcam frame caused significant lag on a CPU-only laptop.

**Solution**: Three-pronged approach:
1. Reduced inference image size from 640 to 320 pixels (`INFER_SIZE = 320`).
2. Frame skipping: only process every 3rd frame (`PROCESS_EVERY_N = 3`).
3. Reuse the last annotated frame for display between inference frames.

**Trade-off**: Detection is ~100ms stale and small/distant objects may be missed at 320px resolution.

## Problem 4: Windows Multiprocessing Crash

**Issue**: Ultralytics' data loaders use Python `multiprocessing` with `workers > 0`. On Windows, `multiprocessing` uses `spawn` (not `fork`), which can cause pickle errors with certain CUDA/OpenCV/torch configurations.

**Solution**: Set `workers=0` in `train_local.py` to disable multiprocessing entirely.

**Trade-off**: Data loading is slower (synchronous in main process), but training completes without crashes.

## Problem 5: Machine Migration

**Issue**: Training was started on a machine with path `F:/programs/python/new_vr/` (seen in `train` and `train2` `args.yaml`), then the project was moved to `c:/Users/patil/Documents/...`.

**Solution**: Updated `can_dataset/data.yaml` with the new absolute path and created `data_colab.yaml` for Colab.

**Trade-off**: Hard-coded absolute paths are fragile. A relative path would be more portable.

---

# 📦 Full Dependency Map

| Dependency | Type | Used For | Files That Use It | Alternatives |
|------------|------|----------|-------------------|-------------|
| `ultralytics` | PyPI package | YOLOv8 model loading, training, inference, validation | `camera_detection.py`, `model_detection.py`, `train_local.py`, `predict_test.py`, `run_validation.py` | `detectron2` (Facebook), `mmdetection` (OpenMMLab), `torch` + manual implementation |
| `opencv-python` | PyPI package | Webcam access, image drawing, window display | `camera_detection.py`, `camera_detection_1.py` | `imageio`, `Pillow` (no webcam), `PyQt5` (heavyweight) |
| `torch` (PyTorch) | PyPI package | Deep learning framework (tensors, autograd, optimizers) | Transitive via `ultralytics` | `tensorflow`, `jax` |
| `torchvision` | PyPI package | Image transforms, model utilities | Transitive via `ultralytics` | `albumentations` (transforms only) |
| `numpy` | Transitive | Array operations for image data | Transitively via `cv2` and `torch` | N/A (fundamental dependency) |
| `pillow` | Transitive | Image I/O inside Ultralytics | Transitively via `ultralytics` | `imageio` |
| `pyyaml` | Transitive | YAML parsing for `data.yaml` config files | Transitively via `ultralytics` | `ruamel.yaml` |
| `matplotlib` | Transitive | Plot generation during training (loss curves, PR curves) | Transitively via `ultralytics` | `plotly`, `seaborn` |
| `scipy` | Transitive | Metric computation inside Ultralytics | Transitively via `ultralytics` | N/A |
| `tqdm` | Transitive | Progress bars during training | Transitively via `ultralytics` | `rich`, `alive-progress` |

---

# 🔗 Inter-File Dependency Graph

```
                    ┌─────────────────┐
                    │    main.py      │
                    │ (entry point)   │
                    └───┬───┬───┬─────┘
          subprocess    │   │   │  subprocess
         ┌──────────────┘   │   └──────────────┐
         ▼                  ▼                   ▼
┌────────────────┐ ┌────────────────┐
│camera_detection│ │camera_detection│
│     .py        │ │   _1.py        │
└───────┬────────┘ └────────────────┘
        │
        │  loads model from
        ▼
┌───────────────────────────────────────────────────────────┐
│         runs/detect/train_fixed_labels/weights/best.pt    │
│                    (shared model file)                     │
└───────────────────────────────────────────────────────────┘
        ▲
        │  produced by
        │
┌───────┴────────┐
│ train_local.py │ ──loads──► runs/detect/train2/weights/best.pt
└───────┬────────┘
        │  reads
        ▼
┌────────────────────┐
│can_dataset/data.yaml│ ──references──► can_dataset/{train,valid}/images/
│                    │ ──references──► plastic_bottle_dataset/{train,valid,test}/images/
└────────────────────┘
        ▲
        │  labels were fixed by
        │
┌───────┴────────┐
│ fix_labels.py  │ ──modifies──► plastic_bottle_dataset/{train,valid,test}/labels/*.txt
└────────────────┘

┌────────────────┐
│predict_test.py │ ──loads──► runs/detect/train_fixed_labels/weights/best.pt
│                │ ──reads──► can_dataset/valid/images/... (1 specific image)
│                │ ──reads──► plastic_bottle_dataset/valid/images/... (1 specific image)
└────────────────┘

┌──────────────────┐
│run_validation.py │ ──loads──► runs/detect/train2/weights/best.pt (LIKELY A BUG)
│                  │ ──reads──► can_dataset/data.yaml
└──────────────────┘
```

## Central Nodes (Most Depended Upon)

1. **`runs/detect/train_fixed_labels/weights/best.pt`** — The trained model file. Used by 2 scripts (`camera_detection.py`, `predict_test.py`).
2. **`can_dataset/data.yaml`** — The dataset configuration. Used by `train_local.py` and `run_validation.py`.
3. **`ultralytics` (YOLO)** — The framework. Used by 5 of 7 Python scripts.
4. **`cv2` (OpenCV)** — Used by 2 scripts.

## Files with Zero Inbound Dependencies

- `main.py` — legacy CLI menu
- `gui.py` — modern graphical launcher
- `camera_detection_1.py` — standalone utility
- `fix_labels.py` — one-time script
- `data_colab.yaml` — only used in Colab (not by any script in this repo)
- All Markdown files — documentation only
- All files in `useless/` — archived, not referenced

---

*End of brain dump. Every file, function, import, constant, and data structure in the project has been documented.*

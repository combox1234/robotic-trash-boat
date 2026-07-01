import sys
import subprocess

def main():
    print("====================================================")
    print("   Robotic Trash Boat - YOLOv8 Detection Platform   ")
    print("====================================================")
    print("1. Start Real-time Object Detection (camera_detection.py)")
    print("2. Run Camera Settings / Focus Test (camera_detection_1.py)")
    print("3. Exit")
    print("====================================================")

    choice = input("Select an option (1-3): ").strip()

    scripts = {
        '1': 'camera_detection.py',
        '2': 'camera_detection_1.py',
    }

    if choice == '3':
        print("\nExiting. Have a great day!")
        sys.exit(0)
    elif choice in scripts:
        script = scripts[choice]
        print(f"\nLaunching {script}...")
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"\n{script} exited with code {result.returncode}")
    else:
        print("\nInvalid selection. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    main()
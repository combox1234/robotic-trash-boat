import sys
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
import threading

class TrashBoatGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Robotic Trash Boat - YOLOv8 Detection Platform")
        self.root.geometry("500x400")
        self.root.configure(bg="#2c3e50")
        
        # Make the GUI look a bit more modern
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Helvetica', 12, 'bold'), padding=10)
        style.configure('TLabel', background="#2c3e50", foreground="#ecf0f1", font=('Helvetica', 16, 'bold'))
        
        # Header Label
        header = ttk.Label(self.root, text="Robotic Trash Boat\nYOLOv8 Detection Platform", justify=tk.CENTER)
        header.pack(pady=30)
        
        # Buttons Frame
        btn_frame = tk.Frame(self.root, bg="#2c3e50")
        btn_frame.pack(fill=tk.BOTH, expand=True, padx=40)
        
        # Define buttons
        self.btn_detect = ttk.Button(btn_frame, text="Start Real-time Object Detection", 
                                     command=lambda: self.run_script('camera_detection.py'))
        self.btn_detect.pack(fill=tk.X, pady=10)
        
        self.btn_focus = ttk.Button(btn_frame, text="Run Camera Settings / Focus Test", 
                                    command=lambda: self.run_script('camera_detection_1.py'))
        self.btn_focus.pack(fill=tk.X, pady=10)
        
        self.btn_exit = ttk.Button(btn_frame, text="Exit", command=self.root.quit)
        self.btn_exit.pack(fill=tk.X, pady=10)

    def run_script(self, script_name):
        # Run in a separate thread so GUI doesn't freeze
        def task():
            self.root.config(cursor="watch")
            try:
                # We use subprocess.Popen to launch it and wait for it to finish
                process = subprocess.Popen([sys.executable, script_name])
                process.communicate()
                
                if process.returncode != 0:
                    messagebox.showerror("Error", f"{script_name} exited with code {process.returncode}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to run {script_name}:\n{str(e)}")
            finally:
                self.root.config(cursor="")
                
        threading.Thread(target=task, daemon=True).start()

def main():
    root = tk.Tk()
    app = TrashBoatGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

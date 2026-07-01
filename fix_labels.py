import os
from pathlib import Path

# The base path of the plastic bottle dataset
base_dir = Path("plastic_bottle_dataset")

# The splits we need to update
splits = ["train", "valid", "test"]

files_updated = 0

for split in splits:
    labels_dir = base_dir / split / "labels"
    if not labels_dir.exists():
        continue
    
    for txt_file in labels_dir.glob("*.txt"):
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        changed = False
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            # If the class ID is 0, change it to 2 (Bottle)
            if parts[0] == '0':
                parts[0] = '2'
                changed = True
            new_lines.append(" ".join(parts) + "\n")
            
        if changed:
            with open(txt_file, 'w') as f:
                f.writelines(new_lines)
            files_updated += 1

print(f"Successfully updated {files_updated} label files in plastic_bottle_dataset.")

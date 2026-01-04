import os
import shutil
import re

# -------- CONFIG --------
LABELS_DIR = r"C:/Users/Rithish Chanalu/Documents/input_annotations/labels"
SOURCE_IMAGES_DIR = r"C:/Users/Rithish Chanalu/Documents/dataset_images"
OUTPUT_IMAGES_DIR = r"C:/Users/Rithish Chanalu/Documents/output_dataset/images"
OUTPUT_LABELS_DIR = r"C:/Users/Rithish Chanalu/Documents/output_dataset/labels"
# ------------------------

pattern = re.compile(r"(IMG_\d+)\.txt")

for fname in os.listdir(LABELS_DIR):
    match = pattern.search(fname)
    if not match:
        print(f"Skipping unrecognized file: {fname}")
        continue

    image_base = match.group(1)
    new_label_name = f"{image_base}.txt"

    # Copy & rename label
    shutil.copy(
        os.path.join(LABELS_DIR, fname),
        os.path.join(OUTPUT_LABELS_DIR, new_label_name)
    )

    # Copy matching image (jpg or png)
    for ext in [".jpg", ".jpeg", ".png"]:
        img_path = os.path.join(SOURCE_IMAGES_DIR, image_base + ext)
        if os.path.exists(img_path):
            shutil.copy(img_path, os.path.join(OUTPUT_IMAGES_DIR, image_base + ext))
            break
    else:
        print(f"⚠ Image not found for {image_base}")

print("✅ Dataset cleanup completed.")

import os
import random
import shutil

BASE_DIR = "C:/Users/Rithish Chanalu/Documents/AI-Checkout-Sytem/dataset"
IMG_DIR = os.path.join(BASE_DIR, "images")
LBL_DIR = os.path.join(BASE_DIR, "labels")

TRAIN_RATIO = 0.8  # 80% train, 20% val

# Create folders
for split in ["train", "val"]:
    os.makedirs(os.path.join(IMG_DIR, split), exist_ok=True)
    os.makedirs(os.path.join(LBL_DIR, split), exist_ok=True)

# Collect images
images = [
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

random.shuffle(images)
split_idx = int(len(images) * TRAIN_RATIO)

train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

def move(files, split):
    for img in files:
        name = os.path.splitext(img)[0]

        shutil.move(
            os.path.join(IMG_DIR, img),
            os.path.join(IMG_DIR, split, img)
        )

        shutil.move(
            os.path.join(LBL_DIR, name + ".txt"),
            os.path.join(LBL_DIR, split, name + ".txt")
        )

move(train_imgs, "train")
move(val_imgs, "val")

print("✅ Train/val split completed successfully.")

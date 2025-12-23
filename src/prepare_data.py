import os
from datasets import load_dataset
from PIL import Image

# -----------------------------
# CONFIG
# -----------------------------
TARGET_CLASSES = [
    'person','car','truck','bus','motorcycle','bicycle','airplane',
    'traffic light','stop sign','bench','dog','cat','horse','bird','cow','elephant',
    'bottle','cup','bowl','pizza','cake','chair','couch','bed','potted plant'
]

SAVE_DIR = "data/classification/train"
MAX_PER_CLASS = 100
MAX_SAMPLES = 8000   # safety cap for Windows

os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
# COCO CATEGORY ID → NAME MAP
# (official COCO mapping)
# -----------------------------
COCO_ID_TO_NAME = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    7: "truck",
    9: "traffic light",
    11: "stop sign",
    13: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    39: "bottle",
    41: "cup",
    45: "bowl",
    53: "pizza",
    55: "cake",
    56: "chair",
    57: "couch",
    59: "bed",
    64: "potted plant"
}

print("📥 Loading COCO dataset (streaming)...")

ds = load_dataset(
    "detection-datasets/coco",
    split="train",
    streaming=True
)

count = {cls: 0 for cls in TARGET_CLASSES}
processed = 0

for sample in ds:
    processed += 1
    if processed >= MAX_SAMPLES:
        print("⚠️ Reached MAX_SAMPLES limit")
        break

    img: Image.Image = sample["image"]
    category_ids = sample["objects"]["category"]

    for cid in category_ids:
        if cid in COCO_ID_TO_NAME:
            label = COCO_ID_TO_NAME[cid]

            if label in TARGET_CLASSES and count[label] < MAX_PER_CLASS:
                class_dir = os.path.join(SAVE_DIR, label)
                os.makedirs(class_dir, exist_ok=True)

                img.save(os.path.join(class_dir, f"{count[label]}.jpg"))
                count[label] += 1

    if all(v >= MAX_PER_CLASS for v in count.values()):
        print("✅ All target classes collected")
        break

# -----------------------------
# SUMMARY
# -----------------------------
print("\n📊 DATA COLLECTION SUMMARY")
for k, v in count.items():
    print(f"{k:15s}: {v}")

print("\n✅ Data preparation complete!")
print(f"📁 Saved to: {SAVE_DIR}")
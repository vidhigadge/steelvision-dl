from pathlib import Path
import random
import shutil

SOURCE_DIR = Path("data/raw/NEU-CLS")
OUTPUT_DIR = Path("data/splits")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

random.seed(SEED)

for class_dir in sorted(SOURCE_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    images = [
        p for p in class_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }

    for split_name, split_images in splits.items():
        destination = OUTPUT_DIR / split_name / class_dir.name
        destination.mkdir(parents=True, exist_ok=True)

        for image_path in split_images:
            shutil.copy2(image_path, destination / image_path.name)

    print(
        f"{class_dir.name}: "
        f"train={len(splits['train'])}, "
        f"val={len(splits['val'])}, "
        f"test={len(splits['test'])}"
    )

print("Dataset split completed.")
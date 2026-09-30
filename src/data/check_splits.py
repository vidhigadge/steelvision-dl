from pathlib import Path

SPLIT_DIR = Path("data/splits")

for split in ["train", "val", "test"]:
    print(f"\n{split.upper()}")

    for class_dir in sorted((SPLIT_DIR / split).iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.*")))
            print(f"{class_dir.name:<20} {count}")
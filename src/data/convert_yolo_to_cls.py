from pathlib import Path
import shutil

SOURCE = Path("data/raw/NEU-CLS-YOLO")
DEST = Path("data/raw/NEU-CLS")

CLASS_NAMES = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled_in_scale",
    5: "scratches",
}

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

for class_name in CLASS_NAMES.values():
    (DEST / class_name).mkdir(parents=True, exist_ok=True)

for split in ["train", "valid"]:
    images_dir = SOURCE / split / "images"
    labels_dir = SOURCE / split / "labels"

    for label_file in labels_dir.glob("*.txt"):
        lines = label_file.read_text().strip().splitlines()

        if not lines:
            print(f"Skipping empty label: {label_file.name}")
            continue

        class_ids = []

        for line in lines:
            parts = line.split()
            if parts:
                class_ids.append(int(parts[0]))

        unique_classes = set(class_ids)

        if len(unique_classes) != 1:
            print(
                f"Skipping {label_file.name}: "
                f"contains multiple classes {unique_classes}"
            )
            continue

        class_id = next(iter(unique_classes))

        if class_id not in CLASS_NAMES:
            print(f"Unknown class {class_id}: {label_file.name}")
            continue

        image_path = None

        for ext in IMAGE_EXTENSIONS:
            candidate = images_dir / f"{label_file.stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            print(f"Image not found for {label_file.name}")
            continue

        class_name = CLASS_NAMES[class_id]

        destination = DEST / class_name / image_path.name

        shutil.copy2(image_path, destination)

print("\nConversion complete.")
print(f"Dataset created at: {DEST}")

from pathlib import Path
import shutil


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SOURCE_ROOT = Path("data/splits")
TARGET_ROOT = Path("data/domain_validation")

SPLITS = [
    "train",
    "val",
    "test",
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
]


# -----------------------------------------------------------------------------
# Copy supported-domain images
# -----------------------------------------------------------------------------

def copy_supported_images():
    total_copied = 0

    print()
    print("=" * 70)
    print("STEELVISION DOMAIN VALIDATION — SUPPORTED DATA PREPARATION")
    print("=" * 70)

    for split in SPLITS:
        source_split = SOURCE_ROOT / split

        target_split = (
            TARGET_ROOT
            / split
            / "supported"
        )

        target_split.mkdir(
            parents=True,
            exist_ok=True,
        )

        split_count = 0

        for class_name in CLASS_NAMES:
            source_class = (
                source_split
                / class_name
            )

            if not source_class.exists():
                print(
                    f"WARNING: Missing folder: "
                    f"{source_class}"
                )
                continue

            for image_path in sorted(
                source_class.iterdir()
            ):
                if (
                    not image_path.is_file()
                    or image_path.suffix.lower()
                    not in IMAGE_EXTENSIONS
                ):
                    continue

                # Prefix the filename with the class name
                # so files from different folders cannot collide.
                target_name = (
                    f"{class_name}__"
                    f"{image_path.name}"
                )

                target_path = (
                    target_split
                    / target_name
                )

                shutil.copy2(
                    image_path,
                    target_path,
                )

                split_count += 1
                total_copied += 1

        print(
            f"{split:<5} → "
            f"{split_count} supported images"
        )

    print("-" * 70)

    print(
        f"Total supported images copied: "
        f"{total_copied}"
    )

    print("=" * 70)
    print()


if __name__ == "__main__":
    copy_supported_images()
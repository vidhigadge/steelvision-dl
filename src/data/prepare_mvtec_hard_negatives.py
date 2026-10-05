from pathlib import Path
import random
import shutil


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SOURCE_ROOT = Path(
    "data/domain_sources"
)

TARGET_ROOT = Path(
    "data/domain_validation"
)

RANDOM_SEED = 42

# We intentionally use only defect-free "good" images.
MVTec_CATEGORIES = [
    "grid",
    "hazelnut",
]

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
}

# Keep the domain-validation dataset roughly balanced.
TARGET_COUNTS = {
    "train": 300,
    "val": 60,
    "test": 60,
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def collect_good_images(
    category: str,
) -> list[Path]:
    category_root = (
        SOURCE_ROOT
        / category
    )

    good_images = []

    # MVTec AD:
    # train/good = defect-free training images
    # test/good  = defect-free test images

    possible_good_dirs = [
        category_root
        / "train"
        / "good",

        category_root
        / "test"
        / "good",
    ]

    for folder in possible_good_dirs:
        if not folder.exists():
            continue

        for image_path in sorted(
            folder.iterdir()
        ):
            if (
                image_path.is_file()
                and image_path.suffix.lower()
                in IMAGE_EXTENSIONS
            ):
                good_images.append(
                    image_path
                )

    return good_images


def copy_image(
    source_path: Path,
    target_path: Path,
) -> None:
    shutil.copy2(
        source_path,
        target_path,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    rng = random.Random(
        RANDOM_SEED
    )

    print()
    print(
        "=" * 76
    )

    print(
        "STEELVISION DOMAIN VALIDATION — MVTEC HARD NEGATIVES"
    )

    print(
        "=" * 76
    )

    all_images = []

    # -------------------------------------------------------------------------
    # Collect defect-free industrial images
    # -------------------------------------------------------------------------

    for category in MVTec_CATEGORIES:
        images = collect_good_images(
            category
        )

        print(
            f"{category:<10} → "
            f"{len(images)} defect-free images found"
        )

        for image_path in images:
            all_images.append(
                (
                    category,
                    image_path,
                )
            )

    total_required = sum(
        TARGET_COUNTS.values()
    )

    print()
    print(
        f"Total defect-free MVTec images available: "
        f"{len(all_images)}"
    )

    print(
        f"Images required for this experiment: "
        f"{total_required}"
    )

    if len(all_images) < total_required:
        raise RuntimeError(
            "Not enough defect-free MVTec images were found. "
            f"Found {len(all_images)}, "
            f"but {total_required} are required."
        )

    # -------------------------------------------------------------------------
    # Shuffle reproducibly
    # -------------------------------------------------------------------------

    rng.shuffle(
        all_images
    )

    selected = all_images[
        :total_required
    ]

    train_end = (
        TARGET_COUNTS[
            "train"
        ]
    )

    val_end = (
        train_end
        + TARGET_COUNTS[
            "val"
        ]
    )

    split_map = {
        "train":
            selected[
                :train_end
            ],

        "val":
            selected[
                train_end:
                val_end
            ],

        "test":
            selected[
                val_end:
            ],
    }

    # -------------------------------------------------------------------------
    # Copy into existing unsupported folders
    #
    # IMPORTANT:
    # We do NOT clear these folders because they already contain Caltech images.
    # -------------------------------------------------------------------------

    print()
    print(
        "ADDING HARD NEGATIVES"
    )

    print(
        "-" * 76
    )

    for split in [
        "train",
        "val",
        "test",
    ]:
        target_dir = (
            TARGET_ROOT
            / split
            / "unsupported"
        )

        target_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        copied_count = 0

        for index, (
            category,
            source_path,
        ) in enumerate(
            split_map[
                split
            ],
            start=1,
        ):
            target_name = (
                f"mvtec_{category}__"
                f"{index:04d}"
                f"{source_path.suffix.lower()}"
            )

            target_path = (
                target_dir
                / target_name
            )

            copy_image(
                source_path,
                target_path,
            )

            copied_count += 1

        total_unsupported = len(
            [
                path
                for path
                in target_dir.iterdir()
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in IMAGE_EXTENSIONS
                )
            ]
        )

        print(
            f"{split:<5} → "
            f"{copied_count} MVTec images added "
            f"| total unsupported now: "
            f"{total_unsupported}"
        )

    print()
    print(
        "=" * 76
    )

    print(
        "HARD-NEGATIVE PREPARATION COMPLETE"
    )

    print(
        "=" * 76
    )

    print()

    print(
        "Expected dataset composition:"
    )

    print(
        "train → "
        "1000 Caltech + 300 MVTec = "
        "about 1300 unsupported"
    )

    print(
        "val   → "
        "200 Caltech + 60 MVTec = "
        "about 260 unsupported"
    )

    print(
        "test  → "
        "200 Caltech + 60 MVTec = "
        "about 260 unsupported"
    )

    print()

    print(
        "NOTE:"
    )

    print(
        "These MVTec images are industrial hard negatives, "
        "not a dedicated clean-steel dataset."
    )

    print()


if __name__ == "__main__":
    main()
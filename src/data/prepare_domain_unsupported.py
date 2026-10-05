from pathlib import Path
import random

from torchvision.datasets import Caltech101


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DOWNLOAD_ROOT = Path(
    "data/domain_sources"
)

TARGET_ROOT = Path(
    "data/domain_validation"
)

RANDOM_SEED = 42

TARGET_COUNTS = {
    "train": 1000,
    "val": 200,
    "test": 200,
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def clear_target_folder(
    folder: Path,
) -> None:
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in folder.iterdir():
        if path.is_file():
            path.unlink()


def save_dataset_image(
    image,
    target_path: Path,
) -> None:
    image = image.convert(
        "RGB"
    )

    image.save(
        target_path,
        format="JPEG",
        quality=95,
    )


def split_indices(
    total: int,
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int,
):
    indices = list(
        range(
            total
        )
    )

    rng = random.Random(
        seed
    )

    rng.shuffle(
        indices
    )

    required = (
        train_count
        + val_count
        + test_count
    )

    if total < required:
        raise RuntimeError(
            f"Caltech-101 contains only {total} images, "
            f"but {required} are required."
        )

    train_indices = indices[
        :train_count
    ]

    val_start = train_count

    val_end = (
        val_start
        + val_count
    )

    val_indices = indices[
        val_start:val_end
    ]

    test_indices = indices[
        val_end:
        val_end
        + test_count
    ]

    return {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print()
    print(
        "=" * 72
    )

    print(
        "STEELVISION DOMAIN VALIDATION — GENERIC UNSUPPORTED DATA"
    )

    print(
        "=" * 72
    )

    print()
    print(
        "Loading Caltech-101..."
    )

    dataset = Caltech101(
        root=str(
            DOWNLOAD_ROOT
        ),
        download=True,
    )

    print(
        f"Caltech-101 images available: "
        f"{len(dataset)}"
    )

    split_map = split_indices(
        total=len(
            dataset
        ),
        train_count=TARGET_COUNTS[
            "train"
        ],
        val_count=TARGET_COUNTS[
            "val"
        ],
        test_count=TARGET_COUNTS[
            "test"
        ],
        seed=RANDOM_SEED,
    )

    print()
    print(
        "TARGET COUNTS"
    )

    print(
        "-" * 72
    )

    for split in [
        "train",
        "val",
        "test",
    ]:
        print(
            f"{split:<5} → "
            f"{TARGET_COUNTS[split]} images"
        )

    print()

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

        clear_target_folder(
            target_dir
        )

        indices = split_map[
            split
        ]

        for output_index, dataset_index in enumerate(
            indices,
            start=1,
        ):
            image, _ = dataset[
                dataset_index
            ]

            target_path = (
                target_dir
                / (
                    f"caltech__"
                    f"{output_index:04d}.jpg"
                )
            )

            save_dataset_image(
                image,
                target_path,
            )

        final_count = len(
            list(
                target_dir.glob(
                    "*.jpg"
                )
            )
        )

        print(
            f"{split:<5} → "
            f"{final_count} unsupported images"
        )

    print()
    print(
        "-" * 72
    )

    total = sum(
        TARGET_COUNTS.values()
    )

    print(
        f"Total generic unsupported images: "
        f"{total}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This dataset contains generic non-domain images only."
    )

    print(
        "Clean steel and plain metal hard negatives "
        "will be added separately."
    )

    print(
        "=" * 72
    )

    print()


if __name__ == "__main__":
    main()
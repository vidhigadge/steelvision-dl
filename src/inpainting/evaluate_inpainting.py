"""Evaluate inpainting reliability for EXP-016."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.inpainting.common import (
    CLASS_NAMES,
    attention_center,
    classification_correct,
    class_flip,
    confidence_change,
    entropy_change,
    generate_gradcam,
    get_device,
    list_images,
    load_classifier,
    load_grayscale_image,
    predict_image,
    seed_everything,
)
from src.inpainting.inpainters import (
    available_methods,
    load_unet_inpainter,
    run_inpainter,
)
from src.inpainting.masks import (
    make_mask,
    mask_coverage,
)
from src.inpainting.metrics import (
    evaluate_reconstruction,
)


DEFAULT_METHODS = (
    "mean_fill",
    "telea",
    "navier_stokes",
)

DEFAULT_MASK_TYPES = (
    "block",
)

DEFAULT_MASK_RATIOS = (
    0.10,
    0.20,
)

DEFAULT_PLACEMENTS = (
    "random",
)


def parse_float_list(
    value: str,
) -> list[float]:
    return [
        float(item.strip())
        for item in value.split(",")
        if item.strip()
    ]


def parse_string_list(
    value: str,
) -> list[str]:
    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


def select_balanced_images(
    images: list[tuple[Path, str]],
    limit: int,
) -> list[tuple[Path, str]]:
    """Select images as evenly as possible across six classes."""
    if limit <= 0:
        return images

    grouped: dict[
        str,
        list[tuple[Path, str]],
    ] = defaultdict(list)

    for item in images:
        grouped[item[1]].append(item)

    selected = []

    class_positions = {
        class_name: 0
        for class_name in CLASS_NAMES
    }

    while len(selected) < limit:
        added_this_round = False

        for class_name in CLASS_NAMES:
            if len(selected) >= limit:
                break

            class_items = grouped.get(
                class_name,
                [],
            )

            position = class_positions[
                class_name
            ]

            if position >= len(class_items):
                continue

            selected.append(
                class_items[position]
            )

            class_positions[
                class_name
            ] += 1

            added_this_round = True

        if not added_this_round:
            break

    return selected


def choose_center(
    placement: str,
    heatmap: np.ndarray | None = None,
) -> tuple[int, int] | None:
    """Choose the center used to generate a mask."""
    if placement == "random":
        return None

    if placement in (
        "high_attention",
        "low_attention",
    ):
        if heatmap is None:
            raise ValueError(
                f"Grad-CAM heatmap required for {placement}"
            )

        return attention_center(
            heatmap,
            placement,
        )

    raise ValueError(
        f"Unknown placement: {placement}"
    )


def evaluate_method(
    *,
    image_path: Path,
    ground_truth_class: str,
    original_image: np.ndarray,
    original_prediction: dict,
    mask: np.ndarray,
    mask_type: str,
    mask_ratio: float,
    placement: str,
    method: str,
    classifier,
    device,
    unet_model=None,
) -> dict:
    """Evaluate one method using the shared condition mask."""
    reconstructed = run_inpainter(
        method=method,
        image=original_image,
        mask=mask,
        unet_model=unet_model,
        device=device,
    )

    reconstructed_prediction = predict_image(
        model=classifier,
        image=reconstructed,
        device=device,
    )

    reconstruction_metrics = evaluate_reconstruction(
        original=original_image,
        reconstructed=reconstructed,
        mask=mask,
    )

    return {
        "image_name":
            image_path.name,

        "image_path":
            str(image_path),

        "ground_truth_class":
            ground_truth_class,

        "original_predicted_class":
            original_prediction[
                "predicted_class"
            ],

        "original_correct":
            int(
                classification_correct(
                    original_prediction,
                    ground_truth_class,
                )
            ),

        "original_confidence":
            original_prediction[
                "confidence"
            ],

        "original_entropy":
            original_prediction[
                "entropy"
            ],

        "mask_type":
            mask_type,

        "mask_ratio_target":
            mask_ratio,

        "mask_ratio_actual":
            mask_coverage(mask),

        "placement":
            placement,

        "method":
            method,

        "reconstructed_predicted_class":
            reconstructed_prediction[
                "predicted_class"
            ],

        "reconstructed_correct":
            int(
                classification_correct(
                    reconstructed_prediction,
                    ground_truth_class,
                )
            ),

        "class_flip":
            int(
                class_flip(
                    original_prediction,
                    reconstructed_prediction,
                )
            ),

        "reconstructed_confidence":
            reconstructed_prediction[
                "confidence"
            ],

        "confidence_change":
            confidence_change(
                original_prediction,
                reconstructed_prediction,
            ),

        "reconstructed_entropy":
            reconstructed_prediction[
                "entropy"
            ],

        "entropy_change":
            entropy_change(
                original_prediction,
                reconstructed_prediction,
            ),

        "psnr":
            reconstruction_metrics[
                "psnr"
            ],

        "ssim":
            reconstruction_metrics[
                "ssim"
            ],

        "mae":
            reconstruction_metrics[
                "mae"
            ],

        "full_image_mae":
            reconstruction_metrics[
                "full_image_mae"
            ],
    }


def write_rows(
    rows: list[dict],
    output_csv: Path,
) -> None:
    if not rows:
        raise RuntimeError(
            "No evaluation rows were produced."
        )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        rows[0].keys()
    )

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path(
            "data/splits/test"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/inpainting/results.csv"
        ),
    )

    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(
            DEFAULT_METHODS
        ),
    )

    parser.add_argument(
        "--mask-types",
        type=str,
        default=",".join(
            DEFAULT_MASK_TYPES
        ),
    )

    parser.add_argument(
        "--mask-ratios",
        type=str,
        default=",".join(
            str(value)
            for value
            in DEFAULT_MASK_RATIOS
        ),
    )

    parser.add_argument(
        "--placements",
        type=str,
        default=",".join(
            DEFAULT_PLACEMENTS
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--balanced",
        action="store_true",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--unet-checkpoint",
        type=Path,
        default=Path(
            "models/inpainting_unet.pth"
        ),
    )

    args = parser.parse_args()

    methods = parse_string_list(
        args.methods
    )

    mask_types = parse_string_list(
        args.mask_types
    )

    mask_ratios = parse_float_list(
        args.mask_ratios
    )

    placements = parse_string_list(
        args.placements
    )

    valid_methods = set(
        available_methods()
    )

    for method in methods:
        if method not in valid_methods:
            raise ValueError(
                f"Unknown method: {method}"
            )

    valid_placements = {
        "random",
        "high_attention",
        "low_attention",
    }

    for placement in placements:
        if placement not in valid_placements:
            raise ValueError(
                f"Unknown placement: {placement}"
            )

    rng = seed_everything(
        args.seed
    )

    device = get_device()

    print(
        "Using device:",
        device,
    )

    classifier = load_classifier(
        device=device
    )

    unet_model = None

    if "unet" in methods:
        unet_model = load_unet_inpainter(
            checkpoint_path=args.unet_checkpoint,
            device=device,
        )

    images = list_images(
        args.test_dir
    )

    if args.balanced:
        images = select_balanced_images(
            images=images,
            limit=args.limit,
        )

    elif args.limit > 0:
        images = images[
            :args.limit
        ]

    print(
        "Images:",
        len(images),
    )

    print(
        "Selected classes:",
        [
            class_name
            for _, class_name
            in images
        ],
    )

    print(
        "Methods:",
        methods,
    )

    print(
        "Mask types:",
        mask_types,
    )

    print(
        "Mask ratios:",
        mask_ratios,
    )

    print(
        "Placements:",
        placements,
    )

    print(
        "Fair-mask mode: same mask reused across methods"
    )

    rows = []

    total_conditions = (
        len(images)
        * len(mask_types)
        * len(mask_ratios)
        * len(placements)
    )

    total_cases = (
        total_conditions
        * len(methods)
    )

    case_number = 0
    condition_number = 0

    for (
        image_path,
        ground_truth_class,
    ) in images:
        original_image = load_grayscale_image(
            image_path
        )

        original_prediction = predict_image(
            model=classifier,
            image=original_image,
            device=device,
        )

        heatmap = None

        if any(
            placement
            in (
                "high_attention",
                "low_attention",
            )
            for placement
            in placements
        ):
            heatmap = generate_gradcam(
                model=classifier,
                image=original_image,
                device=device,
                target_class=original_prediction[
                    "predicted_index"
                ],
            )

        for mask_type in mask_types:
            for mask_ratio in mask_ratios:
                for placement in placements:
                    condition_number += 1

                    center = choose_center(
                        placement=placement,
                        heatmap=heatmap,
                    )

                    mask = make_mask(
                        kind=mask_type,
                        ratio=mask_ratio,
                        rng=rng,
                        size=224,
                        center=center,
                    )

                    actual_coverage = mask_coverage(
                        mask
                    )

                    print()
                    print(
                        f"Condition "
                        f"{condition_number}/"
                        f"{total_conditions} | "
                        f"{image_path.name} | "
                        f"{ground_truth_class} | "
                        f"{mask_type} | "
                        f"target={mask_ratio:.2f} | "
                        f"actual={actual_coverage:.3f} | "
                        f"{placement}"
                    )

                    if center is not None:
                        print(
                            "  center:",
                            center,
                        )

                    for method in methods:
                        case_number += 1

                        print(
                            f"  [{case_number}/"
                            f"{total_cases}] "
                            f"{method}"
                        )

                        row = evaluate_method(
                            image_path=image_path,
                            ground_truth_class=ground_truth_class,
                            original_image=original_image,
                            original_prediction=original_prediction,
                            mask=mask,
                            mask_type=mask_type,
                            mask_ratio=mask_ratio,
                            placement=placement,
                            method=method,
                            classifier=classifier,
                            device=device,
                            unet_model=unet_model,
                        )

                        rows.append(row)

    write_rows(
        rows=rows,
        output_csv=args.output,
    )

    flip_count = sum(
        row["class_flip"]
        for row in rows
    )

    reconstructed_correct = sum(
        row[
            "reconstructed_correct"
        ]
        for row in rows
    )

    print()
    print(
        "Evaluation complete."
    )

    print(
        "Conditions:",
        total_conditions,
    )

    print(
        "Cases:",
        len(rows),
    )

    print(
        "Class flips:",
        flip_count,
    )

    print(
        "Flip rate:",
        f"{flip_count / len(rows):.4f}",
    )

    print(
        "Reconstructed-case accuracy:",
        f"{reconstructed_correct / len(rows):.4f}",
    )

    print(
        "Saved:",
        args.output,
    )


if __name__ == "__main__":
    main()
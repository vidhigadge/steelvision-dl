from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from torchvision import datasets, models, transforms


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MODEL_PATH = Path(
    "models/domain_validator_mobilenet_v3_small.pth"
)

DATA_ROOT = Path(
    "data/domain_validation"
)

IMAGE_SIZE = 224

THRESHOLDS = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.85,
    0.90,
    0.95,
    0.97,
    0.98,
    0.99,
]


# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------

def get_device():
    if torch.backends.mps.is_available():
        return torch.device(
            "mps"
        )

    if torch.cuda.is_available():
        return torch.device(
            "cuda"
        )

    return torch.device(
        "cpu"
    )


# -----------------------------------------------------------------------------
# Transform
# -----------------------------------------------------------------------------

TRANSFORM = transforms.Compose(
    [
        transforms.Grayscale(
            num_output_channels=3
        ),

        transforms.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

def load_model(
    device,
):
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model = models.mobilenet_v3_small(
        weights=None
    )

    in_features = (
        model.classifier[
            3
        ].in_features
    )

    model.classifier[
        3
    ] = nn.Linear(
        in_features,
        2,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    class_to_idx = checkpoint[
        "class_to_idx"
    ]

    return (
        model,
        class_to_idx,
    )


# -----------------------------------------------------------------------------
# Probability collection
# -----------------------------------------------------------------------------

def collect_probabilities(
    model,
    dataset,
    device,
    supported_index,
):
    results = []

    with torch.no_grad():
        for image_path, target in dataset.samples:
            image = Image.open(
                image_path
            ).convert(
                "RGB"
            )

            tensor = TRANSFORM(
                image
            ).unsqueeze(
                0
            ).to(
                device
            )

            logits = model(
                tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )[0]

            supported_probability = float(
                probabilities[
                    supported_index
                ]
                .detach()
                .cpu()
                .item()
            )

            results.append(
                {
                    "path":
                        image_path,

                    "target":
                        int(
                            target
                        ),

                    "supported_probability":
                        supported_probability,
                }
            )

    return results


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

def print_distribution(
    label,
    values,
):
    values = np.array(
        values,
        dtype=np.float64,
    )

    print()
    print(
        label
    )

    print(
        "-" * 72
    )

    print(
        f"Count:     "
        f"{len(values)}"
    )

    print(
        f"Minimum:   "
        f"{values.min():.6f}"
    )

    print(
        f"5th pct:   "
        f"{np.percentile(values, 5):.6f}"
    )

    print(
        f"Median:    "
        f"{np.median(values):.6f}"
    )

    print(
        f"95th pct:  "
        f"{np.percentile(values, 95):.6f}"
    )

    print(
        f"Maximum:   "
        f"{values.max():.6f}"
    )

    print(
        f"Mean:      "
        f"{values.mean():.6f}"
    )


# -----------------------------------------------------------------------------
# Threshold evaluation
# -----------------------------------------------------------------------------

def evaluate_threshold(
    results,
    threshold,
    supported_index,
    unsupported_index,
):
    targets = []

    predictions = []

    for item in results:
        probability = item[
            "supported_probability"
        ]

        prediction = (
            supported_index
            if probability >= threshold
            else unsupported_index
        )

        targets.append(
            item[
                "target"
            ]
        )

        predictions.append(
            prediction
        )

    accuracy = accuracy_score(
        targets,
        predictions,
    )

    supported_precision = precision_score(
        targets,
        predictions,
        pos_label=supported_index,
        zero_division=0,
    )

    supported_recall = recall_score(
        targets,
        predictions,
        pos_label=supported_index,
        zero_division=0,
    )

    supported_f1 = f1_score(
        targets,
        predictions,
        pos_label=supported_index,
        zero_division=0,
    )

    matrix = confusion_matrix(
        targets,
        predictions,
        labels=[
            supported_index,
            unsupported_index,
        ],
    )

    false_accepts = int(
        matrix[
            1,
            0
        ]
    )

    false_rejects = int(
        matrix[
            0,
            1
        ]
    )

    return {
        "threshold":
            threshold,

        "accuracy":
            accuracy,

        "supported_precision":
            supported_precision,

        "supported_recall":
            supported_recall,

        "supported_f1":
            supported_f1,

        "false_accepts":
            false_accepts,

        "false_rejects":
            false_rejects,
    }


# -----------------------------------------------------------------------------
# Error inspection
# -----------------------------------------------------------------------------

def print_errors(
    results,
    threshold,
    supported_index,
    unsupported_index,
):
    false_accepts = []

    false_rejects = []

    for item in results:
        probability = item[
            "supported_probability"
        ]

        prediction = (
            supported_index
            if probability >= threshold
            else unsupported_index
        )

        target = item[
            "target"
        ]

        if (
            target == unsupported_index
            and prediction == supported_index
        ):
            false_accepts.append(
                item
            )

        if (
            target == supported_index
            and prediction == unsupported_index
        ):
            false_rejects.append(
                item
            )

    false_accepts.sort(
        key=lambda item:
            item[
                "supported_probability"
            ],
        reverse=True,
    )

    false_rejects.sort(
        key=lambda item:
            item[
                "supported_probability"
            ],
    )

    print()
    print(
        "FALSE ACCEPTS"
    )

    print(
        "-" * 72
    )

    if not false_accepts:
        print(
            "None"
        )

    for item in false_accepts:
        print(
            f"{item['supported_probability']:.6f} "
            f"| {item['path']}"
        )

    print()
    print(
        "FALSE REJECTS"
    )

    print(
        "-" * 72
    )

    if not false_rejects:
        print(
            "None"
        )

    for item in false_rejects:
        print(
            f"{item['supported_probability']:.6f} "
            f"| {item['path']}"
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    device = get_device()

    print()
    print(
        "=" * 72
    )

    print(
        "STEELVISION DOMAIN VALIDATOR — THRESHOLD ANALYSIS"
    )

    print(
        "=" * 72
    )

    print(
        f"Device: {device}"
    )

    (
        model,
        class_to_idx,
    ) = load_model(
        device
    )

    print(
        f"Class mapping: "
        f"{class_to_idx}"
    )

    supported_index = (
        class_to_idx[
            "supported"
        ]
    )

    unsupported_index = (
        class_to_idx[
            "unsupported"
        ]
    )

    val_dataset = datasets.ImageFolder(
        DATA_ROOT / "val",
        transform=None,
    )

    test_dataset = datasets.ImageFolder(
        DATA_ROOT / "test",
        transform=None,
    )

    print()
    print(
        "Collecting validation probabilities..."
    )

    val_results = collect_probabilities(
        model=model,
        dataset=val_dataset,
        device=device,
        supported_index=supported_index,
    )

    print(
        "Collecting test probabilities..."
    )

    test_results = collect_probabilities(
        model=model,
        dataset=test_dataset,
        device=device,
        supported_index=supported_index,
    )

    val_supported = [
        item[
            "supported_probability"
        ]
        for item in val_results
        if item[
            "target"
        ] == supported_index
    ]

    val_unsupported = [
        item[
            "supported_probability"
        ]
        for item in val_results
        if item[
            "target"
        ] == unsupported_index
    ]

    test_supported = [
        item[
            "supported_probability"
        ]
        for item in test_results
        if item[
            "target"
        ] == supported_index
    ]

    test_unsupported = [
        item[
            "supported_probability"
        ]
        for item in test_results
        if item[
            "target"
        ] == unsupported_index
    ]

    print_distribution(
        "VALIDATION — TRUE SUPPORTED",
        val_supported,
    )

    print_distribution(
        "VALIDATION — TRUE UNSUPPORTED",
        val_unsupported,
    )

    print_distribution(
        "TEST — TRUE SUPPORTED",
        test_supported,
    )

    print_distribution(
        "TEST — TRUE UNSUPPORTED",
        test_unsupported,
    )

    # -------------------------------------------------------------------------
    # Validation threshold table
    # -------------------------------------------------------------------------

    print()
    print(
        "=" * 72
    )

    print(
        "VALIDATION THRESHOLD COMPARISON"
    )

    print(
        "=" * 72
    )

    print(
        f"{'Threshold':>10} "
        f"{'Accuracy':>10} "
        f"{'Precision':>10} "
        f"{'Recall':>10} "
        f"{'F1':>10} "
        f"{'FalseAccept':>12} "
        f"{'FalseReject':>12}"
    )

    print(
        "-" * 82
    )

    for threshold in THRESHOLDS:
        metrics = evaluate_threshold(
            results=val_results,
            threshold=threshold,
            supported_index=supported_index,
            unsupported_index=unsupported_index,
        )

        print(
            f"{threshold:>10.2f} "
            f"{metrics['accuracy'] * 100:>9.2f}% "
            f"{metrics['supported_precision'] * 100:>9.2f}% "
            f"{metrics['supported_recall'] * 100:>9.2f}% "
            f"{metrics['supported_f1'] * 100:>9.2f}% "
            f"{metrics['false_accepts']:>12} "
            f"{metrics['false_rejects']:>12}"
        )

    # -------------------------------------------------------------------------
    # Test threshold table
    # -------------------------------------------------------------------------

    print()
    print(
        "=" * 72
    )

    print(
        "TEST THRESHOLD COMPARISON"
    )

    print(
        "=" * 72
    )

    print(
        f"{'Threshold':>10} "
        f"{'Accuracy':>10} "
        f"{'Precision':>10} "
        f"{'Recall':>10} "
        f"{'F1':>10} "
        f"{'FalseAccept':>12} "
        f"{'FalseReject':>12}"
    )

    print(
        "-" * 82
    )

    for threshold in THRESHOLDS:
        metrics = evaluate_threshold(
            results=test_results,
            threshold=threshold,
            supported_index=supported_index,
            unsupported_index=unsupported_index,
        )

        print(
            f"{threshold:>10.2f} "
            f"{metrics['accuracy'] * 100:>9.2f}% "
            f"{metrics['supported_precision'] * 100:>9.2f}% "
            f"{metrics['supported_recall'] * 100:>9.2f}% "
            f"{metrics['supported_f1'] * 100:>9.2f}% "
            f"{metrics['false_accepts']:>12} "
            f"{metrics['false_rejects']:>12}"
        )

    # -------------------------------------------------------------------------
    # Default threshold errors
    # -------------------------------------------------------------------------

    print()
    print(
        "=" * 72
    )

    print(
        "ERRORS AT DEFAULT THRESHOLD 0.50"
    )

    print(
        "=" * 72
    )

    print_errors(
        results=test_results,
        threshold=0.50,
        supported_index=supported_index,
        unsupported_index=unsupported_index,
    )

    print()
    print(
        "=" * 72
    )

    print(
        "THRESHOLD ANALYSIS COMPLETE"
    )

    print(
        "=" * 72
    )

    print()


if __name__ == "__main__":
    main()
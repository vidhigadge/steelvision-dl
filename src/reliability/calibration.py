from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.resnet50_finetune import create_resnet50_finetune


MODEL_PATH = Path("models/resnet50_finetune.pth")
TEST_DIR = Path("data/splits/test")

NUM_BINS = 10

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def expected_calibration_error(
    confidences,
    correct,
    num_bins=10,
):
    bin_edges = np.linspace(
        0.0,
        1.0,
        num_bins + 1,
    )

    ece = 0.0

    print(
        f"{'Bin':<12}"
        f"{'Count':<8}"
        f"{'Confidence':<14}"
        f"{'Accuracy':<12}"
        f"{'Gap':<10}"
    )

    print("-" * 56)

    for i in range(num_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == 0:
            mask = (
                (confidences >= lower)
                & (confidences <= upper)
            )
        else:
            mask = (
                (confidences > lower)
                & (confidences <= upper)
            )

        count = mask.sum()

        if count == 0:
            continue

        bin_confidence = confidences[mask].mean()
        bin_accuracy = correct[mask].mean()

        gap = abs(
            bin_accuracy - bin_confidence
        )

        ece += (
            count / len(confidences)
        ) * gap

        print(
            f"{lower:.1f}-{upper:.1f}   "
            f"{count:<8}"
            f"{bin_confidence:<14.4f}"
            f"{bin_accuracy:<12.4f}"
            f"{gap:<10.4f}"
        )

    return ece


def main():
    device = get_device()

    transform = transforms.Compose([
        transforms.Grayscale(
            num_output_channels=3
        ),
        transforms.Resize(
            (224, 224)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
    )

    model = create_resnet50_finetune(
        num_classes=len(CLASS_NAMES)
    ).to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
        )
    )

    model.eval()

    all_confidences = []
    all_correct = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            confidences, predictions = torch.max(
                probabilities,
                dim=1,
            )

            correct = predictions.eq(labels)

            all_confidences.extend(
                confidences.cpu().tolist()
            )

            all_correct.extend(
                correct.cpu().tolist()
            )

    confidences = np.array(
        all_confidences
    )

    correct = np.array(
        all_correct,
        dtype=float,
    )

    ece = expected_calibration_error(
        confidences,
        correct,
        NUM_BINS,
    )

    print()
    print(
        f"Overall accuracy: "
        f"{correct.mean():.4f}"
    )

    print(
        f"Mean confidence: "
        f"{confidences.mean():.4f}"
    )

    print(
        f"Expected Calibration Error: "
        f"{ece:.4f}"
    )


if __name__ == "__main__":
    main()
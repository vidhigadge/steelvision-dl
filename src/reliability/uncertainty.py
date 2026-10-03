from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.resnet50_finetune import create_resnet50_finetune


MODEL_PATH = Path("models/resnet50_finetune.pth")
TEST_DIR = Path("data/splits/test")

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


def predictive_entropy(probabilities):
    return -torch.sum(
        probabilities * torch.log(probabilities + 1e-12),
        dim=1,
    )


def main():
    device = get_device()

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
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
    all_entropies = []
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

            entropies = predictive_entropy(
                probabilities
            )

            correct = predictions.eq(labels)

            all_confidences.extend(
                confidences.cpu().tolist()
            )

            all_entropies.extend(
                entropies.cpu().tolist()
            )

            all_correct.extend(
                correct.cpu().tolist()
            )

    confidences = np.array(all_confidences)
    entropies = np.array(all_entropies)
    correct = np.array(all_correct, dtype=bool)

    incorrect = ~correct

    print("Total test images:", len(dataset))
    print("Correct predictions:", correct.sum())
    print("Incorrect predictions:", incorrect.sum())
    print()

    print(
        f"Mean confidence: {confidences.mean():.4f}"
    )

    print(
        f"Mean entropy: {entropies.mean():.4f}"
    )

    print()

    if correct.any():
        print(
            "Correct prediction mean confidence:",
            f"{confidences[correct].mean():.4f}",
        )

        print(
            "Correct prediction mean entropy:",
            f"{entropies[correct].mean():.4f}",
        )

    if incorrect.any():
        print(
            "Incorrect prediction mean confidence:",
            f"{confidences[incorrect].mean():.4f}",
        )

        print(
            "Incorrect prediction mean entropy:",
            f"{entropies[incorrect].mean():.4f}",
        )

        print()
        print("Most uncertain incorrect predictions:")

        incorrect_indices = np.where(incorrect)[0]

        ranked = incorrect_indices[
            np.argsort(
                entropies[incorrect_indices]
            )[::-1]
        ]

        for index in ranked[:10]:
            image_path, true_class = dataset.samples[index]

            print()
            print("Image:", Path(image_path).name)
            print(
                "True class:",
                dataset.classes[true_class],
            )
            print(
                f"Confidence: {confidences[index]:.4f}"
            )
            print(
                f"Entropy: {entropies[index]:.4f}"
            )


if __name__ == "__main__":
    main()
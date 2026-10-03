import sys
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.models.resnet50 import create_resnet50


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def create_test_loader(resolution, batch_size=32):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),

        transforms.Resize(
            (resolution, resolution),
            interpolation=InterpolationMode.BICUBIC,
        ),

        transforms.Resize(
            (224, 224),
            interpolation=InterpolationMode.BICUBIC,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = datasets.ImageFolder(
        "data/splits/test",
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )


def evaluate(model, loader, device):
    predictions = []
    labels_all = []

    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)

            predicted = outputs.argmax(dim=1)

            predictions.extend(
                predicted.cpu().tolist()
            )

            labels_all.extend(
                labels.tolist()
            )

    return accuracy_score(
        labels_all,
        predictions,
    )


def main():
    device = get_device()

    model = create_resnet50(
        num_classes=6,
    ).to(device)

    model.load_state_dict(
        torch.load(
            "models/resnet50_finetune.pth",
            map_location=device,
        )
    )

    resolutions = [
        224,
        128,
        64,
        32,
    ]

    for resolution in resolutions:
        loader = create_test_loader(
            resolution,
        )

        accuracy = evaluate(
            model,
            loader,
            device,
        )

        print(
            f"{resolution}x{resolution} "
            f"-> Accuracy: {accuracy:.4f}"
        )


if __name__ == "__main__":
    main()
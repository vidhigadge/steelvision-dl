import sys
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.models.resnet50_finetune import create_resnet50_finetune


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/evaluation/evaluate_sr.py <resolution>")
        print("Example: python src/evaluation/evaluate_sr.py 128x128")
        return

    resolution = sys.argv[1]

    device = get_device()

    print("Using device:", device)
    print("Evaluating resolution:", resolution)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    data_path = Path(
        f"data/sr/realesrgan_ncnn/{resolution}"
    )

    if not data_path.exists():
        print(f"Dataset folder not found: {data_path}")
        return

    dataset = datasets.ImageFolder(
        data_path,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
    )

    model = create_resnet50_finetune(
        num_classes=6
    ).to(device)

    model.load_state_dict(
        torch.load(
            "models/resnet50_finetune.pth",
            map_location=device,
        )
    )

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    print()
    print(f"Real-ESRGAN NCNN Accuracy: {accuracy:.4f}")
    print()

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=dataset.classes,
            digits=4,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
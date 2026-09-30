import sys
from pathlib import Path

import torch
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.data.loaders import create_dataloaders
from src.models.cnn import SimpleCNN


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_device()

_, _, test_loader = create_dataloaders(batch_size=32)

model = SimpleCNN(num_classes=6).to(device)

model.load_state_dict(
    torch.load(
        "models/simple_cnn.pth",
        map_location=device,
    )
)

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_labels.extend(
            labels.tolist()
        )

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=test_loader.dataset.classes,
    )
)

from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(all_labels, all_predictions)

print(f"Test Accuracy: {accuracy:.4f}")
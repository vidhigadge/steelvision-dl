from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def create_dataloaders(
    data_dir="data/splits",
    batch_size=32,
):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    data_dir = Path(data_dir)

    train_dataset = datasets.ImageFolder(
        data_dir / "train",
        transform=transform,
    )

    val_dataset = datasets.ImageFolder(
        data_dir / "val",
        transform=transform,
    )

    test_dataset = datasets.ImageFolder(
        data_dir / "test",
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader
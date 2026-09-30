from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def create_dataloaders(
    data_dir="data/splits",
    batch_size=32,
    normalize=False,
):
    transform_list = [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]

    if normalize:
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )

    transform = transforms.Compose(transform_list)

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
"""Train the EXP-016 domain-specific U-Net inpainter."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.inpainting.common import (
    IMAGE_SIZE,
    get_device,
    seed_everything,
)
from src.inpainting.inpainters import (
    prepare_unet_input,
)
from src.inpainting.masks import (
    random_training_mask,
)
from src.inpainting.unet_model import (
    create_inpainting_unet,
)


class InpaintingDataset(Dataset):
    """Create synthetic masked training examples from clean images."""

    def __init__(
        self,
        image_paths: list[Path],
        seed: int = 42,
    ) -> None:
        self.image_paths = image_paths
        self.seed = seed

    def __len__(self) -> int:
        return len(
            self.image_paths
        )

    def __getitem__(
        self,
        index: int,
    ):
        image_path = self.image_paths[
            index
        ]

        image = (
            Image.open(
                image_path
            )
            .convert(
                "L"
            )
            .resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                Image.Resampling.BICUBIC,
            )
        )

        image_np = np.asarray(
            image,
            dtype=np.uint8,
        )

        rng = np.random.default_rng(
            self.seed
            + index
            + np.random.randint(
                0,
                1_000_000,
            )
        )

        mask = random_training_mask(
            rng=rng,
            size=IMAGE_SIZE,
        )

        input_tensor = prepare_unet_input(
            image=image_np,
            mask=mask,
        )

        target_tensor = torch.from_numpy(
            image_np.astype(
                np.float32
            )
            / 255.0
        ).unsqueeze(
            0
        )

        mask_tensor = torch.from_numpy(
            (
                mask > 0
            ).astype(
                np.float32
            )
        ).unsqueeze(
            0
        )

        return (
            input_tensor,
            target_tensor,
            mask_tensor,
        )


def list_image_paths(
    root: Path,
) -> list[Path]:
    """Collect image files from ImageFolder-style class directories."""
    valid_suffixes = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    }

    paths = []

    for path in sorted(
        root.rglob(
            "*"
        )
    ):
        if (
            path.is_file()
            and path.suffix.lower()
            in valid_suffixes
        ):
            paths.append(
                path
            )

    return paths


def calculate_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Weighted reconstruction loss.

    Missing-region error receives most of the weight because that
    is the region the U-Net is expected to reconstruct.

    A small full-image term encourages overall stability.
    """
    hole_pixels = mask.sum().clamp(
        min=1.0
    )

    visible_mask = (
        1.0
        - mask
    )

    visible_pixels = (
        visible_mask.sum().clamp(
            min=1.0
        )
    )

    hole_loss = (
        torch.abs(
            prediction
            - target
        )
        * mask
    ).sum() / hole_pixels

    visible_loss = (
        torch.abs(
            prediction
            - target
        )
        * visible_mask
    ).sum() / visible_pixels

    total_loss = (
        hole_loss
        + 0.10
        * visible_loss
    )

    return total_loss


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
) -> float:
    """Run one training or validation epoch."""
    is_training = (
        optimizer is not None
    )

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_items = 0

    for (
        inputs,
        targets,
        masks,
    ) in loader:
        inputs = inputs.to(
            device
        )

        targets = targets.to(
            device
        )

        masks = masks.to(
            device
        )

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(
            is_training
        ):
            predictions = model(
                inputs
            )

            loss = calculate_loss(
                prediction=predictions,
                target=targets,
                mask=masks,
            )

            if is_training:
                loss.backward()

                optimizer.step()

        batch_size = inputs.size(
            0
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_items += batch_size

    return (
        total_loss
        / total_items
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path(
            "data/splits/train"
        ),
    )

    parser.add_argument(
        "--val-dir",
        type=Path,
        default=Path(
            "data/splits/val"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "models/inpainting_unet.pth"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    seed_everything(
        args.seed
    )

    device = get_device()

    print(
        "Using device:",
        device,
    )

    train_paths = list_image_paths(
        args.train_dir
    )

    val_paths = list_image_paths(
        args.val_dir
    )

    if not train_paths:
        raise RuntimeError(
            f"No training images found in "
            f"{args.train_dir}"
        )

    if not val_paths:
        raise RuntimeError(
            f"No validation images found in "
            f"{args.val_dir}"
        )

    print(
        "Training images:",
        len(
            train_paths
        ),
    )

    print(
        "Validation images:",
        len(
            val_paths
        ),
    )

    train_dataset = InpaintingDataset(
        image_paths=train_paths,
        seed=args.seed,
    )

    val_dataset = InpaintingDataset(
        image_paths=val_paths,
        seed=args.seed
        + 10_000,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = create_inpainting_unet()

    model.to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    best_val_loss = float(
        "inf"
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(
        1,
        args.epochs
        + 1,
    ):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
        )

        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
        )

        print(
            f"Epoch {epoch:02d}/"
            f"{args.epochs:02d} | "
            f"Train Loss: "
            f"{train_loss:.6f} | "
            f"Val Loss: "
            f"{val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                args.output,
            )

            print(
                "Saved best checkpoint:",
                args.output,
            )

    print(
        "Best validation loss:",
        f"{best_val_loss:.6f}",
    )


if __name__ == "__main__":
    main()
from pathlib import Path
import copy
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DATA_ROOT = Path(
    "data/domain_validation"
)

MODEL_OUTPUT = Path(
    "models/domain_validator_mobilenet_v3_small.pth"
)

METRICS_OUTPUT = Path(
    "outputs/domain_validator_metrics.json"
)

RANDOM_SEED = 42

IMAGE_SIZE = 224

BATCH_SIZE = 32

NUM_EPOCHS = 10

LEARNING_RATE = 1e-4

NUM_WORKERS = 0


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

def set_seed(
    seed: int,
) -> None:
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(
            seed
        )


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
# Transforms
# -----------------------------------------------------------------------------

def build_transforms():
    train_transform = transforms.Compose(
        [
            # Important:
            # Force BOTH supported and unsupported images
            # into the same grayscale representation.
            transforms.Grayscale(
                num_output_channels=3
            ),

            transforms.Resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                )
            ),

            transforms.RandomHorizontalFlip(
                p=0.5
            ),

            transforms.RandomVerticalFlip(
                p=0.5
            ),

            transforms.RandomRotation(
                degrees=10
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

    eval_transform = transforms.Compose(
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

    return (
        train_transform,
        eval_transform,
    )


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

def build_dataloaders():
    (
        train_transform,
        eval_transform,
    ) = build_transforms()

    train_dataset = datasets.ImageFolder(
        DATA_ROOT / "train",
        transform=train_transform,
    )

    val_dataset = datasets.ImageFolder(
        DATA_ROOT / "val",
        transform=eval_transform,
    )

    test_dataset = datasets.ImageFolder(
        DATA_ROOT / "test",
        transform=eval_transform,
    )

    print()
    print(
        "CLASS MAPPING"
    )
    print(
        "-" * 72
    )

    print(
        train_dataset.class_to_idx
    )

    expected_classes = {
        "supported",
        "unsupported",
    }

    actual_classes = set(
        train_dataset.classes
    )

    if actual_classes != expected_classes:
        raise RuntimeError(
            "Expected exactly two classes: "
            "'supported' and 'unsupported'. "
            f"Found: {train_dataset.classes}"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    )


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

def build_model():
    weights = (
        models.MobileNet_V3_Small_Weights.DEFAULT
    )

    model = models.mobilenet_v3_small(
        weights=weights
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

    return model


# -----------------------------------------------------------------------------
# Training / evaluation
# -----------------------------------------------------------------------------

def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
):
    training = (
        optimizer
        is not None
    )

    if training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0

    all_targets = []
    all_predictions = []

    for images, targets in loader:
        images = images.to(
            device
        )

        targets = targets.to(
            device
        )

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(
            training
        ):
            logits = model(
                images
            )

            loss = criterion(
                logits,
                targets,
            )

            if training:
                loss.backward()

                optimizer.step()

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        running_loss += (
            loss.item()
            * images.size(
                0
            )
        )

        all_targets.extend(
            targets
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_predictions.extend(
            predictions
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

    epoch_loss = (
        running_loss
        / len(
            loader.dataset
        )
    )

    epoch_accuracy = accuracy_score(
        all_targets,
        all_predictions,
    )

    return (
        epoch_loss,
        epoch_accuracy,
    )


def evaluate_model(
    model,
    loader,
    device,
    class_names,
):
    model.eval()

    all_targets = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(
                device
            )

            logits = model(
                images
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predictions = torch.argmax(
                probabilities,
                dim=1,
            )

            all_targets.extend(
                targets.numpy().tolist()
            )

            all_predictions.extend(
                predictions
                .cpu()
                .numpy()
                .tolist()
            )

            all_probabilities.extend(
                probabilities
                .cpu()
                .numpy()
                .tolist()
            )

    accuracy = accuracy_score(
        all_targets,
        all_predictions,
    )

    report = classification_report(
        all_targets,
        all_predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        all_targets,
        all_predictions,
    )

    return {
        "accuracy":
            accuracy,

        "report":
            report,

        "confusion_matrix":
            matrix.tolist(),

        "targets":
            all_targets,

        "predictions":
            all_predictions,

        "probabilities":
            all_probabilities,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    set_seed(
        RANDOM_SEED
    )

    device = get_device()

    MODEL_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "=" * 72
    )

    print(
        "STEELVISION DOMAIN VALIDATOR TRAINING"
    )

    print(
        "=" * 72
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Image size: "
        f"{IMAGE_SIZE} × {IMAGE_SIZE}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Epochs: "
        f"{NUM_EPOCHS}"
    )

    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = build_dataloaders()

    print()
    print(
        "DATASET"
    )

    print(
        "-" * 72
    )

    print(
        f"Train images: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation images: "
        f"{len(val_dataset)}"
    )

    print(
        f"Test images: "
        f"{len(test_dataset)}"
    )

    print()

    print(
        "Train class counts:"
    )

    for class_name in train_dataset.classes:
        class_index = (
            train_dataset
            .class_to_idx[
                class_name
            ]
        )

        count = sum(
            1
            for _,
            target
            in train_dataset.samples
            if target
            == class_index
        )

        print(
            f"  {class_name:<12} "
            f"{count}"
        )

    model = build_model()

    model = model.to(
        device
    )

    criterion = (
        nn.CrossEntropyLoss()
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_accuracy = 0.0

    best_model_state = None

    history = []

    print()
    print(
        "TRAINING"
    )

    print(
        "-" * 72
    )

    start_time = time.time()

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):
        train_loss, train_accuracy = (
            run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
            )
        )

        val_loss, val_accuracy = (
            run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
            )
        )

        history.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "train_accuracy":
                    train_accuracy,

                "val_loss":
                    val_loss,

                "val_accuracy":
                    val_accuracy,
            }
        )

        print(
            f"Epoch "
            f"{epoch:02d}/{NUM_EPOCHS} "
            f"| "
            f"Train loss "
            f"{train_loss:.4f} "
            f"| "
            f"Train acc "
            f"{train_accuracy * 100:.2f}% "
            f"| "
            f"Val loss "
            f"{val_loss:.4f} "
            f"| "
            f"Val acc "
            f"{val_accuracy * 100:.2f}%"
        )

        if (
            val_accuracy
            > best_val_accuracy
        ):
            best_val_accuracy = (
                val_accuracy
            )

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

    elapsed = (
        time.time()
        - start_time
    )

    if best_model_state is None:
        raise RuntimeError(
            "Training finished without "
            "a valid best model."
        )

    model.load_state_dict(
        best_model_state
    )

    torch.save(
        {
            "model_state_dict":
                best_model_state,

            "class_to_idx":
                train_dataset.class_to_idx,

            "image_size":
                IMAGE_SIZE,

            "architecture":
                "mobilenet_v3_small",

            "grayscale_input":
                True,
        },
        MODEL_OUTPUT,
    )

    print()
    print(
        "=" * 72
    )

    print(
        "BEST MODEL"
    )

    print(
        "=" * 72
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        f"Training time: "
        f"{elapsed:.1f} seconds"
    )

    print(
        f"Saved model: "
        f"{MODEL_OUTPUT}"
    )

    # -------------------------------------------------------------------------
    # Final test evaluation
    # -------------------------------------------------------------------------

    test_results = evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
        class_names=train_dataset.classes,
    )

    print()
    print(
        "=" * 72
    )

    print(
        "TEST RESULTS"
    )

    print(
        "=" * 72
    )

    print(
        f"Test accuracy: "
        f"{test_results['accuracy'] * 100:.2f}%"
    )

    print()

    print(
        "Confusion matrix"
    )

    print(
        "Rows = ground truth"
    )

    print(
        "Columns = prediction"
    )

    print()

    print(
        "                 predicted"
    )

    print(
        "                 supported   unsupported"
    )

    matrix = (
        test_results[
            "confusion_matrix"
        ]
    )

    print(
        f"actual supported   "
        f"{matrix[0][0]:>9}   "
        f"{matrix[0][1]:>11}"
    )

    print(
        f"actual unsupported "
        f"{matrix[1][0]:>9}   "
        f"{matrix[1][1]:>11}"
    )

    print()
    print(
        "CLASSIFICATION REPORT"
    )

    print(
        "-" * 72
    )

    for class_name in train_dataset.classes:
        values = (
            test_results[
                "report"
            ][
                class_name
            ]
        )

        print(
            f"{class_name:<12} "
            f"precision="
            f"{values['precision']:.4f} "
            f"recall="
            f"{values['recall']:.4f} "
            f"f1="
            f"{values['f1-score']:.4f}"
        )

    metrics_to_save = {
        "architecture":
            "mobilenet_v3_small",

        "class_to_idx":
            train_dataset.class_to_idx,

        "best_val_accuracy":
            best_val_accuracy,

        "test_accuracy":
            test_results[
                "accuracy"
            ],

        "confusion_matrix":
            test_results[
                "confusion_matrix"
            ],

        "classification_report":
            test_results[
                "report"
            ],

        "history":
            history,
    }

    with open(
        METRICS_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics_to_save,
            file,
            indent=2,
        )

    print()
    print(
        f"Metrics saved: "
        f"{METRICS_OUTPUT}"
    )

    print()
    print(
        "=" * 72
    )

    print(
        "DOMAIN VALIDATOR TRAINING COMPLETE"
    )

    print(
        "=" * 72
    )

    print()


if __name__ == "__main__":
    main()
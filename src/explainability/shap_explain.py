from pathlib import Path

import cv2
import numpy as np
import shap
import torch
from PIL import Image
from torchvision import transforms

from src.models.resnet50_finetune import create_resnet50_finetune


MODEL_PATH = Path("models/resnet50_finetune.pth")

TRAIN_DIR = Path("data/splits/train")
TEST_DIR = Path("data/splits/test")

OUTPUT_DIR = Path("outputs/shap")

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
]


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model():
    device = torch.device("cpu")

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

    return model


def create_background():
    """
    Use a small set of training images as the SHAP baseline.
    """
    background_images = []

    for class_name in CLASS_NAMES:
        class_dir = TRAIN_DIR / class_name

        image_paths = list(
            class_dir.glob("*.jpg")
        )[:2]

        for image_path in image_paths:
            image = Image.open(
                image_path
            ).convert("RGB")

            tensor = TRANSFORM(image)

            background_images.append(tensor)

    return torch.stack(background_images)


def get_class_shap_values(
    shap_values,
    predicted_class,
):
    """
    Handle different SHAP output formats.
    """
    if isinstance(shap_values, list):
        values = shap_values[predicted_class]

        if values.ndim == 4:
            return values[0]

        return values

    values = np.asarray(shap_values)

    # Newer SHAP:
    # (batch, channels, height, width, outputs)
    if values.ndim == 5:
        return values[
            0,
            :,
            :,
            :,
            predicted_class,
        ]

    # Single-output style:
    # (batch, channels, height, width)
    if values.ndim == 4:
        return values[0]

    raise ValueError(
        f"Unexpected SHAP shape: {values.shape}"
    )


def create_overlay(
    image,
    attribution,
):
    image_rgb = np.float32(
        image.resize((224, 224))
    ) / 255.0

    # Combine RGB-channel contributions.
    heatmap = np.mean(
        np.abs(attribution),
        axis=0,
    )

    heatmap = heatmap - heatmap.min()

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    heatmap = np.uint8(
        255 * heatmap
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET,
    )

    image_bgr = cv2.cvtColor(
        np.uint8(image_rgb * 255),
        cv2.COLOR_RGB2BGR,
    )

    overlay = cv2.addWeighted(
        image_bgr,
        0.55,
        heatmap,
        0.45,
        0,
    )

    return overlay


def main():
    print("Loading model...")

    model = load_model()

    print("Creating SHAP background...")

    background = create_background()

    print(
        "Background images:",
        len(background),
    )

    explainer = shap.GradientExplainer(
        model,
        background,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for class_name in CLASS_NAMES:
        class_dir = TEST_DIR / class_name

        image_path = next(
            class_dir.glob("*.jpg")
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        input_tensor = TRANSFORM(
            image
        ).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)

            probabilities = torch.softmax(
                output,
                dim=1,
            )

            predicted_class = int(
                probabilities.argmax(dim=1)
            )

            confidence = float(
                probabilities[
                    0,
                    predicted_class,
                ]
            )

        print()
        print("Explaining:", image_path.name)

        shap_values = explainer.shap_values(
            input_tensor
        )

        attribution = get_class_shap_values(
            shap_values,
            predicted_class,
        )

        overlay = create_overlay(
            image,
            attribution,
        )

        output_path = OUTPUT_DIR / (
            f"{class_name}_shap.jpg"
        )

        cv2.imwrite(
            str(output_path),
            overlay,
        )

        print("True class:", class_name)

        print(
            "Predicted class:",
            CLASS_NAMES[predicted_class],
        )

        print(
            f"Confidence: {confidence:.4f}"
        )

        print(
            "Saved:",
            output_path,
        )


if __name__ == "__main__":
    main()
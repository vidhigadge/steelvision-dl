from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

from src.models.resnet50_finetune import create_resnet50_finetune


MODEL_PATH = Path("models/resnet50_finetune.pth")
IMAGE_DIR = Path("data/splits/test")
OUTPUT_DIR = Path("outputs/gradcam")

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


def main():
    device = get_device()

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

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    target_layers = [
        model.layer4[-1]
    ]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for class_name in CLASS_NAMES:
        class_dir = IMAGE_DIR / class_name

        image_path = next(
            class_dir.glob("*.jpg")
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        rgb_image = np.float32(
            image.resize((224, 224))
        ) / 255.0

        input_tensor = transform(
            image
        ).unsqueeze(0).to(device)

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
                probabilities[0, predicted_class]
            )

        targets = [
            ClassifierOutputTarget(
                predicted_class
            )
        ]

        with GradCAM(
            model=model,
            target_layers=target_layers,
        ) as cam:
            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=targets,
            )[0]

        visualization = show_cam_on_image(
            rgb_image,
            grayscale_cam,
            use_rgb=True,
        )

        output_path = OUTPUT_DIR / (
            f"{class_name}_gradcam.jpg"
        )

        cv2.imwrite(
            str(output_path),
            cv2.cvtColor(
                visualization,
                cv2.COLOR_RGB2BGR,
            ),
        )

        print()
        print("Image:", image_path.name)
        print("True class:", class_name)
        print(
            "Predicted class:",
            CLASS_NAMES[predicted_class],
        )
        print(
            f"Confidence: {confidence:.4f}"
        )
        print("Saved:", output_path)


if __name__ == "__main__":
    main()
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

from src.models.resnet50_finetune import create_resnet50_finetune


MODEL_PATH = Path(
    "models/resnet50_finetune.pth"
)

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


DEVICE = get_device()


MODEL = create_resnet50_finetune(
    num_classes=len(CLASS_NAMES)
).to(DEVICE)

MODEL.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )
)

MODEL.eval()


TRANSFORM = transforms.Compose([
    transforms.Grayscale(
        num_output_channels=3
    ),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


TARGET_LAYERS = [
    MODEL.layer4[-1]
]


def generate_gradcam(image):
    display_image = (
        image
        .convert("L")
        .convert("RGB")
        .resize((224, 224))
    )

    rgb_image = (
        np.float32(display_image)
        / 255.0
    )

    input_tensor = TRANSFORM(
        image
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = MODEL(
            input_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1,
        )

        predicted_index = int(
            probabilities.argmax(dim=1)
        )

        confidence = float(
            probabilities[
                0,
                predicted_index,
            ]
        )

    targets = [
        ClassifierOutputTarget(
            predicted_index
        )
    ]

    with GradCAM(
        model=MODEL,
        target_layers=TARGET_LAYERS,
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

    output_image = Image.fromarray(
        visualization
    )

    buffer = BytesIO()

    output_image.save(
        buffer,
        format="JPEG",
        quality=95,
    )

    return (
        buffer.getvalue(),
        CLASS_NAMES[predicted_index],
        confidence,
    )
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torchvision import transforms

from src.models.resnet50_finetune import create_resnet50_finetune


PYTORCH_MODEL = Path("models/resnet50_finetune.pth")
ONNX_MODEL = Path("models/resnet50_finetune.onnx")
TEST_DIR = Path("data/splits/test")

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
]


TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def softmax(values):
    values = values - np.max(
        values,
        axis=1,
        keepdims=True,
    )

    exp_values = np.exp(values)

    return exp_values / np.sum(
        exp_values,
        axis=1,
        keepdims=True,
    )


def main():
    device = torch.device("cpu")

    # Load PyTorch model
    pytorch_model = create_resnet50_finetune(
        num_classes=len(CLASS_NAMES)
    ).to(device)

    pytorch_model.load_state_dict(
        torch.load(
            PYTORCH_MODEL,
            map_location=device,
        )
    )

    pytorch_model.eval()

    # Load ONNX model
    session = ort.InferenceSession(
        str(ONNX_MODEL),
        providers=["CPUExecutionProvider"],
    )

    image_path = next(
        TEST_DIR.glob("*/*.jpg")
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    input_tensor = TRANSFORM(
        image
    ).unsqueeze(0)

    # PyTorch inference
    with torch.no_grad():
        pytorch_logits = pytorch_model(
            input_tensor
        )

        pytorch_probs = torch.softmax(
            pytorch_logits,
            dim=1,
        ).cpu().numpy()

    # ONNX inference
    onnx_logits = session.run(
        ["logits"],
        {
            "input": input_tensor.numpy()
        },
    )[0]

    onnx_probs = softmax(
        onnx_logits
    )

    pytorch_class = int(
        np.argmax(pytorch_probs)
    )

    onnx_class = int(
        np.argmax(onnx_probs)
    )

    max_difference = np.max(
        np.abs(
            pytorch_probs - onnx_probs
        )
    )

    print("Image:", image_path.name)
    print()

    print(
        "PyTorch prediction:",
        CLASS_NAMES[pytorch_class],
    )

    print(
        "PyTorch confidence:",
        f"{pytorch_probs[0, pytorch_class]:.6f}",
    )

    print()

    print(
        "ONNX prediction:",
        CLASS_NAMES[onnx_class],
    )

    print(
        "ONNX confidence:",
        f"{onnx_probs[0, onnx_class]:.6f}",
    )

    print()

    print(
        "Maximum probability difference:",
        f"{max_difference:.8f}",
    )

    print(
        "Predictions match:",
        pytorch_class == onnx_class,
    )


if __name__ == "__main__":
    main()
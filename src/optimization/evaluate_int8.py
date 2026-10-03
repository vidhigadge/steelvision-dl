from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import datasets, transforms


FP32_MODEL = Path("models/resnet50_finetune.onnx")
INT8_MODEL = Path(
    "models/resnet50_finetune_uint8.onnx"
)
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


def predict(session, image):
    tensor = TRANSFORM(
        image
    ).unsqueeze(0)

    input_name = session.get_inputs()[0].name

    logits = session.run(
        None,
        {
            input_name: np.ascontiguousarray(
                tensor.numpy(),
                dtype=np.float32,
            )
        },
    )[0]

    probabilities = softmax(
        logits
    )

    prediction = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[0, prediction]
    )

    return prediction, confidence


def main():
    fp32_session = ort.InferenceSession(
        str(FP32_MODEL),
        providers=["CPUExecutionProvider"],
    )

    int8_session = ort.InferenceSession(
        str(INT8_MODEL),
        providers=["CPUExecutionProvider"],
    )

    dataset = datasets.ImageFolder(
        TEST_DIR
    )

    fp32_correct = 0
    int8_correct = 0
    matching_predictions = 0

    fp32_confidences = []
    int8_confidences = []

    for image_path, true_class in dataset.samples:
        image = Image.open(
            image_path
        ).convert("RGB")

        fp32_pred, fp32_conf = predict(
            fp32_session,
            image,
        )

        int8_pred, int8_conf = predict(
            int8_session,
            image,
        )

        fp32_correct += (
            fp32_pred == true_class
        )

        int8_correct += (
            int8_pred == true_class
        )

        matching_predictions += (
            fp32_pred == int8_pred
        )

        fp32_confidences.append(
            fp32_conf
        )

        int8_confidences.append(
            int8_conf
        )

    total = len(dataset)

    fp32_accuracy = (
        fp32_correct / total
    )

    int8_accuracy = (
        int8_correct / total
    )

    agreement = (
        matching_predictions / total
    )

    confidence_difference = np.mean(
        np.abs(
            np.array(fp32_confidences)
            - np.array(int8_confidences)
        )
    )

    print("Total test images:", total)
    print()

    print(
        f"FP32 accuracy: "
        f"{fp32_accuracy:.4f}"
    )

    print(
        f"INT8 accuracy: "
        f"{int8_accuracy:.4f}"
    )

    print(
        f"Accuracy difference: "
        f"{int8_accuracy - fp32_accuracy:+.4f}"
    )

    print()

    print(
        f"FP32 / INT8 prediction agreement: "
        f"{agreement:.4f}"
    )

    print(
        f"Mean confidence difference: "
        f"{confidence_difference:.6f}"
    )


if __name__ == "__main__":
    main()
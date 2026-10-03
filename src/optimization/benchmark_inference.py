from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torchvision import transforms

from src.models.resnet50_finetune import create_resnet50_finetune


PYTORCH_MODEL = Path("models/resnet50_finetune.pth")
ONNX_MODEL = Path("models/resnet50_finetune.onnx")
TEST_DIR = Path("data/splits/test")

WARMUP_RUNS = 20
BENCHMARK_RUNS = 100


TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def benchmark_pytorch(model, input_tensor):
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            model(input_tensor)

        start = perf_counter()

        for _ in range(BENCHMARK_RUNS):
            model(input_tensor)

        end = perf_counter()

    return (
        (end - start)
        / BENCHMARK_RUNS
        * 1000
    )


def benchmark_onnx(session, input_array):
    input_name = session.get_inputs()[0].name

    for _ in range(WARMUP_RUNS):
        session.run(
            None,
            {input_name: input_array},
        )

    start = perf_counter()

    for _ in range(BENCHMARK_RUNS):
        session.run(
            None,
            {input_name: input_array},
        )

    end = perf_counter()

    return (
        (end - start)
        / BENCHMARK_RUNS
        * 1000
    )


def main():
    image_path = next(
        TEST_DIR.glob("*/*.jpg")
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    input_tensor = TRANSFORM(
        image
    ).unsqueeze(0)

    input_array = np.ascontiguousarray(
        input_tensor.numpy()
    )

    pytorch_model = create_resnet50_finetune(
        num_classes=6
    )

    pytorch_model.load_state_dict(
        torch.load(
            PYTORCH_MODEL,
            map_location="cpu",
        )
    )

    pytorch_model.eval()

    session = ort.InferenceSession(
        str(ONNX_MODEL),
        providers=["CPUExecutionProvider"],
    )

    pytorch_ms = benchmark_pytorch(
        pytorch_model,
        input_tensor,
    )

    onnx_ms = benchmark_onnx(
        session,
        input_array,
    )

    speedup = pytorch_ms / onnx_ms

    print("Image:", image_path.name)
    print()

    print(
        f"PyTorch CPU latency: "
        f"{pytorch_ms:.3f} ms/image"
    )

    print(
        f"ONNX CPU latency: "
        f"{onnx_ms:.3f} ms/image"
    )

    print(
        f"ONNX speedup: "
        f"{speedup:.2f}x"
    )


if __name__ == "__main__":
    main()
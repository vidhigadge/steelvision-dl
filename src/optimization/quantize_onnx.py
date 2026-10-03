from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms

from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)


INPUT_MODEL = Path(
    "models/resnet50_finetune_preprocessed.onnx"
)

OUTPUT_MODEL = Path(
    "models/resnet50_finetune_uint8.onnx"
)

TRAIN_DIR = Path("data/splits/train")

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
]

IMAGES_PER_CLASS = 10


TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class ImageCalibrationReader(
    CalibrationDataReader
):
    def __init__(self):
        self.data = []

        for class_name in CLASS_NAMES:
            class_dir = TRAIN_DIR / class_name

            image_paths = sorted(
                class_dir.glob("*.jpg")
            )[:IMAGES_PER_CLASS]

            for image_path in image_paths:
                image = Image.open(
                    image_path
                ).convert("RGB")

                tensor = TRANSFORM(
                    image
                ).unsqueeze(0)

                array = np.ascontiguousarray(
                    tensor.numpy(),
                    dtype=np.float32,
                )

                self.data.append(
                    {"input": array}
                )

        self.iterator = iter(self.data)

    def get_next(self):
        return next(
            self.iterator,
            None,
        )


def main():
    print(
        "Calibration images:",
        len(CLASS_NAMES) * IMAGES_PER_CLASS,
    )

    reader = ImageCalibrationReader()

    quantize_static(
        model_input=str(INPUT_MODEL),
        model_output=str(OUTPUT_MODEL),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
        per_channel=False,
        calibrate_method=CalibrationMethod.MinMax,
    )

    print(
        "U8U8 model saved to:",
        OUTPUT_MODEL,
    )


if __name__ == "__main__":
    main()
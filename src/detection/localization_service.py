from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO

from src.detection.coordinate_mapping import (
    map_box_to_original,
)


MODEL_PATH = Path(
    "runs/detect/steel_defect_yolo/weights/best.pt"
)

PROCESSED_SIZE = (800, 800)


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
        return "mps"

    return "cpu"


DEVICE = get_device()


MODEL = YOLO(
    str(MODEL_PATH)
)


def prepare_low_resolution(
    image,
    resolution=None,
):
    """
    Prepare image for detection.

    If resolution is None, use the uploaded image
    without artificial downsampling.

    Otherwise, simulate a lower-resolution observation
    before upscaling to the detector processing size.
    """

    if resolution is None:
        source_image = image.copy()

    else:
        source_image = image.resize(
            (resolution, resolution),
            Image.Resampling.BICUBIC,
        )

    processed = source_image.resize(
        PROCESSED_SIZE,
        Image.Resampling.BICUBIC,
    )

    return source_image, processed


def draw_boxes(
    image,
    detections,
):
    """
    Draw mapped YOLO boxes on an image.
    """

    output = image.copy()

    draw = ImageDraw.Draw(
        output
    )

    line_width = max(
        2,
        int(
            min(output.size) * 0.01
        ),
    )

    for detection in detections:
        x1, y1, x2, y2 = (
            detection["mapped_box"]
        )

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=line_width,
        )

        label = (
            f"{detection['class_name']} "
            f"{detection['confidence']:.2f}"
        )

        draw.text(
            (x1 + 3, y1 + 3),
            label,
            fill="red",
        )

    return output


def localize_defects(
    image,
    resolution=None,
    confidence_threshold=0.25,
):
    """
    Run low-resolution simulation, YOLO detection,
    and map detected boxes back to the original image.
    """

    image = image.convert(
        "RGB"
    )

    original_size = image.size

    low_res, processed = (
        prepare_low_resolution(
            image,
            resolution,
        )
    )

    processed_array = np.array(
        processed
    )

    results = MODEL.predict(
        source=processed_array,
        device=DEVICE,
        conf=confidence_threshold,
        verbose=False,
    )

    result = results[0]

    detections = []

    for box in result.boxes:
        processed_box = (
            box.xyxy[0]
            .cpu()
            .tolist()
        )

        class_id = int(
            box.cls[0]
        )

        confidence = float(
            box.conf[0]
        )

        mapped_box = (
            map_box_to_original(
                box=processed_box,
                processed_size=PROCESSED_SIZE,
                original_size=original_size,
            )
        )

        detections.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[
                    class_id
                ],
                "confidence": confidence,
                "processed_box": [
                    round(value, 2)
                    for value
                    in processed_box
                ],
                "mapped_box": [
                    round(value, 2)
                    for value
                    in mapped_box
                ],
            }
        )

    mapped_image = draw_boxes(
        image,
        detections,
    )

    return {
        "resolution": resolution,
        "original_size": original_size,
        "processed_size": PROCESSED_SIZE,
        "detections": detections,
        "low_res_image": low_res,
        "processed_image": processed,
        "mapped_image": mapped_image,
    }
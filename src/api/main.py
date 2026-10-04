from io import BytesIO
from pathlib import Path
import base64

import numpy as np
import onnxruntime as ort
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

from src.detection.localization_service import (
    localize_defects,
)
from src.explainability.gradcam_service import (
    generate_gradcam,
)


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

MODEL_PATH = Path(
    "models/resnet50_finetune.onnx"
)

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
]


# ----------------------------------------------------------------------------
# Response models
# ----------------------------------------------------------------------------

class TopPrediction(BaseModel):
    class_name: str
    probability: float


class PredictionResponse(BaseModel):
    filename: str
    predicted_class: str
    confidence: float
    entropy: float
    top_3: list[TopPrediction]


class LocalizationDetection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    processed_box: list[float]
    mapped_box: list[float]


class LocalizationResponse(BaseModel):
    filename: str
    resolution: str
    original_size: list[int]
    processed_size: list[int]
    detection_count: int
    detections: list[LocalizationDetection]
    low_res_image: str
    processed_image: str
    mapped_image: str


# ----------------------------------------------------------------------------
# Classification preprocessing
# ----------------------------------------------------------------------------

TRANSFORM = transforms.Compose([
    transforms.Grayscale(
        num_output_channels=3
    ),
    transforms.Resize(
        (224, 224)
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
])


# ----------------------------------------------------------------------------
# FastAPI application
# ----------------------------------------------------------------------------

app = FastAPI(
    title="SteelVision API",
    version="1.0.0",
)


# ----------------------------------------------------------------------------
# ONNX classification model
# ----------------------------------------------------------------------------

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=[
        "CPUExecutionProvider"
    ],
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def softmax(values):
    values = values - np.max(
        values,
        axis=1,
        keepdims=True,
    )

    exp_values = np.exp(
        values
    )

    return (
        exp_values
        / np.sum(
            exp_values,
            axis=1,
            keepdims=True,
        )
    )


def preprocess_image(image):
    tensor = TRANSFORM(
        image
    ).unsqueeze(0)

    return np.ascontiguousarray(
        tensor.numpy(),
        dtype=np.float32,
    )


def image_to_base64(image):
    buffer = BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=90,
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode(
        "utf-8"
    )


def validate_file_type(file):
    if file.content_type not in {
        "image/jpeg",
        "image/png",
    }:
        raise HTTPException(
            status_code=400,
            detail=(
                "Only JPEG and PNG "
                "images are supported"
            ),
        )


async def read_image(file):
    try:
        contents = await file.read()

        image = Image.open(
            BytesIO(
                contents
            )
        ).convert(
            "RGB"
        )

        return image

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file",
        )


# ----------------------------------------------------------------------------
# Basic endpoints
# ----------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "SteelVision API"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# ----------------------------------------------------------------------------
# Classification endpoint
# ----------------------------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
async def predict(
    file: UploadFile = File(...),
):
    validate_file_type(
        file
    )

    image = await read_image(
        file
    )

    input_array = preprocess_image(
        image
    )

    input_name = (
        session
        .get_inputs()[0]
        .name
    )

    logits = session.run(
        None,
        {
            input_name:
            input_array
        },
    )[0]

    probabilities = softmax(
        logits
    )[0]

    predicted_index = int(
        np.argmax(
            probabilities
        )
    )

    confidence = float(
        probabilities[
            predicted_index
        ]
    )

    entropy = float(
        -np.sum(
            probabilities
            * np.log(
                probabilities
                + 1e-12
            )
        )
    )

    top_indices = np.argsort(
        probabilities
    )[::-1][:3]

    top_predictions = [
        {
            "class_name":
                CLASS_NAMES[
                    index
                ],
            "probability":
                round(
                    float(
                        probabilities[
                            index
                        ]
                    ),
                    6,
                ),
        }
        for index
        in top_indices
    ]

    return {
        "filename":
            file.filename,

        "predicted_class":
            CLASS_NAMES[
                predicted_index
            ],

        "confidence":
            round(
                confidence,
                6,
            ),

        "entropy":
            round(
                entropy,
                6,
            ),

        "top_3":
            top_predictions,
    }


# ----------------------------------------------------------------------------
# Grad-CAM explainability endpoint
# ----------------------------------------------------------------------------

@app.post(
    "/explain"
)
async def explain(
    file: UploadFile = File(...),
):
    validate_file_type(
        file
    )

    image = await read_image(
        file
    )

    try:
        (
            gradcam_bytes,
            _,
            _,
        ) = generate_gradcam(
            image
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Grad-CAM generation failed: "
                f"{error}"
            ),
        )

    return Response(
        content=gradcam_bytes,
        media_type="image/jpeg",
    )


# ----------------------------------------------------------------------------
# Localization endpoint
# ----------------------------------------------------------------------------

@app.post(
    "/localize",
    response_model=LocalizationResponse,
)
async def localize(
    file: UploadFile = File(...),
    resolution: str = Form(
        "original"
    ),
):
    """
    Perform defect localization.

    resolution options:

    original
        Use uploaded image without
        artificial downsampling.

    224 / 128 / 64 / 32
        Simulate lower-resolution input
        before YOLO localization.
    """

    validate_file_type(
        file
    )

    resolution = (
        resolution
        .lower()
        .strip()
    )

    allowed_resolutions = {
        "original",
        "224",
        "128",
        "64",
        "32",
    }

    if (
        resolution
        not in allowed_resolutions
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Resolution must be one of: "
                "original, 224, 128, 64, 32"
            ),
        )

    if resolution == "original":
        simulation_resolution = None

    else:
        simulation_resolution = int(
            resolution
        )

    image = await read_image(
        file
    )

    try:
        result = localize_defects(
            image=image,
            resolution=(
                simulation_resolution
            ),
            confidence_threshold=0.25,
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Localization failed: "
                f"{error}"
            ),
        )

    return {
        "filename":
            file.filename,

        "resolution":
            resolution,

        "original_size":
            list(
                result[
                    "original_size"
                ]
            ),

        "processed_size":
            list(
                result[
                    "processed_size"
                ]
            ),

        "detection_count":
            len(
                result[
                    "detections"
                ]
            ),

        "detections":
            result[
                "detections"
            ],

        "low_res_image":
            image_to_base64(
                result[
                    "low_res_image"
                ]
            ),

        "processed_image":
            image_to_base64(
                result[
                    "processed_image"
                ]
            ),

        "mapped_image":
            image_to_base64(
                result[
                    "mapped_image"
                ]
            ),
    }
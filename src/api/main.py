from io import BytesIO
from pathlib import Path
import base64

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
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
from torchvision import models, transforms

from src.detection.localization_service import (
    localize_defects,
)
from src.explainability.gradcam_service import (
    generate_gradcam,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CLASSIFIER_MODEL_PATH = Path(
    "models/resnet50_finetune.onnx"
)

DOMAIN_MODEL_PATH = Path(
    "models/domain_validator_mobilenet_v3_small.pth"
)

DOMAIN_THRESHOLD = 0.90

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
]


# -----------------------------------------------------------------------------
# Response models
# -----------------------------------------------------------------------------

class TopPrediction(BaseModel):
    class_name: str
    probability: float


class PredictionResponse(BaseModel):
    filename: str
    predicted_class: str
    confidence: float
    entropy: float
    top_3: list[TopPrediction]


class DomainValidationResponse(BaseModel):
    filename: str
    supported: bool
    supported_probability: float
    unsupported_probability: float
    threshold: float


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


# -----------------------------------------------------------------------------
# Shared preprocessing
# -----------------------------------------------------------------------------

CLASSIFICATION_TRANSFORM = transforms.Compose(
    [
        transforms.Grayscale(
            num_output_channels=3
        ),
        transforms.Resize(
            (
                224,
                224,
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


DOMAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Grayscale(
            num_output_channels=3
        ),
        transforms.Resize(
            (
                224,
                224,
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


# -----------------------------------------------------------------------------
# FastAPI application
# -----------------------------------------------------------------------------

app = FastAPI(
    title="SteelVision API",
    version="1.1.0",
)


# -----------------------------------------------------------------------------
# ResNet-50 ONNX classifier
# -----------------------------------------------------------------------------

classification_session = ort.InferenceSession(
    str(
        CLASSIFIER_MODEL_PATH
    ),
    providers=[
        "CPUExecutionProvider"
    ],
)


# -----------------------------------------------------------------------------
# MobileNetV3-Small domain validator
# -----------------------------------------------------------------------------

if torch.backends.mps.is_available():
    DOMAIN_DEVICE = torch.device(
        "mps"
    )

elif torch.cuda.is_available():
    DOMAIN_DEVICE = torch.device(
        "cuda"
    )

else:
    DOMAIN_DEVICE = torch.device(
        "cpu"
    )


def load_domain_model():
    if not DOMAIN_MODEL_PATH.exists():
        raise RuntimeError(
            "Domain validator checkpoint was not found at "
            f"{DOMAIN_MODEL_PATH}"
        )

    checkpoint = torch.load(
        DOMAIN_MODEL_PATH,
        map_location=DOMAIN_DEVICE,
    )

    model = models.mobilenet_v3_small(
        weights=None
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

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        DOMAIN_DEVICE
    )

    model.eval()

    class_to_idx = checkpoint[
        "class_to_idx"
    ]

    if (
        "supported"
        not in class_to_idx
        or "unsupported"
        not in class_to_idx
    ):
        raise RuntimeError(
            "Domain validator checkpoint does not contain "
            "the expected supported/unsupported classes."
        )

    return (
        model,
        class_to_idx,
    )


(
    domain_model,
    domain_class_to_idx,
) = load_domain_model()


DOMAIN_SUPPORTED_INDEX = (
    domain_class_to_idx[
        "supported"
    ]
)

DOMAIN_UNSUPPORTED_INDEX = (
    domain_class_to_idx[
        "unsupported"
    ]
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def softmax(
    values,
):
    values = (
        values
        - np.max(
            values,
            axis=1,
            keepdims=True,
        )
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


def preprocess_classification_image(
    image,
):
    tensor = (
        CLASSIFICATION_TRANSFORM(
            image
        )
        .unsqueeze(
            0
        )
    )

    return np.ascontiguousarray(
        tensor.numpy(),
        dtype=np.float32,
    )


def preprocess_domain_image(
    image,
):
    tensor = (
        DOMAIN_TRANSFORM(
            image
        )
        .unsqueeze(
            0
        )
        .to(
            DOMAIN_DEVICE
        )
    )

    return tensor


def image_to_base64(
    image,
):
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


def validate_file_type(
    file,
):
    if file.content_type not in {
        "image/jpeg",
        "image/png",
    }:
        raise HTTPException(
            status_code=400,
            detail=(
                "Only JPEG and PNG images are supported."
            ),
        )


async def read_image(
    file,
):
    try:
        contents = await file.read()

        image = Image.open(
            BytesIO(
                contents
            )
        )

        image.load()

        return image.convert(
            "RGB"
        )

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file.",
        ) from error


def evaluate_domain(
    image,
):
    input_tensor = (
        preprocess_domain_image(
            image
        )
    )

    with torch.no_grad():
        logits = domain_model(
            input_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

    supported_probability = float(
        probabilities[
            DOMAIN_SUPPORTED_INDEX
        ]
        .detach()
        .cpu()
        .item()
    )

    unsupported_probability = float(
        probabilities[
            DOMAIN_UNSUPPORTED_INDEX
        ]
        .detach()
        .cpu()
        .item()
    )

    supported = (
        supported_probability
        >= DOMAIN_THRESHOLD
    )

    return {
        "supported":
            supported,

        "supported_probability":
            supported_probability,

        "unsupported_probability":
            unsupported_probability,

        "threshold":
            DOMAIN_THRESHOLD,
    }


def ensure_supported_domain(
    image,
):
    result = evaluate_domain(
        image
    )

    if not result[
        "supported"
    ]:
        raise HTTPException(
            status_code=422,
            detail={
                "error":
                    "unsupported_input",

                "message":
                    (
                        "The uploaded image does not appear "
                        "to belong to SteelVision's supported "
                        "steel-defect domain."
                    ),

                "supported_probability":
                    round(
                        result[
                            "supported_probability"
                        ],
                        6,
                    ),

                "threshold":
                    DOMAIN_THRESHOLD,
            },
        )

    return result


# -----------------------------------------------------------------------------
# Basic endpoints
# -----------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "message":
            "SteelVision API",

        "version":
            "1.1.0",
    }


@app.get("/health")
def health():
    return {
        "status":
            "ok",

        "classifier":
            "ready",

        "domain_validator":
            "ready",

        "domain_threshold":
            DOMAIN_THRESHOLD,
    }


# -----------------------------------------------------------------------------
# Domain validation endpoint
# -----------------------------------------------------------------------------

@app.post(
    "/validate-domain",
    response_model=DomainValidationResponse,
)
async def validate_domain(
    file: UploadFile = File(...),
):
    validate_file_type(
        file
    )

    image = await read_image(
        file
    )

    result = evaluate_domain(
        image
    )

    return {
        "filename":
            file.filename,

        "supported":
            result[
                "supported"
            ],

        "supported_probability":
            round(
                result[
                    "supported_probability"
                ],
                6,
            ),

        "unsupported_probability":
            round(
                result[
                    "unsupported_probability"
                ],
                6,
            ),

        "threshold":
            DOMAIN_THRESHOLD,
    }


# -----------------------------------------------------------------------------
# Classification endpoint
# -----------------------------------------------------------------------------

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

    # Phase 15 domain gate
    ensure_supported_domain(
        image
    )

    input_array = (
        preprocess_classification_image(
            image
        )
    )

    input_name = (
        classification_session
        .get_inputs()[0]
        .name
    )

    logits = classification_session.run(
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
    )[::-1][
        :3
    ]

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


# -----------------------------------------------------------------------------
# Grad-CAM endpoint
# -----------------------------------------------------------------------------

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

    # Phase 15 domain gate
    ensure_supported_domain(
        image
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


# -----------------------------------------------------------------------------
# Localization endpoint
# -----------------------------------------------------------------------------

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
                "original, 224, 128, 64, 32."
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

    # Phase 15 domain gate
    ensure_supported_domain(
        image
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
from io import BytesIO
from pathlib import Path
from pydantic import BaseModel

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms


MODEL_PATH = Path("models/resnet50_finetune.onnx")

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
]

class TopPrediction(BaseModel):
    class_name: str
    probability: float


class PredictionResponse(BaseModel):
    filename: str
    predicted_class: str
    confidence: float
    entropy: float
    top_3: list[TopPrediction]


TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


app = FastAPI(
    title="SteelVision-DL API",
    version="1.0.0",
)


session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"],
)


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


def preprocess_image(image):
    tensor = TRANSFORM(
        image
    ).unsqueeze(0)

    return np.ascontiguousarray(
        tensor.numpy(),
        dtype=np.float32,
    )


@app.get("/")
def root():
    return {
        "message": "SteelVision-DL API"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
async def predict(
    file: UploadFile = File(...)
):
    if file.content_type not in {
        "image/jpeg",
        "image/png",
    }:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are supported",
        )

    try:
        contents = await file.read()

        image = Image.open(
            BytesIO(contents)
        ).convert("RGB")

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file",
        )

    input_array = preprocess_image(
        image
    )

    input_name = session.get_inputs()[0].name

    logits = session.run(
        None,
        {
            input_name: input_array
        },
    )[0]

    probabilities = softmax(
        logits
    )[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_index]
    )

    entropy = float(
        -np.sum(
            probabilities
            * np.log(
                probabilities + 1e-12
            )
        )
    )

    top_indices = np.argsort(
        probabilities
    )[::-1][:3]

    top_predictions = [
    {
        "class_name": CLASS_NAMES[index],
        "probability": round(
            float(probabilities[index]),
            6,
        ),
    }
    for index in top_indices
]

    return {
        "filename": file.filename,
        "predicted_class": CLASS_NAMES[
            predicted_index
        ],
        "confidence": round(
            confidence,
            6,
        ),
        "entropy": round(
            entropy,
            6,
        ),
        "top_3": top_predictions,
    }
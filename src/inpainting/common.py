"""Shared utilities for EXP-016 inpainting reliability experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50


CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
]

MODEL_PATH = Path(
    "models/resnet50_finetune.pth"
)

IMAGE_SIZE = 224

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


def get_device() -> torch.device:
    """Use Apple MPS when available, otherwise CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def create_classifier(
    num_classes: int = 6,
) -> torch.nn.Module:
    """Recreate SteelVision's fine-tuned ResNet-50 architecture."""
    model = resnet50(
        weights=None
    )

    input_features = (
        model.fc.in_features
    )

    model.fc = torch.nn.Linear(
        input_features,
        num_classes,
    )

    return model


def load_classifier(
    model_path: Path = MODEL_PATH,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """Load the trained SteelVision ResNet-50 classifier."""
    if device is None:
        device = get_device()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Classifier checkpoint not found: {model_path}"
        )

    model = create_classifier(
        num_classes=len(CLASS_NAMES)
    )

    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        state_dict
    )

    model.to(
        device
    )

    model.eval()

    return model


def get_preprocess_transform():
    """Return the same preprocessing used by SteelVision."""
    return transforms.Compose(
        [
            transforms.Grayscale(
                num_output_channels=3
            ),
            transforms.Resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                )
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def load_grayscale_image(
    image_path: Path,
) -> np.ndarray:
    """Load image as a 224×224 grayscale uint8 NumPy array."""
    image = (
        Image.open(
            image_path
        )
        .convert("L")
        .resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            Image.Resampling.BICUBIC,
        )
    )

    return np.asarray(
        image,
        dtype=np.uint8,
    )


def gray_array_to_rgb_pil(
    image: np.ndarray,
) -> Image.Image:
    """Convert grayscale NumPy image to RGB PIL image."""
    image_uint8 = np.clip(
        image,
        0,
        255,
    ).astype(
        np.uint8
    )

    return (
        Image.fromarray(
            image_uint8,
            mode="L",
        )
        .convert("RGB")
    )


def prepare_tensor(
    image: np.ndarray | Image.Image,
    device: torch.device,
) -> torch.Tensor:
    """Convert image into normalized SteelVision input tensor."""
    if isinstance(
        image,
        np.ndarray,
    ):
        image = gray_array_to_rgb_pil(
            image
        )

    transform = get_preprocess_transform()

    tensor = transform(
        image
    ).unsqueeze(
        0
    )

    return tensor.to(
        device
    )


def predict_image(
    model: torch.nn.Module,
    image: np.ndarray | Image.Image,
    device: torch.device,
) -> dict:
    """Run SteelVision classification on one image."""
    tensor = prepare_tensor(
        image=image,
        device=device,
    )

    with torch.no_grad():
        logits = model(
            tensor
        )

        probabilities = F.softmax(
            logits,
            dim=1,
        )

    probabilities_np = (
        probabilities[0]
        .detach()
        .cpu()
        .numpy()
    )

    predicted_index = int(
        probabilities_np.argmax()
    )

    confidence = float(
        probabilities_np[
            predicted_index
        ]
    )

    entropy = calculate_entropy(
        probabilities_np
    )

    return {
        "predicted_index":
            predicted_index,

        "predicted_class":
            CLASS_NAMES[
                predicted_index
            ],

        "confidence":
            confidence,

        "entropy":
            entropy,

        "probabilities":
            probabilities_np,
    }


def calculate_entropy(
    probabilities: np.ndarray,
) -> float:
    """Calculate natural-log softmax entropy."""
    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    probabilities = np.clip(
        probabilities,
        1e-12,
        1.0,
    )

    return float(
        -np.sum(
            probabilities
            * np.log(
                probabilities
            )
        )
    )


def generate_gradcam(
    model: torch.nn.Module,
    image: np.ndarray | Image.Image,
    device: torch.device,
    target_class: int | None = None,
) -> np.ndarray:
    """Generate Grad-CAM heatmap from ResNet-50 layer4[-1].

    Returns:
        224×224 float32 array normalized to [0, 1].

    Important:
        Grad-CAM represents model influence, not physical
        defect ground truth.
    """
    model.eval()

    activations = {}
    gradients = {}

    target_layer = model.layer4[
        -1
    ]

    def forward_hook(
        module,
        inputs,
        output,
    ):
        activations[
            "value"
        ] = output

    def backward_hook(
        module,
        grad_input,
        grad_output,
    ):
        gradients[
            "value"
        ] = grad_output[
            0
        ]

    forward_handle = (
        target_layer.register_forward_hook(
            forward_hook
        )
    )

    backward_handle = (
        target_layer.register_full_backward_hook(
            backward_hook
        )
    )

    try:
        input_tensor = prepare_tensor(
            image=image,
            device=device,
        )

        input_tensor.requires_grad_(
            True
        )

        model.zero_grad(
            set_to_none=True
        )

        logits = model(
            input_tensor
        )

        if target_class is None:
            target_class = int(
                logits[
                    0
                ].argmax().item()
            )

        target_score = logits[
            0,
            target_class,
        ]

        target_score.backward()

        activation_tensor = (
            activations[
                "value"
            ]
        )

        gradient_tensor = (
            gradients[
                "value"
            ]
        )

        weights = gradient_tensor.mean(
            dim=(
                2,
                3,
            ),
            keepdim=True,
        )

        cam = (
            weights
            * activation_tensor
        ).sum(
            dim=1,
            keepdim=True,
        )

        cam = F.relu(
            cam
        )

        cam = F.interpolate(
            cam,
            size=(
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            mode="bilinear",
            align_corners=False,
        )

        cam = (
            cam[
                0,
                0
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        minimum = float(
            cam.min()
        )

        maximum = float(
            cam.max()
        )

        if maximum > minimum:
            cam = (
                cam
                - minimum
            ) / (
                maximum
                - minimum
            )

        else:
            cam = np.zeros_like(
                cam,
                dtype=np.float32,
            )

        return cam

    finally:
        forward_handle.remove()
        backward_handle.remove()

        model.zero_grad(
            set_to_none=True
        )


def attention_peak(
    heatmap: np.ndarray,
) -> tuple[int, int]:
    """Return (x, y) coordinate of strongest Grad-CAM attention."""
    heatmap = np.asarray(
        heatmap,
        dtype=np.float32,
    )

    row, column = np.unravel_index(
        np.argmax(
            heatmap
        ),
        heatmap.shape,
    )

    return (
        int(column),
        int(row),
    )


def attention_minimum(
    heatmap: np.ndarray,
    margin_fraction: float = 0.15,
) -> tuple[int, int]:
    """Return weak-attention coordinate inside a safe interior region.

    The outer 15% border is excluded so low-attention placement does
    not trivially select a corner simply because Grad-CAM is zero there.
    """
    heatmap = np.asarray(
        heatmap,
        dtype=np.float32,
    )

    height, width = heatmap.shape

    margin_y = int(
        round(
            height
            * margin_fraction
        )
    )

    margin_x = int(
        round(
            width
            * margin_fraction
        )
    )

    y1 = margin_y
    y2 = height - margin_y

    x1 = margin_x
    x2 = width - margin_x

    interior = heatmap[
        y1:y2,
        x1:x2,
    ]

    row, column = np.unravel_index(
        np.argmin(
            interior
        ),
        interior.shape,
    )

    return (
        int(
            column
            + x1
        ),
        int(
            row
            + y1
        ),
    )


def attention_center(
    heatmap: np.ndarray,
    mode: str,
) -> tuple[int, int]:
    """Choose mask center using Grad-CAM attention."""
    if mode == "high_attention":
        return attention_peak(
            heatmap
        )

    if mode == "low_attention":
        return attention_minimum(
            heatmap
        )

    raise ValueError(
        f"Unknown attention mode: {mode}"
    )


def class_flip(
    original_prediction: dict,
    reconstructed_prediction: dict,
) -> bool:
    """Return True when reconstruction changes predicted class."""
    return (
        original_prediction[
            "predicted_index"
        ]
        != reconstructed_prediction[
            "predicted_index"
        ]
    )


def confidence_change(
    original_prediction: dict,
    reconstructed_prediction: dict,
) -> float:
    """Reconstructed confidence minus original confidence."""
    return float(
        reconstructed_prediction[
            "confidence"
        ]
        - original_prediction[
            "confidence"
        ]
    )


def entropy_change(
    original_prediction: dict,
    reconstructed_prediction: dict,
) -> float:
    """Reconstructed entropy minus original entropy."""
    return float(
        reconstructed_prediction[
            "entropy"
        ]
        - original_prediction[
            "entropy"
        ]
    )


def list_images(
    root: Path,
) -> list[tuple[Path, str]]:
    """List SteelVision images and their class labels."""
    root = Path(
        root
    )

    if not root.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}"
        )

    items = []

    valid_suffixes = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    }

    class_directories = [
        directory
        for directory in root.iterdir()
        if (
            directory.is_dir()
            and directory.name
            in CLASS_NAMES
        )
    ]

    if class_directories:
        for class_name in CLASS_NAMES:
            class_dir = (
                root
                / class_name
            )

            if not class_dir.exists():
                continue

            for image_path in sorted(
                class_dir.iterdir()
            ):
                if (
                    image_path.is_file()
                    and image_path.suffix.lower()
                    in valid_suffixes
                ):
                    items.append(
                        (
                            image_path,
                            class_name,
                        )
                    )

        return items

    for image_path in sorted(
        root.iterdir()
    ):
        if (
            not image_path.is_file()
            or image_path.suffix.lower()
            not in valid_suffixes
        ):
            continue

        stem = (
            image_path.stem
            .lower()
        )

        matched_class = None

        for class_name in sorted(
            CLASS_NAMES,
            key=len,
            reverse=True,
        ):
            if stem.startswith(
                f"{class_name}_"
            ):
                matched_class = (
                    class_name
                )
                break

        if matched_class is None:
            raise ValueError(
                "Could not infer class from flat filename: "
                f"{image_path.name}"
            )

        items.append(
            (
                image_path,
                matched_class,
            )
        )

    return items


def class_index(
    class_name: str,
) -> int:
    """Convert class name to SteelVision class index."""
    if class_name not in CLASS_NAMES:
        raise ValueError(
            f"Unknown class: {class_name}"
        )

    return CLASS_NAMES.index(
        class_name
    )


def classification_correct(
    prediction: dict,
    ground_truth_class: str,
) -> bool:
    """Check classifier prediction against dataset ground truth."""
    return (
        prediction[
            "predicted_class"
        ]
        == ground_truth_class
    )


def probability_of_class(
    prediction: dict,
    class_name: str,
) -> float:
    """Return predicted probability for a specific class."""
    index = class_index(
        class_name
    )

    return float(
        prediction[
            "probabilities"
        ][
            index
        ]
    )


def normalized_to_uint8(
    image: np.ndarray,
) -> np.ndarray:
    """Convert floating image in [0, 1] to uint8."""
    return np.clip(
        image
        * 255.0,
        0,
        255,
    ).astype(
        np.uint8
    )


def uint8_to_normalized(
    image: np.ndarray,
) -> np.ndarray:
    """Convert uint8 grayscale image to float32 [0, 1]."""
    return (
        image.astype(
            np.float32
        )
        / 255.0
    )


def seed_everything(
    seed: int,
) -> np.random.Generator:
    """Seed NumPy and PyTorch for reproducibility."""
    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    return np.random.default_rng(
        seed
    )
"""Inpainting methods for EXP-016."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from src.inpainting.common import (
    uint8_to_normalized,
)
from src.inpainting.unet_model import (
    create_inpainting_unet,
)


DEFAULT_UNET_PATH = Path(
    "models/inpainting_unet.pth"
)


def mean_fill(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Fill the missing region using the mean visible grayscale value.

    This is the non-inpainting reference baseline.
    """
    image = np.asarray(
        image,
        dtype=np.uint8,
    )

    mask_bool = (
        mask > 0
    )

    output = image.copy()

    visible_pixels = output[
        ~mask_bool
    ]

    if visible_pixels.size == 0:
        fill_value = 127

    else:
        fill_value = int(
            round(
                float(
                    visible_pixels.mean()
                )
            )
        )

    output[
        mask_bool
    ] = fill_value

    return output


def telea_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    radius: float = 3.0,
) -> np.ndarray:
    """OpenCV Telea inpainting."""
    image = np.asarray(
        image,
        dtype=np.uint8,
    )

    mask = np.asarray(
        mask,
        dtype=np.uint8,
    )

    return cv2.inpaint(
        image,
        mask,
        radius,
        cv2.INPAINT_TELEA,
    )


def navier_stokes_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    radius: float = 3.0,
) -> np.ndarray:
    """OpenCV Navier-Stokes inpainting."""
    image = np.asarray(
        image,
        dtype=np.uint8,
    )

    mask = np.asarray(
        mask,
        dtype=np.uint8,
    )

    return cv2.inpaint(
        image,
        mask,
        radius,
        cv2.INPAINT_NS,
    )


def prepare_unet_input(
    image: np.ndarray,
    mask: np.ndarray,
) -> torch.Tensor:
    """Create the 2-channel U-Net input.

    Channel 0:
        masked grayscale image in [0, 1]

    Channel 1:
        binary missing-region mask in {0, 1}
    """
    image_float = uint8_to_normalized(
        image
    )

    mask_float = (
        mask > 0
    ).astype(
        np.float32
    )

    masked_image = image_float.copy()

    masked_image[
        mask_float > 0
    ] = 0.0

    stacked = np.stack(
        [
            masked_image,
            mask_float,
        ],
        axis=0,
    )

    return torch.from_numpy(
        stacked
    ).float()


def combine_unet_output(
    original_image: np.ndarray,
    predicted_image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Use U-Net output only inside the missing region.

    Original visible pixels remain unchanged.
    """
    original_float = uint8_to_normalized(
        original_image
    )

    mask_float = (
        mask > 0
    ).astype(
        np.float32
    )

    combined = (
        original_float
        * (
            1.0
            - mask_float
        )
        + predicted_image
        * mask_float
    )

    return np.clip(
        combined
        * 255.0,
        0,
        255,
    ).astype(
        np.uint8
    )


def load_unet_inpainter(
    checkpoint_path: Path = DEFAULT_UNET_PATH,
    device: torch.device | None = None,
):
    """Load the trained EXP-016 U-Net inpainter."""
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device(
                "mps"
            )
        else:
            device = torch.device(
                "cpu"
            )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "U-Net checkpoint not found: "
            f"{checkpoint_path}"
        )

    model = create_inpainting_unet()

    state_dict = torch.load(
        checkpoint_path,
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


def unet_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    model,
    device: torch.device,
) -> np.ndarray:
    """Reconstruct the missing region using the trained U-Net."""
    input_tensor = prepare_unet_input(
        image=image,
        mask=mask,
    ).unsqueeze(
        0
    ).to(
        device
    )

    with torch.no_grad():
        prediction = model(
            input_tensor
        )

    predicted_image = (
        prediction[
            0,
            0,
        ]
        .detach()
        .cpu()
        .numpy()
    )

    return combine_unet_output(
        original_image=image,
        predicted_image=predicted_image,
        mask=mask,
    )


def available_methods() -> tuple[str, ...]:
    """Return core EXP-016 inpainting methods."""
    return (
        "mean_fill",
        "telea",
        "navier_stokes",
        "unet",
    )


def run_inpainter(
    method: str,
    image: np.ndarray,
    mask: np.ndarray,
    unet_model=None,
    device: torch.device | None = None,
) -> np.ndarray:
    """Run one EXP-016 inpainting method."""
    if method == "mean_fill":
        return mean_fill(
            image,
            mask,
        )

    if method == "telea":
        return telea_inpaint(
            image,
            mask,
        )

    if method == "navier_stokes":
        return navier_stokes_inpaint(
            image,
            mask,
        )

    if method == "unet":
        if unet_model is None:
            raise ValueError(
                "unet_model must be provided "
                "when method='unet'."
            )

        if device is None:
            raise ValueError(
                "device must be provided "
                "when method='unet'."
            )

        return unet_inpaint(
            image=image,
            mask=mask,
            model=unet_model,
            device=device,
        )

    raise ValueError(
        f"Unknown inpainting method: {method}"
    )
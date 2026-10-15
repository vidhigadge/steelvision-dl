"""Synthetic missing-region masks for EXP-016.

Mask convention used throughout the inpainting experiment:

    255 = missing / masked region
      0 = original pixels retained

The masks simulate controlled loss of visual information.

Important:
Mask placement can later be guided by Grad-CAM attention, but an
attention region must not be described as confirmed defect ground truth.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


MASK_TYPES = (
    "block",
    "line",
    "blob",
)


def mask_coverage(
    mask: np.ndarray,
) -> float:
    """Return the fraction of pixels covered by the missing-region mask."""
    return float(
        (
            mask > 0
        ).mean()
    )


def make_mask(
    kind: str,
    ratio: float,
    rng: np.random.Generator,
    size: int = 224,
    center: tuple[int, int] | None = None,
) -> np.ndarray:
    """Create a synthetic missing-region mask.

    Args:
        kind:
            Mask shape. Must be one of:
            - "block"
            - "line"
            - "blob"

        ratio:
            Approximate fraction of the image to remove.
            Example: 0.20 means roughly 20%.

        rng:
            NumPy random-number generator.

        size:
            Output image size. SteelVision uses 224 × 224
            for ResNet-50 classification.

        center:
            Optional (x, y) location around which the mask
            should be generated.

            If None, a random location is selected.

            This can later be used for:
            - random placement
            - high-attention placement
            - low-attention placement

            Attention-based placement comes from Grad-CAM and
            must not be interpreted as physical defect ground truth.

    Returns:
        A uint8 mask of shape (size, size), where:
            255 = missing region
              0 = retained image
    """
    if kind not in MASK_TYPES:
        raise ValueError(
            f"Unknown mask kind '{kind}'. "
            f"Choose from {MASK_TYPES}."
        )

    if not 0.0 < ratio < 0.95:
        raise ValueError(
            "ratio must be between 0 and 0.95."
        )

    height = size
    width = size

    if center is None:
        center = (
            int(
                rng.integers(
                    0,
                    width,
                )
            ),
            int(
                rng.integers(
                    0,
                    height,
                )
            ),
        )

    center_x, center_y = center

    mask = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.uint8,
    )

    if kind == "block":
        _draw_block_mask(
            mask=mask,
            ratio=ratio,
            rng=rng,
            center_x=center_x,
            center_y=center_y,
        )

    elif kind == "line":
        _draw_line_mask(
            mask=mask,
            ratio=ratio,
            rng=rng,
            center_x=center_x,
            center_y=center_y,
            size=size,
        )

    elif kind == "blob":
        _draw_blob_mask(
            mask=mask,
            ratio=ratio,
            rng=rng,
            center_x=center_x,
            center_y=center_y,
            size=size,
        )

    return mask


def _draw_block_mask(
    mask: np.ndarray,
    ratio: float,
    rng: np.random.Generator,
    center_x: int,
    center_y: int,
) -> None:
    """Draw one rectangular missing region."""
    height, width = mask.shape

    aspect_ratio = float(
        rng.uniform(
            0.6,
            1.6,
        )
    )

    target_area = (
        ratio
        * height
        * width
    )

    block_width = int(
        min(
            width,
            max(
                1,
                round(
                    math.sqrt(
                        target_area
                        * aspect_ratio
                    )
                ),
            ),
        )
    )

    block_height = int(
        min(
            height,
            max(
                1,
                round(
                    target_area
                    / block_width
                ),
            ),
        )
    )

    x0 = int(
        np.clip(
            center_x
            - block_width // 2,
            0,
            width
            - block_width,
        )
    )

    y0 = int(
        np.clip(
            center_y
            - block_height // 2,
            0,
            height
            - block_height,
        )
    )

    mask[
        y0:y0 + block_height,
        x0:x0 + block_width,
    ] = 255


def _draw_line_mask(
    mask: np.ndarray,
    ratio: float,
    rng: np.random.Generator,
    center_x: int,
    center_y: int,
    size: int,
) -> None:
    """Draw thick scratch-like missing regions."""
    thickness = max(
        3,
        int(
            size
            * (
                0.02
                + 0.06
                * ratio
            )
        ),
    )

    base_angle = float(
        rng.uniform(
            0,
            math.pi,
        )
    )

    for step in range(
        1,
        801,
    ):
        if (
            mask_coverage(
                mask
            )
            >= ratio
        ):
            break

        spread = (
            size
            * 0.06
            * (
                1
                + step / 20
            )
        )

        angle = (
            base_angle
            + rng.normal(
                0,
                0.25,
            )
        )

        length = (
            rng.uniform(
                0.5,
                1.0,
            )
            * size
        )

        offset_x = (
            center_x
            + rng.normal(
                0,
                spread,
            )
        )

        offset_y = (
            center_y
            + rng.normal(
                0,
                spread,
            )
        )

        delta_x = (
            math.cos(
                angle
            )
            * length
            / 2
        )

        delta_y = (
            math.sin(
                angle
            )
            * length
            / 2
        )

        point_1 = (
            int(
                offset_x
                - delta_x
            ),
            int(
                offset_y
                - delta_y
            ),
        )

        point_2 = (
            int(
                offset_x
                + delta_x
            ),
            int(
                offset_y
                + delta_y
            ),
        )

        cv2.line(
            mask,
            point_1,
            point_2,
            255,
            thickness,
        )


def _draw_blob_mask(
    mask: np.ndarray,
    ratio: float,
    rng: np.random.Generator,
    center_x: int,
    center_y: int,
    size: int,
) -> None:
    """Draw irregular blob-like missing regions."""
    for step in range(
        1,
        801,
    ):
        if (
            mask_coverage(
                mask
            )
            >= ratio
        ):
            break

        spread = (
            size
            * 0.05
            * (
                1
                + step / 15
            )
        )

        offset_x = int(
            center_x
            + rng.normal(
                0,
                spread,
            )
        )

        offset_y = int(
            center_y
            + rng.normal(
                0,
                spread,
            )
        )

        axis_x = int(
            size
            * rng.uniform(
                0.04,
                0.10
                + 0.10
                * ratio,
            )
        )

        axis_y = int(
            size
            * rng.uniform(
                0.04,
                0.10
                + 0.10
                * ratio,
            )
        )

        rotation = float(
            rng.uniform(
                0,
                180,
            )
        )

        cv2.ellipse(
            mask,
            (
                offset_x,
                offset_y,
            ),
            (
                max(
                    axis_x,
                    2,
                ),
                max(
                    axis_y,
                    2,
                ),
            ),
            rotation,
            0,
            360,
            255,
            -1,
        )


def random_training_mask(
    rng: np.random.Generator,
    size: int = 224,
) -> np.ndarray:
    """Generate a random mask for U-Net inpainting training.

    Training masks use:
    - a random mask type
    - random coverage from 5% to 50%
    - random spatial placement

    This exposes the U-Net to different missing-region patterns.
    """
    kind = MASK_TYPES[
        int(
            rng.integers(
                0,
                len(
                    MASK_TYPES
                ),
            )
        )
    ]

    ratio = float(
        rng.uniform(
            0.05,
            0.50,
        )
    )

    return make_mask(
        kind=kind,
        ratio=ratio,
        rng=rng,
        size=size,
    )